# src/train.py
import os, json, argparse, copy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ----------------------------
# Data & Transforms
# ----------------------------
def build_transforms(img_size=224):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tfms, eval_tfms


def make_loaders(data_dir, batch_size=32, img_size=224, num_workers=2, weighted_sampler=False):
    data_dir = Path(data_dir)
    train_tfms, eval_tfms = build_transforms(img_size)

    train_ds = datasets.ImageFolder(data_dir/"train", transform=train_tfms)
    val_ds   = datasets.ImageFolder(data_dir/"val",   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(data_dir/"test",  transform=eval_tfms)

    if weighted_sampler:
        counts = [0]*len(train_ds.classes)
        for _, y in train_ds.samples:
            counts[y] += 1
        total = sum(counts)
        cls_w = [total/c for c in counts]
        samp_w = [cls_w[y] for _, y in train_ds.samples]
        sampler = WeightedRandomSampler(samp_w, num_samples=len(samp_w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader  = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, train_ds.classes


# ----------------------------
# Model
# ----------------------------
def build_model(num_classes):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    return model


# ----------------------------
# Train / Eval
# ----------------------------
def train_one_epoch(model, loader, criterion, optim, device):
    model.train()
    run_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optim.step()

        run_loss += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return run_loss/total, correct/total


@torch.no_grad()
def evaluate(model, loader, criterion, device, tta=1):
    model.eval()
    run_loss, correct, total = 0.0, 0, 0
    all_p, all_t = [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)

        # ---- TTA (Test-Time Augmentation) ----
        if tta > 1:
            outs = []
            for _ in range(tta):
                x_tta = x.clone()
                if torch.rand(1).item() < 0.5:
                    x_tta = torch.flip(x_tta, dims=[-1])  # yatay çevir
                outs.append(model(x_tta))
            out = torch.stack(outs, dim=0).mean(0)  # [B,C]
        else:
            out = model(x)
        # --------------------------------------

        loss = criterion(out, y)
        run_loss += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        all_p.append(pred.cpu().numpy())
        all_t.append(y.cpu().numpy())

    loss = run_loss/total
    acc  = correct/total
    preds = np.concatenate(all_p)
    targets = np.concatenate(all_t)
    return loss, acc, preds, targets


def plot_confusion(cm, class_names, outpath):
    fig = plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix"); plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    thresh = cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha='center', color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel("True"); plt.xlabel("Pred")
    plt.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--save_dir", default="outputs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--weighted_sampler", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--tta", type=int, default=1, help=">1 ise val/test için TTA tekrar sayısı (örn. 4)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    tr, va, te, classes = make_loaders(
        args.data_dir, args.batch_size, args.img_size, weighted_sampler=args.weighted_sampler
    )
    model = build_model(len(classes)).to(device)

    # Loss (label smoothing + opsiyonel class weights)
    if args.use_class_weights:
        counts = [0]*len(classes)
        base_ds = tr.dataset.dataset if hasattr(tr.dataset, "dataset") else tr.dataset
        for _, y in base_ds.samples:
            counts[y] += 1
        total = sum(counts)
        weights = torch.tensor([total/c for c in counts], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, tr, criterion, optim, device)
        va_loss, va_acc, va_p, va_t = evaluate(model, va, criterion, device, tta=args.tta)
        rep = classification_report(va_t, va_p, output_dict=True, zero_division=0)
        macro_f1 = rep["macro avg"]["f1-score"]

        print(f"Epoch {epoch}: tr_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f} macroF1={macro_f1:.4f}")
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"state_dict": best_state, "classes": classes}, os.path.join(args.save_dir, "best.pt"))
        sched.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_acc, te_p, te_t = evaluate(model, te, criterion, device, tta=args.tta)
    report = classification_report(te_t, te_p, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(te_t, te_p)

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump({
            "val_best_macro_f1": best_f1,
            "test_loss": float(te_loss),
            "test_acc": float(te_acc),
            "classification_report": report
        }, f, indent=2)

    plot_confusion(cm, classes, os.path.join(args.save_dir, "confusion_matrix.png"))
    print("Bitti. En iyi model ve metrikler outputs klasörüne yazıldı.")


if __name__ == "__main__":
    main()
