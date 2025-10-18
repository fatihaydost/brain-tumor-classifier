# src/split_data.py
import argparse, os, shutil, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Kök klasör (içinde glioma/ meningioma/ ... var)")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true", help="Kopyalamak yerine taşır")
    args = ap.parse_args()

    random.seed(args.seed)
    data = Path(args.data_dir)
    classes = ["glioma","meningioma","pituitary","no_tumor"]

    # hedef klasörleri oluştur
    for split in ["train","val","test"]:
        for cls in classes:
            (data/split/cls).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        src_dir = data/cls
        imgs = [p for p in src_dir.iterdir() if p.is_file()]
        random.shuffle(imgs)

        n = len(imgs)
        n_train = int(n * args.train_ratio)
        n_val   = int(n * args.val_ratio)
        n_test  = n - n_train - n_val

        splits = [
            ("train", imgs[:n_train]),
            ("val",   imgs[n_train:n_train+n_val]),
            ("test",  imgs[n_train+n_val:]),
        ]

        for split_name, paths in splits:
            dst = data/split_name/cls
            for p in paths:
                if args.move:
                    shutil.move(str(p), dst/p.name)
                else:
                    shutil.copy2(str(p), dst/p.name)

    print("Tamam: train/val/test oluşturuldu.")

if __name__ == "__main__":
    main()
