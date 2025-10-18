"""
Streamlit arayüzü: Tek görsel veya çoklu görsel yükleyip
- sınıf olasılıklarını gösterir,
- 'Tümör yok' sınıfına bakarak basit bir risk göstergesi üretir,
- seçili görsel için Grad-CAM ısı haritası oluşturur.

Not: Bu uygulama tıbbi kullanım için değildir; eğitim/portföy amaçlıdır.
"""

from __future__ import annotations

import io
import os
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


# ---------------------------
# Sabitler (UI + model yolu)
# ---------------------------
WEIGHTS_PATH = "outputs/best.pt"
DEFAULT_CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]

# Arayüzde gösterilecek Türkçe isimler (modelin iç sınıf isimleri değişmez)
FRIENDLY_MAP: Dict[str, str] = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "no_tumor": "Tümör yok",
    "pituitary": "Pituilary",
}

IMG_SIZE = 224
RISK_THRESHOLD = 0.60  # 0.60 ve üstü: “tümör olasılığı yüksek” uyarısı


# ---------------------------
# Yardımcı fonksiyonlar
# ---------------------------
def build_preprocess(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Modelin beklediği ön-işleme adımları."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@st.cache_resource
def load_model(weights_path: str) -> Tuple[torch.nn.Module, List[str]]:
    """Ağırlıkları ve sınıf isimlerini yükle (cache'li)."""
    ckpt = torch.load(weights_path, map_location="cpu")
    classes = ckpt.get("classes", DEFAULT_CLASSES)

    model = models.efficientnet_b0(weights=None)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes


def to_friendly_names(classes: List[str]) -> List[str]:
    """Arayüzde gösterilecek Türkçe sınıf isimleri."""
    return [FRIENDLY_MAP.get(c, c) for c in classes]


def get_last_conv_layer(model: nn.Module) -> nn.Module:
    """Grad-CAM için hedef katman (EfficientNetB0: features)."""
    return model.features


def grad_cam(model: nn.Module, target_layer: nn.Module, x: torch.Tensor, class_idx: int | None = None):
    """
    Temel Grad-CAM: hedef katmandan aktivasyon/gradient alıp ısı haritası üretir.
    x: (1, 3, H, W)
    Dönen: (cam[numpy], kullanılan_sınıf_indeksi)
    """
    activations, gradients = [], []

    def fwd_hook(_m, _inp, out):  # ileri geçişte aktivasyonları yakala
        activations.append(out.detach())

    def bwd_hook(_m, _grad_in, grad_out):  # geri yayılımda gradientleri yakala
        gradients.append(grad_out[0].detach())

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    logits = model(x)
    if class_idx is None:
        class_idx = int(torch.argmax(logits, dim=1).item())

    score = logits[:, class_idx]
    model.zero_grad(set_to_none=True)
    score.backward()

    acts = activations[0][0]           # [C, H, W]
    grads = gradients[0][0]            # [C, H, W]
    weights = grads.mean(dim=(1, 2))   # [C]
    cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))  # [H, W]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    h1.remove()
    h2.remove()
    return cam.cpu().numpy(), class_idx


def overlay_cam(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.35) -> Image.Image:
    """Isı haritasını orijinal görüntünün üzerine bindir."""
    import matplotlib.cm as cm

    cam_uint8 = (cam * 255).astype(np.uint8)
    cam_img = Image.fromarray(cam_uint8).resize(pil_img.size, resample=Image.BILINEAR)
    heat = np.asarray(cam_img) / 255.0
    heat_rgb = (cm.get_cmap("jet")(heat)[..., :3] * 255).astype(np.uint8)

    base = np.asarray(pil_img.convert("RGB")).astype(np.float32)
    over = (alpha * heat_rgb.astype(np.float32) + (1 - alpha) * base).clip(0, 255).astype(np.uint8)
    return Image.fromarray(over)


def find_no_tumor_index(classes: List[str]) -> int:
    """'Tümör yok' sınıfının indeksini (checkpoint sınıf adlarına göre) bul."""
    for i, c in enumerate(classes):
        if c.lower().replace(" ", "").replace("-", "_") in ("no_tumor", "notumor"):
            return i
    return len(classes) - 1  # bulunamazsa güvenlik amaçlı son sınıfı kullan


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Beyin Tümörü Sınıflandırma + Grad-CAM", layout="centered")
st.title("🧠 Beyin Tümörü Sınıflandırma + Grad-CAM")
st.caption("Bu uygulama tıbbi kullanım için değildir — eğitim/portföy amaçlıdır.")

# Modeli yükle
try:
    model, classes = load_model(WEIGHTS_PATH)
except Exception as e:
    st.error(f"Model yüklenemedi: {e}")
    st.stop()

friendly_names = to_friendly_names(classes)
preproc = build_preprocess()

# Dosya yükleme
files = st.file_uploader(
    "MRI görüntüsü yükle (tek ya da birden fazla dosya seçebilirsin)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if not files:
    st.info("Devam etmek için en az bir görüntü yükle.")
    st.stop()

# Toplu tahmin (tablo için)
records = []
display_items = []  # (name, pil_img, tens, probs)
for f in files:
    img = Image.open(f).convert("RGB")
    x = preproc(img).unsqueeze(0)  # (1,3,H,W)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy().squeeze()

    pred_idx = int(np.argmax(probs))
    records.append(
        {
            "dosya": f.name,
            "tahmin_sınıf": friendly_names[pred_idx],
            "tahmin_olasılık": float(probs[pred_idx]),
            **{f"p_{friendly_names[i]}": float(probs[i]) for i in range(len(friendly_names))},
        }
    )
    display_items.append((f.name, img, x, probs))

# Özet tablo
import pandas as pd

df = pd.DataFrame.from_records(records)
st.subheader("Toplu Sonuçlar")
st.dataframe(df, use_container_width=True)

# İndirilebilir CSV
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Sonuçları CSV olarak indir", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

# Risk göstergesi ve Grad-CAM alanı (tek görsel seçimi)
st.markdown("---")
st.subheader("Detay Görünüm + Grad-CAM")

# Görselle çalışılacak hedefi seç
target_name = st.selectbox("Görüntü seç", options=[name for name, *_ in display_items])
name_to_item = {name: (img, x, probs) for name, img, x, probs in display_items}
img, x, probs = name_to_item[target_name]

# Basit risk: 1 - P('Tümör yok')
idx_no_tumor = find_no_tumor_index(classes)
p_no_tumor = float(probs[idx_no_tumor])
p_tumor = float(1.0 - p_no_tumor)

if p_tumor >= RISK_THRESHOLD:
    st.error(f"🔴 Tümör olasılığı **yüksek**: {p_tumor:.2%}")
else:
    st.success(f"🟢 Tümör olasılığı **düşük**: {p_tumor:.2%}")

# Olasılık dökümü
with st.expander("Sınıf olasılıklarını göster"):
    for i, p in enumerate(probs):
        st.write(f"- **{friendly_names[i]}**: {p:.4f}")

# Isı haritası saydamlığı (dosya yüklendikten sonra, aşağıda)
alpha = st.slider("Isı haritası saydamlığı (α)", 0.0, 1.0, 0.35, 0.05)

# Hangi sınıfa göre CAM?
vis_choice = st.radio(
    "Isı haritası sınıfı",
    options=["En olası (otomatik)"] + [f"{i}: {friendly_names[i]}" for i in range(len(friendly_names))],
    index=0,
    horizontal=True,
)
class_for_cam = None if vis_choice.startswith("En olası") else int(vis_choice.split(":")[0])

# Grad-CAM üret ve göster
cam, used_idx = grad_cam(model, get_last_conv_layer(model), x, class_idx=class_for_cam)
overlay = overlay_cam(img, cam, alpha=alpha)

c1, c2 = st.columns(2)
with c1:
    st.image(img, caption="Orijinal", use_container_width=True)
with c2:
    st.image(overlay, caption=f"Grad-CAM (sınıf: {friendly_names[used_idx]})", use_container_width=True)

# İndirme
buf = io.BytesIO()
overlay.save(buf, format="PNG")
st.download_button(
    "Isı haritasını indir (PNG)",
    data=buf.getvalue(),
    file_name=f"gradcam_{target_name}_{friendly_names[used_idx]}.png",
    mime="image/png",
)
