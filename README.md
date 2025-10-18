Beyin Tümörü Sınıflandırma (EfficientNet-B0 + Grad-CAM)

MRI görüntülerinden Glioma, Meningioma, Pituilary, Tümör yok sınıflarını ayırt eden bir görüntü sınıflandırma projesi.
Model, ImageNet ön-eğitimli EfficientNet-B0 üzerine transfer learning ile eğitilir. Arayüz için Streamlit kullanılır ve Grad-CAM ısıl haritalarıyla görselleştirme yapılır.

⚠️ Uyarı: Bu repo eğitim/araştırma/portföy amaçlıdır. Çıktılar tıbbi kullanım için uygun değildir ve klinik kararlarda kesinlikle kullanılmamalıdır.

📦 Veri Seti

Kaynak (Kaggle): Brain Tumor MRI Dataset
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Veri setini indirdikten sonra bu repodaki src/split_data.py ile train/val/test bölmelerini oluşturacağız.

🗂️ Proje Yapısı
.
├─ src/
│  ├─ train.py            # Eğitim betiği (temiz ve TR yorumlu)
│  └─ split_data.py       # Veri setini train/val/test olarak ayırma
├─ streamlit_app_gradcam.py  # Arayüz + Grad-CAM
├─ requirements.txt
├─ README.md
└─ (izlemeye alınmayanlar)
   ├─ data/               # split sonrası veriler (gitignore)
   └─ outputs/            # eğitim çıktıları (best.pt, metrics.json, vs.)


Not: data/ ve outputs/ .gitignore ile takip dışıdır (büyük dosyalar GitHub’a yüklenmez).

🚀 Hızlı Başlangıç

Python için 3.11/3.12 önerilir (PyTorch wheel uyumluluğu açısından).

1) Depoyu indir ve bağımlılıkları kur
git clone https://github.com/<kullanici>/brain-tumor-classifier.git
cd brain-tumor-classifier
pip install -r requirements.txt

2) Veriyi hazırla (Kaggle’dan indir → split et)

Kaggle dataset klasörünü bir yere aç (ör: ./raw_data).

Aşağıdaki komutla train/val/test klasör yapısını oluştur:

# örnek: %70 train / %15 val / %15 test
python src/split_data.py --data_dir ./raw_data --train_ratio 0.7 --val_ratio 0.15 --seed 42


Split sonrası hedef yapı:

data/
  train/<sınıf>/*.jpg
  val/<sınıf>/*.jpg
  test/<sınıf>/*.jpg


Zaten bu yapıya sahipsen split_data.py adımını atlayabilirsin.

3) Modeli eğit
python src/train.py \
  --data_dir ./data \
  --epochs 10 \
  --batch_size 16 \
  --img_size 224 \
  --use_class_weights \
  --save_dir ./outputs


Seçili argümanlar:

--use_class_weights dengesiz sınıfları dengeleyerek eğitir.

--label_smoothing 0.05 (varsayılan) aşırı öğrenmeyi azaltmaya yardımcı olur.

--tta 1 (varsayılan), validasyon/testte basit TTA için --tta 4 deneyebilirsin.

Tam argüman listesi için: python src/train.py -h

Eğitim çıktıları (./outputs)

best.pt → en iyi validasyon macro-F1’e ait ağırlıklar

metrics.json → metrikler (test_acc, val_best_macro_f1, sınıf bazlı rapor)

confusion_matrix.png → karışıklık matrisi görseli

4) Streamlit arayüzü + Grad-CAM
python -m streamlit run ./streamlit_app_gradcam.py


Arayüzde:

Görsel yükle → sınıf olasılıklarını gör

“Tümör yok” olasılığına göre basit risk göstergesi (kırmızı/yeşil)

Seçili sınıf için Grad-CAM ısıl haritası (saydamlık ayarıyla)

Sonuçları CSV ve Grad-CAM görselini indir butonları

Arayüz, ağırlıkları outputs/best.pt yolundan yükler. Dosya yoksa önce eğitmelisin ya da önceden eğitilmiş bir best.pt sağlamalısın.

⚙️ train.py — Öne Çıkan Ayarlar

Model: EfficientNet-B0 (ImageNet ön-eğitimli)

Augmentasyon: Resize, yatay çevirme, hafif rotasyon

Kayıp: CrossEntropy + label smoothing (varsayılan 0.05)

Dengesizlik: --use_class_weights veya --weighted_sampler

TTA: Validasyon/Test için --tta N (örn. 4)

Optimizasyon: AdamW + CosineAnnealingLR

Komut örnekleri:

# Hızlı deneme (1 epoch)
python src/train.py --data_dir ./data --epochs 1 --batch_size 8 --save_dir ./outputs

# Daha güçlü eğitim
python src/train.py --data_dir ./data --epochs 15 --batch_size 16 --img_size 256 --use_class_weights --tta 4 --save_dir ./outputs

🌐 Yayına Alma (Ücretsiz)

Streamlit Community Cloud ile birkaç adımda deploy:

Repo public olsun.

https://streamlit.io/cloud
 → GitHub hesabını bağla → New app

Main file path: streamlit_app_gradcam.py

Ağırlık dosyası (best.pt):

<100MB ise repo içinde weights/best.pt olarak tutup kodda WEIGHTS_PATH = "weights/best.pt" yapabilirsin.

Büyükse: GitHub Releases’a yükleyip uygulama açılışında URL’den indir; dosyayı weights/best.pt olarak cache’e yaz (istersen indirme kod bloğunu ekleyebilirim).

🧪 Sonuçları İzleme

Eğitim sonrası outputs/metrics.json dosyasından temel metriklere bak:

# Linux/macOS
cat outputs/metrics.json

# PowerShell
Get-Content .\outputs\metrics.json | ConvertFrom-Json


test_acc → test doğruluğu

val_best_macro_f1 → en iyi validasyon macro-F1

classification_report → sınıf bazlı precision/recall/F1

Karışıklık matrisi: outputs/confusion_matrix.png

🛠️ Sorun Giderme

PyTorch kurulumu Windows’ta Python 3.14’te sorun çıkarabilir. 3.11/3.12 önerilir.

best.pt bulunamadı → önce modeli eğit veya dosyayı doğru konuma yerleştir.

Yavaşlık/VRAM hatası → --batch_size düşür, --img_size 224 tut.

Sınıf adları streamlit_app_gradcam.py içinde arayüz için Türkçeleştirildi; modelin iç sınıf sırası checkpoint’ten okunur.

📜 Lisans ve Teşekkür

Veri seti: Kaggle’daki “Brain Tumor MRI Dataset” (Masoud Nickparvar) için orijinal lisans/koşullara uyun.

Kod için bir lisans seçmek istersen LICENSE dosyası ekleyebilirsin (ör. MIT).

🔗 Atıf (Citation)

Eğer bu repoyu akademik bir çalışmada kullanırsan, Kaggle veri seti sayfasını ve bu repoyu referans göstermen nezaket olur.

Masoud Nickparvar. Brain Tumor MRI Dataset. Kaggle.
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

✍️ İletişim

Sorular ve katkılar için Pull Request veya Issue açabilirsin.
İyi çalışmalar!
