# Beyin Tümörü Sınıflandırma (EfficientNet-B0 + Grad-CAM)

MRI görüntülerinden **Glioma**, **Meningioma**, **Pituilary**, **Tümör yok** sınıflarını ayırt eden görüntü sınıflandırma projesi.  
Model: ImageNet ön-eğitimli **EfficientNet-B0** (transfer learning) · Arayüz: **Streamlit** · Görselleştirme: **Grad-CAM**

#📈 Model Performansı

Model, 10 epoch boyunca eğitildiğinde doğruluk oranı yaklaşık %97.4, macro-F1 skoru ise 0.96 seviyesine ulaşmıştır. Veri dengesizliğini azaltmak için sınıf ağırlıkları kullanıldığında özellikle “No Tumor” ve “Meningioma” sınıflarında hatalar belirgin şekilde azalmıştır. Modelin performansı kullanılan epoch sayısı, veri artırma çeşitliliği ve çözünürlük parametrelerine bağlı olarak değişebilir.


<img width="718" height="820" alt="resim" src="https://github.com/user-attachments/assets/fd24588b-c03c-445e-a512-ddffdf2bdc10" />


> Bu proje eğitim/araştırma/portföy amaçlıdır; çıktılar tıbbi kullanım için uygun değildir.

## Veri Seti
Kaggle: **Brain Tumor Dataset — Ishans24**  
<https://www.kaggle.com/datasets/ishans24/brain-tumor-dataset>

## Kurulum
Önerilen Python: **3.11** veya **3.12**
```bash
git clone https://github.com/fatihaydost/brain-tumor-classifier.git
cd brain-tumor-classifier
pip install -r requirements.txt
```

## Veriyi Hazırla
Ham veriyi klasöre (`./data`) çıkar ve `train/val/test` böl:
```bash
python src/split_data.py --data_dir ./raw_data --train_ratio 0.7 --val_ratio 0.15 --seed 42
```

## Eğitimi Çalıştır
```bash
python src/train.py   --data_dir ./data   --epochs 10   --batch_size 16   --img_size 224   --use_class_weights   --save_dir ./outputs
```

Eğitim çıktıları (`./outputs`):
- `best.pt` — en iyi ağırlıklar  
- `metrics.json` — metrikler (örn. `test_acc`, `val_best_macro_f1`)  
- `confusion_matrix.png` — karışıklık matrisi

Hızlı deneme:
```bash
python src/train.py --data_dir ./data --epochs 1 --batch_size 8 --save_dir ./outputs
```

## Arayüz (Streamlit) + Grad-CAM
```bash
python -m streamlit run ./streamlit_app_gradcam.py
```
- Görsel yükledikten sonra sınıf olasılıkları ve basit risk göstergesi gösterilir.  
- Seçilen sınıf için Grad-CAM ısıl haritası üretilir.

> Not: Arayüz ağırlıkları `outputs/best.pt` yolundan yükler. Dosya yoksa önce eğitin.

## Proje Yapısı
```
├─ data
│  ├─ glioma
│  ├─ meningioma
│  ├─ no_tumor
│  ├─ pituitary
│  ├─ test
│  ├─ train
│  └─ val
├─ output
│  ├─ best.pt
│  ├─ confusion_matrix.png
│  └─ metrics.json
├─ src/
│  ├─ train.py
│  └─ split_data.py
├─ streamlit_app_gradcam.py
├─ requirements.txt
├─ README.md

```

## Notlar
- `data/` büyük dosyalar içerdiği için repoda bulunmuyor kaggle linkinden indirebilirsiniz.  

