# 🚦 Trafik İşaretleri Sınıflandırma Projesi (Traffic Sign Classification)

Bu proje, **Convolutional Neural Networks (CNN)** kullanarak trafik işaretlerini tanıyan bir yapay zeka uygulamasıdır. GTSRB veri seti kullanılarak eğitilmiş ve **%98** doğruluk oranına ulaşılmıştır.

## 🚀 Özellikler
* **Veri İşleme:** OpenCV kullanılarak görüntü boyutlandırma ve normalizasyon.
* **Model:** TensorFlow/Keras ile oluşturulmuş özel CNN mimarisi.
* **Başarı:** Test verisetinde %98 accuracy.
* **Kullanım:** Eğitilmiş `.h5` model dosyası ile anlık tahmin yapabilme.

## 🛠️ Kurulum

Gerekli kütüphaneleri yüklemek için:
```bash
pip install -r requirements.txt
```

💻 Kullanım
Modeli Eğitmek için: (Opsiyonel, hazır model mevcut)

```bash
python egitim.py
```
Tahmin Yapmak için: test_et.py dosyasındaki resim yolunu güncelleyin ve çalıştırın:


```bash

python test_et.py
```
## 📂 Veri Seti (Dataset)
Bu proje **GTSRB (German Traffic Sign Recognition Benchmark)** veri setini kullanmaktadır. GitHub repo boyutunu düşük tutmak için veri seti buraya yüklenmemiştir.

Projeyi çalıştırmak için:
1. Veri setini [Kaggle GTSRB Linki](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) adresinden indirin.
2. İndirdiğiniz dosyayı zipten çıkarın.
3. `Train` klasörünü projenin olduğu ana dizine kopyalayın. (Klasör yapısı `archive/Train` şeklinde olmalıdır).

📊 Sonuçlar
Model eğitim sürecinde Dropout katmanları kullanılarak overfitting engellenmiş ve yüksek başarım elde edilmiştir.
