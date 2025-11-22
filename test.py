import cv2
import numpy as np
import tensorflow as tf  # Tüm kütüphaneyi 'tf' olarak çağırıyoruz (En garantisi)

# --- 1. MODELİ YÜKLEME ---
print("Model yükleniyor...")

try:
    # 'compile=False' diyerek sadece tahmin yapacağımızı, eğitime devam etmeyeceğimizi belirtiyoruz.
    # Bu, optimizer hatalarını engeller.
    model = tf.keras.models.load_model('my_traffic_model.h5', compile=False)
    print("Model başarıyla yüklendi! Tahmin yapılıyor...")
except OSError:
    print("HATA: 'my_traffic_model.h5' dosyası bulunamadı! Dosyanın bu kodla aynı klasörde olduğundan emin ol.")
    exit()

# --- 2. SINIF İSİMLERİ ---
def getClassName(classNo):
    classes = { 
        0:'Hız Limiti (20km/s)', 
        1:'Hız Limiti (30km/s)', 
        2:'Hız Limiti (50km/s)', 
        3:'Hız Limiti (60km/s)', 
        4:'Hız Limiti (70km/s)', 
        5:'Hız Limiti (80km/s)', 
        6:'Hız Limiti Bitişi (80km/s)', 
        7:'Hız Limiti (100km/s)', 
        8:'Hız Limiti (120km/s)', 
        9:'Geçiş Yok', 
        10:'Kamyon Giremez', 
        11:'Öncelikli Yol', 
        12:'Ana Yol', 
        13:'Yol Ver', 
        14:'DUR (Stop)', 
        15:'Taşıt Giremez', 
        16:'Kamyon Giremez (3.5 Ton Üstü)', 
        17:'Giriş Yasak', 
        18:'Dikkat', 
        19:'Sola Tehlikeli Viraj', 
        20:'Sağa Tehlikeli Viraj', 
        21:'Virajlı Yol', 
        22:'Engebeli Yol', 
        23:'Kaygan Yol', 
        24:'Yol Daralması (Sağdan)', 
        25:'Yol Çalışması', 
        26:'Trafik Işıkları', 
        27:'Yaya Geçidi', 
        28:'Çocuklar Geçebilir', 
        29:'Bisiklet Geçebilir', 
        30:'Buzlanma Riski', 
        31:'Vahşi Hayvan Çıkabilir', 
        32:'Hız Sınırı Sonu', 
        33:'Sağa Dönün', 
        34:'Sola Dönün', 
        35:'İleri ve Sağa', 
        36:'İleri ve Sola', 
        37:'Sola Gitmek Mecburi', 
        38:'Sağa Gitmek Mecburi', 
        39:'Solundan Geçiniz', 
        40:'Dönel Kavşak', 
        41:'Geçiş Yasağı Sonu', 
        42:'Kamyonlar İçin Geçiş Yasağı Sonu' 
    }
    return classes.get(classNo, "Tanımsız Sınıf")

# --- 3. RESMİ İŞLEME VE TAHMİN ---

# Buraya test edeceğin resmin adını yaz (Dosya uzantısına dikkat: .jpg / .png)
path_to_image = "test_resmi.jpg" 

# Resmi oku
imgOriginal = cv2.imread(path_to_image)

if imgOriginal is None:
    print(f"HATA: '{path_to_image}' dosyası okunamadı. İsmi doğru yazdığına emin misin?")
else:
    # Resmi modele uygun hale getir
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0 # Normalize et
    img = img.reshape(1, 32, 32, 3) # Modelin beklediği boyut: (1 adet, 32 boy, 32 en, 3 kanal)

    # Tahmin Yap
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.amax(predictions)

    # Sonucu Ekrana Bas
    print("------------------------------------------------")
    print(f"Tabela: {getClassName(classIndex)}")
    print(f"Eminlik Oranı: %{probabilityValue*100:.2f}")
    print("------------------------------------------------")
    
    # Resmi Ekranda Göster (Pencere açılır, kapatmak için bir tuşa bas)
    cv2.imshow("Sonuc", imgOriginal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()