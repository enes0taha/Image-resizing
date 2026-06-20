import os
import io
import requests
from PIL import Image

print("=== GERÇEK FOTOĞRAF VERİ SETİ İNDİRİLİYOR (50 ADET) ===")

if not os.path.exists("images"):
    os.makedirs("images")


base_url = "https://images.unsplash.com/photo-"

for i in range(1, 51):
    # Çözünürlük dağılımı 
    if i <= 20:
        w, h = 640 + (i * 20), 480 + (i * 15)  
        label = "VGA Sınıfı"
    elif i <= 40:
        w, h = 1280 + ((i - 20) * 32), 720 + ((i - 20) * 18)  
        label = "Full HD Sınıfı"
    else:
        w, h = 3840, 2160  
        label = "4K Sınıfı"

    img_url = f"https://source.unsplash.com/random/{w}x{h}?nature,city,texture&sig={i}"
    
    if i % 2 == 0:
        img_url = f"https://loremflickr.com/{w}/{h}/nature,landscape?lock={i}"
    else:
        img_url = f"https://loremflickr.com/{w}/{h}/abstract,texture?lock={i}"

    try:
        response = requests.get(img_url, timeout=15)
        if response.status_code == 200:
  
            img_bytes = io.BytesIO(response.content)
            img = Image.open(img_bytes).convert("RGB")
            
            
            img = img.resize((w, h))
            
            
            filename = f"images/{i}.ppm"
            img.save(filename, format="PPM")
            
            print(f"[RESİM {i:02d}] {filename} başarıyla indirildi ({w}x{h}) - {label}")
        else:
            print(f"[HATA] Resim {i} indirilemedi. Sunucu hatası.")
    except Exception as e:
        print(f"[HATA] Resim {i} indirilirken bağlantı sorunu oluştu: {e}")

print("\n Tebrikler! 50 adet gerçek doğa, manzara ve nesne fotoğrafı '.ppm' olarak 'images/' klasörüne dolduruldu.")