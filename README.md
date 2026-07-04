# 🕵️‍♂️ Metin Tabanlı Karanlık Tasarım (Dark Pattern) Tespiti

Bu proje, web sitelerinde ve uygulamalarda kullanıcıları manipüle etmek, yanlış yönlendirmek veya istemedikleri kararlar almaya zorlamak amacıyla tasarlanmış **Karanlık Tasarımları (Dark Patterns)** derin öğrenme ve doğal dil işleme (NLP) teknikleri ile tespit etmeyi amaçlar. 

Kullanıcı arayüzü metinleri analiz edilerek "Normal" (0) veya "Dark Pattern" (1) olarak sınıflandırılır. Proje kapsamında **Gelişmiş Çift Yönlü LSTM (Bidirectional LSTM)** mimarisi ile günümüzün modern dil modellerinden **BERT (bert-base-uncased)** karşılaştırmalı olarak uçtan uca bir ardışık düzen (pipeline) ile eğitilmiş ve test edilmiştir.

---

## 📂 Proje Dizin Yapısı

Projenin endüstri standartlarına uygun dizin yapısı aşağıdaki gibidir:

```text
dark-pattern-detection/
├── dataset/                # Veri setlerinin bulunduğu klasör (.gitignore ile gizlenmiştir)
│   ├── train.csv           
│   ├── val.csv             
│   └── test.csv            
├── train.py                # Modellerin eğitimi ve overfitting (aşırı öğrenme) analizi
├── validation.py           # Eğitilmiş modellerin doğrulanması ve hiperparametre testleri
├── test.py                 # Görülmemiş test verisi ile nihai değerlendirme ve ROC-AUC
├── sonuc_train/            # Eğitim süreci çıktıları (Grafikler ve model ağırlıkları)
├── sonuc_val/              # Doğrulama aşaması çıktıları (Hata matrisi ve raporlar)
├── sonuc_test/             # Nihai test çıktıları (Karşılaştırmalı rapor, ROC ve CM grafikleri)
├── .gitignore              # GitHub'a yüklenmeyecek sistem/veri dosyalarının listesi
├── requirements.txt        # Projenin çalışması için gerekli kütüphaneler listesi
└── README.md               # Proje açıklama ve kullanım kılavuzu