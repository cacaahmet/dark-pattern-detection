# 🛍️ Dark Pattern Detection — E-Ticaret Metinlerinde Manipülasyon Analizi ve Tespiti

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/BERT-4285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/LSTM-8A2BE2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<p align="center">
  <b>Derin Öğrenme ve Doğal Dil İşleme (NLP) tabanlı, e-ticaret metinlerindeki manipülatif "Karanlık Desenleri" (Dark Patterns) yüksek doğrulukla tespit eden ikili sınıflandırma sistemi.</b>
</p>

<p align="center">
  <b>LSTM (BiLSTM + Max Pooling)</b> ve <b>BERT (bert-base-uncased)</b> mimarileri karşılaştırmalı olarak eğitilmiş, doğrulanmış ve test edilmiştir.
</p>

---

## 📑 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Dark Pattern Nedir?](#-dark-pattern-nedir)
- [Özellikler](#-özellikler)
- [Proje Mimarisi](#-proje-mimarisi)
  - [LSTM Sınıflandırıcı](#lstm-sınıflandırıcı)
  - [BERT Sınıflandırıcı](#bert-sınıflandırıcı)
- [Veri Seti](#-veri-seti)
- [Kurulum](#-kurulum)
- [Proje Yapısı](#-proje-yapısı)
- [Kullanım](#-kullanım)
  - [Eğitim (Training)](#1️⃣-eğitim-training)
  - [Doğrulama (Validation)](#2️⃣-doğrulama-validation)
  - [Test](#3️⃣-test)
- [Sonuçlar](#-sonuçlar)
- [Yöntemsel Detaylar](#-yöntemsel-detaylar)
- [Kullanım Alanları](#-kullanım-alanları)
- [Yol Haritası](#-yol-haritası)
- [Kaynakça](#-kaynakça)
- [Danışman ve Yazar](#-danışman-ve-yazar)
- [Lisans](#-lisans)

---

## 🎯 Proje Hakkında

E-ticaret platformlarının hızla yaygınlaşması, kullanıcıları kendi çıkarları aleyhine karar vermeye zorlayan manipülatif tasarım ve metin stratejilerinin ("Dark Patterns") artmasına neden olmuştur. Sınırlı stok, aciliyet hissi yaratma veya kullanıcıyı suçlu hissettirme gibi taktikler barındıran bu içerikler, tüketici otonomisini zedelemekte ve dijital güveni sarsmaktadır.

Bu proje, e-ticaret sitelerinde kullanılan manipülatif metinlerin **insan müdahalesine gerek kalmadan, yüksek doğrulukla** tespit edilebilmesi için Derin Öğrenme ve NLP tabanlı bir sınıflandırma sistemi geliştirmeyi amaçlamaktadır. Çalışma kapsamında dört farklı manipülasyon türünü (**Scarcity, Misdirection, Urgency, Social Proof**) içeren bir veri seti kullanılmış; metinsel dizilimleri analiz eden **LSTM** ve çift yönlü bağlamı derinden kavrayan **BERT** modelleri tasarlanıp eğitilmiştir.

Proje, Muş Alparslan Üniversitesi Mühendislik-Mimarlık Fakültesi Yazılım Mühendisliği Bölümü *Derin Öğrenme Dersi* kapsamında bir lisans dönem projesi olarak hazırlanmıştır.

> **Anahtar Kelimeler:** Doğal Dil İşleme, Derin Öğrenme, Dark Patterns, E-Ticaret Manipülasyonu

---

## 🕵️ Dark Pattern Nedir?

"Dark Pattern" (Karanlık Desen) kavramı ilk olarak 2010 yılında UX uzmanı Harry Brignull tarafından, kullanıcıları kendi çıkarları aleyhine hareket etmeye iten, dikkatlice kurgulanmış arayüz ve metin hilelerini tanımlamak için ortaya atılmıştır. Bu çalışmada, literatürdeki sınıflandırmalar temel alınarak dört ana manipülasyon kategorisi ele alınmıştır:

| Kategori | Açıklama | Örnek |
|---|---|---|
| 🔥 **Scarcity** (Kıtlık) | Stok/ürün miktarının sınırlı olduğu algısı yaratarak kaybetme korkusuyla satın almaya iter | *"ONLY 9 LEFT"* |
| 😳 **Misdirection** (Yanıltma / Confirmshaming) | Kullanıcının teklifi reddetmesini duygusal olarak zorlaştıran, suçluluk/utanç uyandıran metinler | *"No thanks, I'd rather pay more"* |
| ⏱️ **Urgency** (Aciliyet) | Geri sayım sayaçları ile rasyonel karar verme süresini azaltan, FOMO yaratan ifadeler | *"... ENDS IN 12h 13m 56s"* |
| 👥 **Social Proof** (Sosyal Kanıt) | Diğer kullanıcıların davranışlarını öne çıkararak "sürü psikolojisi" ile baskı oluşturan ifadeler | *"161 people have viewed this item"* |

---

## ✨ Özellikler

- 🧠 **İki farklı derin öğrenme mimarisi**: Sıfırdan eğitilen BiLSTM ve önceden eğitilmiş (pretrained) BERT
- 🔍 **Kapsamlı değerlendirme metrikleri**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
- 📊 **Otomatik görselleştirme**: Öğrenme eğrileri, confusion matrix ve ROC eğrisi grafikleri
- 📝 **Otomatik raporlama**: Her aşama (train / validation / test) için `.txt` formatında detaylı çıktı raporları
- ⚙️ **Uçtan uca pipeline**: Ayrı `train.py`, `validation.py` ve `test.py` betikleriyle net bir MLOps akışı
- 🛡️ **Overfitting kontrolü**: Dropout, Batch Normalization, AdamW ve `ReduceLROnPlateau` ile düzenlileştirme

---

## 🏗️ Proje Mimarisi

Ham metinler, `bert-base-uncased` tokenizer'ı ile tokenize edilerek sabit uzunluğa (`padding`/`truncation`) getirilir ve `DataLoader` yapılarıyla modele batch halinde beslenir.

### LSTM Sınıflandırıcı

İlk (vanilla) LSTM mimarisi, veri setini ezberleme eğilimi göstererek ~0.69 kayıp değerinde tıkanmıştır. Bu sorunu çözmek amacıyla mimari, "saf bir metin okuyucudan" **akıllı bir özellik çıkarıcıya** dönüştürülmüştür:

```
Input (input_ids)
      │
      ▼
Embedding (vocab=30522, dim=256, padding_idx=0)
      │
      ▼
Dropout (emb_drop)
      │
      ▼
2 Katmanlı Bidirectional LSTM (hidden_dim=128, dropout=0.4)
      │
      ▼
Global Max Pooling  →  "cümlenin neresinde olursa olsun en güçlü sinyali yakala"
      │
      ▼
BatchNorm1d(256)
      │
      ▼
Linear(256 → 64) → ReLU → Dropout(0.4)
      │
      ▼
Linear(64 → 1)  →  Sigmoid (BCEWithLogitsLoss içinde)
```

**Kritik mimari iyileştirmeler:**

| İyileştirme | Amaç |
|---|---|
| **BiLSTM** (`bidirectional=True`) | Metni hem soldan sağa hem sağdan sola okuyarak cümle başındaki manipülatif ifadelerin unutulmasını önler |
| **Global Max Pooling** | Kararı sadece son kelimeye (`hidden[-1]`) değil, tüm zaman adımlarının ürettiği en güçlü sinyale dayandırır |
| **Dropout (0.4)** | Ezberlemeyi (overfitting) önlemek için veri artırma etkisi yaratır |
| **Batch Normalization** | Katmanlar arası veri ölçeğini stabilize ederek gradyan patlaması/kaybolmasını engeller |
| **AdamW + ReduceLROnPlateau** | Ağırlık frenlemesi ve platoya ulaşıldığında dinamik öğrenme oranı düşürme |

### BERT Sınıflandırıcı

Sıfırdan eğitim yerine önceden eğitilmiş `bert-base-uncased` üzerine ince ayar (fine-tuning) yapılır:

```
Input (input_ids, attention_mask)
      │
      ▼
Pretrained BERT (bert-base-uncased)
      │
      ▼
[CLS] token → last_hidden_state[:, 0, :]  (768 boyutlu bağlamsal özet)
      │
      ▼
Linear(768 → 1)  →  Sigmoid (BCEWithLogitsLoss içinde)
```

`attention_mask`, modelin dolgu (padding) token'larını göz ardı edip dikkatini yalnızca gerçek kelimelere vermesini sağlar. Transformer'ın öz-dikkat (self-attention) mekanizması sayesinde model, kelimeler arası mesafeden bağımsız olarak aciliyet/suçluluk gibi psikolojik tonları çift yönlü analiz edebilir.

---

## 📊 Veri Seti

Proje kapsamında özel olarak derlenmiş **"Dark Pattern" veri seti** kullanılmıştır: **2362 satır × 4 sütun**.

| Sütun | Açıklama |
|---|---|
| `page_id` | Kaydın benzersiz kimlik numarası |
| `text` | E-ticaret platformundan alınmış ham metin (model girdisi) |
| `label` | İkili etiket: `0` = Normal, `1` = Dark Pattern |
| `Pattern Category` | Manipülasyon türü (Scarcity / Misdirection / Urgency / Social Proof) |

**Sınıf dağılımı:** 820 normal metin, 829 dark pattern içeren metin — dengeli bir dağılım (class imbalance riski düşük).

**Veri bölünmesi (train / validation / test):**

| Küme | Oran | Boyut |
|---|---|---|
| Train | %70 | (1649, 4) |
| Validation | %15 | (353, 4) |
| Test | %15 | (354, 4) |

> Veri seti bu repoda **yer almamaktadır**; her betik `dataset/train.csv`, `dataset/val.csv`, `dataset/test.csv` yollarını `text` ve `label` sütunlarını içerecek şekilde bekler.

---

## 📁 Proje Yapısı

```
dark-pattern-detection/
├── dataset/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── sonuc_train/              # train.py çıktıları (ağırlıklar + grafikler + rapor)
│   ├── lstm_model.pth
│   ├── bert_model.pth
│   ├── lstm_overfitting_curve.png
│   ├── bert_overfitting_curve.png
│   └── train_raporu.txt
├── sonuc_val/                 # validation.py çıktıları
│   ├── lstm_val_cm.png
│   ├── bert_val_cm.png
│   └── validation_raporu.txt
├── sonuc_test/                # test.py çıktıları
│   ├── lstm_test_cm.png
│   ├── bert_test_cm.png
│   ├── lstm_roc_curve.png
│   ├── bert_roc_curve.png
│   └── nihai_test_raporu.txt
├── train.py
├── validation.py
├── test.py
└── README.md
```

---
## 🚀 Kullanım

### 1️⃣ Eğitim (Training)
```bash
python train.py
```
Bu çalışmada:
- `dataset/train.csv` veri seti kullanılarak model eğitimi gerçekleştirilmiştir.
- LSTM (5 epoch, `lr=1e-3`, AdamW + ReduceLROnPlateau) ve BERT (3 epoch, `lr=2e-5`, Adam) modelleri sırasıyla eğitilmiştir.
- Her epoch sonunda train/validation loss değerleri loglanmıştır.
- Öğrenme eğrileri `sonuc_train/*.png`, model ağırlıkları ise `sonuc_train/*.pth` olarak kaydedilmiştir.
- Overfitting analizi yapılarak sonuçlar `sonuc_train/train_raporu.txt` dosyasına yazılmıştır.

### 2️⃣ Doğrulama (Validation)
```bash
python validation.py
```
Bu çalışmada:
- Eğitilmiş model ağırlıkları (`sonuc_train/*.pth`) yüklenerek doğrulama aşamasına geçilmiştir.
- `dataset/val.csv` üzerinde Accuracy, Precision, Recall ve F1-Score metrikleri hesaplanmıştır.
- Confusion matrix görselleri oluşturularak `sonuc_val/` klasörüne kaydedilmiştir.
- Modellerin karşılaştırmalı sonuçları `sonuc_val/validation_raporu.txt` dosyasına yazılmıştır.

### 3️⃣ Test
```bash
python test.py
```
Bu çalışmada:
- Eğitilmiş model ağırlıkları yüklenerek, modelin daha önce hiç görmediği `dataset/test.csv` verisi üzerinde nihai değerlendirme yapılmıştır.
- Accuracy, Precision, Recall, F1-Score, ROC-AUC metrikleri ve sınıflandırma raporu (`classification_report`) üretilmiştir.
- Confusion matrix ve ROC eğrisi görselleri oluşturularak `sonuc_test/` klasörüne kaydedilmiştir.
- Elde edilen sonuçlar `sonuc_test/nihai_test_raporu.txt` dosyasına yazılmıştır.

---
## 📈 Sonuçlar

İlk LSTM mimarisi ~0.69 kayıp seviyesinde tıkanıp underfitting eğilimi gösterirken, **BiLSTM + Global Max Pooling + Dropout + BatchNorm + AdamW** ile yeniden yapılandırılan LSTM ve fine-tune edilen BERT, test setinde aşağıdaki sonuçlara ulaşmıştır:

| Metrik | LSTM (BiLSTM + Max Pooling) | BERT |
|---|:---:|:---:|
| **Accuracy** | %96.05 | Test betiği çalıştırıldığında `sonuc_test/nihai_test_raporu.txt` içinde raporlanır |
| **Precision** | %95.51 | ” |
| **Recall** | %96.59 | ” |
| **F1-Score** | %96.05 | ” |
| **Test Loss** | 0.1504 | ” |

> BERT modeli eğitim sürecinde çok daha hızlı yakınsamış (train loss: 0.2892 → 0.0058, val loss: 0.2522 → 0.1391) ve genel olarak LSTM'e kıyasla daha üstün bir genelleme performansı sergilemiştir. Kesin BERT test metrikleri, `test.py` çalıştırıldıktan sonra üretilen rapor dosyasında yer alır.

---

### 🔵 LSTM (BiLSTM + Max Pooling) — Test Sonuçları

<div align="center">

| Confusion Matrix | ROC Eğrisi |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/fa73c002-98f5-4a13-a75c-66c5866ed847" width="420"/> | <img src="https://github.com/user-attachments/assets/bb8e133f-0c7d-4705-bc59-87011148da9e" width="420"/> |

</div>

### 🟣 BERT — Test Sonuçları

<div align="center">

| Confusion Matrix | ROC Eğrisi |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/7790efb2-af4b-47cd-9144-426aabb4c387" width="420"/> | <img src="https://github.com/user-attachments/assets/0e4d296f-8623-4912-b595-0f762705e567" width="420"/> |

</div>
---

## 🔬 Yöntemsel Detaylar

- **Tokenizer:** `bert-base-uncased` (max_len = 64, padding/truncation uygulanır)
- **Loss fonksiyonu:** `BCEWithLogitsLoss` (Sigmoid + Binary Cross-Entropy birleşik, sayısal kararlılık sağlar)
- **LSTM Optimizer:** AdamW, `lr=1e-3`, `weight_decay=1e-4`, `ReduceLROnPlateau` (patience=1, factor=0.5)
- **BERT Optimizer:** Adam, `lr=2e-5`
- **Hiperparametre optimizasyonu:** Learning Rate ve Batch Size (16 / 32) üzerinde Grid Search; F1-skorları heatmap ile görselleştirilerek en iyi kombinasyon seçilmiştir
- **Overfitting kontrolü:** Her epoch sonunda validation loss izlenerek erken durdurma sinyalleri değerlendirilmiştir

---

## 🌐 Kullanım Alanları

Bu tespit sisteminin literatürde ve pratikte işaret ettiği başlıca uygulama alanları:

- 🏛️ **RegTech (Düzenleyici Teknolojiler):** Ticaret Bakanlığı / Rekabet Kurumu gibi otoritelerin büyük ölçekli otomatik denetimleri
- 🧩 **Son Kullanıcı Araçları:** Manipülatif metinleri gerçek zamanlı işaretleyen tarayıcı eklentileri
- 🏢 **Kurumsal Uyum (Compliance):** Kampanya metinlerini yayın öncesi etik/yasal standartlara göre test eden iç denetim sistemleri

---

## 🗺️ Yol Haritası

- [ ] Çok sınıflı (multi-class) sınıflandırma ile pattern kategorisi tahmini (Scarcity / Misdirection / Urgency / Social Proof)
- [ ] Model çıktısını tüketen bir tarayıcı eklentisi prototipi
- [ ] Daha geniş ve çok dilli veri seti ile genelleme kabiliyetinin artırılması
- [ ] REST API üzerinden gerçek zamanlı tahmin servisi

---


## 👤 Geliştirici & İletişim


Çalışma hakkında geri bildirimde bulunmak, detaylı bilgi almak, sorularınızı iletmek veya iş birliği yapmak isterseniz benimle aşağıdaki kanallardan iletişime geçebilirsiniz.

<div align="center">
  
  <img width="850" alt="Ahmet Çaça İletişim" src="https://github.com/user-attachments/assets/1688e904-a044-4831-aa06-7c22d4b8822c" />

  <br><br>

  <a href="mailto:ahmetcaca.dev@gmail.com">
    <img src="https://img.shields.io/badge/E_Posta-ahmetcaca.dev@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email" />
  </a>
  
</div>

---
## 📄 Lisans

Bu çalışma akademik bir lisans dönem projesi kapsamında geliştirilmiştir.

---

<p align="center">
  Zaman ayırıp çalışmamı incelediğiniz için teşekkür ederim!
</p>
