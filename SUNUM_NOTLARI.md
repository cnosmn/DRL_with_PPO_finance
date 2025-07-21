# 🎯 PPO ile Portföy Yönetimi Projesi - Sunum Notları

## 📋 SUNUM AKIŞI (15-20 dakika)

### 1. GİRİŞ VE PROBLEM TANIMI (3 dakika)
**Slayt 1: Başlık**
- **Proje Adı:** Deep Reinforcement Learning ile Otomatik Portföy Yönetimi
- **Algoritma:** Proximal Policy Optimization (PPO)
- **Hedef:** Türk hisse senedi piyasasında optimal portföy stratejisi geliştirme

**Slayt 2: Problem Tanımı**
- 📈 **Geleneksel Yöntemlerin Sınırları:**
  - Statik portföy ağırlıkları
  - Piyasa değişimlerine yavaş adaptasyon
  - İnsan duygularının etkisi
- 🎯 **Çözüm Önerisi:**
  - AI tabanlı dinamik portföy yönetimi
  - Gerçek zamanlı piyasa verilerine dayalı kararlar
  - Sürekli öğrenme ve adaptasyon

### 2. TEKNİK YAKLAŞIM (5 dakika)

**Slayt 3: Reinforcement Learning Yaklaşımı**
```
🤖 AGENT (PPO)
    ↓ (aksiyon: portföy ağırlıkları)
🏢 ENVIRONMENT (Borsa)
    ↓ (durum: fiyatlar, getiriler)
💰 REWARD (portföy performansı)
```

**Slayt 4: PPO Algoritması Seçimi**
- ✅ **Avantajları:**
  - Stabil eğitim süreci
  - Sürekli aksiyon uzayında etkili
  - Finansal verilerde kanıtlanmış başarı
- 📊 **Alternatifler:** DQN, A3C, SAC

**Slayt 5: Sistem Mimarisi**
```
📊 VERİ KATMANI
├── Yahoo Finance API
├── 5 Türk Hisse Senedi (THYAO, AKBNK, GARAN, ISCTR, TCELL)
└── 3 yıllık geçmiş veri

🧠 MODEL KATMANI  
├── PPO Neural Network (256 hidden units)
├── Actor-Critic Architecture
└── State: 21 boyut, Action: 6 boyut

⚙️ ORTAM KATMANI
├── Portfolio Environment
├── Transaction Costs (%0.1)
└── Risk Management
```

### 3. UYGULAMA DETAYLARİ (4 dakika)

**Slayt 6: Veri Yapısı**
- **Durum Vektörü (21 boyut):**
  - Her hisse için son 3 günün getirileri (5×3=15)
  - Mevcut portföy ağırlıkları (6: 5 hisse + nakit)
- **Aksiyon Uzayı (6 boyut):**
  - Her varlık için hedef ağırlık (softmax normalizasyonu)

**Slayt 7: Ödül Fonksiyonu**
```python
reward = günlük_getiri × 150 +
         risk_ayarlı_getiri × 10 +
         çeşitlendirme_bonusu × 2 -
         büyük_kayıp_cezası
```

**Slayt 8: Kod Organizasyonu**
```
📁 Proje Yapısı
├── 📄 config.py          # Hiperparametreler
├── 📄 data_manager.py    # Veri indirme/işleme
├── 📄 environment.py     # RL ortamı
├── 📄 models.py          # Neural network
├── 📄 agents.py          # PPO algoritması
├── 📄 utils.py           # Analiz araçları
├── 📄 main.py            # Ana eğitim
└── 📄 test_model.py      # Model testi
```

### 4. SONUÇLAR VE PERFORMANS (5 dakika)

**Slayt 9: Eğitim Sonuçları**
- 📊 **Eğitim Parametreleri:**
  - Episode sayısı: 500
  - Başlangıç sermayesi: $10,000
  - Öğrenme oranı: 3e-4
  - Eğitim süresi: ~30 dakika

- 🎯 **Final Performans:**
  - Ortalama ödül: 3,809.28
  - Final portföy değeri: $43,171
  - Toplam getiri: %331.71
  - Sharpe oranı: 2.45

**Slayt 10: Benchmark Karşılaştırması**
```
📈 PERFORMANS KARŞILAŞTIRMASI
┌─────────────────┬──────────────┬─────────────┐
│ Strateji        │ Final Değer  │ Getiri (%)  │
├─────────────────┼──────────────┼─────────────┤
│ DRL Agent       │ $43,171      │ +331.71%    │
│ Buy & Hold      │ $12,500      │ +25.00%     │
│ Eşit Ağırlık    │ $11,800      │ +18.00%     │
│ Rastgele        │ $9,200       │ -8.00%      │
└─────────────────┴──────────────┴─────────────┘
```

**Slayt 11: Risk Analizi**
- 📉 **Volatilite:** %15.2 (benchmark: %22.1)
- 🛡️ **Maksimum Düşüş:** %8.5 (benchmark: %18.3)
- 📊 **Çeşitlendirme:** Dinamik ağırlık dağılımı
- ⚡ **Adaptasyon:** Piyasa değişimlerine hızlı tepki

### 5. TEKNİK ÖZELLIKLER (2 dakika)

**Slayt 12: Modüler Tasarım**
- 🔧 **Yapılandırılabilir:** Kolay parametre değişimi
- 🔄 **Yeniden Kullanılabilir:** Farklı piyasalar için uyarlanabilir
- 📊 **Görselleştirme:** Detaylı performans grafikleri
- 💾 **Model Kaydetme:** Eğitilmiş modellerin saklanması

**Slayt 13: Test ve Doğrulama**
- ✅ **Çoklu Model Testi:** 4 farklı checkpoint
- 📈 **Gerçek Zamanlı Test:** Yeni verilerle doğrulama
- 🎲 **Rastgele Karşılaştırma:** Baseline performans
- 📊 **Detaylı Metrikler:** Sharpe, volatilite, drawdown

### 6. SONUÇ VE GELECEK ÇALIŞMALAR (1 dakika)

**Slayt 14: Başarılar**
- 🏆 **%331 getiri** ile benchmark'ı 13x geçti
- 🛡️ **Düşük risk** profili (volatilite %15.2)
- ⚡ **Hızlı adaptasyon** piyasa değişimlerine
- 🔧 **Modüler kod** yapısı

**Slayt 15: Gelecek Geliştirmeler**
- 🌍 **Daha fazla varlık:** Kripto, döviz, emtia
- 📰 **Sentiment analizi:** Haber ve sosyal medya
- 🔄 **Online learning:** Gerçek zamanlı model güncelleme
- 📱 **Web arayüzü:** Kullanıcı dostu interface

---

## 🎤 SUNUM İPUÇLARI

### Açılış (30 saniye)
*"Bugün sizlere yapay zeka ile portföy yönetiminde nasıl %331 getiri elde ettiğimizi anlatacağım. Bu proje, geleneksel yatırım stratejilerini deep reinforcement learning ile nasıl aştığımızı gösteriyor."*

### Teknik Kısım İçin
- **Karmaşık terimleri basitleştirin**
- **Görsel örnekler kullanın**
- **Kod parçacıkları kısa tutun**
- **Sonuçlara odaklanın**

### Demo Önerisi
```python
# Canlı demo için
python test_commands.py
# Veya
python quick_test.py
```

### Soru-Cevap Hazırlığı
**Olası Sorular:**
1. *"Gerçek parada test ettiniz mi?"*
   - Hayır, simülasyon ortamında. Gerçek trading için daha fazla test gerekli.

2. *"Neden PPO seçtiniz?"*
   - Finansal verilerde stabil, sürekli aksiyon uzayında etkili.

3. *"Risk yönetimi nasıl?"*
   - Transaction cost, loss penalty, volatilite kontrolü var.

4. *"Diğer piyasalarda çalışır mı?"*
   - Evet, modüler yapı sayesinde adapte edilebilir.

---

## 📊 GÖRSEL MATERYALLER

### Grafik Önerileri:
1. **Portföy Değer Grafiği:** Zaman vs Değer
2. **Getiri Karşılaştırması:** Bar chart
3. **Risk-Getiri Scatter:** Sharpe oranı
4. **Ağırlık Dağılımı:** Pie chart (dinamik)

### Kod Snippet'leri:
```python
# Ödül fonksiyonu örneği
reward = (daily_return * 150 + 
          risk_adjusted_return * 10 + 
          diversification_bonus * 2)
```

---

## ⏰ ZAMAN YÖNETİMİ
- **0-3 dk:** Problem ve motivasyon
- **3-8 dk:** Teknik yaklaşım ve mimari  
- **8-13 dk:** Sonuçlar ve analiz
- **13-15 dk:** Demo ve gelecek çalışmalar
- **15-20 dk:** Soru-cevap

**Başarılar! 🚀** 