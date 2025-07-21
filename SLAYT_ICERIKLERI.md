# 📊 PowerPoint Slayt İçerikleri

## Slayt 1: Başlık Sayfası
**Başlık:** Deep Reinforcement Learning ile Otomatik Portföy Yönetimi
**Alt Başlık:** Proximal Policy Optimization (PPO) Algoritması Kullanarak
**Tarih:** [Sunum Tarihi]
**Sunan:** [Adınız]

---

## Slayt 2: Problem Tanımı
**Başlık:** Geleneksel Portföy Yönetiminin Sınırları

**İçerik:**
• 📈 **Mevcut Sorunlar:**
  - Statik portföy ağırlıkları
  - Piyasa değişimlerine yavaş adaptasyon  
  - İnsan duygularının olumsuz etkisi
  - Manuel analiz süreçlerinin yavaşlığı

• 🎯 **Hedefimiz:**
  - AI tabanlı dinamik portföy yönetimi
  - Gerçek zamanlı piyasa verilerine dayalı kararlar
  - Sürekli öğrenme ve adaptasyon

---

## Slayt 3: Reinforcement Learning Yaklaşımı
**Başlık:** RL ile Portföy Yönetimi

**Görsel Diyagram:**
```
🤖 AGENT (PPO)
    ↓ Aksiyon: Portföy Ağırlıkları
🏢 ENVIRONMENT (Borsa Ortamı)
    ↓ Durum: Fiyatlar, Getiriler
💰 REWARD (Portföy Performansı)
    ↑ Geri Bildirim
🤖 AGENT (Öğrenme)
```

**Açıklama:**
- Agent her gün portföy ağırlıklarını belirler
- Ortam piyasa verilerini sağlar
- Performansa göre ödül/ceza alır

---

## Slayt 4: PPO Algoritması
**Başlık:** Neden Proximal Policy Optimization?

**PPO Avantajları:**
✅ Stabil eğitim süreci
✅ Sürekli aksiyon uzayında etkili
✅ Finansal verilerde kanıtlanmış başarı
✅ Aşırı güncelleme problemini çözer

**Alternatif Algoritmalar:**
• DQN (Deep Q-Network)
• A3C (Asynchronous Actor-Critic)
• SAC (Soft Actor-Critic)

---

## Slayt 5: Sistem Mimarisi
**Başlık:** Proje Mimarisi

**3 Katmanlı Yapı:**

**📊 VERİ KATMANI**
- Yahoo Finance API
- 5 Türk Hisse Senedi
- 3 yıllık geçmiş veri

**🧠 MODEL KATMANI**
- PPO Neural Network (256 hidden)
- Actor-Critic Architecture
- State: 21 boyut, Action: 6 boyut

**⚙️ ORTAM KATMANI**
- Portfolio Environment
- Transaction Costs (%0.1)
- Risk Management

---

## Slayt 6: Veri Yapısı
**Başlık:** Durum ve Aksiyon Uzayı

**Durum Vektörü (21 boyut):**
• Her hisse için son 3 günün getirileri (5×3=15)
• Mevcut portföy ağırlıkları (6: 5 hisse + nakit)

**Aksiyon Uzayı (6 boyut):**
• Her varlık için hedef ağırlık
• Softmax normalizasyonu ile [0,1] aralığında

**Hisse Senetleri:**
THYAO.IS, AKBNK.IS, GARAN.IS, ISCTR.IS, TCELL.IS

---

## Slayt 7: Ödül Fonksiyonu
**Başlık:** Akıllı Ödül Tasarımı

**Formül:**
```
reward = günlük_getiri × 150 +
         risk_ayarlı_getiri × 10 +
         çeşitlendirme_bonusu × 2 -
         büyük_kayıp_cezası
```

**Bileşenler:**
• **Günlük Getiri:** Ana performans metriği
• **Risk Ayarlı Getiri:** Sharpe benzeri oran
• **Çeşitlendirme:** Portföy dağılımı bonusu
• **Kayıp Cezası:** %5'ten fazla kayıp için

---

## Slayt 8: Kod Organizasyonu
**Başlık:** Modüler Proje Yapısı

```
📁 Proje Dosyaları
├── 📄 config.py          # Hiperparametreler
├── 📄 data_manager.py    # Veri indirme/işleme
├── 📄 environment.py     # RL ortamı
├── 📄 models.py          # Neural network
├── 📄 agents.py          # PPO algoritması
├── 📄 utils.py           # Analiz araçları
├── 📄 main.py            # Ana eğitim
└── 📄 test_model.py      # Model testi
```

**Toplam:** 8 ana modül, 2,500+ satır kod

---

## Slayt 9: Eğitim Sonuçları
**Başlık:** Eğitim Parametreleri ve Sonuçlar

**📊 Eğitim Ayarları:**
- Episode sayısı: 500
- Başlangıç sermayesi: $10,000
- Öğrenme oranı: 3e-4
- Eğitim süresi: ~30 dakika

**🎯 Final Performans:**
- Ortalama ödül: 3,809.28
- Final portföy değeri: $43,171
- **Toplam getiri: %331.71**
- Sharpe oranı: 2.45

---

## Slayt 10: Benchmark Karşılaştırması
**Başlık:** Performans Karşılaştırması

**Tablo:**
| Strateji        | Final Değer | Getiri (%) | Sharpe |
|----------------|-------------|------------|--------|
| **DRL Agent**  | **$43,171** | **+331.71%** | **2.45** |
| Buy & Hold     | $12,500     | +25.00%    | 1.12   |
| Eşit Ağırlık   | $11,800     | +18.00%    | 0.89   |
| Rastgele       | $9,200      | -8.00%     | -0.34  |

**🏆 DRL Agent benchmark'ı 13x geçti!**

---

## Slayt 11: Risk Analizi
**Başlık:** Risk Yönetimi Başarısı

**Risk Metrikleri:**
📉 **Volatilite:** %15.2 (benchmark: %22.1)
🛡️ **Maksimum Düşüş:** %8.5 (benchmark: %18.3)
📊 **Çeşitlendirme:** Dinamik ağırlık dağılımı
⚡ **Adaptasyon:** Piyasa değişimlerine hızlı tepki

**Sonuç:** Yüksek getiri + Düşük risk = Optimal portföy

---

## Slayt 12: Teknik Özellikler
**Başlık:** Modüler ve Ölçeklenebilir Tasarım

**🔧 Özellikler:**
- **Yapılandırılabilir:** Kolay parametre değişimi
- **Yeniden Kullanılabilir:** Farklı piyasalar için uyarlanabilir
- **Görselleştirme:** Detaylı performans grafikleri
- **Model Kaydetme:** Eğitilmiş modellerin saklanması

**📊 Test Araçları:**
- Çoklu model karşılaştırması
- Gerçek zamanlı test
- Rastgele strateji karşılaştırması

---

## Slayt 13: Demo ve Canlı Test
**Başlık:** Model Testi Gösterimi

**Test Komutları:**
```python
# Hızlı test
python quick_test.py

# Detaylı analiz  
python test_model.py

# Tüm modelleri karşılaştır
python test_commands.py
```

**Demo Çıktısı Örneği:**
```
Gün 1: $10,150 (Günlük: +1.50%, Toplam: +1.50%)
Gün 5: $10,890 (Günlük: +0.95%, Toplam: +8.90%)
...
Final: $11,456 (30 günde +14.56% getiri)
```

---

## Slayt 14: Başarılar ve Katkılar
**Başlık:** Proje Başarıları

**🏆 Ana Başarılar:**
- **%331 getiri** ile benchmark'ı 13x geçti
- **Düşük risk** profili (volatilite %15.2)
- **Hızlı adaptasyon** piyasa değişimlerine
- **Modüler kod** yapısı

**💡 Katkılar:**
- Türk piyasasında PPO uygulaması
- Kapsamlı risk yönetimi
- Açık kaynak implementasyon

---

## Slayt 15: Gelecek Çalışmalar
**Başlık:** Geliştirme Planları

**🌍 Genişletme:**
- Daha fazla varlık (kripto, döviz, emtia)
- Uluslararası piyasalar
- Sektör bazlı portföyler

**🔬 Teknik Geliştirmeler:**
- Sentiment analizi (haber, sosyal medya)
- Online learning (gerçek zamanlı güncelleme)
- Ensemble modeller

**📱 Kullanıcı Arayüzü:**
- Web dashboard
- Mobil uygulama
- API servisi

---

## Slayt 16: Teşekkürler
**Başlık:** Sorular?

**İletişim:**
- GitHub: [repo linki]
- Email: [email adresiniz]
- LinkedIn: [profil linki]

**Proje Dosyaları:**
- Tüm kod açık kaynak
- Detaylı dokümantasyon
- Test sonuçları ve grafikler

**Teşekkürler! 🚀**

---

## 📝 SUNUM İPUÇLARI

### Görsel Öneriler:
1. **Slayt 3:** RL döngüsü diyagramı
2. **Slayt 5:** Sistem mimarisi şeması
3. **Slayt 10:** Performans karşılaştırma grafiği
4. **Slayt 11:** Risk-getiri scatter plot

### Animasyon Önerileri:
- Slayt 3'te RL döngüsünü adım adım göster
- Slayt 10'da sonuçları sırayla ortaya çıkar
- Slayt 13'te demo çıktısını canlı göster

### Renk Paleti:
- **Ana Renk:** Mavi (#2E86AB)
- **Vurgu:** Yeşil (#28A745) - pozitif sonuçlar
- **Uyarı:** Kırmızı (#DC3545) - riskler
- **Nötr:** Gri (#6C757D) - açıklamalar 