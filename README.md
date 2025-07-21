# PPO Portföy Yönetimi - Derin Pekiştirmeli Öğrenme Projesi

Bu proje, PPO (Proximal Policy Optimization) algoritmasını kullanarak otomatik portföy yönetimi yapan bir derin pekiştirmeli öğrenme sistemidir. Türk hisse senetleri üzerinde çalışarak optimal yatırım kararları vermeyi öğrenir.

## 🎯 Proje Özeti

- **Algoritma**: PPO (Proximal Policy Optimization)
- **Hedef**: Türk hisse senetlerinde optimal portföy yönetimi
- **Veri Kaynağı**: Yahoo Finance (Borsa İstanbul)
- **Framework**: PyTorch
- **Hisse Senetleri**: THYAO, AKBNK, GARAN, ISCTR, TCELL

## 📁 Proje Yapısı

```
rl_code/
├── config.py              # Konfigürasyon ayarları
├── data_manager.py         # Veri indirme ve işleme
├── environment.py          # Portföy ortamı (RL Environment)
├── models.py              # PPO ağ mimarisi
├── agents.py              # PPO agent sınıfı
├── utils.py               # Yardımcı fonksiyonlar ve görselleştirme
├── main.py                # Ana eğitim dosyası
├── requirements.txt       # Gerekli paketler
└── README.md             # Bu dosya
```

## 🛠️ Kurulum

### 1. Gereksinimler

```bash
pip install -r requirements.txt
```

### 2. Paketler

- numpy>=1.21.0
- pandas>=1.3.0  
- torch>=1.9.0
- matplotlib>=3.4.0
- yfinance>=0.1.63
- tqdm>=4.62.0

## 🚀 Kullanım

### Temel Eğitim

```bash
python main.py
```

### Hızlı Test (50 episode)

```bash
python main.py --quick
```

### Python Script'ten Çağırma

```python
from main import train_portfolio_agent

# Tam eğitim
agent, env, results = train_portfolio_agent()

# Hızlı test
from main import quick_test
agent, env, results = quick_test()
```

## ⚙️ Konfigürasyon

`config.py` dosyasından ana parametreleri değiştirebilirsiniz:

```python
# Hisse senetleri
STOCK_SYMBOLS = ['THYAO.IS', 'AKBNK.IS', 'GARAN.IS', 'ISCTR.IS', 'TCELL.IS']

# Eğitim parametreleri
NUM_EPISODES = 500
LEARNING_RATE = 3e-4
INITIAL_BALANCE = 10000

# PPO parametreleri
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4
```

## 📊 Çıktılar

### Eğitim Sırasında
- Episode ödülleri ve portföy değeri gelişimi
- Periyodik performans raporları
- Loss metrikleri

### Eğitim Sonunda
- Detaylı performans analizi
- Agent vs Benchmark karşılaştırması
- Grafik görselleştirmeler
- Sharpe oranı, maksimum drawdown, volatilite metrikleri

### Kaydedilen Dosyalar
- `best_portfolio_agent.pt` - En iyi performans gösteren model
- `portfolio_results_agent_YYYYMMDD_HHMMSS.pt` - Final model
- `portfolio_results_metrics_YYYYMMDD_HHMMSS.json` - Performans metrikleri

## 🧠 Model Mimarisi

### PPO Network
- **Giriş**: Durum vektörü (geçmiş getiriler + portföy ağırlıkları)
- **Paylaşılan Katmanlar**: 2x256 nöronlu fully connected
- **Actor Head**: Politika ağı (aksiyon olasılıkları)
- **Critic Head**: Değer ağı (durum değeri)

### Durum Uzayı
- Her hisse senedi için son 3 günün getirisi
- Mevcut portföy ağırlıkları (hisse senetleri + nakit)

### Aksiyon Uzayı
- Her hisse senedi + nakit için portföy ağırlığı kararı
- Softmax normalizasyonu ile toplam %100

## 📈 Özellikler

### Portföy Yönetimi
- ✅ Gerçek hisse senedi verileri
- ✅ İşlem maliyetleri
- ✅ Nakit pozisyonu yönetimi
- ✅ Risk ayarlı ödül fonksiyonu

### Risk Yönetimi
- ✅ Volatilite tabanlı risk hesaplaması
- ✅ Maksimum drawdown takibi
- ✅ Çeşitlendirme bonusu
- ✅ Büyük kayıp cezaları

### Teknik Özellikler
- ✅ Gradient clipping
- ✅ Experience replay
- ✅ Model kaydetme/yükleme
- ✅ Kapsamlı görselleştirme

## 🔧 Modül Detayları

### `data_manager.py`
- Yahoo Finance'dan veri indirme
- Veri temizleme ve normalizasyon
- Getiri hesaplamaları

### `environment.py`
- RL ortamı implementasyonu
- Portföy değeri hesaplama
- Ödül fonksiyonu
- Yeniden dengeleme (rebalancing)

### `models.py`
- PPO network mimarisi
- Actor-Critic yapısı
- Model kaydetme/yükleme

### `agents.py`
- PPO algoritması implementasyonu
- Experience buffer yönetimi
- Policy güncelleme

### `utils.py`
- Performans analizi
- Görselleştirme araçları
- Benchmark karşılaştırma

## 🎯 Performans Metrikleri

- **Toplam Getiri**: Final portföy değeri vs başlangıç
- **Sharpe Oranı**: Risk ayarlı getiri
- **Maksimum Drawdown**: En büyük kayıp
- **Volatilite**: Günlük getiri standardı sapması
- **Kazanma Oranı**: Pozitif günlerin yüzdesi

## 🐛 Sorun Giderme

### Veri İndirme Sorunları
```python
# İnternet bağlantısını kontrol edin
# Farklı zaman aralığı deneyin
DATA_PERIOD = "2y"  # config.py'de
```

### Memory Sorunları
```python
# Episode sayısını azaltın
NUM_EPISODES = 100  # config.py'de
```

### GPU Kullanımı
Model otomatik olarak GPU'yu algılar ve kullanır (varsa).

## 📝 Notlar

- Bu proje eğitim amaçlıdır, gerçek yatırım tavsiyesi değildir
- Geçmiş performans gelecek performansı garanti etmez
- Risk toleransınıza uygun yatırım yapın

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje eğitim amaçlı MIT lisansı altında dağıtılmaktadır.

## 👨‍💻 Geliştirici

- Derin Pekiştirmeli Öğrenme Dersi Projesi
- PPO + Portföy Yönetimi Implementation

---

**Not**: Bu proje akademik çalışma amaçlıdır. Gerçek finansal kararlar için profesyonel finansal danışmanlık alınız. 