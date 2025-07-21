"""
Konfigürasyon ayarları - Portföy Yönetimi DRL Projesi
"""

# Hisse senedi sembolleri
STOCK_SYMBOLS = ['THYAO.IS', 'AKBNK.IS', 'GARAN.IS', 'ISCTR.IS', 'TCELL.IS']

# Veri parametreleri
DATA_PERIOD = "3y"  # Veri indirme süresi
MIN_DATA_LENGTH = 100  # Minimum veri uzunluğu

# Ortam parametreleri
INITIAL_BALANCE = 10000  # Başlangıç sermayesi
TRANSACTION_COST = 0.001  # İşlem maliyeti

# PPO parametreleri
LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
EPS_CLIP = 0.2  # Clipping parameter
K_EPOCHS = 4  # Update epochs
HIDDEN_DIM = 256  # Hidden layer boyutu

# Eğitim parametreleri
NUM_EPISODES = 1000
UPDATE_FREQUENCY = 10  # Kaç episode'da bir güncelleme
REPORT_FREQUENCY = 50  # Kaç episode'da bir rapor

# Görselleştirme
FIGURE_SIZE = (15, 10)
PLOT_ALPHA = 0.7

# Risk parametreleri
MAX_LOSS_THRESHOLD = -0.05  # %5'den fazla kayıp cezası
LOSS_PENALTY = 50
VOLATILITY_WINDOW = 10  # Risk hesaplama penceresi 