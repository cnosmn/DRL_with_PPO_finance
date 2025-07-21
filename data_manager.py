"""
Veri Yönetimi Modülü - Hisse senedi verilerini indirme ve işleme
"""

import numpy as np
import yfinance as yf
import warnings
from config import MIN_DATA_LENGTH

warnings.filterwarnings('ignore')


class DataManager:
    """Hisse senedi verilerini yöneten sınıf"""
    
    def __init__(self):
        self.raw_data = {}
        self.processed_data = {}
    
    def download_stock_data(self, symbols, period="2y"):
        """
        Gerçek hisse senedi verilerini indir
        
        Args:
            symbols (list): Hisse senedi sembolleri
            period (str): Veri indirme süresi
            
        Returns:
            dict: İşlenmiş hisse senedi verileri
        """
        data = {}
        print(f"Toplam {len(symbols)} hisse senedi için veri indiriliyor...")
        
        for symbol in symbols:
            try:
                stock = yf.download(symbol, period=period, progress=False)
                if not stock.empty:
                    prices = stock['Close'].values
                    # NaN değerleri temizle
                    prices = prices[~np.isnan(prices)]
                    if len(prices) > MIN_DATA_LENGTH:
                        data[symbol] = prices
                        print(f"{symbol}: {len(prices)} günlük veri indirild")
                    else:
                        print(f"Yetersiz veri: {symbol} - {len(prices)} gün")
                else:
                    print(f"Veri bulunamadı: {symbol}")
            except Exception as e:
                print(f"Hata {symbol}: {e}")
        
        self.raw_data = data
        return data
    
    def process_data(self, stock_data):
        """
        Ham veriyi işle ve normalize et
        
        Args:
            stock_data (dict): Ham hisse senedi verileri
            
        Returns:
            dict: İşlenmiş veriler
        """
        if not stock_data:
            return {}
        
        stock_names = list(stock_data.keys())
        n_stocks = len(stock_names)
        
        # Tüm hisse senetleri aynı uzunlukta olmalı
        min_length = min(len(stock_data[name]) for name in stock_names)
        print(f"Minimum veri uzunluğu: {min_length}")
        
        # Fiyat matrisi oluştur
        prices = np.zeros((n_stocks, min_length))
        for i, name in enumerate(stock_names):
            prices[i] = stock_data[name][:min_length]
        
        # Getiri hesaplama
        returns = self._calculate_returns(prices)
        
        # Normalizasyon
        normalized_prices = prices / prices[:, 0:1]
        
        processed = {
            'prices': prices,
            'returns': returns,
            'normalized_prices': normalized_prices,
            'stock_names': stock_names,
            'n_stocks': n_stocks,
            'n_days': min_length
        }
        
        self.processed_data = processed
        return processed
    
    def _calculate_returns(self, prices):
        """
        Günlük getiri hesapla
        
        Args:
            prices (np.array): Fiyat matrisi
            
        Returns:
            np.array: Getiri matrisi
        """
        returns = np.zeros_like(prices)
        n_stocks, n_days = prices.shape
        
        for i in range(n_stocks):
            for j in range(1, n_days):
                if prices[i, j-1] != 0:
                    returns[i, j] = (prices[i, j] - prices[i, j-1]) / prices[i, j-1]
                else:
                    returns[i, j] = 0.0
        
        # NaN kontrolü ve temizleme
        nan_count = np.sum(np.isnan(returns))
        if nan_count > 0:
            print(f"Uyarı: {nan_count} NaN değer bulundu, sıfırla değiştiriliyor")
            returns = np.nan_to_num(returns, nan=0.0)
        
        return returns
    
    def get_data_summary(self):
        """Veri özeti döndür"""
        if not self.processed_data:
            return "Henüz işlenmiş veri yok"
        
        summary = f"""
        Veri Özeti:
        -----------
        Hisse Senedi Sayısı: {self.processed_data['n_stocks']}
        Gün Sayısı: {self.processed_data['n_days']}
        Hisse Senetleri: {', '.join(self.processed_data['stock_names'])}
        
        Fiyat İstatistikleri:
        - Ortalama: {np.mean(self.processed_data['prices']):.2f}
        - Standart Sapma: {np.std(self.processed_data['prices']):.2f}
        
        Getiri İstatistikleri:
        - Ortalama Günlük Getiri: {np.mean(self.processed_data['returns']) * 100:.4f}%
        - Getiri Volatilitesi: {np.std(self.processed_data['returns']) * 100:.4f}%
        """
        return summary 