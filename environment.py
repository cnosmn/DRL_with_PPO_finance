"""
Portföy Yönetimi Ortamı - RL Environment
"""

import numpy as np
from config import INITIAL_BALANCE, TRANSACTION_COST, MAX_LOSS_THRESHOLD, LOSS_PENALTY, VOLATILITY_WINDOW

class PortfolioEnvironment:
    """
    Portföy yönetimi için pekiştirmeli öğrenme ortamı
    """
    
    def __init__(self, processed_data, initial_balance=INITIAL_BALANCE, transaction_cost=TRANSACTION_COST):
        # Veri yükleme
        self.prices = processed_data['prices']
        self.returns = processed_data['returns']
        self.normalized_prices = processed_data['normalized_prices']
        self.stock_names = processed_data['stock_names']
        self.n_stocks = processed_data['n_stocks']
        self.n_days = processed_data['n_days']
        
        # Ortam parametreleri
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Durum değişkenleri
        self.current_step = 0
        self.balance = initial_balance
        self.portfolio = np.zeros(self.n_stocks)  # Her hisse senedinden kaç adet
        self.portfolio_weights = np.zeros(self.n_stocks + 1)  # Son eleman nakit
        self.portfolio_weights[-1] = 1.0  # Başlangıçta tüm para nakit
        
        self.total_portfolio_value = initial_balance
        self.portfolio_history = [initial_balance]
        
        print(f"Portföy ortamı oluşturuldu:")
        print(f"- Hisse senedi sayısı: {self.n_stocks}")
        print(f"- Toplam gün sayısı: {self.n_days}")
        print(f"- Başlangıç sermayesi: ${initial_balance:,}")
    
    def reset(self):
        """Ortamı başlangıç durumuna sıfırla"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.portfolio = np.zeros(self.n_stocks)
        self.portfolio_weights = np.zeros(self.n_stocks + 1)
        self.portfolio_weights[-1] = 1.0  # Tüm para nakit
        
        self.total_portfolio_value = self.initial_balance
        self.portfolio_history = [self.initial_balance]
        
        return self.get_state()
    
    def get_state(self):
        """
        Mevcut durum vektörünü döndür
        
        Returns:
            np.array: Durum vektörü [returns_history + portfolio_weights]
        """
        if self.current_step >= self.n_days - 1:
            # Terminal durum
            return np.zeros(self.n_stocks * 3 + self.n_stocks + 1, dtype=np.float32)
        
        state_components = []
        
        # Her hisse senedi için son 3 günün getirilerini ekle
        for i in range(self.n_stocks):
            returns_for_stock = []
            for j in range(3):  # Son 3 gün
                idx = self.current_step - (2 - j)
                if idx >= 0 and idx < self.returns.shape[1]:
                    return_val = self.returns[i, idx]
                    # Güvenlik kontrolü
                    if np.isnan(return_val) or np.isinf(return_val):
                        return_val = 0.0
                    returns_for_stock.append(return_val)
                else:
                    returns_for_stock.append(0.0)
            
            state_components.extend(returns_for_stock)
        
        # Mevcut portföy ağırlıklarını ekle
        for weight in self.portfolio_weights:
            if np.isnan(weight) or np.isinf(weight):
                weight = 0.0
            state_components.append(weight)
        
        state = np.array(state_components, dtype=np.float32)
        
        # Boyut kontrolü
        expected_size = self.n_stocks * 3 + self.n_stocks + 1
        if len(state) != expected_size:
            print(f"Durum vektörü boyut hatası: {len(state)} != {expected_size}")
            state = np.zeros(expected_size, dtype=np.float32)
        
        return state
    
    def step(self, action):
        """
        Bir adım at
        
        Args:
            action (np.array): Aksiyon vektörü (portföy ağırlıkları)
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        if self.current_step >= self.n_days - 1:
            return self.get_state(), 0, True, {}
        
        # Önceki portföy değeri
        prev_value = self.calculate_portfolio_value()
        
        # Yeni ağırlıkları normalize et
        new_weights = self._normalize_weights(action)
        
        # Portföyü yeniden dengele
        self._rebalance_portfolio(new_weights)
        
        # Bir gün ileri
        self.current_step += 1
        
        # Yeni portföy değeri
        current_value = self.calculate_portfolio_value()
        
        # Ödül hesapla
        reward = self._calculate_reward(prev_value, current_value)
        
        # Geçmişi güncelle
        self.portfolio_history.append(current_value)
        
        # Terminal kontrolü
        done = self.current_step >= self.n_days - 1
        
        info = {
            'portfolio_value': current_value,
            'portfolio_weights': self.portfolio_weights.copy(),
            'daily_return': (current_value - prev_value) / prev_value if prev_value > 0 else 0
        }
        
        return self.get_state(), reward, done, info
    
    def _normalize_weights(self, action):
        """Aksiyon vektörünü normalize edilmiş portföy ağırlıklarına çevir"""
        # Softmax normalizasyonu
        exp_action = np.exp(action - np.max(action))
        weights = exp_action / np.sum(exp_action)
        return weights
    
    def _rebalance_portfolio(self, new_weights):
        """Portföyü yeniden dengele"""
        current_value = self.calculate_portfolio_value()
        
        # İşlem maliyetini hesapla
        weight_changes = np.abs(new_weights - self.portfolio_weights)
        transaction_cost_amount = np.sum(weight_changes) * self.transaction_cost * current_value
        
        # Yeni pozisyonları ayarla
        cash_amount = new_weights[-1] * current_value
        
        # Hisse senedi pozisyonlarını güncelle
        for i in range(self.n_stocks):
            target_value = new_weights[i] * current_value
            current_price = self.prices[i, self.current_step]
            if current_price > 0:
                self.portfolio[i] = target_value / current_price
            else:
                self.portfolio[i] = 0
        
        # Nakit pozisyonunu güncelle (işlem maliyeti düşüldükten sonra)
        self.balance = max(0, cash_amount - transaction_cost_amount)
        self.portfolio_weights = new_weights.copy()
    
    def calculate_portfolio_value(self):
        """Toplam portföy değerini hesapla"""
        if self.current_step >= self.n_days:
            return self.total_portfolio_value
        
        # Hisse senedi değeri
        stock_value = 0
        for i in range(self.n_stocks):
            stock_value += self.portfolio[i] * self.prices[i, self.current_step]
        
        # Toplam değer = hisse senedi değeri + nakit
        total_value = stock_value + self.balance
        self.total_portfolio_value = total_value
        
        return total_value
    
    def _calculate_reward(self, prev_value, current_value):
        """
        Ödül fonksiyonu
        
        Args:
            prev_value (float): Önceki portföy değeri
            current_value (float): Mevcut portföy değeri
            
        Returns:
            float: Hesaplanan ödül
        """
        # Ana ödül: günlük getiri
        if prev_value > 0:
            daily_return = (current_value - prev_value) / prev_value
        else:
            daily_return = 0
        
        # Risk ayarlı getiri (Sharpe oranı benzeri)
        risk_adjusted_return = self._calculate_risk_adjusted_return(daily_return)
        
        # Çeşitlendirme bonusu
        diversification_bonus = self._calculate_diversification_bonus()
        
        # Toplam ödül
        reward = (
            daily_return * 150 +  # 100'den 150'ye
            risk_adjusted_return * 10 +
            diversification_bonus * 2  # 5'ten 2'ye
        )
        
        # Büyük kayıp cezası
        if daily_return < MAX_LOSS_THRESHOLD:
            reward -= LOSS_PENALTY
        
        # Güvenlik kontrolü
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        
        return float(reward)
    
    def _calculate_risk_adjusted_return(self, daily_return):
        """Risk ayarlı getiri hesapla"""
        if len(self.portfolio_history) > VOLATILITY_WINDOW:
            # Son VOLATILITY_WINDOW günün getirilerini al
            recent_values = self.portfolio_history[-VOLATILITY_WINDOW:]
            recent_returns = np.diff(recent_values) / np.array(recent_values[:-1])
            
            # NaN ve inf değerleri temizle
            recent_returns = recent_returns[np.isfinite(recent_returns)]
            
            if len(recent_returns) > 1:
                volatility = np.std(recent_returns)
                risk_adjusted = daily_return / (volatility + 1e-8)
                return risk_adjusted
        
        return daily_return
    
    def _calculate_diversification_bonus(self):
        """Çeşitlendirme bonusu hesapla"""
        # Nakit dışındaki ağırlıklar
        non_cash_weights = self.portfolio_weights[:-1]
        
        if np.sum(non_cash_weights) > 0:
            # Herfindahl indeksinin tersi (1 - sum(w_i^2))
            concentration = np.sum(non_cash_weights ** 2)
            diversification_score = 1 - concentration
            return diversification_score
        
        return 0
    
    def get_portfolio_summary(self):
        """Portföy özetini döndür"""
        current_value = self.calculate_portfolio_value()
        total_return = (current_value / self.initial_balance - 1) * 100
        
        summary = f"""
        Portföy Durumu (Gün {self.current_step}):
        --------------------------------
        Toplam Değer: ${current_value:,.2f}
        Başlangıç Değeri: ${self.initial_balance:,.2f}
        Toplam Getiri: %{total_return:.2f}
        Nakit Oranı: %{self.portfolio_weights[-1] * 100:.1f}
        
        Hisse Senedi Dağılımı:
        """
        
        for i, stock_name in enumerate(self.stock_names):
            weight_pct = self.portfolio_weights[i] * 100
            summary += f"        {stock_name}: %{weight_pct:.1f}\n"
        
        return summary 