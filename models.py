"""
Sinir Ağı Modelleri - PPO mimarisi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import HIDDEN_DIM


class PPONetwork(nn.Module):
    """
    PPO (Proximal Policy Optimization) için Actor-Critic ağı
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super(PPONetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Paylaşılan katmanlar (feature extraction)
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Actor kafası (politika ağı)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic kafası (değer ağı)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Ağırlıkları başlat
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Ağırlıkları Xavier uniform ile başlat"""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state):
        """
        İleri yayılım
        
        Args:
            state (torch.Tensor): Durum vektörü
            
        Returns:
            tuple: (action_logits, state_value)
        """
        # Girdi boyut kontrolü
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # Paylaşılan özellik çıkarımı
        shared_features = self.shared_layers(state)
        
        # Actor çıktısı (aksiyon logitleri)
        action_logits = self.actor_head(shared_features)
        
        # Critic çıktısı (durum değeri)
        state_value = self.critic_head(shared_features)
        
        return action_logits, state_value.squeeze(-1)
    
    def get_action_probabilities(self, state):
        """
        Aksiyon olasılıklarını hesapla
        
        Args:
            state (torch.Tensor): Durum vektörü
            
        Returns:
            torch.Tensor: Aksiyon olasılıkları
        """
        action_logits, _ = self.forward(state)
        return F.softmax(action_logits, dim=-1)
    
    def get_state_value(self, state):
        """
        Durum değerini hesapla
        
        Args:
            state (torch.Tensor): Durum vektörü
            
        Returns:
            torch.Tensor: Durum değeri
        """
        _, state_value = self.forward(state)
        return state_value
    
    def save_model(self, filepath):
        """Modeli kaydet"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim
        }, filepath)
        print(f"Model kaydedildi: {filepath}")
    
    def load_model(self, filepath):
        """Modeli yükle"""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model yüklendi: {filepath}")
        return checkpoint
    
    def get_model_info(self):
        """Model bilgilerini döndür"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = f"""
        PPO Network Bilgileri:
        ---------------------
        Durum Boyutu: {self.state_dim}
        Aksiyon Boyutu: {self.action_dim}
        Gizli Katman Boyutu: {self.hidden_dim}
        Toplam Parametre: {total_params:,}
        Eğitilebilir Parametre: {trainable_params:,}
        """
        return info


class PortfolioEncoder(nn.Module):
    """
    Portföy durumunu encode eden yardımcı ağ (gelecekteki geliştirmeler için)
    """
    
    def __init__(self, n_stocks, lookback_window=5, encoding_dim=64):
        super(PortfolioEncoder, self).__init__()
        
        self.n_stocks = n_stocks
        self.lookback_window = lookback_window
        self.encoding_dim = encoding_dim
        
        # LSTM katmanı geçmiş verileri işlemek için
        self.lstm = nn.LSTM(
            input_size=n_stocks,
            hidden_size=encoding_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Çıktı katmanı
        self.output_layer = nn.Linear(encoding_dim // 2, encoding_dim)
        
    def forward(self, price_sequence):
        """
        Fiyat serisini encode et
        
        Args:
            price_sequence (torch.Tensor): [batch, sequence_length, n_stocks]
            
        Returns:
            torch.Tensor: Encoded representation
        """
        lstm_out, _ = self.lstm(price_sequence)
        # Son zaman adımının çıktısını al
        last_output = lstm_out[:, -1, :]
        encoded = self.output_layer(last_output)
        return encoded 