"""
PPO Agent - Proximal Policy Optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from models import PPONetwork
from config import LEARNING_RATE, GAMMA, EPS_CLIP, K_EPOCHS


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) Agent
    """
    
    def __init__(self, state_dim, action_dim, lr=LEARNING_RATE, gamma=GAMMA, 
                 eps_clip=EPS_CLIP, k_epochs=K_EPOCHS):
        """
        PPO Agent başlatma
        
        Args:
            state_dim (int): Durum vektörü boyutu
            action_dim (int): Aksiyon sayısı
            lr (float): Öğrenme oranı
            gamma (float): Discount factor
            eps_clip (float): PPO clipping parametresi
            k_epochs (int): Her güncellemede kaç epoch eğitim
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # Ağları oluştur
        self.policy = PPONetwork(state_dim, action_dim)
        self.policy_old = PPONetwork(state_dim, action_dim)
        
        # Eski politikayı başlat
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Loss fonksiyonu
        self.mse_loss = nn.MSELoss()
        
        # Experience buffer
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }
        
        # Eğitim metrikleri
        self.training_metrics = {
            'actor_losses': [],
            'critic_losses': [],
            'total_losses': [],
            'entropies': []
        }
        
        print(f"PPO Agent oluşturuldu:")
        print(f"- Durum boyutu: {state_dim}")
        print(f"- Aksiyon boyutu: {action_dim}")
        print(f"- Öğrenme oranı: {lr}")
    
    def select_action(self, state, training=True):
        """
        Durum için aksiyon seç
        
        Args:
            state (np.array): Durum vektörü
            training (bool): Eğitim modunda mı
            
        Returns:
            tuple: (action_index, log_probability, state_value)
        """
        with torch.no_grad():
            # Durum tensörlere çevir
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
            else:
                state_tensor = state.unsqueeze(0) if len(state.shape) == 1 else state
            
            # Ağdan çıktı al
            action_logits, state_value = self.policy_old(state_tensor)
            
            # Aksiyon olasılıklarını hesapla
            action_probs = F.softmax(action_logits, dim=-1)
            
            if training:
                # Eğitim modunda: stokastik sampling
                action_dist = Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            else:
                # Test modunda: en yüksek olasılıklı aksiyonu seç
                action = torch.argmax(action_probs, dim=-1)
                action_dist = Categorical(action_probs)
                log_prob = action_dist.log_prob(action)
            
            return action.item(), log_prob.item(), state_value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """
        Deneyimi buffer'a sakla
        
        Args:
            state: Durum
            action: Aksiyon
            reward: Ödül
            log_prob: Log probability
            value: State value
            done: Terminal durum mu
        """
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['values'].append(value)
        self.buffer['dones'].append(done)
    
    def update(self):
        """
        Politikayı güncelle (PPO algoritması)
        
        Returns:
            dict: Eğitim metrikleri
        """
        if len(self.buffer['states']) == 0:
            return {}
        
        # Buffer'ı tensörlere çevir
        states = torch.FloatTensor(np.array(self.buffer['states']))
        actions = torch.LongTensor(self.buffer['actions'])
        old_log_probs = torch.FloatTensor(self.buffer['log_probs'])
        
        # Discounted rewards hesapla
        discounted_rewards = self._calculate_discounted_rewards()
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # PPO update döngüsü
        epoch_metrics = {
            'actor_loss': 0,
            'critic_loss': 0, 
            'total_loss': 0,
            'entropy': 0
        }
        
        for epoch in range(self.k_epochs):
            # Yeni politikadan çıktı al
            action_logits, state_values = self.policy(states)
            state_values = state_values.squeeze()
            
            # Yeni log probabilities
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)
            
            # Importance sampling ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Advantages hesapla
            advantages = discounted_rewards - state_values.detach()
            
            # Actor loss (PPO-Clip)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = self.mse_loss(state_values, discounted_rewards)
            
            # Entropy loss (exploration bonus)
            entropy_loss = -action_dist.entropy().mean()
            
            # Toplam loss
            total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            
            self.optimizer.step()
            
            # Metrikleri kaydet
            epoch_metrics['actor_loss'] += actor_loss.item()
            epoch_metrics['critic_loss'] += critic_loss.item()
            epoch_metrics['total_loss'] += total_loss.item()
            epoch_metrics['entropy'] += -entropy_loss.item()
        
        # Ortalama metrikleri hesapla
        for key in epoch_metrics:
            epoch_metrics[key] /= self.k_epochs
        
        # Training metrics'i güncelle (doğru anahtar eşleştirmesi)
        self.training_metrics['actor_losses'].append(epoch_metrics['actor_loss'])
        self.training_metrics['critic_losses'].append(epoch_metrics['critic_loss'])
        self.training_metrics['total_losses'].append(epoch_metrics['total_loss'])
        self.training_metrics['entropies'].append(epoch_metrics['entropy'])
        
        # Eski politikayı güncelle
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Buffer'ı temizle
        self.clear_buffer()
        
        return epoch_metrics
    
    def _calculate_discounted_rewards(self):
        """Discounted rewards hesapla"""
        rewards = []
        discounted_reward = 0
        
        for reward, done in zip(reversed(self.buffer['rewards']), 
                               reversed(self.buffer['dones'])):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        return rewards
    
    def clear_buffer(self):
        """Experience buffer'ı temizle"""
        for key in self.buffer:
            self.buffer[key] = []
    
    def save_agent(self, filepath):
        """Agent'ı kaydet"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'eps_clip': self.eps_clip,
            'k_epochs': self.k_epochs,
            'training_metrics': self.training_metrics
        }, filepath)
        print(f"Agent kaydedildi: {filepath}")
    
    def load_agent(self, filepath):
        """Agent'ı yükle"""
        checkpoint = torch.load(filepath)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'training_metrics' in checkpoint:
            self.training_metrics = checkpoint['training_metrics']
        
        print(f"Agent yüklendi: {filepath}")
        return checkpoint
    
    def get_buffer_size(self):
        """Buffer boyutunu döndür"""
        return len(self.buffer['states'])
    
    def get_training_summary(self):
        """Eğitim özetini döndür"""
        if not self.training_metrics['total_losses']:
            return "Henüz eğitim verisi yok"
        
        summary = f"""
        Eğitim Özeti:
        -------------
        Toplam Güncelleme: {len(self.training_metrics['total_losses'])}
        Son Actor Loss: {self.training_metrics['actor_losses'][-1]:.4f}
        Son Critic Loss: {self.training_metrics['critic_losses'][-1]:.4f}
        Son Total Loss: {self.training_metrics['total_losses'][-1]:.4f}
        Son Entropy: {self.training_metrics['entropies'][-1]:.4f}
        
        Ortalama Metrikler (son 10 güncelleme):
        """
        
        if len(self.training_metrics['total_losses']) >= 10:
            recent_actor = np.mean(self.training_metrics['actor_losses'][-10:])
            recent_critic = np.mean(self.training_metrics['critic_losses'][-10:])
            recent_total = np.mean(self.training_metrics['total_losses'][-10:])
            recent_entropy = np.mean(self.training_metrics['entropies'][-10:])
            
            summary += f"""
        Actor Loss: {recent_actor:.4f}
        Critic Loss: {recent_critic:.4f}
        Total Loss: {recent_total:.4f}
        Entropy: {recent_entropy:.4f}
        """
        
        return summary 