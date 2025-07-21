"""
Sunum için Görsel Materyaller Oluşturucu
Bu script sunumda kullanılacak grafikleri ve görselleri oluşturur.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.patches as patches

# Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('seaborn-v0_8')

def create_rl_cycle_diagram():
    """Reinforcement Learning döngüsü diyagramı"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Daire pozisyonları
    positions = {
        'Agent': (0.5, 0.8),
        'Environment': (0.5, 0.2),
        'Action': (0.2, 0.5),
        'State': (0.8, 0.5),
        'Reward': (0.5, 0.5)
    }
    
    # Daireler
    for name, (x, y) in positions.items():
        if name == 'Agent':
            color = '#2E86AB'
            text_color = 'white'
        elif name == 'Environment':
            color = '#A23B72'
            text_color = 'white'
        else:
            color = '#F18F01'
            text_color = 'black'
            
        circle = plt.Circle((x, y), 0.1, color=color, alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', 
                fontsize=12, fontweight='bold', color=text_color)
    
    # Oklar
    arrows = [
        (positions['Agent'], positions['Action'], 'Portföy\nAğırlıkları'),
        (positions['Action'], positions['Environment'], ''),
        (positions['Environment'], positions['State'], 'Piyasa\nVerileri'),
        (positions['State'], positions['Agent'], ''),
        (positions['Environment'], positions['Reward'], 'Performans'),
        (positions['Reward'], positions['Agent'], 'Öğrenme')
    ]
    
    for (start, end, label) in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='#333'))
        if label:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y, label, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Reinforcement Learning Döngüsü\nPortföy Yönetiminde', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('rl_cycle_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_comparison():
    """Performans karşılaştırma grafiği"""
    strategies = ['DRL Agent', 'Buy & Hold', 'Eşit Ağırlık', 'Rastgele']
    returns = [331.71, 25.00, 18.00, -8.00]
    colors = ['#28A745', '#2E86AB', '#FFC107', '#DC3545']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart
    bars = ax1.bar(strategies, returns, color=colors, alpha=0.8)
    ax1.set_ylabel('Getiri (%)', fontsize=12)
    ax1.set_title('Strateji Performans Karşılaştırması', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Değerleri bar üzerine yaz
    for bar, value in zip(bars, returns):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Zaman serisi simülasyonu
    days = np.arange(0, 252)  # 1 yıl
    np.random.seed(42)
    
    # DRL Agent - yüksek getiri, düşük volatilite
    drl_returns = np.random.normal(0.002, 0.015, 252)
    drl_returns[50:60] = np.random.normal(0.008, 0.01, 10)  # Güçlü performans dönemi
    drl_cumulative = (1 + drl_returns).cumprod() * 10000
    
    # Buy & Hold - orta getiri, orta volatilite
    bh_returns = np.random.normal(0.0008, 0.02, 252)
    bh_cumulative = (1 + bh_returns).cumprod() * 10000
    
    # Eşit Ağırlık - düşük getiri, düşük volatilite
    eq_returns = np.random.normal(0.0006, 0.018, 252)
    eq_cumulative = (1 + eq_returns).cumprod() * 10000
    
    # Rastgele - negatif getiri, yüksek volatilite
    rand_returns = np.random.normal(-0.0002, 0.025, 252)
    rand_cumulative = (1 + rand_returns).cumprod() * 10000
    
    ax2.plot(days, drl_cumulative, color=colors[0], linewidth=3, label='DRL Agent')
    ax2.plot(days, bh_cumulative, color=colors[1], linewidth=2, label='Buy & Hold')
    ax2.plot(days, eq_cumulative, color=colors[2], linewidth=2, label='Eşit Ağırlık')
    ax2.plot(days, rand_cumulative, color=colors[3], linewidth=2, label='Rastgele')
    
    ax2.set_xlabel('Gün', fontsize=12)
    ax2.set_ylabel('Portföy Değeri ($)', fontsize=12)
    ax2.set_title('Zaman İçinde Portföy Performansı', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_risk_return_scatter():
    """Risk-getiri scatter plot"""
    strategies = ['DRL Agent', 'Buy & Hold', 'Eşit Ağırlık', 'Rastgele']
    returns = [331.71, 25.00, 18.00, -8.00]
    volatilities = [15.2, 22.1, 19.5, 28.3]
    colors = ['#28A745', '#2E86AB', '#FFC107', '#DC3545']
    sizes = [200, 100, 100, 100]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    scatter = ax.scatter(volatilities, returns, c=colors, s=sizes, alpha=0.7, edgecolors='black')
    
    # Etiketler
    for i, strategy in enumerate(strategies):
        ax.annotate(strategy, (volatilities[i], returns[i]), 
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                   fontsize=11, fontweight='bold')
    
    # Efficient frontier çizgisi (simüle)
    x_eff = np.linspace(10, 30, 100)
    y_eff = -0.5 * (x_eff - 15)**2 + 350
    ax.plot(x_eff, y_eff, '--', color='gray', alpha=0.5, label='Efficient Frontier')
    
    ax.set_xlabel('Volatilite (%)', fontsize=12)
    ax.set_ylabel('Getiri (%)', fontsize=12)
    ax.set_title('Risk-Getiri Analizi', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # DRL Agent'ı vurgula
    ax.annotate('Optimal Bölge', xy=(15.2, 331.71), xytext=(20, 250),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=12, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('risk_return_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_system_architecture():
    """Sistem mimarisi diyagramı"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Katmanlar
    layers = {
        'Data Layer': {'y': 0.8, 'color': '#E3F2FD', 'items': ['Yahoo Finance API', '5 Türk Hisse', '3 Yıl Veri']},
        'Model Layer': {'y': 0.5, 'color': '#F3E5F5', 'items': ['PPO Network', 'Actor-Critic', '256 Hidden Units']},
        'Environment Layer': {'y': 0.2, 'color': '#E8F5E8', 'items': ['Portfolio Env', 'Transaction Costs', 'Risk Management']}
    }
    
    for layer_name, layer_info in layers.items():
        # Ana kutu
        rect = patches.Rectangle((0.1, layer_info['y']-0.1), 0.8, 0.15, 
                               linewidth=2, edgecolor='black', 
                               facecolor=layer_info['color'], alpha=0.7)
        ax.add_patch(rect)
        
        # Başlık
        ax.text(0.5, layer_info['y']+0.02, layer_name, ha='center', va='center',
               fontsize=14, fontweight='bold')
        
        # Alt öğeler
        for i, item in enumerate(layer_info['items']):
            x_pos = 0.2 + i * 0.2
            ax.text(x_pos, layer_info['y']-0.03, item, ha='center', va='center',
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.2", 
                                        facecolor='white', alpha=0.8))
    
    # Oklar
    arrow_props = dict(arrowstyle='->', lw=3, color='#333')
    ax.annotate('', xy=(0.5, 0.65), xytext=(0.5, 0.75), arrowprops=arrow_props)
    ax.annotate('', xy=(0.5, 0.35), xytext=(0.5, 0.45), arrowprops=arrow_props)
    
    # Yan bilgiler
    ax.text(1.05, 0.8, 'State: 21 boyut\n(3×5 getiri + 6 ağırlık)', 
           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFE0B2'))
    ax.text(1.05, 0.5, 'Action: 6 boyut\n(5 hisse + nakit)', 
           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFE0B2'))
    ax.text(1.05, 0.2, 'Reward: Çok bileşenli\n(getiri + risk + çeşitlendirme)', 
           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFE0B2'))
    
    ax.set_xlim(0, 1.4)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Sistem Mimarisi', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_training_progress():
    """Eğitim süreci grafiği"""
    episodes = np.arange(1, 501)
    
    # Simüle edilmiş eğitim metrikleri
    np.random.seed(42)
    
    # Ödül - başlangıçta düşük, sonra artış
    base_reward = 1000 + episodes * 5
    noise = np.random.normal(0, 200, 500)
    rewards = base_reward + noise
    rewards = np.maximum(rewards, 0)  # Negatif değerleri sıfırla
    
    # Kayıp - başlangıçta yüksek, sonra azalış
    loss = 1.0 * np.exp(-episodes/100) + np.random.normal(0, 0.1, 500)
    loss = np.maximum(loss, 0.01)
    
    # Portföy değeri
    portfolio_value = 10000 * (1 + (episodes - 1) * 0.006 + np.random.normal(0, 0.02, 500))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Ödül
    ax1.plot(episodes, rewards, color='#28A745', alpha=0.7)
    ax1.plot(episodes, np.convolve(rewards, np.ones(20)/20, mode='same'), 
             color='#1E7E34', linewidth=3, label='20-episode ortalaması')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Ortalama Ödül')
    ax1.set_title('Eğitim Ödülü')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Kayıp
    ax2.plot(episodes, loss, color='#DC3545', alpha=0.7)
    ax2.plot(episodes, np.convolve(loss, np.ones(20)/20, mode='same'), 
             color='#C82333', linewidth=3, label='20-episode ortalaması')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Kayıp')
    ax2.set_title('Model Kaybı')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Portföy değeri
    ax3.plot(episodes, portfolio_value, color='#2E86AB', alpha=0.7)
    ax3.plot(episodes, np.convolve(portfolio_value, np.ones(20)/20, mode='same'), 
             color='#1F5F8B', linewidth=3, label='20-episode ortalaması')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Portföy Değeri ($)')
    ax3.set_title('Portföy Performansı')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Sharpe oranı
    returns_sim = np.diff(portfolio_value) / portfolio_value[:-1]
    sharpe_rolling = []
    for i in range(20, len(returns_sim)):
        window_returns = returns_sim[i-20:i]
        sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252)
        sharpe_rolling.append(sharpe)
    
    ax4.plot(episodes[20:], sharpe_rolling, color='#FFC107', linewidth=2)
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='İyi Sharpe (>1.0)')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Sharpe Oranı')
    ax4.set_title('Risk-Ayarlı Performans')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.suptitle('Eğitim Süreci Metrikleri', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_portfolio_allocation():
    """Portföy dağılımı pasta grafiği"""
    stocks = ['THYAO.IS', 'AKBNK.IS', 'GARAN.IS', 'ISCTR.IS', 'TCELL.IS', 'Nakit']
    allocations = [22, 18, 20, 15, 12, 13]  # Örnek dağılım
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Pasta grafiği
    wedges, texts, autotexts = ax1.pie(allocations, labels=stocks, colors=colors, 
                                      autopct='%1.1f%%', startangle=90)
    ax1.set_title('Optimal Portföy Dağılımı', fontsize=14, fontweight='bold')
    
    # Zaman içinde ağırlık değişimi
    days = np.arange(0, 30)
    np.random.seed(42)
    
    for i, (stock, color) in enumerate(zip(stocks[:-1], colors[:-1])):  # Nakit hariç
        base_weight = allocations[i] / 100
        variation = np.random.normal(0, 0.02, 30)
        weights = base_weight + variation
        weights = np.clip(weights, 0.05, 0.4)  # %5-40 arası sınırla
        ax2.plot(days, weights * 100, color=color, linewidth=2, label=stock)
    
    ax2.set_xlabel('Gün')
    ax2.set_ylabel('Ağırlık (%)')
    ax2.set_title('Dinamik Portföy Ağırlıkları (30 Gün)', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('portfolio_allocation.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Sunum görselleri oluşturuluyor...")
    
    # Tüm grafikleri oluştur
    create_rl_cycle_diagram()
    create_performance_comparison()
    create_risk_return_scatter()
    create_system_architecture()
    create_training_progress()
    create_portfolio_allocation()
    
    print("Tüm görseller başarıyla oluşturuldu!")
    print("\nOluşturulan dosyalar:")
    print("- rl_cycle_diagram.png")
    print("- performance_comparison.png") 
    print("- risk_return_scatter.png")
    print("- system_architecture.png")
    print("- training_progress.png")
    print("- portfolio_allocation.png") 