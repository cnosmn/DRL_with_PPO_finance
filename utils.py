"""
Yardımcı Fonksiyonlar ve Görselleştirme Araçları
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from config import FIGURE_SIZE, PLOT_ALPHA, INITIAL_BALANCE


class PerformanceAnalyzer:
    """Performans analizi ve görselleştirme sınıfı"""
    
    def __init__(self):
        self.results = {}
    
    def calculate_performance_metrics(self, portfolio_history, initial_balance=INITIAL_BALANCE):
        """
        Portföy performans metriklerini hesapla
        
        Args:
            portfolio_history (list): Portföy değer geçmişi
            initial_balance (float): Başlangıç sermayesi
            
        Returns:
            dict: Performans metrikleri
        """
        portfolio_values = np.array(portfolio_history)
        
        # Temel metrikler
        final_value = portfolio_values[-1]
        total_return = (final_value / initial_balance - 1) * 100
        
        # Günlük getiriler
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        daily_returns = daily_returns[np.isfinite(daily_returns)]  # NaN ve inf temizle
        
        # Risk metrikleri
        volatility = np.std(daily_returns) * np.sqrt(252) * 100  # Yıllık volatilite
        sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
        
        # Drawdown hesaplama
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - cumulative_max) / cumulative_max * 100
        max_drawdown = np.min(drawdowns)
        
        # Kazanan/Kaybeden günler
        positive_days = np.sum(daily_returns > 0)
        negative_days = np.sum(daily_returns < 0)
        win_rate = positive_days / (positive_days + negative_days) * 100 if len(daily_returns) > 0 else 0
        
        metrics = {
            'final_value': final_value,
            'total_return_pct': total_return,
            'annualized_volatility_pct': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'win_rate_pct': win_rate,
            'total_days': len(portfolio_values),
            'positive_days': positive_days,
            'negative_days': negative_days
        }
        
        return metrics
    
    def create_benchmark_portfolio(self, stock_data, portfolio_length):
        """
        Eşit ağırlık benchmark portföyü oluştur
        
        Args:
            stock_data (dict): Hisse senedi verileri
            portfolio_length (int): Portföy uzunluğu
            
        Returns:
            list: Benchmark portföy değerleri
        """
        n_stocks = len(stock_data)
        stock_names = list(stock_data.keys())
        
        benchmark_portfolio = []
        
        for day in range(portfolio_length):
            portfolio_value = 0
            
            for stock_name in stock_names:
                if day < len(stock_data[stock_name]):
                    # Her hisse senedine eşit ağırlık
                    daily_return = stock_data[stock_name][day] / stock_data[stock_name][0]
                    portfolio_value += (INITIAL_BALANCE / n_stocks) * daily_return
                else:
                    portfolio_value += INITIAL_BALANCE / n_stocks
            
            benchmark_portfolio.append(portfolio_value)
        
        return benchmark_portfolio
    
    def plot_training_results(self, episode_rewards, portfolio_values, training_metrics=None):
        """
        Eğitim sonuçlarını görselleştir
        
        Args:
            episode_rewards (list): Episode ödülleri
            portfolio_values (list): Portföy değerleri
            training_metrics (dict): Eğitim metrikleri (opsiyonel)
        """
        fig_height = 15 if training_metrics else 10
        fig, axes = plt.subplots(2, 2, figsize=(FIGURE_SIZE[0], fig_height))
        
        # Episode ödülleri
        axes[0, 0].plot(episode_rewards, alpha=PLOT_ALPHA)
        axes[0, 0].set_title('Episode Ödülleri', fontsize=14)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Ödül')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Portföy değeri gelişimi
        axes[0, 1].plot(portfolio_values, alpha=PLOT_ALPHA, label='DRL Agent')
        axes[0, 1].axhline(y=INITIAL_BALANCE, color='r', linestyle='--', 
                          label=f'Başlangıç (${INITIAL_BALANCE:,})', alpha=PLOT_ALPHA)
        axes[0, 1].set_title('Portföy Değeri Gelişimi', fontsize=14)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Portföy Değeri ($)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Hareketli ortalama ödüller
        if len(episode_rewards) > 50:
            moving_avg = pd.Series(episode_rewards).rolling(window=50).mean()
            axes[1, 0].plot(moving_avg, color='orange', alpha=PLOT_ALPHA)
            axes[1, 0].set_title('Hareketli Ortalama Ödüller (50 Episode)', fontsize=14)
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Ortalama Ödül')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Portföy getiri dağılımı
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1]) * 100
            returns = returns[np.isfinite(returns)]
            axes[1, 1].hist(returns, bins=30, alpha=PLOT_ALPHA, edgecolor='black')
            axes[1, 1].set_title('Günlük Getiri Dağılımı', fontsize=14)
            axes[1, 1].set_xlabel('Günlük Getiri (%)')
            axes[1, 1].set_ylabel('Frekans')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Eğitim metrikleri varsa ayrı bir figür oluştur
        if training_metrics and training_metrics.get('total_losses'):
            self._plot_training_metrics(training_metrics)
    
    def _plot_training_metrics(self, training_metrics):
        """Eğitim metriklerini çiz"""
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
        
        # Actor Loss
        axes[0, 0].plot(training_metrics['actor_losses'])
        axes[0, 0].set_title('Actor Loss')
        axes[0, 0].set_xlabel('Güncelleme')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Critic Loss
        axes[0, 1].plot(training_metrics['critic_losses'])
        axes[0, 1].set_title('Critic Loss')
        axes[0, 1].set_xlabel('Güncelleme')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Total Loss
        axes[1, 0].plot(training_metrics['total_losses'])
        axes[1, 0].set_title('Total Loss')
        axes[1, 0].set_xlabel('Güncelleme')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Entropy
        axes[1, 1].plot(training_metrics['entropies'])
        axes[1, 1].set_title('Policy Entropy')
        axes[1, 1].set_xlabel('Güncelleme')
        axes[1, 1].set_ylabel('Entropy')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_final_comparison(self, agent_portfolio, benchmark_portfolio, stock_data):
        """
        Final portföy karşılaştırması
        
        Args:
            agent_portfolio (list): Agent portföy geçmişi
            benchmark_portfolio (list): Benchmark portföy geçmişi
            stock_data (dict): Ham hisse senedi verileri
        """
        fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE)
        
        # Portföy karşılaştırması
        days = range(len(agent_portfolio))
        axes[0].plot(days, agent_portfolio, label='DRL Agent', linewidth=2)
        axes[0].plot(days, benchmark_portfolio[:len(agent_portfolio)], 
                    label='Eşit Ağırlık Portföyü', alpha=PLOT_ALPHA, linewidth=2)
        axes[0].axhline(y=INITIAL_BALANCE, color='r', linestyle='--', 
                       label=f'Başlangıç (${INITIAL_BALANCE:,})', alpha=PLOT_ALPHA)
        
        axes[0].set_title('Portföy Performansı Karşılaştırması', fontsize=16)
        axes[0].set_xlabel('Gün')
        axes[0].set_ylabel('Portföy Değeri ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Hisse senedi fiyat gelişimi (normalize edilmiş)
        stock_names = list(stock_data.keys())
        for i, stock_name in enumerate(stock_names):
            if len(stock_data[stock_name]) > 0:
                normalized_prices = np.array(stock_data[stock_name]) / stock_data[stock_name][0]
                stock_days = range(len(normalized_prices))
                axes[1].plot(stock_days, normalized_prices, 
                           label=stock_name.replace('.IS', ''), alpha=PLOT_ALPHA)
        
        axes[1].set_title('Bireysel Hisse Senedi Performansı (Normalize)', fontsize=16)
        axes[1].set_xlabel('Gün')
        axes[1].set_ylabel('Normalize Fiyat')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_performance_report(self, agent_metrics, benchmark_metrics):
        """
        Detaylı performans raporu yazdır
        
        Args:
            agent_metrics (dict): Agent performans metrikleri
            benchmark_metrics (dict): Benchmark performans metrikleri
        """
        print("\n" + "="*70)
        print("                    PERFORMANS RAPORU")
        print("="*70)
        
        print(f"{'Metrik':<25} {'DRL Agent':<20} {'Benchmark':<20} {'Fark':<10}")
        print("-"*70)
        
        # Finansal metrikler
        metrics_to_compare = [
            ('Final Değer', 'final_value', '${:,.2f}'),
            ('Toplam Getiri', 'total_return_pct', '{:.2f}%'),
            ('Yıllık Volatilite', 'annualized_volatility_pct', '{:.2f}%'),
            ('Sharpe Oranı', 'sharpe_ratio', '{:.3f}'),
            ('Max Drawdown', 'max_drawdown_pct', '{:.2f}%'),
            ('Kazanma Oranı', 'win_rate_pct', '{:.1f}%')
        ]
        
        for metric_name, key, format_str in metrics_to_compare:
            agent_val = agent_metrics[key]
            benchmark_val = benchmark_metrics[key]
            
            if key in ['total_return_pct', 'sharpe_ratio']:
                diff = agent_val - benchmark_val
                diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
            else:
                diff_str = "-"
            
            print(f"{metric_name:<25} {format_str.format(agent_val):<20} "
                  f"{format_str.format(benchmark_val):<20} {diff_str:<10}")
        
        print("="*70)
        
        # Özet yorum
        agent_return = agent_metrics['total_return_pct']
        benchmark_return = benchmark_metrics['total_return_pct']
        
        if agent_return > benchmark_return:
            performance = "DRL Agent benchmark'dan daha iyi performans gösterdi! 🎉"
        elif agent_return > 0:
            performance = "DRL Agent pozitif getiri sağladı ancak benchmark'ı geçemedi."
        else:
            performance = "DRL Agent negatif getiri verdi. Model iyileştirmesi gerekebilir."
        
        print(f"\nSonuç: {performance}")
        print(f"Risk-Getiri Profili: Agent'ın Sharpe oranı {agent_metrics['sharpe_ratio']:.3f}")
        print("="*70)


def setup_plotting():
    """Matplotlib ayarlarını yapılandır"""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = FIGURE_SIZE
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def convert_numpy_types(obj):
    """
    NumPy tiplerini JSON serileştirilebilir tiplere çevir
    
    Args:
        obj: Dönüştürülecek nesne
        
    Returns:
        JSON serileştirilebilir nesne
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # NumPy scalar'lar için
        return obj.item()
    else:
        return obj


def save_results(agent, environment, metrics, filename_prefix="portfolio_results"):
    """
    Sonuçları kaydet
    
    Args:
        agent: Eğitilmiş agent
        environment: Ortam
        metrics: Performans metrikleri
        filename_prefix: Dosya adı öneki
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Agent'ı kaydet
    agent_filename = f"{filename_prefix}_agent_{timestamp}.pt"
    agent.save_agent(agent_filename)
    
    # Metrikleri JSON serileştirilebilir hale getir
    serializable_metrics = convert_numpy_types(metrics)
    
    # Metrikleri kaydet
    metrics_filename = f"{filename_prefix}_metrics_{timestamp}.json"
    import json
    with open(metrics_filename, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"Sonuçlar kaydedildi:")
    print(f"- Agent: {agent_filename}")
    print(f"- Metrikler: {metrics_filename}")


def print_system_info():
    """Sistem bilgilerini yazdır"""
    import torch
    import sys
    
    print("="*50)
    print("SİSTEM BİLGİLERİ")
    print("="*50)
    print(f"Python Versiyonu: {sys.version}")
    print(f"PyTorch Versiyonu: {torch.__version__}")
    print(f"CUDA Mevcut: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Versiyonu: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*50) 