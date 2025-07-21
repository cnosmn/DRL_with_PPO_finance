"""
Eğitilmiş PPO Modelini Test Etme
"""

import numpy as np
import matplotlib.pyplot as plt
from data_manager import DataManager
from environment import PortfolioEnvironment  
from agents import PPOAgent
from utils import PerformanceAnalyzer
from config import *


def load_and_test_model(model_path="best_portfolio_agent.pt"):
    """
    Kaydedilmiş modeli yükle ve test et
    
    Args:
        model_path (str): Model dosya yolu
        
    Returns:
        dict: Test sonuçları
    """
    print("🔬 Eğitilmiş Model Test Ediliyor...")
    print("=" * 50)
    
    # Veri hazırlama
    data_manager = DataManager()
    print("📊 Test verileri indiriliyor...")
    
    # EĞİTİM İLE AYNI HİSSE SENETLERİNİ KULLAN
    raw_data = data_manager.download_stock_data(STOCK_SYMBOLS, "1y")  # Tüm hisse senetleri
    
    if len(raw_data) < 2:
        print("❌ Test verisi bulunamadı!")
        return None
    
    processed_data = data_manager.process_data(raw_data)
    
    # Ortam oluştur
    env = PortfolioEnvironment(processed_data)
    
    # Durum ve aksiyon boyutları
    state_dim = len(env.get_state())
    action_dim = processed_data['n_stocks'] + 1
    
    print(f"✅ Test ortamı hazır:")
    print(f"   - Hisse sayısı: {processed_data['n_stocks']}")
    print(f"   - Test günü: {processed_data['n_days']}")
    print(f"   - Durum boyutu: {state_dim}")
    
    # Agent oluştur ve modeli yükle
    agent = PPOAgent(state_dim, action_dim)
    
    try:
        agent.load_agent(model_path)
        print(f"✅ Model yüklendi: {model_path}")
    except FileNotFoundError:
        print(f"❌ Model dosyası bulunamadı: {model_path}")
        print("   Önce modeli eğitmeniz gerekiyor!")
        return None
    
    # Test çalıştır
    print("\n🚀 Test başlıyor...")
    portfolio_history = test_episode(agent, env, action_dim)
    
    # Performans analizi
    analyzer = PerformanceAnalyzer()
    
    # Benchmark portföy oluştur
    benchmark_portfolio = analyzer.create_benchmark_portfolio(
        raw_data, len(portfolio_history)
    )
    
    # Metrikleri hesapla
    agent_metrics = analyzer.calculate_performance_metrics(portfolio_history)
    benchmark_metrics = analyzer.calculate_performance_metrics(benchmark_portfolio)
    
    # Sonuçları görselleştir
    plot_test_results(portfolio_history, benchmark_portfolio, raw_data)
    
    # Performans raporu
    analyzer.print_performance_report(agent_metrics, benchmark_metrics)
    
    return {
        'agent_metrics': agent_metrics,
        'benchmark_metrics': benchmark_metrics,
        'portfolio_history': portfolio_history,
        'benchmark_portfolio': benchmark_portfolio
    }


def test_episode(agent, env, action_dim):
    """
    Tek bir episode test et
    
    Args:
        agent: Eğitilmiş agent
        env: Test ortamı
        action_dim: Aksiyon boyutu
        
    Returns:
        list: Portföy değer geçmişi
    """
    state = env.reset()
    portfolio_history = [env.initial_balance]
    done = False
    day = 0
    
    print("Günlük işlemler:")
    print("-" * 40)
    
    while not done:
        # Deterministik aksiyon seç (test modunda)
        action_idx, _, _ = agent.select_action(state, training=False)
        
        # Aksiyon vektörü oluştur
        action_vector = np.zeros(action_dim)
        action_vector[action_idx] = 1.0
        
        # Adım at
        state, _, done, info = env.step(action_vector)
        portfolio_value = info['portfolio_value']
        portfolio_history.append(portfolio_value)
        
        # Her 10 günde bir rapor
        if day % 10 == 0:
            daily_return = info['daily_return'] * 100
            total_return = (portfolio_value / INITIAL_BALANCE - 1) * 100
            
            print(f"Gün {day:3d}: ${portfolio_value:8,.0f} "
                  f"(Günlük: %{daily_return:5.2f}, "
                  f"Toplam: %{total_return:6.2f})")
        
        day += 1
    
    print("-" * 40)
    print(f"✅ Test tamamlandı! Toplam {day} gün")
    
    return portfolio_history


def plot_test_results(agent_portfolio, benchmark_portfolio, stock_data):
    """Test sonuçlarını görselleştir"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Portföy karşılaştırması
    days = range(len(agent_portfolio))
    axes[0, 0].plot(days, agent_portfolio, 
                   label='DRL Agent', linewidth=2, color='blue')
    axes[0, 0].plot(days, benchmark_portfolio[:len(agent_portfolio)], 
                   label='Benchmark', linewidth=2, color='red', alpha=0.7)
    axes[0, 0].axhline(y=INITIAL_BALANCE, color='gray', linestyle='--', 
                      label=f'Başlangıç (${INITIAL_BALANCE:,})')
    
    axes[0, 0].set_title('Test Performansı Karşılaştırması', fontsize=14)
    axes[0, 0].set_xlabel('Gün')
    axes[0, 0].set_ylabel('Portföy Değeri ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Günlük getiriler
    agent_returns = np.diff(agent_portfolio) / np.array(agent_portfolio[:-1]) * 100
    benchmark_returns = np.diff(benchmark_portfolio[:len(agent_portfolio)]) / np.array(benchmark_portfolio[:len(agent_portfolio)-1]) * 100
    
    axes[0, 1].plot(agent_returns, label='DRL Agent', alpha=0.7)
    axes[0, 1].plot(benchmark_returns, label='Benchmark', alpha=0.7)
    axes[0, 1].set_title('Günlük Getiriler', fontsize=14)
    axes[0, 1].set_xlabel('Gün')
    axes[0, 1].set_ylabel('Günlük Getiri (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Getiri dağılımı
    axes[1, 0].hist(agent_returns, bins=20, alpha=0.7, label='DRL Agent', color='blue')
    axes[1, 0].hist(benchmark_returns, bins=20, alpha=0.7, label='Benchmark', color='red')
    axes[1, 0].set_title('Getiri Dağılımı', fontsize=14)
    axes[1, 0].set_xlabel('Günlük Getiri (%)')
    axes[1, 0].set_ylabel('Frekans')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Kümülatif getiri
    agent_cumulative = np.cumprod(1 + np.array(agent_returns) / 100) - 1
    benchmark_cumulative = np.cumprod(1 + np.array(benchmark_returns) / 100) - 1
    
    axes[1, 1].plot(agent_cumulative * 100, label='DRL Agent', linewidth=2)
    axes[1, 1].plot(benchmark_cumulative * 100, label='Benchmark', linewidth=2)
    axes[1, 1].set_title('Kümülatif Getiri', fontsize=14)
    axes[1, 1].set_xlabel('Gün')
    axes[1, 1].set_ylabel('Kümülatif Getiri (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_multiple_models():
    """Birden fazla modeli karşılaştır"""
    print("🔍 Birden Fazla Model Karşılaştırması")
    print("=" * 50)
    
    models = [
        "best_portfolio_agent.pt",
        "portfolio_results_agent_*.pt"  # En son kaydedilen model
    ]
    
    results = {}
    
    for model_path in models:
        if model_path.endswith("*.pt"):
            # En son modeli bul
            import glob
            model_files = glob.glob(model_path)
            if model_files:
                model_path = sorted(model_files)[-1]  # En son dosya
            else:
                continue
        
        print(f"\n📊 Test ediliyor: {model_path}")
        result = load_and_test_model(model_path)
        if result:
            results[model_path] = result
    
    # Karşılaştırma tablosu
    if len(results) > 1:
        print("\n🏆 MODEL KARŞILAŞTIRMA TABLOSU")
        print("=" * 70)
        print(f"{'Model':<30} {'Final Değer':<15} {'Toplam Getiri':<15} {'Sharpe':<10}")
        print("-" * 70)
        
        for model_name, result in results.items():
            metrics = result['agent_metrics']
            short_name = model_name.split('/')[-1][:25]
            print(f"{short_name:<30} ${metrics['final_value']:>10,.0f} "
                  f"{metrics['total_return_pct']:>12.2f}% "
                  f"{metrics['sharpe_ratio']:>8.3f}")


if __name__ == "__main__":
    # Tek model testi
    result = load_and_test_model()
    
    if result:
        print(f"\n📈 Test başarılı!")
        print(f"Final portföy değeri: ${result['agent_metrics']['final_value']:,.2f}")
        print(f"Toplam getiri: %{result['agent_metrics']['total_return_pct']:.2f}")
        
        # Birden fazla model varsa karşılaştır
        print("\n" + "="*50)
        compare_multiple_models() 