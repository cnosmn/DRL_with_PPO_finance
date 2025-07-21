"""
Ana Eğitim Dosyası - PPO Portföy Yönetimi
"""

import numpy as np
import warnings
from tqdm import tqdm

# Kendi modüllerimizi içe aktar
from config import *
import config
from data_manager import DataManager
from environment import PortfolioEnvironment
from agents import PPOAgent
from utils import PerformanceAnalyzer, setup_plotting, save_results, print_system_info

warnings.filterwarnings('ignore')


def train_portfolio_agent():
    """
    Ana eğitim fonksiyonu
    
    Returns:
        tuple: (trained_agent, environment, results)
    """
    # Sistem bilgilerini yazdır
    print_system_info()
    
    # Matplotlib ayarlarını yap
    setup_plotting()
    
    # Veri yöneticisi oluştur
    data_manager = DataManager()
    
    print(f"\n{STOCK_SYMBOLS} hisse senetleri için veri indiriliyor...")
    raw_data = data_manager.download_stock_data(STOCK_SYMBOLS, DATA_PERIOD)
    
    if len(raw_data) < 2:
        print("❌ Yeterli veri bulunamadı!")
        return None, None, None
    
    print(f"✅ Toplam {len(raw_data)} hisse senedi verisi indirildi")
    
    # Veriyi işle
    processed_data = data_manager.process_data(raw_data)
    print(data_manager.get_data_summary())
    
    # Ortamı oluştur
    env = PortfolioEnvironment(processed_data)
    
    # Durum ve aksiyon boyutlarını belirle
    state_dim = len(env.get_state())
    action_dim = processed_data['n_stocks'] + 1  # Hisse senetleri + nakit
    
    print(f"\n📊 Model Parametreleri:")
    print(f"- Durum boyutu: {state_dim}")
    print(f"- Aksiyon boyutu: {action_dim}")
    print(f"- Episode sayısı: {config.NUM_EPISODES}")
    
    # Agent'ı oluştur
    agent = PPOAgent(state_dim, action_dim)
    
    # Performans analiz aracı
    analyzer = PerformanceAnalyzer()
    
    # Eğitim döngüsü
    episode_rewards = []
    episode_portfolio_values = []
    best_portfolio_value = 0
    
    print(f"\n🎯 Eğitim başlıyor...")
    print("="*60)
    
    for episode in tqdm(range(config.NUM_EPISODES), desc="Eğitim"):
        # Episode başlangıcı
        state = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        # Episode döngüsü
        while not done:
            # Aksiyon seç
            action_idx, log_prob, value = agent.select_action(state, training=True)
            
            # Aksiyon vektörünü oluştur
            action_vector = np.zeros(action_dim)
            action_vector[action_idx] = 1.0
            
            # Adım at
            next_state, reward, done, info = env.step(action_vector)
            
            # Deneyimi kaydet
            agent.store_transition(state, action_idx, reward, log_prob, value, done)
            
            # Durumu güncelle
            state = next_state
            episode_reward += reward
            step_count += 1
        
        # Episode sonuçlarını kaydet
        final_portfolio_value = env.portfolio_history[-1]
        episode_rewards.append(episode_reward)
        episode_portfolio_values.append(final_portfolio_value)
        
        # En iyi modeli kaydet
        if final_portfolio_value > best_portfolio_value:
            best_portfolio_value = final_portfolio_value
            agent.save_agent("best_portfolio_agent.pt")
        
        # Belirli aralıklarla güncelle
        if (episode + 1) % config.UPDATE_FREQUENCY == 0:
            update_metrics = agent.update()
        
        # İlerleme raporu
        if (episode + 1) % config.REPORT_FREQUENCY == 0:
            avg_reward = np.mean(episode_rewards[-config.REPORT_FREQUENCY:])
            avg_portfolio_value = np.mean(episode_portfolio_values[-config.REPORT_FREQUENCY:])
            total_return = (avg_portfolio_value / INITIAL_BALANCE - 1) * 100
            
            print(f"\n📈 Episode {episode + 1}/{config.NUM_EPISODES}")
            print(f"   Ortalama Ödül (son {config.REPORT_FREQUENCY}): {avg_reward:.2f}")
            print(f"   Ortalama Portföy Değeri: ${avg_portfolio_value:,.2f}")
            print(f"   Getiri: %{total_return:.2f}")
            
            if agent.training_metrics['total_losses']:
                print(f"   Son Total Loss: {agent.training_metrics['total_losses'][-1]:.4f}")
            
            print("-" * 50)
    
    print("\n✅ Eğitim tamamlandı!")
    
    # Final performans testi
    print("\n🧪 Final performans testi çalıştırılıyor...")
    final_agent_portfolio = run_final_test(agent, env, action_dim)
    
    # Benchmark portföy oluştur
    benchmark_portfolio = analyzer.create_benchmark_portfolio(
        raw_data, len(final_agent_portfolio)
    )
    
    # Performans metrikleri hesapla
    agent_metrics = analyzer.calculate_performance_metrics(final_agent_portfolio)
    benchmark_metrics = analyzer.calculate_performance_metrics(benchmark_portfolio)
    
    # Sonuçları görselleştir
    print("\n📊 Sonuçlar görselleştiriliyor...")
    
    # Eğitim sonuçları
    analyzer.plot_training_results(
        episode_rewards, 
        episode_portfolio_values, 
        agent.training_metrics
    )
    
    # Final karşılaştırma
    analyzer.plot_final_comparison(
        final_agent_portfolio, 
        benchmark_portfolio, 
        raw_data
    )
    
    # Performans raporu
    analyzer.print_performance_report(agent_metrics, benchmark_metrics)
    
    # Sonuçları kaydet
    results = {
        'agent_metrics': agent_metrics,
        'benchmark_metrics': benchmark_metrics,
        'episode_rewards': episode_rewards,
        'episode_portfolio_values': episode_portfolio_values,
        'final_agent_portfolio': final_agent_portfolio,
        'config': {
            'stock_symbols': STOCK_SYMBOLS,
            'num_episodes': config.NUM_EPISODES,
            'initial_balance': INITIAL_BALANCE,
            'learning_rate': LEARNING_RATE
        }
    }
    
    save_results(agent, env, results)
    
    return agent, env, results


def run_final_test(agent, env, action_dim):
    """
    Final test - eğitilmiş agent'la bir episode çalıştır
    
    Args:
        agent: Eğitilmiş agent
        env: Ortam
        action_dim: Aksiyon boyutu
        
    Returns:
        list: Portföy değer geçmişi
    """
    state = env.reset()
    portfolio_history = [env.initial_balance]
    done = False
    
    while not done:
        # Test modunda deterministik aksiyon seç
        action_idx, _, _ = agent.select_action(state, training=False)
        
        # Aksiyon vektörü oluştur
        action_vector = np.zeros(action_dim)
        action_vector[action_idx] = 1.0
        
        # Adım at
        state, _, done, info = env.step(action_vector)
        portfolio_history.append(info['portfolio_value'])
    
    return portfolio_history


def quick_test():
    """
    Hızlı test için küçük ölçekli eğitim
    """
    print("🔥 Hızlı test modu - sadece 50 episode")
    
    # Config modülünü doğrudan güncelle
    import config
    original_episodes = config.NUM_EPISODES
    original_frequency = config.REPORT_FREQUENCY
    
    config.NUM_EPISODES = 50
    config.REPORT_FREQUENCY = 10
    
    try:
        result = train_portfolio_agent()
        return result
    finally:
        # Parametreleri geri yükle
        config.NUM_EPISODES = original_episodes
        config.REPORT_FREQUENCY = original_frequency


if __name__ == "__main__":
    import sys
    
    print("🚀 PPO Portföy Yönetimi Projesi")
    print("=" * 50)
    
    # Komut satırı argümanlarını kontrol et
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        trained_agent, environment, results = quick_test()
    else:
        trained_agent, environment, results = train_portfolio_agent()
    
    if trained_agent is not None:
        print("\n🎉 Proje başarıyla tamamlandı!")
        print("\nEğitilmiş modeli test etmek için:")
        print("python -c \"from main import *; agent, env, _ = train_portfolio_agent()\"")
    else:
        print("\n❌ Proje başarısız oldu. Lütfen veri bağlantınızı kontrol edin.") 