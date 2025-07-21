"""
Hızlı Model Testi - Eğitilmiş Modeli Basit Test Et
"""

import numpy as np
from data_manager import DataManager
from environment import PortfolioEnvironment
from agents import PPOAgent
from config import *


def quick_test_model(model_path="best_portfolio_agent.pt", test_days=30):
    """
    Eğitilmiş modeli hızlıca test et
    
    Args:
        model_path (str): Model dosya yolu
        test_days (int): Test edilecek gün sayısı
    """
    print("⚡ HIZLI MODEL TESTİ")
    print("=" * 40)
    
    try:
        # Veri hazırla - EĞİTİM İLE AYNI HİSSE SENETLERİNİ KULLAN
        data_manager = DataManager()
        raw_data = data_manager.download_stock_data(STOCK_SYMBOLS, "6mo")  # Tüm hisse senetleri
        processed_data = data_manager.process_data(raw_data)
        
        # Ortam oluştur
        env = PortfolioEnvironment(processed_data)
        state_dim = len(env.get_state())
        action_dim = processed_data['n_stocks'] + 1
        
        print(f"📊 Test ortamı bilgileri:")
        print(f"   - Hisse sayısı: {processed_data['n_stocks']}")
        print(f"   - Durum boyutu: {state_dim}")
        print(f"   - Aksiyon boyutu: {action_dim}")
        
        # Modeli yükle
        agent = PPOAgent(state_dim, action_dim)
        agent.load_agent(model_path)
        print(f"✅ Model yüklendi: {model_path}")
        
        # Test çalıştır
        state = env.reset()
        portfolio_values = [INITIAL_BALANCE]
        
        print(f"\n📊 {test_days} günlük test başlıyor...")
        print("Gün | Portföy Değeri | Günlük Getiri | Toplam Getiri")
        print("-" * 55)
        
        for day in range(min(test_days, processed_data['n_days']-1)):
            # Aksiyon seç
            action_idx, _, _ = agent.select_action(state, training=False)
            action_vector = np.zeros(action_dim)
            action_vector[action_idx] = 1.0
            
            # Adım at
            state, _, done, info = env.step(action_vector)
            portfolio_value = info['portfolio_value']
            portfolio_values.append(portfolio_value)
            
            # Getiri hesapla
            daily_return = info['daily_return'] * 100
            total_return = (portfolio_value / INITIAL_BALANCE - 1) * 100
            
            # Rapor
            if day % 5 == 0 or day < 5:  # İlk 5 gün ve her 5 günde bir
                print(f"{day+1:3d} | ${portfolio_value:>12,.0f} | {daily_return:>9.2f}% | {total_return:>9.2f}%")
            
            if done:
                break
        
        print("-" * 55)
        
        # Özet sonuçlar
        final_value = portfolio_values[-1]
        total_return = (final_value / INITIAL_BALANCE - 1) * 100
        
        print(f"\n🎯 TEST SONUÇLARI:")
        print(f"   Başlangıç: ${INITIAL_BALANCE:,}")
        print(f"   Final Değer: ${final_value:,.2f}")
        print(f"   Toplam Getiri: %{total_return:.2f}")
        print(f"   Test Gün Sayısı: {len(portfolio_values)-1}")
        
        # Performans değerlendirmesi
        if total_return > 10:
            print("   📈 Mükemmel performans!")
        elif total_return > 5:
            print("   📊 İyi performans!")
        elif total_return > 0:
            print("   ↗️  Pozitif getiri")
        else:
            print("   📉 Negatif getiri")
            
        return {
            'final_value': final_value,
            'total_return': total_return,
            'portfolio_history': portfolio_values
        }
        
    except FileNotFoundError:
        print(f"❌ Model dosyası bulunamadı: {model_path}")
        print("   Şu komutla modeli eğitin:")
        print("   python main.py --quick")
        return None
    except Exception as e:
        print(f"❌ Test hatası: {e}")
        return None


def compare_with_random():
    """Eğitilmiş model ile rastgele stratejinin karşılaştırması"""
    print("\n🎲 RASTGELE STRATEJİ KARŞILAŞTIRMASI")
    print("=" * 50)
    
    # Eğitilmiş model sonucu
    trained_result = quick_test_model()
    if not trained_result:
        return
    
    # Rastgele strateji simülasyonu
    print("\n🎯 Rastgele strateji test ediliyor...")
    
    # Basit rastgele getiri simülasyonu
    np.random.seed(42)  # Tekrarlanabilir sonuçlar için
    random_returns = np.random.normal(0.001, 0.02, 30)  # Ortalama %0.1, volatilite %2
    
    random_portfolio = [INITIAL_BALANCE]
    for daily_return in random_returns:
        new_value = random_portfolio[-1] * (1 + daily_return)
        random_portfolio.append(new_value)
    
    random_final = random_portfolio[-1]
    random_return = (random_final / INITIAL_BALANCE - 1) * 100
    
    print(f"\n📊 KARŞILAŞTIRMA:")
    print("-" * 40)
    print(f"Eğitilmiş Model:")
    print(f"  Final Değer: ${trained_result['final_value']:,.2f}")
    print(f"  Getiri: %{trained_result['total_return']:.2f}")
    print()
    print(f"Rastgele Strateji:")
    print(f"  Final Değer: ${random_final:,.2f}")
    print(f"  Getiri: %{random_return:.2f}")
    print()
    
    # Kazanan
    if trained_result['total_return'] > random_return:
        diff = trained_result['total_return'] - random_return
        print(f"🏆 Eğitilmiş model kazandı! ({diff:.2f}% daha iyi)")
    else:
        diff = random_return - trained_result['total_return']
        print(f"😅 Rastgele strateji daha iyi ({diff:.2f}% fark)")


if __name__ == "__main__":
    # Hızlı test
    result = quick_test_model()
    
    if result:
        # Rastgele ile karşılaştır
        compare_with_random()
        
        print("\n🔬 Daha detaylı test için:")
        print("python test_model.py")
    
    print("\n✨ Test tamamlandı!") 