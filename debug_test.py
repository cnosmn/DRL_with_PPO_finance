"""
Debug Test - Model boyutlarını kontrol et
"""

import torch
from data_manager import DataManager
from environment import PortfolioEnvironment
from agents import PPOAgent
from config import *


def check_model_dimensions():
    """Model ve ortam boyutlarını kontrol et"""
    print("🔍 MODEL VE ORTAM BOYUTLARI KONTROLÜ")
    print("=" * 50)
    
    # Eğitim parametrelerini yazdır
    print(f"📋 Eğitim Konfigürasyonu:")
    print(f"   - Hisse Senetleri: {STOCK_SYMBOLS}")
    print(f"   - Hisse Sayısı: {len(STOCK_SYMBOLS)}")
    print(f"   - Veri Süresi: {DATA_PERIOD}")
    print(f"   - Başlangıç Sermaye: ${INITIAL_BALANCE:,}")
    
    # Veri hazırla
    print(f"\n📊 Veri hazırlanıyor...")
    data_manager = DataManager()
    raw_data = data_manager.download_stock_data(STOCK_SYMBOLS, "6mo")
    processed_data = data_manager.process_data(raw_data)
    
    # Ortam oluştur
    env = PortfolioEnvironment(processed_data)
    state_dim = len(env.get_state())
    action_dim = processed_data['n_stocks'] + 1
    
    print(f"\n🎯 Mevcut Ortam Boyutları:")
    print(f"   - Hisse Sayısı: {processed_data['n_stocks']}")
    print(f"   - Durum Boyutu: {state_dim}")
    print(f"   - Aksiyon Boyutu: {action_dim}")
    
    # Model dosyasını kontrol et
    model_path = "best_portfolio_agent.pt"
    
    try:
        # Model dosyasını yükle ve boyutları kontrol et
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"\n🔧 Kaydedilmiş Model Boyutları:")
        
        # Network state dict'ten boyutları çıkar
        state_dict = checkpoint['network_state_dict']
        
        # İlk layer'dan input boyutunu al
        first_layer_weight = state_dict['shared_layers.0.weight']
        saved_input_dim = first_layer_weight.shape[1]
        
        # Son layer'dan output boyutunu al  
        actor_output_weight = state_dict['actor_head.2.weight']
        saved_output_dim = actor_output_weight.shape[0]
        
        print(f"   - Kaydedilmiş Durum Boyutu: {saved_input_dim}")
        print(f"   - Kaydedilmiş Aksiyon Boyutu: {saved_output_dim}")
        
        # Boyut uyumluluğunu kontrol et
        print(f"\n⚖️  BOYUT KARŞILAŞTIRMASI:")
        print(f"   Durum Boyutu - Mevcut: {state_dim}, Kaydedilmiş: {saved_input_dim}")
        print(f"   Aksiyon Boyutu - Mevcut: {action_dim}, Kaydedilmiş: {saved_output_dim}")
        
        if state_dim == saved_input_dim and action_dim == saved_output_dim:
            print("   ✅ BOYUTLAR UYUMLU! Test yapılabilir.")
            return True, env, state_dim, action_dim
        else:
            print("   ❌ BOYUTLAR UYUMSUZ!")
            
            # Hangi hisse senetlerinin eksik olduğunu tahmin et
            expected_stocks = saved_output_dim - 1  # -1 nakit için
            current_stocks = action_dim - 1
            
            print(f"\n💡 ÇÖZÜM ÖNERİLERİ:")
            print(f"   - Model {expected_stocks} hisse senedi için eğitilmiş")
            print(f"   - Mevcut ortam {current_stocks} hisse senedi kullanıyor")
            
            if expected_stocks > current_stocks:
                missing = expected_stocks - current_stocks
                print(f"   - {missing} hisse senedi eksik!")
                print(f"   - Tüm {len(STOCK_SYMBOLS)} hisse senedini kullanın")
            
            return False, env, state_dim, action_dim
            
    except Exception as e:
        print(f"❌ Model dosyası okuma hatası: {e}")
        return False, None, 0, 0


def test_with_correct_dimensions():
    """Doğru boyutlarla test yap"""
    success, env, state_dim, action_dim = check_model_dimensions()
    
    if not success:
        print("\n❌ Boyut uyumsuzluğu nedeniyle test yapılamıyor!")
        return None
    
    print(f"\n🚀 Test başlatılıyor...")
    
    try:
        # Agent oluştur
        agent = PPOAgent(state_dim, action_dim)
        
        # Modeli yükle
        agent.load_agent("best_portfolio_agent.pt")
        print(f"✅ Model başarıyla yüklendi!")
        
        # Kısa bir test yap
        state = env.reset()
        portfolio_values = [env.initial_balance]
        
        print(f"\n📊 5 günlük hızlı test:")
        
        for day in range(5):
            action_idx, _, _ = agent.select_action(state, training=False)
            action_vector = np.zeros(action_dim)
            action_vector[action_idx] = 1.0
            
            state, _, done, info = env.step(action_vector)
            portfolio_value = info['portfolio_value']
            portfolio_values.append(portfolio_value)
            
            daily_return = info['daily_return'] * 100
            total_return = (portfolio_value / env.initial_balance - 1) * 100
            
            print(f"   Gün {day+1}: ${portfolio_value:,.0f} (Günlük: %{daily_return:.2f}, Toplam: %{total_return:.2f})")
            
            if done:
                break
        
        final_value = portfolio_values[-1]
        total_return = (final_value / env.initial_balance - 1) * 100
        
        print(f"\n✅ TEST BAŞARILI!")
        print(f"   Final Değer: ${final_value:,.2f}")
        print(f"   5 günlük getiri: %{total_return:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test hatası: {e}")
        return False


if __name__ == "__main__":
    import numpy as np
    
    print("🔧 MODEL DEBUG VE TEST ARACI")
    print("=" * 60)
    
    # Boyutları kontrol et ve test yap
    test_with_correct_dimensions() 