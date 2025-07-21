"""
Model Test Komutları - Pratik Test Fonksiyonları
"""

import os
import subprocess
import sys
import glob

def test_best_model():
    """En iyi modeli test et"""
    print("🏆 En iyi model test ediliyor: best_portfolio_agent.pt")
    print("-" * 50)
    
    try:
        from quick_test import quick_test_model
        result = quick_test_model("best_portfolio_agent.pt", test_days=30)
        
        if result:
            print(f"\n✅ Test başarılı!")
            print(f"📈 Final değer: ${result['final_value']:,.2f}")
            print(f"📊 Toplam getiri: %{result['total_return']:.2f}")
            return result
        else:
            print("❌ Test başarısız!")
            return None
            
    except Exception as e:
        print(f"❌ Hata: {e}")
        return None


def test_latest_model():
    """En son kaydedilen modeli test et"""
    print("🔥 En son model test ediliyor...")
    print("-" * 50)
    
    # En son model dosyasını bul
    model_files = glob.glob("portfolio_results_agent_*.pt")
    
    if not model_files:
        print("❌ Portfolio results modeli bulunamadı!")
        return None
    
    latest_model = sorted(model_files)[-1]  # En son tarihli
    print(f"📁 Test edilen model: {latest_model}")
    
    try:
        from quick_test import quick_test_model
        result = quick_test_model(latest_model, test_days=30)
        
        if result:
            print(f"\n✅ Test başarılı!")
            print(f"📈 Final değer: ${result['final_value']:,.2f}")
            print(f"📊 Toplam getiri: %{result['total_return']:.2f}")
            return result
        else:
            print("❌ Test başarısız!")
            return None
            
    except Exception as e:
        print(f"❌ Hata: {e}")
        return None


def compare_all_models():
    """Tüm modelleri karşılaştır"""
    print("🔍 TÜM MODELLER KARŞILAŞTIRILIYOR")
    print("=" * 60)
    
    # Tüm model dosyalarını bul
    model_files = ["best_portfolio_agent.pt"]
    model_files.extend(glob.glob("portfolio_results_agent_*.pt"))
    
    results = {}
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"\n📊 Test ediliyor: {model_file}")
            
            try:
                from quick_test import quick_test_model
                result = quick_test_model(model_file, test_days=20)  # Daha hızlı
                
                if result:
                    results[model_file] = result
                    print(f"   ✅ Final: ${result['final_value']:,.0f} | Getiri: %{result['total_return']:.2f}")
                else:
                    print(f"   ❌ Test başarısız!")
                    
            except Exception as e:
                print(f"   ❌ Hata: {e}")
    
    # Karşılaştırma tablosu
    if len(results) > 1:
        print(f"\n🏆 KARŞILAŞTIRMA TABLOSU")
        print("=" * 70)
        print(f"{'Model':<35} {'Final Değer':<15} {'Getiri (%)':<12}")
        print("-" * 70)
        
        # Getiriye göre sırala
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1]['total_return'], 
                              reverse=True)
        
        for i, (model_name, result) in enumerate(sorted_results):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            short_name = model_name.replace("portfolio_results_agent_", "").replace(".pt", "")[:30]
            
            print(f"{rank} {short_name:<30} ${result['final_value']:>12,.0f} {result['total_return']:>10.2f}%")
    
    return results


def quick_performance_check():
    """Hızlı performans kontrolü"""
    print("⚡ HIZLI PERFORMANS KONTROLÜ")
    print("=" * 40)
    
    # Sadece en iyi modeli test et
    result = test_best_model()
    
    if result:
        total_return = result['total_return']
        
        print(f"\n🎯 PERFORMANS DEĞERLENDİRMESİ:")
        print("-" * 30)
        
        if total_return > 15:
            print("🔥 MÜKEMMEL! Model çok başarılı!")
            rating = "A+"
        elif total_return > 10:
            print("🚀 HARIKA! Model başarılı!")
            rating = "A"
        elif total_return > 5:
            print("👍 İYİ! Model umut verici!")
            rating = "B+"
        elif total_return > 0:
            print("📈 TAMAM! Pozitif getiri!")
            rating = "B"
        else:
            print("📉 Geliştirilebilir...")
            rating = "C"
        
        print(f"📊 Model Notu: {rating}")
        
        # Öneriler
        print(f"\n💡 ÖNERİLER:")
        if total_return < 5:
            print("   - Daha fazla episode ile eğitim yapın")
            print("   - Hiperparametreleri ayarlayın")
            print("   - Farklı hisse senetleri deneyin")
        else:
            print("   - Modeli canlı trading için değerlendirin")
            print("   - Farklı zaman dilimleri test edin")
            print("   - Risk yönetimi stratejileri ekleyin")
    
    return result


def help_menu():
    """Yardım menüsü"""
    print("🔧 MODEL TEST KOMUTLARI")
    print("=" * 40)
    print("1. test_best_model()       - En iyi modeli test et")
    print("2. test_latest_model()     - En son modeli test et") 
    print("3. compare_all_models()    - Tüm modelleri karşılaştır")
    print("4. quick_performance_check() - Hızlı performans kontrolü")
    print("\nDetaylı test için: python test_model.py")
    print("Hızlı test için: python quick_test.py")


if __name__ == "__main__":
    help_menu()
    print("\n🚀 Hızlı başlangıç için:")
    print(">>> quick_performance_check()")
    
    # Otomatik hızlı kontrol
    quick_performance_check() 