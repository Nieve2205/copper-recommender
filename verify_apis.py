"""
Script para verificar que las API keys están configuradas correctamente
"""

import os
from dotenv import load_dotenv
from colorama import init, Fore, Style
import requests

init(autoreset=True)

# Cargar variables de entorno
load_dotenv()

def print_header():
    print("\n" + "=" * 60)
    print(Fore.CYAN + Style.BRIGHT + "🔑 VERIFICADOR DE API KEYS")
    print("=" * 60 + "\n")

def check_newsapi():
    """Verifica NewsAPI"""
    print(Fore.YELLOW + "📰 Verificando NewsAPI...")
    
    api_key = os.getenv('NEWS_API_KEY')
    
    if not api_key or api_key == 'demo' or api_key == 'your_newsapi_key_here':
        print(Fore.RED + "   ❌ NewsAPI no configurada")
        print(Fore.YELLOW + "   💡 Obtén tu key en: https://newsapi.org/register")
        return False
    
    # Probar la API
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'copper',
            'pageSize': 1,
            'apiKey': api_key
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'ok':
                print(Fore.GREEN + "   ✅ NewsAPI funcionando correctamente")
                print(Fore.CYAN + f"   📊 API Key válida: {api_key[:8]}...")
                return True
            else:
                print(Fore.RED + f"   ❌ Error: {data.get('message', 'Unknown')}")
                return False
        elif response.status_code == 401:
            print(Fore.RED + "   ❌ API Key inválida")
            print(Fore.YELLOW + "   💡 Verifica que copiaste la key correctamente")
            return False
        elif response.status_code == 429:
            print(Fore.YELLOW + "   ⚠️ Límite de requests alcanzado (espera 24hrs)")
            print(Fore.GREEN + "   ✅ Pero la key es válida")
            return True
        else:
            print(Fore.RED + f"   ❌ Error HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(Fore.RED + f"   ❌ Error: {e}")
        return False

def check_alpha_vantage():
    """Verifica Alpha Vantage"""
    print(Fore.YELLOW + "\n📈 Verificando Alpha Vantage...")
    
    api_key = os.getenv('ALPHA_VANTAGE_KEY')
    
    if not api_key or api_key == 'demo' or api_key == 'your_alphavantage_key_here':
        print(Fore.YELLOW + "   ⚠️ Alpha Vantage no configurada (opcional)")
        print(Fore.CYAN + "   ℹ️ Obtén tu key en: https://www.alphavantage.co/support/#api-key")
        return False
    
    # Probar la API
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'CURRENCY_EXCHANGE_RATE',
            'from_currency': 'USD',
            'to_currency': 'EUR',
            'apikey': api_key
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'Realtime Currency Exchange Rate' in data:
                print(Fore.GREEN + "   ✅ Alpha Vantage funcionando")
                return True
            elif 'Error Message' in data or 'Note' in data:
                print(Fore.YELLOW + "   ⚠️ API Key válida pero límite alcanzado")
                return True
            else:
                print(Fore.RED + "   ❌ Respuesta inesperada")
                return False
        else:
            print(Fore.RED + f"   ❌ Error HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(Fore.RED + f"   ❌ Error: {e}")
        return False

def check_fred():
    """Verifica FRED API"""
    print(Fore.YELLOW + "\n📊 Verificando FRED API...")
    
    api_key = os.getenv('FRED_API_KEY')
    
    if not api_key or api_key == 'demo' or api_key == 'your_fred_key_here':
        print(Fore.YELLOW + "   ⚠️ FRED API no configurada (opcional)")
        print(Fore.CYAN + "   ℹ️ Obtén tu key en: https://fred.stlouisfed.org/docs/api/api_key.html")
        return False
    
    # Probar la API
    try:
        url = "https://api.stlouisfed.org/fred/series"
        params = {
            'series_id': 'CPIAUCSL',  # CPI data
            'api_key': api_key,
            'file_type': 'json'
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'seriess' in data:
                print(Fore.GREEN + "   ✅ FRED API funcionando")
                return True
            else:
                print(Fore.RED + "   ❌ Respuesta inesperada")
                return False
        elif response.status_code == 400:
            print(Fore.RED + "   ❌ API Key inválida")
            return False
        else:
            print(Fore.RED + f"   ❌ Error HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(Fore.RED + f"   ❌ Error: {e}")
        return False

def check_public_apis():
    """Verifica APIs públicas (sin key)"""
    print(Fore.YELLOW + "\n🌐 Verificando APIs Públicas...")
    
    all_ok = True
    
    # World Bank
    try:
        print(Fore.CYAN + "   🌍 World Bank API...")
        url = "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD"
        params = {'format': 'json', 'per_page': 1}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            print(Fore.GREEN + "      ✅ World Bank OK")
        else:
            print(Fore.RED + f"      ❌ Error {response.status_code}")
            all_ok = False
    except Exception as e:
        print(Fore.RED + f"      ❌ Error: {e}")
        all_ok = False
    
    # Reddit
    try:
        print(Fore.CYAN + "   🔴 Reddit API...")
        url = "https://www.reddit.com/r/investing/hot.json"
        params = {'limit': 1}
        headers = {'User-Agent': 'API Checker'}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            print(Fore.GREEN + "      ✅ Reddit OK")
        else:
            print(Fore.RED + f"      ❌ Error {response.status_code}")
            all_ok = False
    except Exception as e:
        print(Fore.RED + f"      ❌ Error: {e}")
        all_ok = False
    
    return all_ok

def generate_report(results):
    """Genera reporte final"""
    print("\n" + "=" * 60)
    print(Fore.CYAN + Style.BRIGHT + "📋 REPORTE FINAL")
    print("=" * 60 + "\n")
    
    total = len(results)
    passed = sum(results.values())
    
    # Estado general
    if passed >= 3:
        status_color = Fore.GREEN
        status = "✅ EXCELENTE"
        message = "Tu sistema está completamente configurado"
    elif passed >= 1:
        status_color = Fore.YELLOW
        status = "⚠️ BUENO"
        message = "Funcionalidad básica disponible"
    else:
        status_color = Fore.RED
        status = "❌ LIMITADO"
        message = "El sistema usará datos simulados"
    
    print(status_color + Style.BRIGHT + f"{status}: {passed}/{total} APIs configuradas\n")
    print(Fore.CYAN + message + "\n")
    
    # Detalles
    print(Fore.CYAN + "Detalles por API:")
    for api_name, is_ok in results.items():
        icon = "✅" if is_ok else "❌"
        color = Fore.GREEN if is_ok else Fore.RED
        print(f"{color}  {icon} {api_name}")
    
    # Recomendaciones
    print(Fore.YELLOW + "\n💡 Recomendaciones:")
    
    if not results.get('NewsAPI', False):
        print(Fore.YELLOW + """
   1. 🔴 CRÍTICO: Configura NewsAPI
      - Es la más importante para análisis de sentimiento
      - Registro gratis: https://newsapi.org/register
      - Solo toma 3-5 minutos
        """)
    
    if not results.get('Alpha Vantage', False):
        print(Fore.CYAN + """
   2. ⭐ Opcional: Alpha Vantage
      - Datos financieros adicionales
      - Registro: https://www.alphavantage.co/support/#api-key
        """)
    
    if not results.get('FRED', False):
        print(Fore.CYAN + """
   3. ⭐ Opcional: FRED API
      - Indicadores macroeconómicos
      - Registro: https://fred.stlouisfed.org/docs/api/api_key.html
        """)
    
    # Próximos pasos
    print(Fore.GREEN + "\n🚀 Próximos pasos:")
    print("""
   1. Si configuraste APIs: ¡Listo! Ejecuta el sistema
   2. Si no: El proyecto funciona con datos simulados
   3. Para probar: streamlit run dashboard.py
   4. Para más info: lee CONFIGURAR_APIS.md
    """)

def main():
    print_header()
    
    # Verificar archivo .env
    if not os.path.exists('.env'):
        print(Fore.YELLOW + "⚠️ Archivo .env no encontrado")
        print(Fore.CYAN + "💡 Copia .env.template a .env y configura tus keys\n")
        print(Fore.YELLOW + "   Comando: cp .env.template .env\n")
    
    results = {}
    
    # Verificar cada API
    results['NewsAPI'] = check_newsapi()
    results['Alpha Vantage'] = check_alpha_vantage()
    results['FRED'] = check_fred()
    results['APIs Públicas'] = check_public_apis()
    
    # Generar reporte
    generate_report(results)
    
    print("\n" + "=" * 60)
    print(Fore.GREEN + Style.BRIGHT + "Verificación completada".center(60))
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()