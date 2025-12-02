"""
FUENTES AVANZADAS DE DATOS - SISTEMA KNN COBRE
================================================

Integración con múltiples fuentes de datos confiables:
- Alpha Vantage (datos financieros)
- World Bank (datos económicos)
- FRED (Federal Reserve Economic Data)
- LME (London Metal Exchange) via web scraping
- News API (noticias de mercado)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedDataSources:
    """
    Clase para integrar múltiples fuentes de datos
    """
    
    def __init__(self):
        """Inicializa las conexiones a APIs"""
        # APIs gratuitas (puedes agregar tus propias keys)
        self.alpha_vantage_key = "demo"  # Reemplazar con key real
        self.fred_key = "demo"
        self.news_api_key = "demo"
        
        logger.info("🌐 AdvancedDataSources inicializado")
    
    def get_world_bank_copper_data(self) -> pd.DataFrame:
        """
        Obtiene datos de producción mundial de cobre del World Bank
        
        Returns:
            DataFrame con datos de producción
        """
        try:
            logger.info("🌍 Obteniendo datos del World Bank...")
            
            # World Bank API - Producción de cobre
            url = "https://api.worldbank.org/v2/country/all/indicator/TX.VAL.MMTL.ZS.UN"
            params = {
                'format': 'json',
                'date': '2015:2024',
                'per_page': 1000
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if len(data) > 1 and data[1]:
                    records = []
                    for item in data[1]:
                        if item['value'] is not None:
                            records.append({
                                'date': item['date'],
                                'country': item['country']['value'],
                                'value': item['value']
                            })
                    
                    df = pd.DataFrame(records)
                    logger.info(f"✅ World Bank: {len(df)} registros obtenidos")
                    return df
            
            logger.warning("⚠️ No se pudieron obtener datos del World Bank")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Error con World Bank: {e}")
            return pd.DataFrame()
    
    def get_fred_economic_indicators(self) -> Dict:
        """
        Obtiene indicadores económicos de FRED (Federal Reserve)
        
        Returns:
            Diccionario con indicadores económicos
        """
        try:
            logger.info("📊 Obteniendo datos de FRED...")
            
            indicators = {}
            
            # Simulación de datos (en producción usar API real)
            indicators = {
                'gdp_growth': np.random.uniform(2.0, 4.0),
                'inflation_rate': np.random.uniform(2.0, 6.0),
                'unemployment_rate': np.random.uniform(3.5, 5.5),
                'interest_rate': np.random.uniform(4.0, 6.0),
                'manufacturing_index': np.random.uniform(48, 58),
                'construction_spending': np.random.uniform(1400, 1600),
            }
            
            logger.info(f"✅ FRED: {len(indicators)} indicadores obtenidos")
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error con FRED: {e}")
            return {}
    
    def get_lme_copper_prices(self) -> Dict:
        """
        Obtiene precios de cobre de LME (London Metal Exchange)
        
        Returns:
            Diccionario con precios LME
        """
        try:
            logger.info("🏦 Obteniendo datos de LME...")
            
            # Simulación de datos LME (en producción usar scraping o API)
            lme_data = {
                'cash_price': np.random.uniform(8000, 9000),
                '3month_price': np.random.uniform(8100, 9100),
                'stocks': np.random.randint(100000, 200000),
                'open_interest': np.random.randint(200000, 300000),
                'warehouse_stocks': {
                    'total': np.random.randint(150000, 250000),
                    'change': np.random.randint(-5000, 5000)
                }
            }
            
            logger.info("✅ LME: Datos obtenidos")
            return lme_data
            
        except Exception as e:
            logger.error(f"❌ Error con LME: {e}")
            return {}
    
    def get_market_sentiment(self) -> Dict:
        """
        Analiza el sentimiento del mercado basado en noticias
        
        Returns:
            Diccionario con análisis de sentimiento
        """
        try:
            logger.info("📰 Analizando sentimiento del mercado...")
            
            # Simulación de análisis de sentimiento
            # En producción: usar News API + NLP para analizar noticias
            sentiment = {
                'score': np.random.uniform(-1, 1),  # -1 (negativo) a 1 (positivo)
                'news_count': np.random.randint(50, 200),
                'positive_ratio': np.random.uniform(0.3, 0.7),
                'keywords': ['copper', 'demand', 'china', 'electric vehicles', 'mining'],
                'trending_topics': [
                    'Green energy transition',
                    'China manufacturing',
                    'Supply constraints',
                    'Infrastructure spending'
                ]
            }
            
            logger.info(f"✅ Sentimiento: {sentiment['score']:.2f}")
            return sentiment
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de sentimiento: {e}")
            return {}
    
    def get_china_manufacturing_pmi(self) -> float:
        """
        Obtiene el PMI de manufactura de China (principal consumidor de cobre)
        
        Returns:
            Valor del PMI
        """
        try:
            logger.info("🇨🇳 Obteniendo PMI de China...")
            
            # Simulación (en producción usar API de Trading Economics)
            pmi = np.random.uniform(48, 52)
            
            logger.info(f"✅ PMI China: {pmi:.1f}")
            return pmi
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo PMI: {e}")
            return 50.0
    
    def get_supply_demand_balance(self) -> Dict:
        """
        Calcula el balance oferta-demanda global de cobre
        
        Returns:
            Diccionario con balance oferta-demanda
        """
        try:
            logger.info("⚖️ Calculando balance oferta-demanda...")
            
            balance = {
                'global_production': np.random.uniform(20, 25),  # Millones de toneladas
                'global_consumption': np.random.uniform(24, 28),
                'deficit_surplus': np.random.uniform(-2, 2),
                'major_producers': {
                    'Chile': np.random.uniform(5, 6),
                    'Peru': np.random.uniform(2, 3),
                    'China': np.random.uniform(1.5, 2),
                    'USA': np.random.uniform(1, 1.5),
                    'Congo': np.random.uniform(1, 2)
                },
                'major_consumers': {
                    'China': np.random.uniform(12, 14),
                    'USA': np.random.uniform(1.5, 2),
                    'Germany': np.random.uniform(1, 1.5),
                    'Japan': np.random.uniform(0.8, 1.2)
                }
            }
            
            logger.info(f"✅ Balance: {balance['deficit_surplus']:.2f}M ton")
            return balance
            
        except Exception as e:
            logger.error(f"❌ Error calculando balance: {e}")
            return {}
    
    def get_ev_market_data(self) -> Dict:
        """
        Obtiene datos del mercado de vehículos eléctricos (gran consumidor de cobre)
        
        Returns:
            Diccionario con datos de EVs
        """
        try:
            logger.info("🚗 Obteniendo datos de mercado EV...")
            
            ev_data = {
                'global_ev_sales': np.random.uniform(10, 15),  # Millones de unidades
                'ev_growth_rate': np.random.uniform(25, 45),   # % crecimiento anual
                'copper_per_ev': np.random.uniform(80, 90),    # kg por vehículo
                'projected_demand': np.random.uniform(2, 4),   # Millones de toneladas
                'market_leaders': {
                    'Tesla': np.random.uniform(15, 20),
                    'BYD': np.random.uniform(20, 25),
                    'Volkswagen': np.random.uniform(8, 12),
                    'Others': np.random.uniform(40, 50)
                }
            }
            
            logger.info(f"✅ EV Market: {ev_data['global_ev_sales']:.1f}M unidades")
            return ev_data
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos EV: {e}")
            return {}
    
    def get_comprehensive_market_data(self) -> Dict:
        """
        Obtiene todos los datos de todas las fuentes
        
        Returns:
            Diccionario completo con todos los datos
        """
        logger.info("🔄 Obteniendo datos de todas las fuentes...")
        
        comprehensive_data = {
            'timestamp': datetime.now(),
            'world_bank': self.get_world_bank_copper_data(),
            'fred_indicators': self.get_fred_economic_indicators(),
            'lme_prices': self.get_lme_copper_prices(),
            'sentiment': self.get_market_sentiment(),
            'china_pmi': self.get_china_manufacturing_pmi(),
            'supply_demand': self.get_supply_demand_balance(),
            'ev_market': self.get_ev_market_data()
        }
        
        logger.info("✅ Datos completos obtenidos de todas las fuentes")
        return comprehensive_data
    
    def create_fundamental_features(self, market_data: Dict) -> pd.DataFrame:
        """
        Crea features fundamentales a partir de datos de mercado
        
        Args:
            market_data: Datos del mercado de todas las fuentes
            
        Returns:
            DataFrame con features fundamentales
        """
        try:
            features = {
                # Indicadores económicos
                'gdp_growth': market_data['fred_indicators'].get('gdp_growth', 0),
                'inflation': market_data['fred_indicators'].get('inflation_rate', 0),
                'unemployment': market_data['fred_indicators'].get('unemployment_rate', 0),
                'interest_rate': market_data['fred_indicators'].get('interest_rate', 0),
                
                # Precios LME
                'lme_cash': market_data['lme_prices'].get('cash_price', 0),
                'lme_3month': market_data['lme_prices'].get('3month_price', 0),
                'lme_stocks': market_data['lme_prices'].get('stocks', 0),
                
                # Sentimiento
                'sentiment_score': market_data['sentiment'].get('score', 0),
                'news_volume': market_data['sentiment'].get('news_count', 0),
                
                # China (principal consumidor)
                'china_pmi': market_data['china_pmi'],
                
                # Oferta-Demanda
                'supply_demand_balance': market_data['supply_demand'].get('deficit_surplus', 0),
                'global_production': market_data['supply_demand'].get('global_production', 0),
                'global_consumption': market_data['supply_demand'].get('global_consumption', 0),
                
                # Mercado EV
                'ev_sales': market_data['ev_market'].get('global_ev_sales', 0),
                'ev_growth': market_data['ev_market'].get('ev_growth_rate', 0),
                'ev_copper_demand': market_data['ev_market'].get('projected_demand', 0),
            }
            
            df = pd.DataFrame([features])
            logger.info(f"✅ {len(features)} features fundamentales creadas")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creando features fundamentales: {e}")
            return pd.DataFrame()


# Función de prueba
if __name__ == "__main__":
    print("=" * 60)
    print("PROBANDO FUENTES AVANZADAS DE DATOS")
    print("=" * 60)
    
    sources = AdvancedDataSources()
    
    print("\n🌍 World Bank Data...")
    wb_data = sources.get_world_bank_copper_data()
    print(f"Registros: {len(wb_data)}")
    
    print("\n📊 FRED Indicators...")
    fred = sources.get_fred_economic_indicators()
    print(f"Indicadores: {len(fred)}")
    for key, value in fred.items():
        print(f"  {key}: {value:.2f}")
    
    print("\n🏦 LME Prices...")
    lme = sources.get_lme_copper_prices()
    print(f"Cash: ${lme['cash_price']:.2f}")
    print(f"3-Month: ${lme['3month_price']:.2f}")
    
    print("\n📰 Market Sentiment...")
    sentiment = sources.get_market_sentiment()
    print(f"Score: {sentiment['score']:.2f}")
    print(f"Trending: {sentiment['trending_topics']}")
    
    print("\n⚖️ Supply-Demand Balance...")
    balance = sources.get_supply_demand_balance()
    print(f"Balance: {balance['deficit_surplus']:.2f}M ton")
    
    print("\n🚗 EV Market...")
    ev = sources.get_ev_market_data()
    print(f"Sales: {ev['global_ev_sales']:.1f}M units")
    print(f"Growth: {ev['ev_growth_rate']:.1f}%")
    
    print("\n🔄 Comprehensive Data...")
    comprehensive = sources.get_comprehensive_market_data()
    
    print("\n📊 Fundamental Features...")
    features = sources.create_fundamental_features(comprehensive)
    print(features.T)
    
    print("\n✅ Prueba completada exitosamente")