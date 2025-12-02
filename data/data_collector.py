"""
Recolector de datos en tiempo real para el cobre
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict
import time

from config.settings import (
    COPPER_SYMBOL, HISTORICAL_PERIOD, DATA_INTERVAL,
    MIN_DATA_POINTS, ALTERNATIVE_SYMBOLS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """
    Clase para recolectar datos del mercado de cobre en tiempo real
    """
    
    def __init__(self, symbol: str = COPPER_SYMBOL):
        """
        Inicializa el recolector de datos
        
        Args:
            symbol: Símbolo del activo a rastrear
        """
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        self.last_update = None
        logger.info(f"🔧 DataCollector inicializado para {symbol}")
    
    def get_historical_data(self, 
                           period: str = HISTORICAL_PERIOD,
                           interval: str = DATA_INTERVAL) -> pd.DataFrame:
        """
        Obtiene datos históricos del cobre
        
        Args:
            period: Período de datos (ej: '1y', '2y', '5y')
            interval: Intervalo de datos (ej: '1d', '1h')
            
        Returns:
            DataFrame con datos históricos
        """
        try:
            logger.info(f"📊 Descargando datos históricos ({period}, {interval})...")
            
            # Descargar datos
            df = self.ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.error("❌ No se pudieron obtener datos")
                return pd.DataFrame()
            
            # Limpiar columnas
            df.columns = df.columns.str.lower()
            
            # Resetear índice para tener la fecha como columna
            df.reset_index(inplace=True)
            
            # Renombrar columna de fecha (puede ser 'date' o 'index')
            if 'date' in df.columns:
                df.rename(columns={'date': 'datetime'}, inplace=True)
            elif 'index' in df.columns:
                df.rename(columns={'index': 'datetime'}, inplace=True)
            else:
                # Si el índice no tiene nombre, la primera columna será la fecha
                df.columns = ['datetime'] + list(df.columns[1:])
            
            # Validar datos mínimos
            if len(df) < MIN_DATA_POINTS:
                logger.warning(f"⚠️ Solo se obtuvieron {len(df)} puntos de datos (mínimo: {MIN_DATA_POINTS})")
            
            logger.info(f"✅ {len(df)} registros descargados exitosamente")
            logger.info(f"📈 Rango: {df['datetime'].min()} a {df['datetime'].max()}")
            logger.info(f"💰 Precio actual: ${df['close'].iloc[-1]:.2f}")
            
            self.last_update = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos históricos: {e}")
            return pd.DataFrame()
    
    def get_realtime_data(self) -> Dict:
        """
        Obtiene datos en tiempo real del cobre
        
        Returns:
            Diccionario con datos actuales
        """
        try:
            logger.info("🔄 Obteniendo datos en tiempo real...")
            
            # Obtener información actual
            info = self.ticker.info
            
            # Obtener último precio
            df = self.ticker.history(period='1d', interval='1m')
            
            if df.empty:
                logger.error("❌ No se pudieron obtener datos en tiempo real")
                return {}
            
            latest = df.iloc[-1]
            
            realtime_data = {
                'datetime': datetime.now(),
                'symbol': self.symbol,
                'price': latest['Close'],
                'open': latest['Open'],
                'high': latest['High'],
                'low': latest['Low'],
                'volume': latest['Volume'],
                'change': latest['Close'] - latest['Open'],
                'change_pct': ((latest['Close'] - latest['Open']) / latest['Open']) * 100,
            }
            
            logger.info(f"✅ Precio actual: ${realtime_data['price']:.2f} ({realtime_data['change_pct']:+.2f}%)")
            
            return realtime_data
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos en tiempo real: {e}")
            return {}
    
    def get_market_info(self) -> Dict:
        """
        Obtiene información adicional del mercado
        
        Returns:
            Diccionario con información del mercado
        """
        try:
            info = self.ticker.info
            
            market_info = {
                'name': info.get('shortName', 'N/A'),
                'exchange': info.get('exchange', 'N/A'),
                'currency': info.get('currency', 'USD'),
                'market_cap': info.get('marketCap', 0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                '52w_high': info.get('fiftyTwoWeekHigh', 0),
                '52w_low': info.get('fiftyTwoWeekLow', 0),
            }
            
            return market_info
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo información del mercado: {e}")
            return {}
    
    def get_related_assets(self) -> Dict[str, pd.DataFrame]:
        """
        Obtiene datos de activos relacionados con el cobre
        
        Returns:
            Diccionario con DataFrames de activos relacionados
        """
        try:
            logger.info("📊 Obteniendo datos de activos relacionados...")
            
            related_data = {}
            
            for name, symbol in ALTERNATIVE_SYMBOLS.items():
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period='1mo', interval='1d')
                    
                    if not df.empty:
                        related_data[name] = df
                        logger.info(f"  ✓ {name} ({symbol}): {len(df)} registros")
                    
                except Exception as e:
                    logger.warning(f"  ✗ Error con {name} ({symbol}): {e}")
                    continue
            
            return related_data
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo activos relacionados: {e}")
            return {}
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Verifica la calidad de los datos
        
        Args:
            df: DataFrame a verificar
            
        Returns:
            Diccionario con métricas de calidad
        """
        if df.empty:
            return {'quality_score': 0, 'issues': ['DataFrame vacío']}
        
        issues = []
        
        # Verificar valores nulos
        null_counts = df.isnull().sum()
        if null_counts.any():
            issues.append(f"Valores nulos encontrados: {null_counts[null_counts > 0].to_dict()}")
        
        # Verificar duplicados
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"{duplicates} filas duplicadas")
        
        # Verificar valores negativos en volumen
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                issues.append(f"{negative_volume} valores de volumen negativos")
        
        # Verificar precios anómalos
        if 'close' in df.columns:
            price_std = df['close'].std()
            price_mean = df['close'].mean()
            outliers = ((df['close'] - price_mean).abs() > 3 * price_std).sum()
            if outliers > 0:
                issues.append(f"{outliers} precios potencialmente anómalos")
        
        # Calcular score de calidad
        quality_score = max(0, 100 - len(issues) * 10)
        
        return {
            'quality_score': quality_score,
            'total_records': len(df),
            'null_values': null_counts.sum(),
            'duplicates': duplicates,
            'issues': issues if issues else ['Sin problemas detectados']
        }
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """
        Guarda datos en archivo CSV
        
        Args:
            df: DataFrame a guardar
            filename: Nombre del archivo
        """
        try:
            from config.settings import DATA_DIR
            filepath = f"{DATA_DIR}/{filename}"
            df.to_csv(filepath, index=False)
            logger.info(f"💾 Datos guardados en {filepath}")
        except Exception as e:
            logger.error(f"❌ Error guardando datos: {e}")


# Función de prueba
if __name__ == "__main__":
    print("=" * 60)
    print("PROBANDO DATA COLLECTOR - SISTEMA KNN COBRE")
    print("=" * 60)
    
    collector = DataCollector()
    
    # Obtener datos históricos
    df = collector.get_historical_data()
    print(f"\n📊 Datos históricos obtenidos: {len(df)} registros")
    print(df.head())
    
    # Verificar calidad
    quality = collector.check_data_quality(df)
    print(f"\n🔍 Calidad de datos: {quality['quality_score']}%")
    print(f"Problemas: {quality['issues']}")
    
    # Obtener datos en tiempo real
    realtime = collector.get_realtime_data()
    print(f"\n💰 Precio en tiempo real: ${realtime.get('price', 0):.2f}")
    
    # Información del mercado
    market_info = collector.get_market_info()
    print(f"\n📈 Información del mercado:")
    for key, value in market_info.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Prueba completada exitosamente")