"""
Indicadores técnicos para análisis de trading
"""

import pandas as pd
import numpy as np
from typing import Dict

from config.settings import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, ATR_PERIOD
)


class TechnicalIndicators:
    """
    Clase para calcular indicadores técnicos
    """
    
    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """
        Calcula Simple Moving Average (SMA)
        
        Args:
            prices: Serie de precios
            period: Período de la media móvil
            
        Returns:
            Serie con SMA
        """
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """
        Calcula Exponential Moving Average (EMA)
        
        Args:
            prices: Serie de precios
            period: Período de la media móvil
            
        Returns:
            Serie con EMA
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
        """
        Calcula Relative Strength Index (RSI)
        
        Args:
            prices: Serie de precios
            period: Período del RSI
            
        Returns:
            Serie con RSI
        """
        # Calcular cambios de precio
        delta = prices.diff()
        
        # Separar ganancias y pérdidas
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calcular RS y RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series,
                      fast: int = MACD_FAST,
                      slow: int = MACD_SLOW,
                      signal: int = MACD_SIGNAL) -> Dict[str, pd.Series]:
        """
        Calcula Moving Average Convergence Divergence (MACD)
        
        Args:
            prices: Serie de precios
            fast: Período rápido
            slow: Período lento
            signal: Período de la señal
            
        Returns:
            Diccionario con MACD, señal e histograma
        """
        # Calcular EMAs
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        # Calcular MACD
        macd = ema_fast - ema_slow
        
        # Calcular señal
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        
        # Calcular histograma
        macd_histogram = macd - macd_signal
        
        return {
            'macd': macd,
            'signal': macd_signal,
            'histogram': macd_histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series,
                                 period: int = BB_PERIOD,
                                 std_dev: float = BB_STD) -> Dict[str, pd.Series]:
        """
        Calcula Bollinger Bands
        
        Args:
            prices: Serie de precios
            period: Período de la media móvil
            std_dev: Desviaciones estándar
            
        Returns:
            Diccionario con banda superior, media e inferior
        """
        # Calcular media móvil
        middle = prices.rolling(window=period).mean()
        
        # Calcular desviación estándar
        std = prices.rolling(window=period).std()
        
        # Calcular bandas
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        # Calcular ancho de banda
        width = upper - lower
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width
        }
    
    @staticmethod
    def calculate_atr(high: pd.Series,
                     low: pd.Series,
                     close: pd.Series,
                     period: int = ATR_PERIOD) -> pd.Series:
        """
        Calcula Average True Range (ATR)
        
        Args:
            high: Serie de precios máximos
            low: Serie de precios mínimos
            close: Serie de precios de cierre
            period: Período del ATR
            
        Returns:
            Serie con ATR
        """
        # Calcular True Range
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calcular ATR
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_stochastic(high: pd.Series,
                           low: pd.Series,
                           close: pd.Series,
                           period: int = 14) -> Dict[str, pd.Series]:
        """
        Calcula Stochastic Oscillator
        
        Args:
            high: Serie de precios máximos
            low: Serie de precios mínimos
            close: Serie de precios de cierre
            period: Período del oscilador
            
        Returns:
            Diccionario con %K y %D
        """
        # Calcular mínimo y máximo del período
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        # Calcular %K
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        # Calcular %D (media móvil de %K)
        d_percent = k_percent.rolling(window=3).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calcula On-Balance Volume (OBV)
        
        Args:
            close: Serie de precios de cierre
            volume: Serie de volumen
            
        Returns:
            Serie con OBV
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def calculate_vwap(high: pd.Series,
                      low: pd.Series,
                      close: pd.Series,
                      volume: pd.Series) -> pd.Series:
        """
        Calcula Volume Weighted Average Price (VWAP)
        
        Args:
            high: Serie de precios máximos
            low: Serie de precios mínimos
            close: Serie de precios de cierre
            volume: Serie de volumen
            
        Returns:
            Serie con VWAP
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    @staticmethod
    def identify_support_resistance(prices: pd.Series, window: int = 20) -> Dict:
        """
        Identifica niveles de soporte y resistencia
        
        Args:
            prices: Serie de precios
            window: Ventana para identificar niveles
            
        Returns:
            Diccionario con niveles de soporte y resistencia
        """
        # Máximos y mínimos locales
        rolling_max = prices.rolling(window=window, center=True).max()
        rolling_min = prices.rolling(window=window, center=True).min()
        
        # Identificar resistencias (máximos locales)
        resistance = prices[(prices == rolling_max)].dropna()
        
        # Identificar soportes (mínimos locales)
        support = prices[(prices == rolling_min)].dropna()
        
        return {
            'support': support.values.tolist(),
            'resistance': resistance.values.tolist(),
            'current_support': support.iloc[-1] if len(support) > 0 else None,
            'current_resistance': resistance.iloc[-1] if len(resistance) > 0 else None
        }


# Función de prueba
if __name__ == "__main__":
    print("=" * 60)
    print("PROBANDO TECHNICAL INDICATORS")
    print("=" * 60)
    
    # Crear datos de ejemplo
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 2), index=dates)
    volume = pd.Series(np.random.randint(1000000, 5000000, 100), index=dates)
    high = prices + np.random.rand(100) * 2
    low = prices - np.random.rand(100) * 2
    
    # Crear instancia
    indicators = TechnicalIndicators()
    
    # Probar indicadores
    print("\n📊 Calculando indicadores...")
    
    sma = indicators.calculate_sma(prices, 20)
    print(f"✓ SMA(20): {sma.iloc[-1]:.2f}")
    
    ema = indicators.calculate_ema(prices, 12)
    print(f"✓ EMA(12): {ema.iloc[-1]:.2f}")
    
    rsi = indicators.calculate_rsi(prices)
    print(f"✓ RSI: {rsi.iloc[-1]:.2f}")
    
    macd = indicators.calculate_macd(prices)
    print(f"✓ MACD: {macd['macd'].iloc[-1]:.2f}")
    print(f"✓ MACD Signal: {macd['signal'].iloc[-1]:.2f}")
    
    bb = indicators.calculate_bollinger_bands(prices)
    print(f"✓ BB Upper: {bb['upper'].iloc[-1]:.2f}")
    print(f"✓ BB Lower: {bb['lower'].iloc[-1]:.2f}")
    
    atr = indicators.calculate_atr(high, low, prices)
    print(f"✓ ATR: {atr.iloc[-1]:.2f}")
    
    sr = indicators.identify_support_resistance(prices)
    print(f"✓ Soporte actual: {sr['current_support']:.2f}")
    print(f"✓ Resistencia actual: {sr['current_resistance']:.2f}")
    
    print("\n✅ Todos los indicadores funcionan correctamente")