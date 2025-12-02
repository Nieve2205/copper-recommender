"""
Procesador de datos y creación de features para el modelo KNN
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List

from utils.indicators import TechnicalIndicators
from config.settings import (
    FEATURES, TARGET_PRICE, MIN_VOLUME_MILLIONS,
    SMA_PERIODS, EMA_PERIODS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Clase para procesar datos y crear features para el modelo
    """
    
    def __init__(self):
        """
        Inicializa el procesador de datos
        """
        self.indicators = TechnicalIndicators()
        logger.info("🔧 DataProcessor inicializado")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y prepara los datos
        
        Args:
            df: DataFrame con datos crudos
            
        Returns:
            DataFrame limpio
        """
        try:
            logger.info("🧹 Limpiando datos...")
            
            # Copiar para no modificar original
            df_clean = df.copy()
            
            # Eliminar duplicados
            df_clean = df_clean.drop_duplicates()
            
            # Ordenar por fecha
            if 'datetime' in df_clean.columns:
                df_clean = df_clean.sort_values('datetime')
                df_clean.reset_index(drop=True, inplace=True)
            
            # Rellenar valores nulos (forward fill, luego backward fill)
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_columns] = df_clean[numeric_columns].ffill()
            df_clean[numeric_columns] = df_clean[numeric_columns].bfill()
            
            # Eliminar filas con valores nulos restantes
            df_clean = df_clean.dropna()
            
            logger.info(f"✅ Datos limpios: {len(df_clean)} registros")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"❌ Error limpiando datos: {e}")
            return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea todas las features necesarias para el modelo
        
        Args:
            df: DataFrame con datos limpios
            
        Returns:
            DataFrame con features creadas
        """
        try:
            logger.info("🔨 Creando features...")
            
            df_features = df.copy()
            
            # ===== FEATURES DE PRECIO =====
            df_features['price_change'] = df_features['close'].diff()
            df_features['price_change_pct'] = df_features['close'].pct_change() * 100
            
            # Momentum de precio
            df_features['price_momentum_5'] = df_features['close'].pct_change(periods=5) * 100
            df_features['price_momentum_10'] = df_features['close'].pct_change(periods=10) * 100
            
            # ===== FEATURES DE VOLUMEN =====
            df_features['volume_change_pct'] = df_features['volume'].pct_change() * 100
            df_features['volume_sma_20'] = df_features['volume'].rolling(window=20).mean()
            
            # ===== MEDIAS MÓVILES =====
            for period in SMA_PERIODS:
                df_features[f'sma_{period}'] = self.indicators.calculate_sma(
                    df_features['close'], period
                )
            
            for period in EMA_PERIODS:
                df_features[f'ema_{period}'] = self.indicators.calculate_ema(
                    df_features['close'], period
                )
            
            # ===== INDICADORES TÉCNICOS =====
            
            # RSI
            df_features['rsi'] = self.indicators.calculate_rsi(df_features['close'])
            
            # MACD
            macd_data = self.indicators.calculate_macd(df_features['close'])
            df_features['macd'] = macd_data['macd']
            df_features['macd_signal'] = macd_data['signal']
            df_features['macd_diff'] = macd_data['histogram']
            
            # Bollinger Bands
            bb_data = self.indicators.calculate_bollinger_bands(df_features['close'])
            df_features['bb_upper'] = bb_data['upper']
            df_features['bb_middle'] = bb_data['middle']
            df_features['bb_lower'] = bb_data['lower']
            df_features['bb_width'] = bb_data['width']
            
            # ATR
            df_features['atr'] = self.indicators.calculate_atr(
                df_features['high'],
                df_features['low'],
                df_features['close']
            )
            
            # ===== RELACIONES DE PRECIO =====
            # Usar np.where para evitar división por cero
            df_features['price_to_sma20'] = np.where(
                df_features['sma_20'] != 0,
                (df_features['close'] / df_features['sma_20'] - 1) * 100,
                0
            )
            df_features['price_to_sma50'] = np.where(
                df_features['sma_50'] != 0,
                (df_features['close'] / df_features['sma_50'] - 1) * 100,
                0
            )
            df_features['price_to_sma200'] = np.where(
                df_features['sma_200'] != 0,
                (df_features['close'] / df_features['sma_200'] - 1) * 100,
                0
            )
            
            # Reemplazar infinitos y NaN
            df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Eliminar filas con NaN generados por los indicadores
            df_features = df_features.dropna()
            
            logger.info(f"✅ Features creadas: {len(df_features)} registros con {len(df_features.columns)} columnas")
            
            return df_features
            
        except Exception as e:
            logger.error(f"❌ Error creando features: {e}")
            return df
    
    def create_target(self, df: pd.DataFrame, lookahead: int = 1) -> pd.DataFrame:
        """
        Crea la variable objetivo (target) para el modelo
        
        Args:
            df: DataFrame con features
            lookahead: Períodos hacia adelante para predecir
            
        Returns:
            DataFrame con target creado
        """
        try:
            logger.info("🎯 Creando variable objetivo...")
            
            df_target = df.copy()
            
            # Precio futuro
            df_target['future_close'] = df_target['close'].shift(-lookahead)
            
            # Cambio de precio futuro
            df_target['future_change_pct'] = (
                (df_target['future_close'] - df_target['close']) / df_target['close']
            ) * 100
            
            # Señal de trading (1 = COMPRA, 0 = HOLD, -1 = VENTA)
            conditions = [
                df_target['future_change_pct'] > 2,   # Subida > 2% = COMPRA
                df_target['future_change_pct'] < -2,  # Bajada > 2% = VENTA
            ]
            choices = [1, -1]
            df_target['target'] = np.select(conditions, choices, default=0)
            
            # Señal binaria simplificada (1 = COMPRA/HOLD, 0 = VENTA)
            df_target['target_binary'] = (df_target['target'] >= 0).astype(int)
            
            # Eliminar filas sin target (últimas filas)
            df_target = df_target.dropna(subset=['future_close', 'target'])
            
            # Estadísticas del target
            target_counts = df_target['target'].value_counts()
            logger.info(f"✅ Target creado:")
            logger.info(f"  COMPRA (1): {target_counts.get(1, 0)} registros")
            logger.info(f"  HOLD (0): {target_counts.get(0, 0)} registros")
            logger.info(f"  VENTA (-1): {target_counts.get(-1, 0)} registros")
            
            return df_target
            
        except Exception as e:
            logger.error(f"❌ Error creando target: {e}")
            return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara datos para entrenamiento (X, y)
        
        Args:
            df: DataFrame con features y target
            
        Returns:
            Tupla (X, y) con features y target
        """
        try:
            logger.info("📦 Preparando datos de entrenamiento...")
            
            # Verificar que existan las features necesarias
            missing_features = [f for f in FEATURES if f not in df.columns]
            if missing_features:
                logger.warning(f"⚠️ Features faltantes: {missing_features}")
            
            # Seleccionar solo features disponibles
            available_features = [f for f in FEATURES if f in df.columns]
            
            # Separar features (X) y target (y)
            X = df[available_features].copy()
            y = df['target'].copy()
            
            # Normalizar features
            X = self.normalize_features(X)
            
            logger.info(f"✅ Datos preparados: X={X.shape}, y={y.shape}")
            logger.info(f"📊 Features utilizadas: {len(available_features)}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Error preparando datos de entrenamiento: {e}")
            return pd.DataFrame(), pd.Series()
    
    def normalize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza las features usando min-max scaling
        
        Args:
            X: DataFrame con features
            
        Returns:
            DataFrame con features normalizadas
        """
        try:
            from sklearn.preprocessing import MinMaxScaler
            
            X_normalized = X.copy()
            
            # Reemplazar infinitos con NaN
            X_normalized.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Rellenar NaN con la mediana de cada columna
            for col in X_normalized.columns:
                if X_normalized[col].isna().any():
                    median_val = X_normalized[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    X_normalized[col].fillna(median_val, inplace=True)
            
            # Normalizar
            scaler = MinMaxScaler()
            X_normalized[X.columns] = scaler.fit_transform(X_normalized)
            
            return X_normalized
            
        except Exception as e:
            logger.error(f"❌ Error normalizando features: {e}")
            return X
    
    def split_train_test(self, 
                        X: pd.DataFrame, 
                        y: pd.Series, 
                        test_size: float = 0.2) -> Tuple:
        """
        Divide datos en conjuntos de entrenamiento y prueba
        
        Args:
            X: Features
            y: Target
            test_size: Proporción del conjunto de prueba
            
        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
        try:
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            logger.info(f"✅ Datos divididos:")
            logger.info(f"  Entrenamiento: {len(X_train)} registros")
            logger.info(f"  Prueba: {len(X_test)} registros")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"❌ Error dividiendo datos: {e}")
            return None, None, None, None


# Función de prueba
if __name__ == "__main__":
    from data.data_collector import DataCollector
    
    print("=" * 60)
    print("PROBANDO DATA PROCESSOR - SISTEMA KNN COBRE")
    print("=" * 60)
    
    # Obtener datos
    collector = DataCollector()
    df = collector.get_historical_data()
    
    # Procesar datos
    processor = DataProcessor()
    
    # Limpiar
    df_clean = processor.clean_data(df)
    print(f"\n🧹 Datos limpios: {len(df_clean)} registros")
    
    # Crear features
    df_features = processor.create_features(df_clean)
    print(f"\n🔨 Features creadas: {df_features.shape}")
    print(f"Columnas: {list(df_features.columns)}")
    
    # Crear target
    df_target = processor.create_target(df_features)
    print(f"\n🎯 Target creado: {df_target.shape}")
    
    # Preparar datos de entrenamiento
    X, y = processor.prepare_training_data(df_target)
    print(f"\n📦 Datos preparados: X={X.shape}, y={y.shape}")
    
    # Dividir en train/test
    X_train, X_test, y_train, y_test = processor.split_train_test(X, y)
    print(f"\n✂️ Train: {X_train.shape}, Test: {X_test.shape}")
    
    print("\n✅ Prueba completada exitosamente")