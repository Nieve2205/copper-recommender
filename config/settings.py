"""
Configuración principal del sistema KNN para trading de cobre
"""

import os
from datetime import datetime, timedelta

# ==========================================
# CONFIGURACIÓN DEL MODELO KNN
# ==========================================

# Número de vecinos más cercanos a considerar
K_NEIGHBORS = 50

# Peso de los vecinos ('uniform' o 'distance')
WEIGHTS = 'distance'

# Algoritmo para calcular vecinos más cercanos
ALGORITHM = 'auto'  # 'ball_tree', 'kd_tree', 'brute', 'auto'

# Métrica de distancia
METRIC = 'euclidean'

# ==========================================
# CONFIGURACIÓN DE DATOS
# ==========================================

# Símbolo del cobre en Yahoo Finance
COPPER_SYMBOL = 'HG=F'  # Cobre Futuro

# Símbolos alternativos para análisis
ALTERNATIVE_SYMBOLS = {
    'copper_futures': 'HG=F',
    'copper_etf': 'CPER',  # ETF de Cobre
    'freeport': 'FCX',     # Freeport-McMoRan (minera de cobre)
}

# Período de datos históricos
HISTORICAL_PERIOD = '2y'  # 2 años de datos

# Intervalo de datos
DATA_INTERVAL = '1d'  # 1 día

# Datos mínimos requeridos
MIN_DATA_POINTS = 100

# ==========================================
# CONFIGURACIÓN DE TRADING
# ==========================================

# Precio objetivo para señales
TARGET_PRICE = 8500  # $8,500 por tonelada

# Volumen mínimo (en millones)
MIN_VOLUME_MILLIONS = 20

# Umbral de confianza para señales
CONFIDENCE_THRESHOLD = 0.70  # 70%

# Stop loss (porcentaje)
STOP_LOSS_PCT = 0.03  # 3%

# Take profit (porcentaje)
TAKE_PROFIT_PCT = 0.05  # 5%

# ==========================================
# INDICADORES TÉCNICOS
# ==========================================

# Medias móviles
SMA_PERIODS = [20, 50, 200]
EMA_PERIODS = [12, 26]

# RSI
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2

# ATR
ATR_PERIOD = 14

# ==========================================
# FEATURES PARA EL MODELO
# ==========================================

FEATURES = [
    # Precio y variación
    'close',
    'price_change_pct',
    'price_momentum_5',
    'price_momentum_10',
    
    # Volumen
    'volume',
    'volume_change_pct',
    'volume_sma_20',
    
    # Medias móviles
    'sma_20',
    'sma_50',
    'sma_200',
    'ema_12',
    'ema_26',
    
    # Indicadores técnicos
    'rsi',
    'macd',
    'macd_signal',
    'macd_diff',
    'bb_upper',
    'bb_middle',
    'bb_lower',
    'bb_width',
    'atr',
    
    # Relaciones de precio
    'price_to_sma20',
    'price_to_sma50',
    'price_to_sma200',
]

# ==========================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# ==========================================

# Colores
COLOR_BUY = 'green'
COLOR_SELL = 'red'
COLOR_HOLD = 'gray'

# Tamaño de figuras
FIGURE_SIZE = (15, 10)

# Estilo de gráficos
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# ==========================================
# RUTAS Y DIRECTORIOS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data_cache')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Crear directorios si no existen
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==========================================
# CONFIGURACIÓN DE LOGGING
# ==========================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(LOGS_DIR, f'knn_trading_{datetime.now().strftime("%Y%m%d")}.log')

# ==========================================
# CONFIGURACIÓN DE NOTIFICACIONES
# ==========================================

ENABLE_NOTIFICATIONS = False
NOTIFICATION_EMAIL = ''
NOTIFICATION_PHONE = ''

# ==========================================
# MODO DE OPERACIÓN
# ==========================================

# Modo debug
DEBUG_MODE = True

# Actualización automática
AUTO_UPDATE = True
UPDATE_INTERVAL_MINUTES = 60

# Trading en vivo (desactivado por defecto)
LIVE_TRADING = False

print(f"✅ Configuración cargada - Símbolo: {COPPER_SYMBOL}, K={K_NEIGHBORS}")