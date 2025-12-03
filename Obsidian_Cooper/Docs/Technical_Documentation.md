# Documentación Técnica - Sistema KNN para Trading de Cobre

## 📋 Resumen Ejecutivo

Sistema profesional de recomendación de trading basado en Machine Learning (K-Nearest Neighbors) que integra **múltiples fuentes de datos**, **análisis técnico y fundamental**, **Business Intelligence avanzado** y **gestión de riesgo** para predecir movimientos del precio del cobre.

**Tecnología Principal**: Python 3.8+ con scikit-learn y análisis avanzado  
**Algoritmo Core**: K-Nearest Neighbors (KNN) con 50 vecinos  
**Objetivo**: Generar señales COMPRA/VENTA/HOLD con confianza ≥70%  
**Características Únicas**: Multi-source data, Monte Carlo, VaR, Backtesting, Análisis fundamental

---

## 🏗️ Arquitectura del Sistema

### Estructura Modular

```
copper-recommender/
├── config/               # Configuración centralizada
│   ├── settings.py      # Parámetros del sistema
│   └── __init__.py
├── data/                 # Gestión y procesamiento de datos
│   ├── data_collector.py       # Recolección Yahoo Finance
│   ├── data_processor.py       # Procesamiento y features
│   ├── advanced_sources.py     # 🆕 Multi-source (WB, FRED, LME)
│   ├── advanced_analytics.py   # 🆕 BI avanzado (MC, VaR, Backtest)
│   └── __init__.py
├── models/               # Modelos de Machine Learning
│   ├── knn_model.py     # Modelo KNN principal
│   └── __init__.py
├── utils/                # Utilidades
│   ├── indicators.py    # Indicadores técnicos (RSI, MACD, etc.)
│   ├── visualizer.py    # Generación de gráficos
│   └── __init__.py
├── main.py               # Orquestador principal (CLI)
├── dashboard.py          # 🆕 Dashboard web interactivo (Streamlit)
├── requirements.txt      # Dependencias Python
├── data_cache/           # Cache de datos
├── saved_models/         # Modelos entrenados (.pkl)
├── logs/                 # Logs del sistema
└── Obsidian_Cooper/      # 🆕 Documentación técnica
    ├── Docs/
    │   └── Technical_Documentation.md
    └── Flows/
        ├── System_Flow.canvas
        └── Architecture_Overview.canvas
```

### Componentes Principales

#### Módulos Core
1. **DataCollector**: Recolección datos Yahoo Finance (históricos, tiempo real, market info)
2. **DataProcessor**: Pipeline de procesamiento (limpieza, features, target, split)
3. **KNNTradingModel**: Modelo ML (train, predict, evaluate, cross-validate)
4. **TechnicalIndicators**: Cálculo de 24+ indicadores técnicos
5. **Visualizer**: Gráficos interactivos (Matplotlib/Plotly)

#### Módulos Avanzados 🆕
6. **AdvancedDataSources**: Integración multi-fuente (World Bank, FRED, LME, News)
7. **AdvancedAnalytics**: BI profesional (Monte Carlo, VaR/CVaR, Backtesting, Escenarios)

#### Interfaces
8. **main.py**: Sistema CLI con análisis completo y visualizaciones
9. **dashboard.py**: Dashboard web interactivo con Streamlit y Plotly

---

## 🔬 Metodología del Algoritmo KNN

### Funcionamiento Core

1. **Recolección Multi-Fuente**: 
   - Yahoo Finance: Datos técnicos (precio, volumen, OHLC)
   - World Bank: Producción mundial de metales
   - FRED: Indicadores macroeconómicos (GDP, inflación, PMI)
   - LME: Precios institucionales y stocks
   - News API: Sentimiento del mercado

2. **Feature Engineering Avanzado**: 
   - 24 features técnicas (precio, volumen, medias, indicadores, ratios)
   - Features fundamentales (balance O/D, China PMI, demanda EV)
   - Features de sentimiento de mercado

3. **Normalización**: Min-Max scaling para comparabilidad

4. **Búsqueda de Patrones**: 
   - K=50 momentos históricos más similares (distancia euclidiana)
   - Ponderación por distancia (vecinos más cercanos pesan más)

5. **Predicción Probabilística**: 
   - Analiza qué ocurrió después de situaciones similares
   - Calcula probabilidades por clase (VENTA/HOLD/COMPRA)
   - Genera nivel de confianza

6. **Análisis de Riesgo**:
   - Value at Risk (VaR) histórico y paramétrico
   - Conditional VaR (Expected Shortfall)
   - Simulación Monte Carlo (1000+ escenarios)

7. **Señal Final**: Recomendación con confianza ≥70% y análisis completo

### Parámetros del Modelo

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `K_NEIGHBORS` | 50 | Número de vecinos más cercanos |
| `WEIGHTS` | 'distance' | Ponderación por distancia |
| `ALGORITHM` | 'auto' | Algoritmo de búsqueda |
| `METRIC` | 'euclidean' | Métrica de distancia |

### Variables Objetivo

- **Target Multi-clase**: 
  - `1` = COMPRA (subida > 2%)
  - `0` = HOLD (cambio entre -2% y +2%)
  - `-1` = VENTA (bajada > 2%)

---

## 🌐 Fuentes de Datos Múltiples 🆕

### Integración Multi-Source

El sistema integra **7 fuentes de datos** diferentes para análisis holístico:

#### 1. Yahoo Finance (Datos Técnicos) ✅
- **Tipo**: Precios y volúmenes en tiempo real
- **Datos**: OHLC, volumen, precio ajustado
- **Frecuencia**: Diaria, actualización continua
- **Historial**: 2 años por defecto (configurable)

#### 2. World Bank (Producción Global)
- **Tipo**: Datos macroeconómicos
- **Datos**: Producción mundial de cobre por país
- **Frecuencia**: Anual
- **API**: pública, formato JSON

#### 3. FRED - Federal Reserve (Indicadores Económicos)
- **Tipo**: Indicadores macroeconómicos USA
- **Datos**: GDP growth, inflación, desempleo, tasas de interés, PMI manufacturero
- **Frecuencia**: Mensual/trimestral
- **Relevancia**: USA es gran consumidor de cobre

#### 4. London Metal Exchange - LME (Precios Institucionales)
- **Tipo**: Precios oficiales de metales
- **Datos**: Cash prices, futuros 3 meses, stocks en almacenes, open interest
- **Frecuencia**: Diaria
- **Método**: Web scraping (API de pago disponible)

#### 5. China PMI (Demanda Industrial)
- **Tipo**: Indicador de actividad manufacturera
- **Datos**: Manufacturing PMI de China
- **Relevancia**: China consume ~50% del cobre mundial
- **Interpretación**: >50 = expansión, <50 = contracción

#### 6. Electric Vehicle Market (Demanda Futura)
- **Tipo**: Proyecciones de demanda
- **Datos**: Ventas globales de vehículos eléctricos, proyecciones de crecimiento
- **Relevancia**: Cada EV usa 2.5x más cobre que vehículo tradicional
- **Fuente**: IEA, Bloomberg NEF

#### 7. Sentiment Analysis (Percepción de Mercado)
- **Tipo**: Análisis de sentimiento de noticias
- **Datos**: Volumen de noticias, sentimiento (-1 a +1), trending topics
- **Método**: NLP sobre artículos financieros
- **API**: News API, Google News

### Balance Oferta-Demanda

El sistema calcula el **balance global** integrando:
- **Oferta**: Producción mundial (World Bank) + stocks LME
- **Demanda**: Consumo industrial + demanda EV proyectada + actividad manufacturera
- **Resultado**: Déficit/Superávit que afecta precios

---

## 📊 Features del Modelo

### Categorías de Features (24 total)

#### 1. Precio y Momentum (4 features)
- `close`: Precio de cierre
- `price_change_pct`: Cambio porcentual
- `price_momentum_5`: Momentum a 5 períodos
- `price_momentum_10`: Momentum a 10 períodos

#### 2. Volumen (3 features)
- `volume`: Volumen actual
- `volume_change_pct`: Cambio de volumen
- `volume_sma_20`: Media móvil de volumen

#### 3. Medias Móviles (5 features)
- `sma_20`, `sma_50`, `sma_200`: Simple Moving Average
- `ema_12`, `ema_26`: Exponential Moving Average

#### 4. Indicadores Técnicos (9 features)
- `rsi`: Relative Strength Index
- `macd`, `macd_signal`, `macd_diff`: MACD indicators
- `bb_upper`, `bb_middle`, `bb_lower`, `bb_width`: Bollinger Bands
- `atr`: Average True Range

#### 5. Relaciones de Precio (3 features)
- `price_to_sma20`: Ratio precio/SMA20
- `price_to_sma50`: Ratio precio/SMA50
- `price_to_sma200`: Ratio precio/SMA200

---

## 🔄 Pipeline de Ejecución

### main.py - Flujo CLI Completo

1. **Inicialización**: Configuración, logging, validación de entorno
2. **Recolección Multi-Fuente**: 
   - Datos técnicos (Yahoo Finance)
   - Datos económicos (World Bank, FRED)
   - Sentimiento de mercado (News API)
3. **Validación de Calidad**: Quality score, null values, outliers
4. **Procesamiento Avanzado**: 
   - Limpieza y normalización
   - 24 features técnicas
   - Features fundamentales
   - Variable objetivo (COMPRA=1, HOLD=0, VENTA=-1)
5. **División Estratificada**: Train/Test split (80/20)
6. **Entrenamiento KNN**: K=50, weights='distance', metric='euclidean'
7. **Evaluación Multi-Métrica**: 
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - Classification Report
8. **Validación Cruzada**: 5-fold CV con scores detallados
9. **Predicción Actual**: Señal + confianza + probabilidades
10. **Análisis de Riesgo**: VaR, CVaR, simulación Monte Carlo
11. **Visualización**: 
    - Gráfico 1: Precio + SMA + Bollinger Bands
    - Gráfico 2: RSI + MACD + Volumen + ATR
    - Gráfico 3: Predicciones históricas
    - Gráfico 4: Confusion Matrix
12. **Persistencia**: Modelo (.pkl), logs, cache

### dashboard.py - Dashboard Web Profesional 🆕

- **Framework**: Streamlit + Plotly (interactividad avanzada)
- **Arquitectura**: Modular con caché inteligente (@st.cache_data, TTL=1h)

**Características Principales**:
1. **Señal Principal**:
   - Badge visual coloreado (verde/rojo/amarillo)
   - Gauge de confianza animado (0-100%)
   - Recomendación clara (EJECUTAR/ESPERAR/CONSIDERAR)

2. **Análisis Multi-Dimensional**:
   - **Tab 1 - Gráficos**: Precio + SMA + BB + RSI + MACD (Plotly interactivo)
   - **Tab 2 - Probabilidades**: Distribución 3 clases + barras + progress bars
   - **Tab 3 - Condiciones**: Tabla de verificación de condiciones de trading
   - **Tab 4 - Modelo**: Métricas KNN (train/test) + gráfico comparativo

3. **Análisis Avanzado** (si módulos disponibles):
   - Simulación Monte Carlo con distribución de precios futuros
   - VaR/CVaR con múltiples niveles de confianza
   - Backtesting con métricas (Sharpe, Drawdown, Win Rate)
   - Análisis fundamental (balance O/D, China PMI, demanda EV)
   - Análisis de escenarios (optimista/base/pesimista)

4. **Información de Mercado**:
   - Métricas clave (52w high/low, volumen, cambio %)
   - Indicadores técnicos actuales
   - Contexto macroeconómico

5. **UX/UI Profesional**:
   - CSS personalizado con colores temáticos
   - Responsive design
   - Sidebar con configuración
   - Disclaimers y advertencias
   - Timestamp de última actualización

---

## 📈 Indicadores Técnicos Utilizados

### 1. RSI (Relative Strength Index)
- **Período**: 14
- **Interpretación**: 
  - > 70: Sobrecompra
  - < 30: Sobreventa
- **Fórmula**: RSI = 100 - (100 / (1 + RS)), donde RS = Avg Gain / Avg Loss

### 2. MACD (Moving Average Convergence Divergence)
- **Parámetros**: Fast=12, Slow=26, Signal=9
- **Componentes**:
  - MACD Line: EMA(12) - EMA(26)
  - Signal Line: EMA(9) del MACD
  - Histogram: MACD - Signal

### 3. Bollinger Bands
- **Período**: 20
- **Desviación**: 2σ
- **Componentes**:
  - Upper Band: SMA(20) + 2σ
  - Middle Band: SMA(20)
  - Lower Band: SMA(20) - 2σ
  - Width: (Upper - Lower) / Middle

### 4. ATR (Average True Range)
- **Período**: 14
- **Propósito**: Medir volatilidad del mercado
- **Fórmula**: Media móvil del True Range

---

## 🎯 Sistema de Señales

### Generación de Señales

```python
if confidence >= 70% and prediction == 1:
    signal = "COMPRA"
    recommendation = "EJECUTAR"
elif confidence >= 70% and prediction == -1:
    signal = "VENTA"
    recommendation = "CONSIDERAR"
else:
    signal = "HOLD"
    recommendation = "ESPERAR"
```

### Niveles de Confianza

| Rango | Categoría | Acción |
|-------|-----------|--------|
| 80-100% | MUY ALTA | Ejecutar con alta convicción |
| 70-80% | ALTA | Ejecutar con cautela |
| 60-70% | MEDIA | Esperar confirmación |
| < 60% | BAJA | No operar |

---

## 📦 Stack Tecnológico

### Machine Learning & Data Science
```python
scikit-learn >= 1.5.0  # KNN, métricas, cross-validation
numpy >= 1.26.0        # Operaciones numéricas, arrays, álgebra lineal
pandas >= 2.1.0        # DataFrames, manipulación de series temporales
scipy >= 1.11.0        # Estadística avanzada (VaR, distribuciones)
```

### Fuentes de Datos Financieros
```python
yfinance >= 0.2.40     # Yahoo Finance API (precios, volumen)
ta >= 0.11.0           # Technical Analysis Library (indicadores)
requests >= 2.31.0     # HTTP para APIs (World Bank, FRED, LME)
```

### Visualización & Dashboard
```python
matplotlib >= 3.8.0    # Gráficos estáticos (baseline, reports)
seaborn >= 0.13.0      # Gráficos estadísticos elegantes
plotly >= 5.18.0       # Gráficos interactivos (hover, zoom, pan)
streamlit >= 1.28.0    # Framework web para dashboard
```

### Utilidades & Productividad
```python
colorama >= 0.4.6      # Colores en terminal (CLI mejorado)
tabulate >= 0.9.0      # Tablas formateadas en consola
python-dotenv >= 1.0.0 # Gestión de variables de entorno (.env)
```

### Opcional (Notificaciones)
```python
# twilio == 8.10.0              # SMS notifications
# python-telegram-bot == 20.6   # Telegram bot
```

### Requisitos del Sistema
- **Python**: 3.8+ (recomendado 3.11 o 3.12)
- **RAM**: Mínimo 4GB (recomendado 8GB para Monte Carlo)
- **Almacenamiento**: 500MB (datos + modelos + cache)
- **Internet**: Conexión estable para APIs

---

## 📊 Técnicas de Business Intelligence Avanzado 🆕

### 1. Simulación Monte Carlo
**Objetivo**: Proyectar distribución de precios futuros mediante 1000+ escenarios

**Metodología**:
- Modelo: Movimiento Browniano Geométrico (GBM)
- Fórmula: `S(t+1) = S(t) * exp((μ - σ²/2) + σ*Z)`
  - S(t): Precio en tiempo t
  - μ: Retorno promedio histórico
  - σ: Volatilidad (desviación estándar)
  - Z: Shock aleatorio (distribución normal)

**Outputs**:
- Precio esperado (media de simulaciones)
- Intervalos de confianza (5%, 25%, 75%, 95%)
- Probabilidad de subida/bajada
- Rango de precios más probable

### 2. Value at Risk (VaR) y CVaR
**Objetivo**: Cuantificar riesgo de pérdida máxima esperada

**Métodos Implementados**:

#### VaR Histórico
- Basado en distribución empírica de retornos históricos
- Percentil de la distribución (ej: 5% para 95% confianza)
- No asume distribución normal

#### VaR Paramétrico
- Asume distribución normal de retornos
- Fórmula: `VaR = μ + σ * Z(α)`
- Más rápido pero menos preciso en colas gordas

#### CVaR (Conditional VaR / Expected Shortfall)
- Pérdida esperada **dado que** se excedió VaR
- CVaR = E[Retorno | Retorno ≤ VaR]
- Métrica más conservadora y coherente

**Niveles de Confianza**: 90%, 95%, 99%

### 3. Backtesting de Estrategias
**Objetivo**: Validar rentabilidad histórica de señales KNN

**Métricas Calculadas**:

| Métrica | Descripción | Interpretación |
|---------|-------------|----------------|
| **Total Return** | Retorno acumulado | >0% es ganancia |
| **Sharpe Ratio** | Retorno/Riesgo | >1 bueno, >2 excelente |
| **Max Drawdown** | Caída máxima desde pico | Menor es mejor |
| **Win Rate** | % operaciones ganadoras | >50% es positivo |
| **Profit Factor** | Ganancias/Pérdidas | >1.5 es bueno |
| **Avg Win/Loss** | Ratio ganancia/pérdida promedio | >2 es ideal |

**Proceso**:
1. Generar señales en datos históricos
2. Simular operaciones (compra/venta según señal)
3. Aplicar costos de transacción (slippage, comisiones)
4. Calcular curva de equity
5. Computar métricas de rendimiento

### 4. Análisis de Escenarios (What-If Analysis)
**Objetivo**: Evaluar impacto de diferentes escenarios macroeconómicos

**Escenarios Definidos**:

#### Escenario Optimista (30% probabilidad)
- Fuerte adopción de vehículos eléctricos (+25% YoY)
- China PMI >52 (expansión robusta)
- Déficit de oferta global
- **Precio proyectado**: +15% a +25%

#### Escenario Base (50% probabilidad)
- Crecimiento EV moderado (+15% YoY)
- China PMI ~50 (estable)
- Balance oferta-demanda equilibrado
- **Precio proyectado**: -5% a +10%

#### Escenario Pesimista (20% probabilidad)
- Recesión global (GDP negativo)
- China PMI <48 (contracción)
- Superávit de oferta
- **Precio proyectado**: -15% a -25%

**Precio Ponderado** = Σ(Precio_escenario * Probabilidad)

### 5. Optimización de Cartera (Kelly Criterion)
**Objetivo**: Calcular tamaño óptimo de posición

**Fórmula de Kelly**:
```
f* = (p * b - q) / b
```
Donde:
- f*: Fracción óptima del capital a invertir
- p: Probabilidad de ganancia (win rate)
- q: Probabilidad de pérdida (1 - p)
- b: Ratio ganancia/pérdida promedio

**Implementación**:
- Kelly completo (agresivo)
- Half-Kelly (conservador, recomendado)
- Quarter-Kelly (muy conservador)

**Output**: % de capital a invertir por operación

### 6. Análisis de Correlaciones
**Objetivo**: Identificar relaciones entre variables

**Métodos**:
- Matriz de correlación de Pearson
- Correlación de Spearman (no lineal)
- Rolling correlations (ventana móvil)

**Variables Analizadas**:
- Precio cobre vs USD Index
- Precio cobre vs China PMI
- Precio cobre vs tasas de interés
- Precio cobre vs S&P 500
- Precio cobre vs oro

---

## 🔍 Evaluación del Modelo

### Métricas Utilizadas

1. **Accuracy**: Porcentaje de predicciones correctas
2. **Precision**: Ratio de verdaderos positivos sobre predicciones positivas
3. **Recall**: Ratio de verdaderos positivos sobre positivos reales
4. **F1-Score**: Media armónica de Precision y Recall
5. **Confusion Matrix**: Matriz de confusión detallada
6. **Cross-Validation**: 5-fold para validar robustez

### Interpretación de Métricas

- **Accuracy > 60%**: Modelo supera probabilidad aleatoria (33.3% para 3 clases)
- **Precision alta**: Pocas falsas alarmas en señales de compra
- **Recall alto**: Captura la mayoría de oportunidades reales
- **F1-Score balanceado**: Buen equilibrio entre Precision y Recall

---

## 💾 Persistencia y Caché

### Modelos Guardados
- **Formato**: Pickle (.pkl)
- **Contenido**: Modelo entrenado + metadatos
- **Ubicación**: `saved_models/`
- **Naming**: `knn_model_YYYYMMDD_HHMMSS.pkl`

### Caché de Datos
- **Ubicación**: `data_cache/`
- **TTL Dashboard**: 1 hora
- **Propósito**: Reducir llamadas API

### Logs
- **Ubicación**: `logs/`
- **Formato**: `knn_trading_YYYYMMDD.log`
- **Nivel**: INFO

---

## ⚙️ Configuración Avanzada

### config/settings.py

```python
# Trading
TARGET_PRICE = 8500          # Precio objetivo
CONFIDENCE_THRESHOLD = 0.70  # Confianza mínima
STOP_LOSS_PCT = 0.03         # Stop loss 3%
TAKE_PROFIT_PCT = 0.05       # Take profit 5%

# Datos
HISTORICAL_PERIOD = '2y'     # 2 años de histórico
DATA_INTERVAL = '1d'         # Intervalos diarios

# Modelo
K_NEIGHBORS = 50             # 50 vecinos
WEIGHTS = 'distance'         # Ponderación por distancia
```

---

## 🚀 Casos de Uso y Modos de Operación

### 1. Análisis Diario (Modo CLI)
```bash
python main.py
```
**Salida**:
- ✅ Señal de trading (COMPRA/VENTA/HOLD) con confianza
- ✅ Métricas del modelo (accuracy, precision, recall, F1)
- ✅ 4 gráficos interactivos guardados
- ✅ Tabla de condiciones de trading
- ✅ Recomendación final ejecutable
- ✅ Logs detallados en `logs/`
- ✅ Modelo guardado en `saved_models/`

**Tiempo de ejecución**: ~30-60 segundos

### 2. Dashboard Interactivo (Modo Web)
```bash
streamlit run dashboard.py
```
**Características**:
- 🌐 Acceso vía navegador: `http://localhost:8501`
- 🔄 Actualización automática (caché 1 hora)
- 📊 Gráficos interactivos Plotly (zoom, pan, hover)
- 🎯 Gauge de confianza animado
- 📈 Múltiples tabs (gráficos, probabilidades, condiciones, modelo)
- 🆕 Análisis avanzado (Monte Carlo, VaR, Backtesting)
- 📱 Responsive (funciona en móvil)

**Ideal para**: Monitoreo continuo, presentaciones, análisis exploratorio

### 3. Backtesting Histórico
```python
from data.advanced_analytics import AdvancedAnalytics

analytics = AdvancedAnalytics()
results = analytics.backtest_strategy(df, signals, initial_capital=10000)
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Total Return: {results['total_return']:.2%}")
```
**Aplicación**: Validar rentabilidad en diferentes períodos históricos

### 4. Simulación de Escenarios
```python
from data.advanced_analytics import AdvancedAnalytics

analytics = AdvancedAnalytics()
scenarios = analytics.scenario_analysis(
    current_price=8500,
    ev_growth=0.25,  # Crecimiento EV agresivo
    china_pmi=52     # PMI expansivo
)
print(f"Precio optimista: ${scenarios['optimistic']:.2f}")
```
**Aplicación**: Planificación estratégica, análisis de sensibilidad

### 5. Optimización de Hiperparámetros
Editar `config/settings.py`:
```python
K_NEIGHBORS = 30  # Probar con 30 vecinos
CONFIDENCE_THRESHOLD = 0.75  # Requerir 75% confianza
HISTORICAL_PERIOD = '5y'  # Usar 5 años de datos
```
Luego ejecutar `python main.py` y comparar métricas

**Aplicación**: Tuning del modelo, diferentes perfiles de riesgo

### 6. Integración con Trading Bot
```python
from models.knn_model import KNNTradingModel

model = KNNTradingModel()
model.load_model('knn_model_20241203_120000.pkl')

prediction = model.predict_next(current_data)

if prediction['confidence'] >= 0.80 and prediction['signal'] == 'COMPRA':
    # Ejecutar orden de compra vía API de broker
    broker.place_order('BUY', symbol='HG=F', quantity=100)
```
**Aplicación**: Trading automatizado (usar con extrema cautela)

### 7. Análisis Académico/Investigación
**Casos de estudio**:
- Comparación KNN vs LSTM vs Random Forest
- Impacto de features fundamentales en predicción
- Eficiencia de diferentes valores de K
- Análisis de eficiencia de mercado (EMH)
- Backtesting en crisis históricas (2008, 2020)

**Documentación disponible**: Métodos, métricas, visualizaciones para papers

---

## ⚠️ Limitaciones y Consideraciones

### Limitaciones Técnicas

#### Limitaciones del Algoritmo KNN
1. **Maldición de la dimensionalidad**: 
   - Con 24 features, el espacio es muy "vacío"
   - Distancias euclídeanas pueden perder significado
   - **Mitigación**: Selección cuidadosa de features, normalización

2. **Dependencia de patrones históricos**:
   - Solo encuentra situaciones similares previas
   - Crisis sin precedentes no se predicen bien (ej: COVID-19)
   - **Mitigación**: Integrar análisis fundamental y sentimiento

3. **Sensibilidad a outliers**:
   - Eventos extremos distorsionan distancias
   - **Mitigación**: Limpieza de datos, detección de outliers

4. **Lag en señales**:
   - KNN es reactivo, no predictivo de cambios abruptos
   - **Mitigación**: Usar con trailing stop-loss

#### Limitaciones de Datos
5. **Latencia de APIs**:
   - Yahoo Finance: delay de ~15 minutos
   - World Bank: datos anuales (rezago significativo)
   - **Impacto**: No apto para trading de alta frecuencia

6. **Calidad de datos**:
   - Posibles gaps, valores nulos, datos incorrectos
   - **Mitigación**: Validación de calidad automática

7. **Cobertura limitada**:
   - No todas las APIs disponibles en todos los países
   - Algunas requieren suscripción de pago

#### Limitaciones Computacionales
8. **Escalabilidad**:
   - KNN requiere almacenar todos los datos de entrenamiento
   - Predicción es O(n*d) donde n=muestras, d=dimensiones
   - **Impacto**: No escala a millones de registros

9. **Simulaciones intensivas**:
   - Monte Carlo con 1000+ simulaciones: ~5-10 segundos
   - **Mitigación**: Caché, paralelización (futuro)

### Consideraciones de Trading

#### Riesgo Financiero
1. **NO es asesoramiento financiero**: 
   - Sistema educativo/investigación únicamente
   - No sustituye análisis profesional
   - **DISCLAIMER obligatorio**

2. **Gestión de riesgo mandatoria**:
   - **Stop-loss**: Mínimo 3% (configurable)
   - **Take-profit**: 5% recomendado
   - **Position sizing**: Máximo 5% del capital por operación
   - **Kelly Criterion**: Usar fracción (Half-Kelly)

3. **Diversificación**:
   - No poner todos los fondos en cobre
   - Diversificar por clases de activos
   - Considerar correlaciones

4. **Contexto macroeconómico**:
   - Modelo no captura noticias de última hora
   - Eventos geopoliticos pueden invalidar señales
   - **Recomendación**: Leer noticias antes de ejecutar

5. **Costos de transacción**:
   - Comisiones, spreads, slippage no incluidos en backtesting
   - Trading frecuente reduce retornos
   - **Impacto**: ~0.1-0.5% por transacción

#### Consideraciones Regulatorias
6. **Cumplimiento legal**:
   - Verificar regulaciones locales
   - Algunas jurisdicciones restringen trading algoritmico
   - Impuestos sobre ganancias de capital

7. **Responsabilidad**:
   - Usuario es 100% responsable de sus operaciones
   - Desarrolladores NO asumen responsabilidad por pérdidas
   - **Usar bajo su propio riesgo**

### Supuestos del Modelo

1. **Mercados semi-eficientes**: Patrones históricos tienen valor predictivo
2. **Estacionariedad débil**: Propiedades estadísticas relativamente estables
3. **Costos de transacción despreciables**: No considerados en modelo base
4. **Liquidez suficiente**: Puede entrar/salir sin mover el mercado
5. **No hay manipulación de mercado**: Precios reflejan información real

### Mejores Prácticas Recomendadas

✅ **Empezar con paper trading** (simulación sin dinero real)  
✅ **Usar confianza mínima 70%** para ejecutar señales  
✅ **Combinar con análisis fundamental** (noticias, reportes)  
✅ **Mantener diario de operaciones** para aprender  
✅ **Re-entrenar modelo mensualmente** con datos frescos  
✅ **Monitorear degradación de métricas** (accuracy baja = re-entrenar)  
✅ **No operar en alta volatilidad** (VIX >30, eventos mayores)  
✅ **Respetar estrictamente stop-loss** automáticos  
✅ **Invertir solo capital que puede permitirse perder**

---

## 🔧 Mantenimiento y Mejoras Futuras

### Mejoras Potenciales
- [ ] Implementar LSTM/GRU para series temporales
- [ ] Añadir análisis de sentimiento (Twitter/News)
- [ ] Integración con APIs de brokers para trading automático
- [ ] Optimización de hiperparámetros con Grid Search
- [ ] Sistema de notificaciones (Email/Telegram)
- [ ] Backtesting framework completo con métricas Sharpe

### Mantenimiento Regular
- Actualizar dependencias mensualmente
- Validar calidad de datos de Yahoo Finance
- Re-entrenar modelo con datos frescos
- Monitorear degradación de métricas

---

## 📚 Referencias Técnicas

### Papers y Documentación
- [scikit-learn KNN](https://scikit-learn.org/stable/modules/neighbors.html)
- [Technical Analysis Library](https://technical-analysis-library-in-python.readthedocs.io/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)

### Conceptos Clave
- **K-Nearest Neighbors**: Algoritmo de clasificación basado en proximidad
- **Feature Engineering**: Creación de variables predictivas desde datos raw
- **Cross-Validation**: Técnica para validar generalización del modelo
- **Time Series Analysis**: Análisis de series temporales financieras

---

---

## 📊 Resumen de Capacidades del Sistema

| Categoría | Capacidad | Estado |
|-----------|-----------|--------|
| **Machine Learning** | K-Nearest Neighbors (K=50) | ✅ Implementado |
| **Datos Técnicos** | Yahoo Finance (OHLC, volumen) | ✅ Implementado |
| **Indicadores** | RSI, MACD, BB, ATR, SMA, EMA | ✅ 24 features |
| **Multi-Source** | World Bank, FRED, LME | ✅ Integrado |
| **Análisis Fundamental** | Balance O/D, PMI, Demanda EV | ✅ Integrado |
| **Monte Carlo** | 1000+ simulaciones GBM | ✅ Implementado |
| **Riesgo** | VaR, CVaR (90%, 95%, 99%) | ✅ Implementado |
| **Backtesting** | Sharpe, Drawdown, Win Rate | ✅ Implementado |
| **Escenarios** | Optimista/Base/Pesimista | ✅ Implementado |
| **Optimización** | Kelly Criterion | ✅ Implementado |
| **Visualización** | Matplotlib + Plotly | ✅ Implementado |
| **Dashboard** | Streamlit interactivo | ✅ Implementado |
| **CLI** | Sistema completo en terminal | ✅ Implementado |
| **Persistencia** | Modelos .pkl, logs, cache | ✅ Implementado |
| **Documentación** | Técnica + Obsidian Canvas | ✅ Completa |

---

## 🎯 Conclusión

Este sistema representa un **enfoque integral** para el trading de cobre que combina:
- **Machine Learning** (KNN para buscar patrones históricos)
- **Análisis Técnico** (24 indicadores calculados automáticamente)  
- **Análisis Fundamental** (oferta-demanda global, indicadores macro)
- **Business Intelligence** (Monte Carlo, VaR, backtesting, escenarios)
- **Visualización Profesional** (dashboard interactivo con Plotly)

Es ideal para:
- 🎓 **Educación**: Aprender ML aplicado a finanzas
- 🔬 **Investigación**: Estudios académicos sobre trading algorítmico
- 💼 **Trading asistido**: Generación de señales como herramienta de apoyo
- 📊 **Business Intelligence**: Análisis de riesgo y proyecciones

**⚠️ RECORDATORIO FINAL**: Esta herramienta es para **fines educativos e informativos únicamente**. No garantiza ganancias. El trading conlleva riesgos significativos. Siempre consulte con asesores financieros profesionales antes de tomar decisiones de inversión.

---

**Versión**: 2.0 (Actualizada con módulos avanzados)  
**Última Actualización**: Diciembre 2024  
**Autor**: Sistema KNN Trading - Copper Recommender  
**Repositorio**: copper-recommender  
**Licencia**: MIT (Uso educativo)
