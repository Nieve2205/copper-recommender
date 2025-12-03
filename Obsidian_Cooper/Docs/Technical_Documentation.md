# Documentación Técnica - Sistema KNN para Trading de Cobre

## 📋 Resumen Ejecutivo

Sistema de recomendación de trading basado en Machine Learning que utiliza K-Nearest Neighbors (KNN) para predecir movimientos del precio del cobre mediante el análisis de patrones históricos similares del mercado.

**Tecnología Principal**: Python 3.8+ con scikit-learn  
**Algoritmo**: K-Nearest Neighbors (KNN)  
**Objetivo**: Generar señales de COMPRA/VENTA/HOLD con niveles de confianza

---

## 🏗️ Arquitectura del Sistema

### Estructura Modular

```
copper-recommender/
├── config/           # Configuración centralizada
├── data/             # Gestión y procesamiento de datos
├── models/           # Modelos de Machine Learning
├── utils/            # Utilidades (indicadores, visualización)
├── main.py           # Orquestador principal (CLI)
└── dashboard.py      # Interfaz web (Streamlit)
```

### Componentes Principales

1. **DataCollector**: Recolección de datos en tiempo real vía Yahoo Finance
2. **DataProcessor**: Procesamiento y creación de features (indicadores técnicos)
3. **KNNTradingModel**: Modelo de Machine Learning para predicciones
4. **TechnicalIndicators**: Cálculo de indicadores técnicos (RSI, MACD, BB, ATR)
5. **Visualizer**: Generación de gráficos y análisis visual

---

## 🔬 Metodología del Algoritmo KNN

### Funcionamiento

1. **Recolección**: Obtiene datos históricos del cobre (2 años por defecto)
2. **Feature Engineering**: Calcula 24+ indicadores técnicos
3. **Normalización**: Min-Max scaling de features
4. **Búsqueda**: Encuentra los K=50 momentos históricos más similares
5. **Predicción**: Analiza qué ocurrió después de esos momentos
6. **Señal**: Genera recomendación con nivel de confianza

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

### main.py - Flujo CLI

1. **Inicialización**: Configuración y logging
2. **Recolección**: Descarga datos históricos (Yahoo Finance)
3. **Validación**: Verifica calidad de datos
4. **Procesamiento**: Limpieza y creación de features
5. **División**: Train/Test split (80/20)
6. **Entrenamiento**: Entrena modelo KNN
7. **Evaluación**: Accuracy, Precision, Recall, F1-Score
8. **Validación Cruzada**: 5-fold cross-validation
9. **Predicción**: Genera señal actual con confianza
10. **Visualización**: 4 gráficos interactivos
11. **Persistencia**: Guarda modelo entrenado (.pkl)

### dashboard.py - Interfaz Web

- **Framework**: Streamlit con Plotly
- **Características**:
  - Dashboard interactivo en tiempo real
  - Gauge de confianza visual
  - Análisis técnico completo
  - Métricas del modelo
  - Verificación de condiciones de trading
  - Actualización con caché (1 hora TTL)

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

## 📦 Dependencias Principales

### Machine Learning
```
scikit-learn >= 1.5.0  # KNN y métricas
numpy >= 1.26.0        # Operaciones numéricas
pandas >= 2.1.0        # Manipulación de datos
```

### Datos Financieros
```
yfinance >= 0.2.40     # API de Yahoo Finance
ta >= 0.11.0           # Indicadores técnicos
```

### Visualización
```
matplotlib >= 3.8.0    # Gráficos estáticos
seaborn >= 0.13.0      # Gráficos estadísticos
plotly >= 5.18.0       # Gráficos interactivos
streamlit >= 1.28.0    # Dashboard web
```

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

## 🚀 Casos de Uso

### 1. Análisis Diario
```bash
python main.py
```
Genera señal de trading con análisis completo y visualizaciones.

### 2. Dashboard Interactivo
```bash
streamlit run dashboard.py
```
Interfaz web con actualización en tiempo real.

### 3. Backtesting Histórico
Modificar `HISTORICAL_PERIOD` y analizar rendimiento en períodos específicos.

### 4. Optimización de Hiperparámetros
Ajustar `K_NEIGHBORS`, `CONFIDENCE_THRESHOLD` para diferentes estrategias.

---

## ⚠️ Limitaciones y Consideraciones

### Limitaciones Técnicas
1. **Dependencia de datos históricos**: Requiere patrones similares previos
2. **No captura eventos únicos**: Crisis sin precedentes no se predicen bien
3. **Latencia de datos**: Yahoo Finance puede tener delay
4. **Overfitting potencial**: En mercados muy volátiles

### Consideraciones de Trading
1. **No es asesoramiento financiero**: Solo herramienta educativa
2. **Gestión de riesgo**: Siempre usar stop-loss
3. **Diversificación**: No depender de una sola señal
4. **Contexto fundamental**: Considerar noticias y eventos macroeconómicos

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

**Versión**: 1.0  
**Última Actualización**: Diciembre 2024  
**Autor**: Sistema KNN Trading  
**Licencia**: MIT
