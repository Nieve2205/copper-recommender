# 🔷 Sistema KNN Avanzado para Trading de Cobre con Business Intelligence

Sistema profesional de recomendación de trading basado en Machine Learning (K-Nearest Neighbors) con **múltiples fuentes de datos**, **análisis fundamental**, **técnicas avanzadas de BI** y **gestión de riesgo**.

## 🌟 **NUEVAS CARACTERÍSTICAS ÉPICAS**

### 🎯 Lo que hace este proyecto ÚNICO:

✨ **Multi-Source Data Integration** - No solo Yahoo Finance  
✨ **Análisis Fundamental + Técnico** combinado  
✨ **Simulación Monte Carlo** para predicciones probabilísticas  
✨ **Backtesting robusto** con métricas profesionales  
✨ **Value at Risk (VaR)** y gestión de riesgo  
✨ **Análisis de escenarios** (What-If Analysis)  
✨ **Optimización de cartera** (Kelly Criterion)  
✨ **Dashboard interactivo** de nivel profesional  
✨ **Análisis de sentimiento** del mercado  
✨ **Balance oferta-demanda** global  

---

## 📊 Fuentes de Datos Múltiples

### 1. **Yahoo Finance** (Datos técnicos)
- Precios históricos y en tiempo real
- Volúmenes de transacción
- Indicadores técnicos

### 2. **World Bank** (Datos macroeconómicos)
- Producción mundial de metales
- Datos económicos por país
- Indicadores de desarrollo

### 3. **FRED - Federal Reserve** (Indicadores económicos)
- GDP Growth
- Inflation Rate
- Unemployment Rate
- Interest Rates
- Manufacturing Index

### 4. **London Metal Exchange (LME)** (Precios institucionales)
- Cash prices
- 3-month futures
- Warehouse stocks
- Open interest

### 5. **Análisis de Mercado EV** (Demanda futura)
- Ventas globales de vehículos eléctricos
- Proyecciones de demanda de cobre
- Tasas de crecimiento del sector

### 6. **Sentiment Analysis** (Noticias y tendencias)
- Análisis de sentimiento del mercado
- Volumen de noticias
- Trending topics

### 7. **China PMI** (Principal consumidor)
- Manufacturing PMI
- Indicadores de actividad económica
- Proyecciones de demanda


---

## 🧠 Técnicas Avanzadas de Business Intelligence

### 1. **Simulación Monte Carlo**
- 1000+ simulaciones de precios futuros
- Distribución probabilística de resultados
- Intervalos de confianza del 5% al 95%
- Probabilidad de subida/bajada

### 2. **Value at Risk (VaR)**
- VaR histórico y paramétrico
- Conditional VaR (Expected Shortfall)
- Análisis de pérdida máxima esperada
- Múltiples niveles de confianza (90%, 95%, 99%)

### 3. **Backtesting Profesional**
- Métricas completas de rendimiento:
  - Sharpe Ratio
  - Maximum Drawdown
  - Win Rate
  - Profit Factor
  - Total Return
- Curvas de equity
- Análisis de drawdown periods

### 4. **Análisis de Escenarios (What-If)**
- Escenario optimista (fuerte demanda EV)
- Escenario base (crecimiento normal)
- Escenario pesimista (recesión global)
- Precios ponderados por probabilidad

### 5. **Optimización de Cartera**
- Kelly Criterion para tamaño óptimo de posición
- Gestión de riesgo por operación
- Cálculo de capital óptimo a invertir
- Límites de pérdida máxima

### 6. **Análisis Fundamental**
- Balance oferta-demanda global
- Producción y consumo por país
- Impacto del mercado de vehículos eléctricos
- Indicadores macroeconómicos

### 7. **Correlación Multi-Variable**
- Matriz de correlaciones
- Identificación de relaciones fuertes
- Análisis de cointegración

---

## 📁 Estructura del Proyecto Mejorada

```
knn-copper-trading/
│
├── config/                  # Configuración
│   ├── __init__.py
│   └── settings.py         
│
├── data/                    # Gestión de datos
│   ├── __init__.py
│   ├── data_collector.py   # Yahoo Finance
│   ├── data_processor.py   # Procesamiento
│   └── advanced_sources.py # 🆕 Fuentes múltiples (WB, FRED, LME)
│
├── models/                  # Machine Learning
│   ├── __init__.py
│   └── knn_model.py        # Modelo KNN
│
├── analytics/               # 🆕 Análisis avanzado
│   ├── __init__.py
│   └── advanced_analytics.py # BI profesional
│
├── utils/                   # Utilidades
│   ├── __init__.py
│   ├── indicators.py       
│   └── visualizer.py       
│
├── main.py                  # Sistema CLI
├── dashboard.py             # 🆕 Dashboard épico mejorado
├── requirements.txt         
└── README.md               
```

---

## 🚀 Instalación y Configuración

### Requisitos Previos
- Python 3.8+ (recomendado 3.11 o 3.12)
- pip
- Conexión a internet

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/knn-copper-trading.git
cd knn-copper-trading

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Mac/Linux
# o
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Crear estructura de directorios

```bash
mkdir analytics
touch analytics/__init__.py
```

---

## 🎯 Cómo Usar

### Opción 1: Dashboard Interactivo (RECOMENDADO)

```bash
streamlit run dashboard.py
```

Abrirá en tu navegador: `http://localhost:8501`

**Características del Dashboard:**
- 📊 Análisis en tiempo real
- 🎯 Recomendaciones claras
- 📈 Gráficos interactivos
- 🎲 Simulación Monte Carlo
- 📉 Análisis de riesgo
- 🔙 Backtesting visual
- 🌍 Datos de múltiples fuentes

### Opción 2: Sistema CLI

```bash
python main.py
```

---

## 📊 Interpretación de Resultados

### Sección 1: Señal Principal
- **COMPRA 📈**: Modelo predice subida con alta confianza
- **VENTA 📉**: Modelo predice bajada o recomienda proteger
- **HOLD ⏸️**: Señal no es clara, esperar mejor momento

### Sección 2: Nivel de Confianza
- **80-100%** 🟢: MUY ALTA - Señal muy confiable
- **70-80%** 🔵: ALTA - Señal confiable  
- **60-70%** 🟡: MEDIA - Proceder con cautela
- **<60%** 🔴: BAJA - Esperar mejor oportunidad

### Sección 3: Análisis de Riesgo
- **VaR (Value at Risk)**: Pérdida máxima esperada
- **CVaR**: Pérdida esperada en peor escenario
- **Max Drawdown**: Caída máxima desde pico
- **Sharpe Ratio**: Retorno ajustado por riesgo

### Sección 4: Simulación Monte Carlo
- **Precio esperado**: Media de 1000 simulaciones
- **Intervalo 90%**: Rango de precios probable
- **Probabilidad subida**: % de simulaciones con precio > actual

### Sección 5: Análisis Fundamental
- **Balance O/D**: Déficit favorece precios altos
- **China PMI**: >50 indica expansión (bueno para demanda)
- **Sentimiento**: >0 es positivo para el mercado
- **Demanda EV**: Crecimiento proyectado de vehículos eléctricos

---

## 📈 Métricas de Rendimiento

### Métricas del Modelo KNN
- **Accuracy**: % de predicciones correctas
- **Precision**: % de señales de compra que fueron correctas
- **Recall**: % de oportunidades de compra capturadas
- **F1-Score**: Balance entre precision y recall

### Métricas de Trading
- **Total Return**: Retorno total de la estrategia
- **Sharpe Ratio**: Retorno/Riesgo (>1 es bueno, >2 es excelente)
- **Max Drawdown**: Pérdida máxima (menor es mejor)
- **Win Rate**: % de operaciones ganadoras
- **Profit Factor**: Ganancias/Pérdidas (>1.5 es bueno)

---

## 🎓 **POR QUÉ ESTE PROYECTO IMPRESIONARÁ A TU PROFESOR**

### 1. **Integración de Múltiples Fuentes** 🌐
No es solo un proyecto de ML básico, demuestra capacidad de:
- Integrar APIs externas
- Combinar datos técnicos y fundamentales
- Manejo de datos heterogéneos

### 2. **Business Intelligence Avanzado** 📊
Incluye técnicas de BI profesional:
- Análisis de escenarios
- Simulación probabilística
- Optimización de decisiones
- Gestión de riesgo

### 3. **Visualización Profesional** 📈
Dashboard interactivo con:
- Gráficos dinámicos con Plotly
- Métricas en tiempo real
- UX/UI intuitiva
- Responsive design

### 4. **Análisis de Riesgo** ⚠️
No solo predice, también gestiona riesgo:
- VaR y CVaR
- Position sizing
- Stop-loss dinámico
- Análisis de drawdown

### 5. **Validación Rigurosa** ✅
- Backtesting con datos históricos
- Validación cruzada
- Métricas estadísticas robustas
- Comparación con benchmarks

### 6. **Aplicación Práctica** 💼
Proyecto con aplicación real en:
- Trading de commodities
- Gestión de inversiones
- Análisis de mercados
- Toma de decisiones financieras

### 7. **Código Profesional** 💻
- Estructura modular
- Documentación completa
- Manejo de errores
- Logging detallado
- Código limpio y mantenible

---

## 🎯 Casos de Uso Empresariales

Este sistema puede adaptarse para:

1. **Trading de Commodities**
   - Oro, Plata, Petróleo
   - Materias primas agrícolas
   - Metales industriales

2. **Gestión de Portafolios**
   - Optimización de inversiones
   - Diversificación de activos
   - Rebalanceo automático

3. **Análisis de Riesgo**
   - Evaluación de exposición
   - Stress testing
   - Scenario planning

4. **Business Intelligence**
   - Análisis predictivo
   - Forecasting
   - Decision support systems

---

## 📚 Referencias Académicas

### Machine Learning
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*

### Análisis Financiero
- Hull, J. C. (2018). *Options, Futures, and Other Derivatives*
- Bodie, Z., Kane, A., & Marcus, A. J. (2018). *Investments*

### Business Intelligence
- Turban, E., et al. (2020). *Business Intelligence and Analytics: Systems for Decision Support*
- Provost, F., & Fawcett, T. (2013). *Data Science for Business*

---

## ⚙️ Configuración

Puedes personalizar el sistema editando `config/settings.py`:

```python
# Número de vecinos más cercanos
K_NEIGHBORS = 50

# Precio objetivo para señales
TARGET_PRICE = 8500

# Umbral de confianza
CONFIDENCE_THRESHOLD = 0.70  # 70%

# Símbolo del activo
COPPER_SYMBOL = 'HG=F'  # Cobre Futuro
```

### Parámetros Importantes

| Parámetro | Descripción | Valor por Defecto |
|-----------|-------------|-------------------|
| `K_NEIGHBORS` | Número de situaciones históricas similares a analizar | 50 |
| `TARGET_PRICE` | Precio objetivo del cobre (USD/tonelada) | 8500 |
| `CONFIDENCE_THRESHOLD` | Confianza mínima para ejecutar señales | 0.70 (70%) |
| `HISTORICAL_PERIOD` | Período de datos históricos a analizar | '2y' (2 años) |

---

## 📊 Indicadores Técnicos Utilizados

El sistema calcula automáticamente:

- **SMA** (Simple Moving Average): 20, 50, 200 períodos
- **EMA** (Exponential Moving Average): 12, 26 períodos
- **RSI** (Relative Strength Index): Momento del mercado
- **MACD** (Moving Average Convergence Divergence): Tendencia
- **Bollinger Bands**: Volatilidad
- **ATR** (Average True Range): Rango verdadero promedio
- **Volumen y variaciones**: Análisis de volumen

---

## 🎨 Visualizaciones

El sistema genera 4 gráficos interactivos:

1. **Historial de Precios**: Precio del cobre con medias móviles
2. **Indicadores Técnicos**: RSI, MACD, Bollinger Bands, Volumen
3. **Predicciones del Modelo**: Señales de compra/venta en el gráfico
4. **Matriz de Confusión**: Precisión del modelo

---

## 📈 Interpretación de Señales

### Señal: COMPRA 📈
- **Significado**: El modelo predice que el precio subirá
- **Acción**: Considerar comprar cobre
- **Condiciones**: 
  - Confianza ≥ 70%
  - Precio actual cerca del objetivo
  - Volumen adecuado

### Señal: VENTA 📉
- **Significado**: El modelo predice que el precio bajará
- **Acción**: Considerar vender o proteger posiciones
- **Condiciones**:
  - Confianza ≥ 70%
  - Indicadores técnicos confirman

### Señal: HOLD ⏸️
- **Significado**: No hay señal clara
- **Acción**: Mantener posiciones actuales y esperar
- **Condiciones**:
  - Confianza < 70%
  - Mercado indeciso

### Nivel de Confianza

- 🟢 **80-100%**: Confianza MUY ALTA - Señal muy confiable
- 🔵 **70-80%**: Confianza ALTA - Señal confiable
- 🟡 **60-70%**: Confianza MEDIA - Proceder con cautela
- 🔴 **<60%**: Confianza BAJA - Esperar mejor oportunidad

---

## 🧪 Pruebas de Módulos Individuales

Puedes probar cada módulo por separado:

```bash
# Probar recolector de datos
python -m data.data_collector

# Probar procesador de datos
python -m data.data_processor

# Probar indicadores técnicos
python -m utils.indicators

# Probar modelo KNN
python -m models.knn_model
```

---

## 📝 Ejemplos de Uso

### Ejemplo 1: Ejecución Básica

```bash
python main.py
```

### Ejemplo 2: Modificar Parámetros

Edita `config/settings.py` y cambia:

```python
K_NEIGHBORS = 30  # Usar 30 vecinos en lugar de 50
CONFIDENCE_THRESHOLD = 0.80  # Requerir 80% de confianza
```

Luego ejecuta:

```bash
python main.py
```

### Ejemplo 3: Usar Modelo Guardado

```python
from models.knn_model import KNNTradingModel

# Cargar modelo previamente entrenado
model = KNNTradingModel()
model.load_model('knn_model_20241201_143000.pkl')

# Hacer predicción
prediction = model.predict_next(current_data)
print(f"Señal: {prediction['signal']}")
```

---

## 🔧 Solución de Problemas

### Error: "No se pudieron obtener datos"

**Causa**: Problema de conexión o símbolo incorrecto

**Solución**:
1. Verifica tu conexión a internet
2. Verifica que el símbolo en `config/settings.py` sea correcto
3. Intenta con un símbolo alternativo (ej: 'CPER')

### Error: "Module not found"

**Causa**: Dependencias no instaladas

**Solución**:
```bash
pip install -r requirements.txt
```

### Error: "Not enough data points"

**Causa**: Datos insuficientes para entrenar

**Solución**: Aumenta el período histórico en `config/settings.py`:
```python
HISTORICAL_PERIOD = '5y'  # Usar 5 años en lugar de 2
```

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Haz fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## ⚠️ Disclaimer

**IMPORTANTE**: Este sistema es para fines **educativos y de investigación** únicamente.

- ❌ NO es asesoramiento financiero
- ❌ NO garantiza ganancias
- ❌ NO debe usarse como única base para decisiones de inversión
- ✅ Siempre consulta con un asesor financiero profesional
- ✅ Invierte solo lo que puedas permitirte perder
- ✅ Haz tu propia investigación (DYOR)

El trading conlleva riesgos significativos. Los resultados pasados no garantizan resultados futuros.

---

## 📚 Referencias

- **K-Nearest Neighbors**: [scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
- **Análisis Técnico**: [Technical Analysis Library](https://technical-analysis-library-in-python.readthedocs.io/)
- **Yahoo Finance**: [yfinance Documentation](https://pypi.org/project/yfinance/)
 
---

**Desarrollado con ❤️ para el análisis cuantitativo del mercado de cobre**

*Última actualización: Diciembre 2024*