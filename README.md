# 🔷 Sistema KNN para Trading de Cobre

Sistema completo de recomendación de trading basado en Machine Learning (K-Nearest Neighbors) que analiza momentos históricos similares del mercado del cobre para predecir movimientos futuros de precio.

## 📋 Descripción

Este sistema utiliza el algoritmo **K-Nearest Neighbors (KNN)** para encontrar situaciones de mercado similares en el historial y predecir si el precio del cobre subirá o bajará, generando señales de **COMPRA**, **VENTA** o **HOLD**.

### ¿Cómo funciona?

1. **Recopila datos en tiempo real** del mercado de cobre
2. **Calcula indicadores técnicos** (RSI, MACD, Bollinger Bands, etc.)
3. **Busca los 50 momentos históricos más similares** a la situación actual
4. **Analiza qué pasó después** de esas situaciones similares
5. **Genera una recomendación** con nivel de confianza

**Analogía simple**: Es como Netflix recomendando películas. Si a 50 personas con gustos similares a los tuyos les gustó una película, probablemente a ti también te gustará.

---

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Conexión a internet para descargar datos

### Paso 1: Clonar o Descargar el Proyecto

```bash
# Si tienes git
git clone https://github.com/Nieve2205/copper-recommender.git
cd copper-recommender

# O simplemente descarga y descomprime el ZIP
```

### Paso 2: Crear Entorno Virtual (Recomendado)

```bash
# Windows
python -m venv venv_bigdata
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv_bigdata/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

---

## 📁 Estructura del Proyecto

```
knn-copper-trading/
│
├── config/                  # Configuración del sistema
│   ├── __init__.py
│   └── settings.py         # Parámetros configurables
│
├── data/                    # Módulo de gestión de datos
│   ├── __init__.py
│   ├── data_collector.py   # Recolección de datos en tiempo real
│   └── data_processor.py   # Procesamiento y creación de features
│
├── models/                  # Modelos de Machine Learning
│   ├── __init__.py
│   └── knn_model.py        # Modelo KNN para trading
│
├── utils/                   # Utilidades
│   ├── __init__.py
│   ├── indicators.py       # Indicadores técnicos
│   └── visualizer.py       # Visualizaciones
│
├── data_cache/              # Caché de datos (se crea automáticamente)
├── saved_models/            # Modelos guardados (se crea automáticamente)
├── logs/                    # Logs del sistema (se crea automáticamente)
│
├── main.py                  # Archivo principal
├── requirements.txt         # Dependencias
└── README.md               # Este archivo
```

---

## 🎯 Uso

### Ejecución Básica

```bash
python main.py
```

El sistema ejecutará automáticamente:

1. ✅ Descarga de datos históricos del cobre
2. ✅ Cálculo de indicadores técnicos
3. ✅ Entrenamiento del modelo KNN
4. ✅ Evaluación del modelo
5. ✅ Generación de señal de trading actual
6. ✅ Visualizaciones interactivas

### Salida del Sistema

El sistema mostrará:

- 📊 **Información del mercado** (precio actual, cambio, volumen)
- 📈 **Métricas del modelo** (accuracy, precision, recall)
- 🎯 **Señal de trading** con nivel de confianza
- 📉 **Gráficos interactivos** con análisis técnico

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

## 📧 Contacto

Para preguntas, sugerencias o reportar bugs:

- 📧 Email: tu-email@ejemplo.com
- 🐛 Issues: [GitHub Issues](https://github.com/tu-usuario/knn-copper-trading/issues)

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

---

## 🙏 Agradecimientos

- Comunidad de scikit-learn
- Contribuidores de yfinance
- Comunidad de análisis técnico

---

**Desarrollado con ❤️ para el análisis cuantitativo del mercado de cobre**

*Última actualización: Diciembre 2024*