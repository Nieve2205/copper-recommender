# 📚 Documentación del Proyecto - Sistema KNN Trading

Esta carpeta contiene la documentación técnica y diagramas de flujo del sistema de trading de cobre basado en K-Nearest Neighbors.

## 📂 Estructura

```
Obsidian_Cooper/
├── Docs/
│   └── Technical_Documentation.md  # Documentación técnica completa
└── Flows/
    ├── System_Flow.canvas          # Flujo detallado del sistema (paso a paso)
    └── Architecture_Overview.canvas # Vista arquitectónica general
```

## 📖 Contenido

### 📄 Docs/Technical_Documentation.md

Documentación técnica exhaustiva que incluye:

- **Resumen Ejecutivo**: Visión general del sistema
- **Arquitectura**: Estructura modular y componentes
- **Metodología KNN**: Funcionamiento del algoritmo
- **Features**: 24 indicadores técnicos utilizados
- **Pipeline de Ejecución**: Flujo CLI y Dashboard
- **Indicadores Técnicos**: RSI, MACD, Bollinger Bands, ATR
- **Sistema de Señales**: Lógica de decisión y confianza
- **Dependencias**: Stack tecnológico completo
- **Evaluación**: Métricas y validación del modelo
- **Configuración**: Parámetros avanzados
- **Limitaciones**: Consideraciones técnicas y de trading
- **Referencias**: Documentación adicional

### 🎨 Flows/System_Flow.canvas

**Canvas de Obsidian** con el flujo detallado del sistema (de izquierda a derecha):

1. **Inicio** → Usuario ejecuta main.py o dashboard.py
2. **Configuración** → Carga de parámetros desde settings.py
3. **Data Collector** → Descarga de datos de Yahoo Finance
4. **Data Processor** → Limpieza y creación de features
5. **Technical Indicators** → Cálculo de RSI, MACD, BB, ATR
6. **Features** → 24 variables predictivas normalizadas
7. **KNN Model** → Algoritmo de Machine Learning
8. **Entrenamiento** → Fit con datos históricos + validación cruzada
9. **Evaluación** → Métricas de rendimiento (Accuracy, Precision, Recall, F1)
10. **Predicción Actual** → Genera señal para momento presente
11. **Lógica de Señales** → Decisión COMPRA/VENTA/HOLD con confianza
12. **Visualización** → Gráficos y dashboard interactivo
13. **Persistencia** → Guardado de modelo y logs
14. **Output Final** → Presentación de resultados (CLI/Web)
15. **Acción del Usuario** → Decisión informada de trading

**Colores:**
- 🟢 Verde: Input/Output
- 🔵 Azul: Procesamiento de datos
- 🟣 Morado: Machine Learning
- 🟡 Amarillo: Configuración
- 🟠 Naranja: Visualización
- 🔴 Rojo: Evaluación

### 🏗️ Flows/Architecture_Overview.canvas

**Canvas de Obsidian** con vista arquitectónica del sistema:

**Flujo Principal (horizontal):**
- INPUT → PROCESSING → ML MODEL → EVALUATION → SIGNAL → OUTPUT

**Capas de Arquitectura (vertical):**
1. **Data Layer**: Recolección y procesamiento
2. **Model Layer**: KNN y predicciones
3. **Utils Layer**: Indicadores y visualización
4. **Config Layer**: Configuración centralizada
5. **App Layer**: CLI y Dashboard

**Componentes Adicionales:**
- **Tech Stack**: Tecnologías utilizadas
- **Architecture**: Patrón modular y principios
- **Workflow**: Pipeline secuencial
- **Decision System**: Algoritmo de decisión detallado

## 🔧 Cómo Usar

### Abrir en Obsidian

1. Abre Obsidian
2. Abre esta carpeta como vault o agrégala a un vault existente
3. Los archivos `.canvas` se abrirán como diagramas interactivos
4. La documentación `.md` se renderizará con formato

### Sin Obsidian

- **Technical_Documentation.md**: Se puede leer en cualquier visualizador Markdown (VS Code, GitHub, etc.)
- **Canvas files**: Son JSON, pero se visualizan mejor en Obsidian

## 📊 Diagramas Interactivos

Los archivos `.canvas` son **interactivos en Obsidian**:
- ✅ Zoom in/out
- ✅ Navegación arrastrando
- ✅ Conexiones visuales entre componentes
- ✅ Colores para categorizar elementos
- ✅ Layout modular de izquierda a derecha

## 🎯 Propósito

Esta documentación sirve para:

- **Onboarding**: Nuevos desarrolladores entiendan el sistema
- **Mantenimiento**: Referencia técnica para cambios
- **Educación**: Aprender sobre trading algorítmico y KNN
- **Auditoría**: Validar decisiones técnicas y arquitectónicas
- **Presentaciones**: Material para demos y explicaciones

## 📝 Notas

- La documentación técnica está sincronizada con el código fuente
- Los diagramas reflejan la arquitectura actual del sistema
- Para actualizaciones, modificar los archivos correspondientes
- Usar Obsidian para mejor experiencia visual

## 🔗 Enlaces Útiles

- [Obsidian](https://obsidian.md/) - Aplicación para visualizar los canvas
- [scikit-learn KNN](https://scikit-learn.org/stable/modules/neighbors.html)
- [Technical Analysis Library](https://technical-analysis-library-in-python.readthedocs.io/)
- [yfinance](https://pypi.org/project/yfinance/)

---

**Última actualización**: Diciembre 2024  
**Versión**: 1.0
