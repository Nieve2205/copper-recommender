# 📚 Documentación del Proyecto - Sistema KNN Trading 🆕 v2.0

Sistema profesional de trading de cobre con **Machine Learning (KNN)**, **Business Intelligence Avanzado** (Monte Carlo, VaR, Backtesting), **Integración Multi-Fuente** (Yahoo Finance, World Bank, FRED, LME) y **Dashboard Interactivo** (Streamlit).

Esta carpeta contiene documentación técnica completa, diagramas de flujo y arquitectura del sistema.

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

### 📄 Docs/Technical_Documentation.md 🆕 Actualizada v2.0

Documentación técnica exhaustiva (20+ páginas) que incluye:

#### Core del Sistema
- **Resumen Ejecutivo**: Visión general con características únicas
- **Arquitectura Modular**: Estructura completa con todos los módulos
- **Metodología KNN**: Funcionamiento detallado del algoritmo

#### Fuentes de Datos 🆕
- **Multi-Source Integration**: 7 fuentes de datos
  - Yahoo Finance (precios técnicos)
  - World Bank (producción global)
  - FRED (indicadores macroeconómicos)
  - LME (precios institucionales)
  - China PMI (demanda industrial)
  - EV Market (demanda futura)
  - Sentiment Analysis (análisis de noticias)

#### Machine Learning & BI
- **Features del Modelo**: 24 indicadores técnicos detallados
- **Técnicas de BI Avanzado** 🆕:
  - Simulación Monte Carlo (1000+ escenarios)
  - Value at Risk (VaR) y CVaR
  - Backtesting profesional (Sharpe, Drawdown, Win Rate)
  - Análisis de Escenarios (What-If)
  - Optimización de Cartera (Kelly Criterion)
  - Análisis de Correlaciones

#### Implementación
- **Pipeline de Ejecución**: Flujo completo CLI y Dashboard Web
- **Indicadores Técnicos**: RSI, MACD, Bollinger Bands, ATR, SMA, EMA
- **Sistema de Señales**: Lógica de decisión con confianza ≥70%
- **Dashboard Interactivo** 🆕: Streamlit con Plotly, tabs, gauge, visualizaciones

#### Técnico
- **Stack Tecnológico**: Completo con versiones (scikit-learn, pandas, numpy, scipy, plotly, streamlit)
- **Evaluación**: Métricas ML + métricas de trading
- **Casos de Uso**: 7 escenarios (análisis diario, dashboard, backtesting, etc.)
- **Limitaciones**: Técnicas, de datos, computacionales, financieras (exhaustivas)
- **Mejores Prácticas**: Recomendaciones detalladas
- **Referencias**: Papers, documentación, conceptos clave

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

### 🏗️ Flows/Architecture_Overview.canvas 🆕 Actualizada

**Canvas de Obsidian** con arquitectura completa del sistema v2.0:

**Flujo Principal (horizontal):**
- **INPUT (Multi-Source)** → PROCESSING → ML MODEL → EVALUATION + ADVANCED ANALYTICS → SIGNAL → OUTPUT

**Nuevos Componentes** 🆕:
- **Multi-Source Data**: 7 fuentes integradas (Yahoo, WB, FRED, LME, etc.)
- **Advanced Analytics Layer**: Monte Carlo, VaR, Backtesting, Escenarios
- **Dashboard Web**: Streamlit interactivo con Plotly
- **Advanced Sources Module**: AdvancedDataSources class
- **Analytics Module**: AdvancedAnalytics class

**Capas de Arquitectura (vertical):**
1. **Data Layer**: Recolección multi-fuente y procesamiento avanzado
2. **Processing Layer**: Features técnicas + fundamentales
3. **Model Layer**: KNN training, evaluation, prediction
4. **Analytics Layer** 🆕: BI avanzado (MC, VaR, Backtest)
5. **Visualization Layer**: Matplotlib + Plotly
6. **Presentation Layer**: CLI + Dashboard Web
7. **Persistence Layer**: Models, logs, cache

**Componentes Técnicos:**
- **Tech Stack**: Actualizado con todas las librerías y versiones
- **Decision System**: Algoritmo completo con análisis de riesgo
- **Workflow**: Pipeline con feedback loops
- **Architecture**: Patrón modular actualizado

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

## 🆕 Novedades en Versión 2.0

### Documentación Actualizada:
✅ **Technical_Documentation.md**: Expandida de 10 a 20+ páginas  
✅ Nueva sección: "Fuentes de Datos Múltiples" (7 fuentes detalladas)  
✅ Nueva sección: "Técnicas de Business Intelligence Avanzado" (Monte Carlo, VaR, Backtesting, Escenarios)  
✅ Expandida: "Pipeline de Ejecución" con análisis avanzado completo  
✅ Expandida: "Dashboard Web" con todas las características interactivas  
✅ Actualizada: "Stack Tecnológico" con versiones y descripciones  
✅ Expandidas: "Limitaciones" con análisis profundo (técnicas, datos, computacionales, financieras)  
✅ Nuevos: "Casos de Uso" detallados (7 escenarios prácticos)  
✅ Nueva: Tabla resumen de capacidades del sistema  

### Diagramas Canvas Actualizados:
✅ **System_Flow.canvas**: Mantiene flujo modular detallado  
✅ **Architecture_Overview.canvas**: Actualizada con:
   - Multi-source data layer
   - Advanced analytics layer
   - Dashboard web interactivo
   - Módulos avanzados (AdvancedDataSources, AdvancedAnalytics)
   - Tech stack completo actualizado

### Lo Que Hace Única Esta Documentación:
🎯 **Completitud**: Cubre desde basics hasta técnicas avanzadas de BI  
🎯 **Actualizada**: Refleja 100% el código actual (Dic 2024)  
🎯 **Visual**: 2 canvas interactivos para entender rápido  
🎯 **Práctica**: Casos de uso, mejores prácticas, troubleshooting  
🎯 **Académica**: Referencias, limitaciones, supuestos del modelo  
🎯 **Profesional**: Formato técnico apto para papers o presentaciones  

---

## 🎓 Para Profesores/Evaluadores

Este proyecto destaca por:

1. **Integración Multi-Fuente**: No solo Yahoo Finance, sino 7 fuentes distintas
2. **BI Avanzado**: Monte Carlo, VaR/CVaR, Backtesting con métricas profesionales
3. **ML Riguroso**: KNN con validación cruzada, métricas completas, análisis de vecinos
4. **Visualización Profesional**: Dashboard interactivo con Plotly + Streamlit
5. **Documentación Excepcional**: Técnica completa + diagramas interactivos
6. **Código Limpio**: Modular, comentado, con logging y manejo de errores
7. **Aplicación Real**: Sistema funcional para trading con análisis de riesgo

**Complejidad**: Alta (combina ML, finanzas, BI, desarrollo web, integración APIs)  
**Completitud**: 100% (funcional end-to-end con todos los módulos)  
**Innovación**: Análisis fundamental + técnico + BI en un solo sistema  

---

**Última actualización**: Diciembre 3, 2025  
**Versión**: 2.0 (Documentación completa actualizada)  
**Proyecto**: Copper Recommender - KNN Trading System  
**Mantenedor**: Business Intelligence Team
