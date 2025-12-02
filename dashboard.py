"""
DASHBOARD INTERACTIVO - SISTEMA KNN PARA TRADING DE COBRE
===========================================================

Dashboard web interactivo con Streamlit para visualizar resultados
y recomendaciones del sistema de trading.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import sys

# Importar módulos del sistema
from data.data_collector import DataCollector
from data.data_processor import DataProcessor
from models.knn_model import KNNTradingModel
from config.settings import (
    COPPER_SYMBOL, K_NEIGHBORS, TARGET_PRICE, 
    MIN_VOLUME_MILLIONS, CONFIDENCE_THRESHOLD
)

# Configuración de la página
st.set_page_config(
    page_title="KNN Trading - Cobre",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .buy-signal {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .sell-signal {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .hold-signal {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data():
    """Carga y procesa los datos"""
    collector = DataCollector(COPPER_SYMBOL)
    df = collector.get_historical_data()
    
    if df.empty:
        st.error("❌ No se pudieron obtener datos del mercado")
        st.stop()
    
    processor = DataProcessor()
    df_clean = processor.clean_data(df)
    df_features = processor.create_features(df_clean)
    df_target = processor.create_target(df_features)
    
    # Solo retornar DataFrames (serializables)
    return df_clean, df_features, df_target


@st.cache_data(ttl=3600)
def get_market_info():
    """Obtiene información del mercado"""
    collector = DataCollector(COPPER_SYMBOL)
    return collector.get_market_info()


def train_model(X_train, y_train):
    """Entrena el modelo KNN"""
    model = KNNTradingModel(n_neighbors=K_NEIGHBORS)
    model.train(X_train, y_train)
    return model


def create_price_chart(df):
    """Crea gráfico de precio con indicadores"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Precio y Medias Móviles', 'RSI', 'MACD'),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Precio y medias móviles
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['close'], name='Precio', 
                  line=dict(color='#1f77b4', width=2)),
        row=1, col=1
    )
    
    if 'sma_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['sma_20'], name='SMA 20',
                      line=dict(color='orange', width=1, dash='dash')),
            row=1, col=1
        )
    
    if 'sma_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['sma_50'], name='SMA 50',
                      line=dict(color='green', width=1, dash='dash')),
            row=1, col=1
        )
    
    # Bollinger Bands
    if all(col in df.columns for col in ['bb_upper', 'bb_lower']):
        fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['bb_upper'], name='BB Superior',
                      line=dict(color='gray', width=1), opacity=0.3),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['bb_lower'], name='BB Inferior',
                      line=dict(color='gray', width=1), fill='tonexty', opacity=0.3),
            row=1, col=1
        )
    
    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['rsi'], name='RSI',
                      line=dict(color='purple', width=2)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    
    # MACD
    if all(col in df.columns for col in ['macd', 'macd_signal']):
        fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['macd'], name='MACD',
                      line=dict(color='blue', width=2)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['macd_signal'], name='Señal',
                      line=dict(color='red', width=2)),
            row=3, col=1
        )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text="Fecha", row=3, col=1)
    fig.update_yaxes(title_text="Precio (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig


def create_signal_gauge(confidence, signal):
    """Crea un gauge de confianza"""
    color = "green" if signal == "COMPRA" else "red" if signal == "VENTA" else "orange"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Nivel de Confianza", 'font': {'size': 24}},
        delta={'reference': CONFIDENCE_THRESHOLD * 100, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 60], 'color': '#ffcccc'},
                {'range': [60, 70], 'color': '#ffffcc'},
                {'range': [70, 80], 'color': '#ccffcc'},
                {'range': [80, 100], 'color': '#99ff99'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': CONFIDENCE_THRESHOLD * 100
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_performance_chart(train_metrics, test_metrics):
    """Crea gráfico de rendimiento del modelo"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    train_values = [
        train_metrics.get('accuracy', 0),
        train_metrics.get('precision', 0),
        train_metrics.get('recall', 0),
        train_metrics.get('f1_score', 0)
    ]
    test_values = [
        test_metrics.get('accuracy', 0),
        test_metrics.get('precision', 0),
        test_metrics.get('recall', 0),
        test_metrics.get('f1_score', 0)
    ]
    
    fig = go.Figure(data=[
        go.Bar(name='Entrenamiento', x=metrics, y=train_values, marker_color='#1f77b4'),
        go.Bar(name='Prueba', x=metrics, y=test_values, marker_color='#ff7f0e')
    ])
    
    fig.update_layout(
        title='Métricas de Rendimiento del Modelo',
        yaxis_title='Score',
        barmode='group',
        height=400,
        template='plotly_white'
    )
    
    return fig


def main():
    """Función principal del dashboard"""
    
    # Header
    st.markdown('<p class="main-header">🔷 Sistema KNN para Trading de Cobre</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/copper.png", width=100)
        st.title("⚙️ Configuración")
        
        st.info(f"""
        **Símbolo:** {COPPER_SYMBOL}  
        **K-Vecinos:** {K_NEIGHBORS}  
        **Precio Objetivo:** ${TARGET_PRICE:,}  
        **Confianza Mínima:** {CONFIDENCE_THRESHOLD:.0%}
        """)
        
        if st.button("🔄 Actualizar Datos", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        
        st.divider()
        
        st.subheader("📚 Acerca de")
        st.markdown("""
        Este sistema utiliza **K-Nearest Neighbors (KNN)** para analizar 
        situaciones históricas similares del mercado y predecir movimientos futuros.
        
        **¿Cómo funciona?**
        1. Busca los 50 momentos más similares en el historial
        2. Analiza qué pasó después de esas situaciones
        3. Genera una recomendación con nivel de confianza
        """)
    
    # Cargar datos
    with st.spinner('📊 Descargando datos del mercado...'):
        df_clean, df_features, df_target = load_data()
    
    # Obtener información del mercado
    market_info = get_market_info()
    
    # Preparar datos para modelo
    processor = DataProcessor()
    X, y = processor.prepare_training_data(df_target)
    X_train, X_test, y_train, y_test = processor.split_train_test(X, y, test_size=0.2)
    
    # Entrenar modelo
    with st.spinner('🤖 Entrenando modelo KNN...'):
        model = train_model(X_train, y_train)
        train_metrics = model.training_metrics
        test_metrics = model.evaluate(X_test, y_test)
    
    # Obtener predicción actual
    current_data = X.iloc[[-1]]
    prediction = model.predict_next(current_data)
    current_price = df_clean['close'].iloc[-1]
    
    # ==================== SECCIÓN 1: SEÑAL PRINCIPAL ====================
    st.header("🎯 Recomendación de Trading")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        signal = prediction['signal']
        confidence = prediction['confidence']
        
        if signal == "COMPRA":
            st.markdown(f'<div class="buy-signal">📈 SEÑAL: COMPRA<br>Confianza: {confidence:.1%}</div>', 
                       unsafe_allow_html=True)
            st.success("✅ El modelo recomienda COMPRAR basado en patrones históricos similares")
        elif signal == "VENTA":
            st.markdown(f'<div class="sell-signal">📉 SEÑAL: VENTA<br>Confianza: {confidence:.1%}</div>', 
                       unsafe_allow_html=True)
            st.error("⚠️ El modelo sugiere VENDER o proteger posiciones")
        else:
            st.markdown(f'<div class="hold-signal">⏸️ SEÑAL: MANTENER<br>Confianza: {confidence:.1%}</div>', 
                       unsafe_allow_html=True)
            st.warning("⏸️ El modelo sugiere ESPERAR por una señal más clara")
    
    with col2:
        st.metric("Precio Actual", f"${current_price:.2f}", 
                 f"{df_clean['close'].pct_change().iloc[-1]*100:+.2f}%")
    
    with col3:
        recommendation = prediction['recommendation']
        if recommendation == "EJECUTAR":
            st.success(f"**{recommendation}**")
        else:
            st.warning(f"**{recommendation}**")
    
    # Gauge de confianza
    st.plotly_chart(create_signal_gauge(confidence, signal), use_container_width=True)
    
    # ==================== SECCIÓN 2: ANÁLISIS DETALLADO ====================
    st.header("📊 Análisis Detallado")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Gráficos", "🎲 Probabilidades", "📋 Condiciones", "🤖 Modelo"])
    
    with tab1:
        st.subheader("Análisis Técnico del Precio")
        fig_price = create_price_chart(df_features.tail(100))
        st.plotly_chart(fig_price, use_container_width=True)
    
    with tab2:
        st.subheader("Distribución de Probabilidades")
        
        col1, col2, col3 = st.columns(3)
        
        probs = prediction['probabilities']
        
        with col1:
            st.metric("📉 Probabilidad VENTA", f"{probs['venta']:.1%}")
            st.progress(probs['venta'])
        
        with col2:
            st.metric("⏸️ Probabilidad HOLD", f"{probs['hold']:.1%}")
            st.progress(probs['hold'])
        
        with col3:
            st.metric("📈 Probabilidad COMPRA", f"{probs['compra']:.1%}")
            st.progress(probs['compra'])
        
        # Gráfico de barras
        fig_probs = go.Figure(data=[
            go.Bar(
                x=['VENTA', 'HOLD', 'COMPRA'],
                y=[probs['venta'], probs['hold'], probs['compra']],
                marker_color=['#dc3545', '#ffc107', '#28a745'],
                text=[f"{probs['venta']:.1%}", f"{probs['hold']:.1%}", f"{probs['compra']:.1%}"],
                textposition='auto',
            )
        ])
        fig_probs.update_layout(
            title='Probabilidades por Clase',
            yaxis_title='Probabilidad',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_probs, use_container_width=True)
    
    with tab3:
        st.subheader("Verificación de Condiciones de Trading")
        
        # Tabla de condiciones
        conditions_data = {
            'Condición': [
                f'Precio >= ${TARGET_PRICE:,}',
                f'Confianza >= {CONFIDENCE_THRESHOLD:.0%}',
                'Tendencia Alcista (SMA 20 > SMA 50)',
                'RSI en Rango Normal (30-70)',
                'Volumen Adecuado'
            ],
            'Valor Actual': [
                f'${current_price:.2f}',
                f'{confidence:.1%}',
                '✅' if df_features['sma_20'].iloc[-1] > df_features['sma_50'].iloc[-1] else '❌',
                '✅' if 30 <= df_features['rsi'].iloc[-1] <= 70 else '❌',
                '✅' if df_clean['volume'].iloc[-1] > df_clean['volume'].mean() else '❌'
            ],
            'Estado': [
                '✅' if current_price >= TARGET_PRICE else '❌',
                '✅' if confidence >= CONFIDENCE_THRESHOLD else '❌',
                '✅' if df_features['sma_20'].iloc[-1] > df_features['sma_50'].iloc[-1] else '❌',
                '✅' if 30 <= df_features['rsi'].iloc[-1] <= 70 else '❌',
                '✅' if df_clean['volume'].iloc[-1] > df_clean['volume'].mean() else '❌'
            ]
        }
        
        df_conditions = pd.DataFrame(conditions_data)
        st.dataframe(df_conditions, use_container_width=True, hide_index=True)
        
        # Indicadores técnicos actuales
        st.subheader("Indicadores Técnicos Actuales")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RSI", f"{df_features['rsi'].iloc[-1]:.1f}")
        
        with col2:
            st.metric("MACD", f"{df_features['macd'].iloc[-1]:.4f}")
        
        with col3:
            st.metric("ATR", f"{df_features['atr'].iloc[-1]:.4f}")
        
        with col4:
            st.metric("BB Width", f"{df_features['bb_width'].iloc[-1]:.4f}")
    
    with tab4:
        st.subheader("Rendimiento del Modelo KNN")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Métricas de Entrenamiento**")
            st.metric("Accuracy", f"{train_metrics.get('accuracy', 0):.2%}")
            st.metric("Precision", f"{train_metrics.get('precision', 0):.2%}")
            st.metric("Recall", f"{train_metrics.get('recall', 0):.2%}")
            st.metric("F1-Score", f"{train_metrics.get('f1_score', 0):.2%}")
        
        with col2:
            st.markdown("**Métricas de Prueba**")
            st.metric("Accuracy", f"{test_metrics.get('accuracy', 0):.2%}")
            st.metric("Precision", f"{test_metrics.get('precision', 0):.2%}")
            st.metric("Recall", f"{test_metrics.get('recall', 0):.2%}")
            st.metric("F1-Score", f"{test_metrics.get('f1_score', 0):.2%}")
        
        # Gráfico de comparación
        fig_performance = create_performance_chart(train_metrics, test_metrics)
        st.plotly_chart(fig_performance, use_container_width=True)
        
        # Información del modelo
        st.info(f"""
        **Configuración del Modelo:**
        - Algoritmo: K-Nearest Neighbors (KNN)
        - Número de vecinos: {K_NEIGHBORS}
        - Peso de vecinos: distance
        - Muestras de entrenamiento: {train_metrics.get('samples', 0):,}
        - Features utilizadas: {train_metrics.get('features', 0)}
        """)
    
    # ==================== SECCIÓN 3: INFORMACIÓN DEL MERCADO ====================
    st.header("📈 Información del Mercado")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Máximo 52 Semanas", f"${market_info.get('52w_high', 0):.2f}")
    
    with col2:
        st.metric("Mínimo 52 Semanas", f"${market_info.get('52w_low', 0):.2f}")
    
    with col3:
        st.metric("Volumen Actual", f"{market_info.get('volume', 0):,}")
    
    with col4:
        st.metric("Volumen Promedio", f"{market_info.get('avg_volume', 0):,}")
    
    # ==================== FOOTER ====================
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption(f"⏰ Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col2:
        st.caption("📊 Datos proporcionados por Yahoo Finance")
    
    with col3:
        st.caption("⚠️ Solo para fines educativos. No es asesoramiento financiero.")


if __name__ == "__main__":
    main()