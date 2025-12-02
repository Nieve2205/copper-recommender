"""
Visualizador de datos y resultados del sistema KNN
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List
import warnings

warnings.filterwarnings('ignore')

from config.settings import (
    COLOR_BUY, COLOR_SELL, COLOR_HOLD,
    FIGURE_SIZE, PLOT_STYLE
)


class Visualizer:
    """
    Clase para visualizar datos y resultados
    """
    
    def __init__(self):
        """
        Inicializa el visualizador
        """
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        self.fig_count = 0
    
    def plot_price_history(self, df: pd.DataFrame, title: str = "Historial de Precios del Cobre"):
        """
        Grafica el historial de precios
        
        Args:
            df: DataFrame con datos
            title: Título del gráfico
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGURE_SIZE, height_ratios=[3, 1])
        
        # Gráfico de precios
        ax1.plot(df['datetime'], df['close'], label='Precio de Cierre', color='blue', linewidth=2)
        
        # Agregar medias móviles si existen
        if 'sma_20' in df.columns:
            ax1.plot(df['datetime'], df['sma_20'], label='SMA 20', alpha=0.7, linestyle='--')
        if 'sma_50' in df.columns:
            ax1.plot(df['datetime'], df['sma_50'], label='SMA 50', alpha=0.7, linestyle='--')
        if 'sma_200' in df.columns:
            ax1.plot(df['datetime'], df['sma_200'], label='SMA 200', alpha=0.7, linestyle='--')
        
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('Precio (USD)', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de volumen
        ax2.bar(df['datetime'], df['volume'], color='gray', alpha=0.5, label='Volumen')
        ax2.set_xlabel('Fecha', fontsize=12)
        ax2.set_ylabel('Volumen', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_technical_indicators(self, df: pd.DataFrame):
        """
        Grafica indicadores técnicos
        
        Args:
            df: DataFrame con datos e indicadores
        """
        fig, axes = plt.subplots(4, 1, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] + 4))
        
        # 1. Precio con Bollinger Bands
        if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            axes[0].plot(df['datetime'], df['close'], label='Precio', color='blue', linewidth=2)
            axes[0].plot(df['datetime'], df['bb_upper'], label='BB Superior', color='red', linestyle='--', alpha=0.7)
            axes[0].plot(df['datetime'], df['bb_middle'], label='BB Media', color='gray', linestyle='--', alpha=0.7)
            axes[0].plot(df['datetime'], df['bb_lower'], label='BB Inferior', color='green', linestyle='--', alpha=0.7)
            axes[0].fill_between(df['datetime'], df['bb_upper'], df['bb_lower'], alpha=0.1, color='gray')
            axes[0].set_title('Precio con Bollinger Bands', fontweight='bold')
            axes[0].legend(loc='best')
            axes[0].grid(True, alpha=0.3)
        
        # 2. RSI
        if 'rsi' in df.columns:
            axes[1].plot(df['datetime'], df['rsi'], label='RSI', color='purple', linewidth=2)
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Sobrecompra (70)')
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Sobreventa (30)')
            axes[1].fill_between(df['datetime'], 30, 70, alpha=0.1, color='gray')
            axes[1].set_title('RSI (Relative Strength Index)', fontweight='bold')
            axes[1].set_ylim(0, 100)
            axes[1].legend(loc='best')
            axes[1].grid(True, alpha=0.3)
        
        # 3. MACD
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            axes[2].plot(df['datetime'], df['macd'], label='MACD', color='blue', linewidth=2)
            axes[2].plot(df['datetime'], df['macd_signal'], label='Señal', color='red', linewidth=2)
            
            if 'macd_diff' in df.columns:
                colors = ['green' if val >= 0 else 'red' for val in df['macd_diff']]
                axes[2].bar(df['datetime'], df['macd_diff'], color=colors, alpha=0.3, label='Histograma')
            
            axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[2].set_title('MACD', fontweight='bold')
            axes[2].legend(loc='best')
            axes[2].grid(True, alpha=0.3)
        
        # 4. Volumen
        axes[3].bar(df['datetime'], df['volume'], color='gray', alpha=0.5, label='Volumen')
        if 'volume_sma_20' in df.columns:
            axes[3].plot(df['datetime'], df['volume_sma_20'], color='red', linewidth=2, label='Vol SMA 20')
        axes[3].set_title('Volumen', fontweight='bold')
        axes[3].set_xlabel('Fecha', fontsize=12)
        axes[3].legend(loc='best')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, df: pd.DataFrame, predictions: np.ndarray, title: str = "Predicciones del Modelo KNN"):
        """
        Grafica predicciones del modelo
        
        Args:
            df: DataFrame con datos
            predictions: Array con predicciones
            title: Título del gráfico
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        # Graficar precio
        ax.plot(df['datetime'], df['close'], label='Precio Real', color='blue', linewidth=2, alpha=0.7)
        
        # Marcar predicciones
        buy_signals = df[predictions == 1]
        sell_signals = df[predictions == -1]
        
        ax.scatter(buy_signals['datetime'], buy_signals['close'], 
                  color=COLOR_BUY, marker='^', s=100, label='Señal COMPRA', zorder=5)
        ax.scatter(sell_signals['datetime'], sell_signals['close'], 
                  color=COLOR_SELL, marker='v', s=100, label='Señal VENTA', zorder=5)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Precio (USD)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Grafica matriz de confusión
        
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['VENTA', 'HOLD', 'COMPRA'],
                   yticklabels=['VENTA', 'HOLD', 'COMPRA'],
                   cbar_kws={'label': 'Frecuencia'})
        
        ax.set_title('Matriz de Confusión', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicción', fontsize=12)
        ax.set_ylabel('Real', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], importances: np.ndarray, top_n: int = 15):
        """
        Grafica importancia de features
        
        Args:
            feature_names: Nombres de las features
            importances: Importancia de cada feature
            top_n: Número de features más importantes a mostrar
        """
        # Crear DataFrame y ordenar
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        bars = ax.barh(df_importance['feature'], df_importance['importance'], color='steelblue')
        
        # Colorear las barras según importancia
        colors = plt.cm.RdYlGn(df_importance['importance'] / df_importance['importance'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_title(f'Top {top_n} Features Más Importantes', fontsize=16, fontweight='bold')
        ax.set_xlabel('Importancia', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_performance(self, history: dict):
        """
        Grafica métricas de rendimiento del modelo
        
        Args:
            history: Diccionario con métricas históricas
        """
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
        
        # Accuracy
        if 'accuracy' in history:
            axes[0, 0].plot(history['accuracy'], marker='o', linewidth=2)
            axes[0, 0].set_title('Accuracy', fontweight='bold')
            axes[0, 0].set_xlabel('Época')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Precision
        if 'precision' in history:
            axes[0, 1].plot(history['precision'], marker='o', linewidth=2, color='green')
            axes[0, 1].set_title('Precision', fontweight='bold')
            axes[0, 1].set_xlabel('Época')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in history:
            axes[1, 0].plot(history['recall'], marker='o', linewidth=2, color='orange')
            axes[1, 0].set_title('Recall', fontweight='bold')
            axes[1, 0].set_xlabel('Época')
            axes[1, 0].set_ylabel('Recall')
            axes[1, 0].grid(True, alpha=0.3)
        
        # F1 Score
        if 'f1_score' in history:
            axes[1, 1].plot(history['f1_score'], marker='o', linewidth=2, color='red')
            axes[1, 1].set_title('F1 Score', fontweight='bold')
            axes[1, 1].set_xlabel('Época')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Rendimiento del Modelo KNN', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    def plot_returns(self, df: pd.DataFrame):
        """
        Grafica retornos acumulados
        
        Args:
            df: DataFrame con datos de retornos
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        if 'strategy_returns' in df.columns and 'market_returns' in df.columns:
            cumulative_strategy = (1 + df['strategy_returns']).cumprod()
            cumulative_market = (1 + df['market_returns']).cumprod()
            
            ax.plot(df['datetime'], cumulative_strategy, label='Estrategia KNN', 
                   color='green', linewidth=2)
            ax.plot(df['datetime'], cumulative_market, label='Buy & Hold', 
                   color='blue', linewidth=2, alpha=0.7)
            
            ax.set_title('Retornos Acumulados: Estrategia vs Mercado', fontsize=16, fontweight='bold')
            ax.set_xlabel('Fecha', fontsize=12)
            ax.set_ylabel('Retorno Acumulado', fontsize=12)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()


# Función de prueba
if __name__ == "__main__":
    print("=" * 60)
    print("PROBANDO VISUALIZER")
    print("=" * 60)
    
    # Crear datos de ejemplo
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    df_example = pd.DataFrame({
        'datetime': dates,
        'close': 100 + np.cumsum(np.random.randn(100) * 2),
        'volume': np.random.randint(1000000, 5000000, 100),
        'sma_20': 100 + np.cumsum(np.random.randn(100) * 1.5),
        'sma_50': 100 + np.cumsum(np.random.randn(100) * 1),
        'rsi': np.random.uniform(30, 70, 100),
        'macd': np.random.randn(100) * 2,
        'macd_signal': np.random.randn(100) * 1.5,
        'bb_upper': 105 + np.cumsum(np.random.randn(100) * 2),
        'bb_middle': 100 + np.cumsum(np.random.randn(100) * 2),
        'bb_lower': 95 + np.cumsum(np.random.randn(100) * 2),
    })
    
    # Crear instancia
    viz = Visualizer()
    
    print("\n📊 Generando gráficos de ejemplo...")
    print("(Cierra las ventanas para continuar)")
    
    # Probar visualizaciones
    viz.plot_price_history(df_example)
    
    print("\n✅ Visualizer funcionando correctamente")