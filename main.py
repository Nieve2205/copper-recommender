"""
SISTEMA KNN PARA TRADING DE COBRE
===================================

Sistema completo de recomendación de trading basado en Machine Learning
que analiza momentos históricos similares para predecir movimientos del precio del cobre.

Autor: Sistema KNN Trading
Fecha: 2024
"""

import sys
import logging
from datetime import datetime
from colorama import init, Fore, Style
from tabulate import tabulate
import pandas as pd
import numpy as np

# Inicializar colorama para colores en consola
init(autoreset=True)

# Importar módulos del sistema
from data.data_collector import DataCollector
from data.data_processor import DataProcessor
from models.knn_model import KNNTradingModel
from utils.visualizer import Visualizer
from config.settings import (
    COPPER_SYMBOL, K_NEIGHBORS, TARGET_PRICE, 
    MIN_VOLUME_MILLIONS, CONFIDENCE_THRESHOLD
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/knn_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def print_header():
    """Imprime el encabezado del sistema"""
    print("\n" + "=" * 80)
    print(Fore.CYAN + Style.BRIGHT + "🔷 SISTEMA KNN PARA TRADING DE COBRE 🔷".center(80))
    print("=" * 80)
    print(Fore.YELLOW + f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(Fore.YELLOW + f"Símbolo: {COPPER_SYMBOL}")
    print(Fore.YELLOW + f"K-Vecinos: {K_NEIGHBORS}")
    print("=" * 80 + "\n")


def print_section(title: str):
    """Imprime un título de sección"""
    print("\n" + Fore.GREEN + Style.BRIGHT + f"{'='*60}")
    print(Fore.GREEN + Style.BRIGHT + f" {title}")
    print(Fore.GREEN + Style.BRIGHT + f"{'='*60}\n")


def display_market_info(collector: DataCollector):
    """Muestra información del mercado"""
    print_section("📊 INFORMACIÓN DEL MERCADO")
    
    # Obtener información
    market_info = collector.get_market_info()
    realtime_data = collector.get_realtime_data()
    
    if market_info and realtime_data:
        info_table = [
            ["Nombre", market_info.get('name', 'N/A')],
            ["Bolsa", market_info.get('exchange', 'N/A')],
            ["Moneda", market_info.get('currency', 'N/A')],
            ["Precio Actual", f"${realtime_data.get('price', 0):.2f}"],
            ["Cambio", f"${realtime_data.get('change', 0):.2f} ({realtime_data.get('change_pct', 0):+.2f}%)"],
            ["Máximo 52s", f"${market_info.get('52w_high', 0):.2f}"],
            ["Mínimo 52s", f"${market_info.get('52w_low', 0):.2f}"],
            ["Volumen", f"{market_info.get('volume', 0):,}"],
        ]
        
        print(tabulate(info_table, headers=["Métrica", "Valor"], tablefmt="fancy_grid"))
    else:
        print(Fore.RED + "⚠️ No se pudo obtener información del mercado")


def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Entrena y evalúa el modelo KNN"""
    print_section("🤖 ENTRENAMIENTO DEL MODELO KNN")
    
    # Crear modelo
    model = KNNTradingModel(n_neighbors=K_NEIGHBORS)
    
    # Entrenar
    print(Fore.YELLOW + "🎓 Entrenando modelo...")
    train_metrics = model.train(X_train, y_train)
    
    # Mostrar métricas de entrenamiento
    train_table = [
        ["Accuracy", f"{train_metrics.get('accuracy', 0):.4f}"],
        ["Precision", f"{train_metrics.get('precision', 0):.4f}"],
        ["Recall", f"{train_metrics.get('recall', 0):.4f}"],
        ["F1-Score", f"{train_metrics.get('f1_score', 0):.4f}"],
        ["Muestras", f"{train_metrics.get('samples', 0):,}"],
        ["Features", f"{train_metrics.get('features', 0)}"],
    ]
    print("\n" + Fore.CYAN + "Métricas de Entrenamiento:")
    print(tabulate(train_table, headers=["Métrica", "Valor"], tablefmt="fancy_grid"))
    
    # Evaluar
    print("\n" + Fore.YELLOW + "📊 Evaluando modelo en datos de prueba...")
    test_metrics = model.evaluate(X_test, y_test)
    
    test_table = [
        ["Accuracy", f"{test_metrics.get('accuracy', 0):.4f}"],
        ["Precision", f"{test_metrics.get('precision', 0):.4f}"],
        ["Recall", f"{test_metrics.get('recall', 0):.4f}"],
        ["F1-Score", f"{test_metrics.get('f1_score', 0):.4f}"],
        ["Muestras", f"{test_metrics.get('samples', 0):,}"],
    ]
    print("\n" + Fore.CYAN + "Métricas de Evaluación:")
    print(tabulate(test_table, headers=["Métrica", "Valor"], tablefmt="fancy_grid"))
    
    # Validación cruzada
    print("\n" + Fore.YELLOW + "🔄 Realizando validación cruzada...")
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)
    cv_results = model.cross_validate(X_full, y_full, cv=5)
    
    cv_table = [
        ["Accuracy Promedio", f"{cv_results.get('mean_score', 0):.4f}"],
        ["Desviación Estándar", f"{cv_results.get('std_score', 0):.4f}"],
        ["Mínimo", f"{cv_results.get('min_score', 0):.4f}"],
        ["Máximo", f"{cv_results.get('max_score', 0):.4f}"],
    ]
    print("\n" + Fore.CYAN + "Validación Cruzada:")
    print(tabulate(cv_table, headers=["Métrica", "Valor"], tablefmt="fancy_grid"))
    
    return model


def generate_trading_signal(model: KNNTradingModel, current_data, current_price: float):
    """Genera y muestra la señal de trading actual"""
    print_section("🎯 SEÑAL DE TRADING ACTUAL")
    
    # Obtener predicción
    prediction = model.predict_next(current_data)
    
    if not prediction:
        print(Fore.RED + "❌ No se pudo generar predicción")
        return
    
    # Determinar color de la señal
    signal = prediction['signal']
    if signal == 'COMPRA':
        signal_color = Fore.GREEN
        emoji = "📈"
    elif signal == 'VENTA':
        signal_color = Fore.RED
        emoji = "📉"
    else:
        signal_color = Fore.YELLOW
        emoji = "⏸️"
    
    # Mostrar señal principal
    print(signal_color + Style.BRIGHT + f"\n{emoji} SEÑAL: {signal} {emoji}\n")
    
    # Tabla de detalles
    signal_table = [
        ["Señal", f"{signal}"],
        ["Confianza", f"{prediction['confidence']:.2%}"],
        ["Recomendación", prediction['recommendation']],
        ["Precio Actual", f"${current_price:.2f}"],
        ["Timestamp", prediction['timestamp'].strftime('%Y-%m-%d %H:%M:%S')],
    ]
    print(tabulate(signal_table, headers=["Parámetro", "Valor"], tablefmt="fancy_grid"))
    
    # Tabla de probabilidades
    prob_table = [
        ["VENTA", f"{prediction['probabilities']['venta']:.2%}"],
        ["HOLD", f"{prediction['probabilities']['hold']:.2%}"],
        ["COMPRA", f"{prediction['probabilities']['compra']:.2%}"],
    ]
    print("\n" + Fore.CYAN + "Probabilidades por Clase:")
    print(tabulate(prob_table, headers=["Clase", "Probabilidad"], tablefmt="fancy_grid"))
    
    # Análisis de confianza
    confidence = prediction['confidence']
    if confidence >= 0.80:
        conf_msg = Fore.GREEN + "✅ Confianza MUY ALTA - Señal muy confiable"
    elif confidence >= 0.70:
        conf_msg = Fore.CYAN + "✓ Confianza ALTA - Señal confiable"
    elif confidence >= 0.60:
        conf_msg = Fore.YELLOW + "⚠ Confianza MEDIA - Proceder con cautela"
    else:
        conf_msg = Fore.RED + "⚠ Confianza BAJA - Esperar mejor oportunidad"
    
    print("\n" + conf_msg)
    
    # Verificar condiciones del sistema
    print("\n" + Fore.CYAN + "Verificación de Condiciones:")
    conditions = []
    
    # Condición 1: Precio objetivo
    price_condition = "✅" if current_price >= TARGET_PRICE else "❌"
    conditions.append([
        f"Precio >= ${TARGET_PRICE}",
        f"{price_condition} (${current_price:.2f})"
    ])
    
    # Condición 2: Confianza
    conf_condition = "✅" if confidence >= CONFIDENCE_THRESHOLD else "❌"
    conditions.append([
        f"Confianza >= {CONFIDENCE_THRESHOLD:.0%}",
        f"{conf_condition} ({confidence:.2%})"
    ])
    
    print(tabulate(conditions, headers=["Condición", "Estado"], tablefmt="fancy_grid"))
    
    # Recomendación final
    if signal == 'COMPRA' and confidence >= CONFIDENCE_THRESHOLD:
        print("\n" + Fore.GREEN + Style.BRIGHT + "✅ RECOMENDACIÓN: EJECUTAR COMPRA")
        print(Fore.GREEN + "El sistema recomienda COMPRAR basado en situaciones históricas similares.")
    elif signal == 'VENTA' and confidence >= CONFIDENCE_THRESHOLD:
        print("\n" + Fore.RED + Style.BRIGHT + "⚠️ RECOMENDACIÓN: CONSIDERAR VENTA")
        print(Fore.RED + "El sistema sugiere VENDER o proteger posiciones.")
    else:
        print("\n" + Fore.YELLOW + Style.BRIGHT + "⏸️ RECOMENDACIÓN: MANTENER / ESPERAR")
        print(Fore.YELLOW + "El sistema sugiere ESPERAR por una señal más clara.")


def main():
    """Función principal del sistema"""
    try:
        # Mostrar encabezado
        print_header()
        
        # ========== PASO 1: RECOLECCIÓN DE DATOS ==========
        print_section("📥 RECOLECCIÓN DE DATOS")
        
        collector = DataCollector(COPPER_SYMBOL)
        
        # Mostrar información del mercado
        display_market_info(collector)
        
        # Obtener datos históricos
        print("\n" + Fore.YELLOW + "📊 Descargando datos históricos...")
        df = collector.get_historical_data()
        
        if df.empty:
            print(Fore.RED + "❌ No se pudieron obtener datos. Abortando.")
            return
        
        # Verificar calidad de datos
        quality = collector.check_data_quality(df)
        quality_table = [
            ["Score de Calidad", f"{quality['quality_score']}%"],
            ["Total Registros", f"{quality['total_records']:,}"],
            ["Valores Nulos", f"{quality['null_values']}"],
            ["Duplicados", f"{quality['duplicates']}"],
        ]
        print("\n" + Fore.CYAN + "Calidad de Datos:")
        print(tabulate(quality_table, headers=["Métrica", "Valor"], tablefmt="fancy_grid"))
        
        # ========== PASO 2: PROCESAMIENTO DE DATOS ==========
        print_section("🔨 PROCESAMIENTO DE DATOS")
        
        processor = DataProcessor()
        
        print(Fore.YELLOW + "🧹 Limpiando datos...")
        df_clean = processor.clean_data(df)
        
        print(Fore.YELLOW + "🔨 Creando features...")
        df_features = processor.create_features(df_clean)
        
        print(Fore.YELLOW + "🎯 Creando variable objetivo...")
        df_target = processor.create_target(df_features)
        
        print(Fore.GREEN + f"✅ Procesamiento completado: {len(df_target)} registros listos")
        
        # ========== PASO 3: PREPARACIÓN DE DATOS ==========
        print_section("📦 PREPARACIÓN DE DATOS PARA ENTRENAMIENTO")
        
        print(Fore.YELLOW + "📦 Preparando datos de entrenamiento...")
        X, y = processor.prepare_training_data(df_target)
        
        print(Fore.YELLOW + "✂️ Dividiendo en conjuntos de entrenamiento y prueba...")
        X_train, X_test, y_train, y_test = processor.split_train_test(X, y, test_size=0.2)
        
        split_table = [
            ["Entrenamiento", f"{len(X_train):,} registros ({len(X_train)/len(X)*100:.1f}%)"],
            ["Prueba", f"{len(X_test):,} registros ({len(X_test)/len(X)*100:.1f}%)"],
            ["Features", f"{X_train.shape[1]}"],
        ]
        print(tabulate(split_table, headers=["Conjunto", "Detalles"], tablefmt="fancy_grid"))
        
        # ========== PASO 4: ENTRENAMIENTO Y EVALUACIÓN ==========
        model = train_and_evaluate_model(X_train, X_test, y_train, y_test)
        
        # ========== PASO 5: GENERAR SEÑAL ACTUAL ==========
        current_price = df_clean['close'].iloc[-1]
        current_data = X.iloc[[-1]]
        
        generate_trading_signal(model, current_data, current_price)
        
        # ========== PASO 6: VISUALIZACIONES ==========
        print_section("📊 GENERANDO VISUALIZACIONES")
        
        visualizer = Visualizer()
        
        print(Fore.YELLOW + "📈 Generando gráficos...")
        print(Fore.CYAN + "Gráfico 1: Historial de precios")
        visualizer.plot_price_history(df_features)
        
        print(Fore.CYAN + "Gráfico 2: Indicadores técnicos")
        visualizer.plot_technical_indicators(df_features)
        
        print(Fore.CYAN + "Gráfico 3: Predicciones del modelo")
        predictions = model.predict(X_test)
        df_plot = df_target.iloc[-len(X_test):].copy()
        visualizer.plot_predictions(df_plot, predictions)
        
        print(Fore.CYAN + "Gráfico 4: Matriz de confusión")
        visualizer.plot_confusion_matrix(y_test, predictions)
        
        # ========== PASO 7: GUARDAR MODELO ==========
        print_section("💾 GUARDANDO MODELO")
        
        print(Fore.YELLOW + "💾 Guardando modelo entrenado...")
        model.save_model()
        print(Fore.GREEN + "✅ Modelo guardado exitosamente")
        
        # ========== FINALIZACIÓN ==========
        print("\n" + "=" * 80)
        print(Fore.GREEN + Style.BRIGHT + "✅ SISTEMA KNN EJECUTADO EXITOSAMENTE".center(80))
        print("=" * 80 + "\n")
        
        print(Fore.CYAN + "📌 Próximos pasos recomendados:")
        print("   1. Revisar las visualizaciones generadas")
        print("   2. Analizar la señal de trading con su confianza")
        print("   3. Considerar factores externos (noticias, eventos)")
        print("   4. Ejecutar el sistema regularmente para actualizar señales")
        print("   5. Mantener un registro de las operaciones realizadas\n")
        
    except KeyboardInterrupt:
        print("\n" + Fore.RED + "⚠️ Ejecución interrumpida por el usuario")
    except Exception as e:
        logger.error(f"❌ Error en la ejecución del sistema: {e}", exc_info=True)
        print(Fore.RED + f"\n❌ Error: {e}")
        print(Fore.YELLOW + "Revisa el archivo de logs para más detalles")


if __name__ == "__main__":
    main()