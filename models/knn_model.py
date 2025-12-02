"""
Modelo KNN para predicción de movimientos del cobre
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
import pickle
import os
from datetime import datetime

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score

from config.settings import (
    K_NEIGHBORS, WEIGHTS, ALGORITHM, METRIC,
    CONFIDENCE_THRESHOLD, MODELS_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KNNTradingModel:
    """
    Modelo KNN para trading de cobre
    
    El modelo busca los K vecinos más cercanos (situaciones de mercado similares)
    y predice el movimiento futuro basándose en lo que pasó después de esas
    situaciones históricas similares.
    """
    
    def __init__(self, 
                 n_neighbors: int = K_NEIGHBORS,
                 weights: str = WEIGHTS,
                 algorithm: str = ALGORITHM,
                 metric: str = METRIC):
        """
        Inicializa el modelo KNN
        
        Args:
            n_neighbors: Número de vecinos más cercanos a considerar
            weights: Peso de los vecinos ('uniform' o 'distance')
            algorithm: Algoritmo para encontrar vecinos
            metric: Métrica de distancia
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric
        
        # Crear modelo
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            metric=metric
        )
        
        # Métricas de entrenamiento
        self.training_metrics = {}
        self.is_trained = False
        
        logger.info(f"🤖 Modelo KNN inicializado (K={n_neighbors}, weights={weights})")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Entrena el modelo KNN
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            
        Returns:
            Diccionario con métricas de entrenamiento
        """
        try:
            logger.info("🎓 Entrenando modelo KNN...")
            
            # Entrenar modelo
            self.model.fit(X_train, y_train)
            
            # Hacer predicciones en datos de entrenamiento
            y_train_pred = self.model.predict(X_train)
            
            # Calcular métricas
            self.training_metrics = {
                'accuracy': accuracy_score(y_train, y_train_pred),
                'precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
                'samples': len(X_train),
                'features': X_train.shape[1]
            }
            
            self.is_trained = True
            
            logger.info("✅ Modelo entrenado exitosamente")
            logger.info(f"📊 Accuracy: {self.training_metrics['accuracy']:.4f}")
            logger.info(f"📊 Precision: {self.training_metrics['precision']:.4f}")
            logger.info(f"📊 Recall: {self.training_metrics['recall']:.4f}")
            logger.info(f"📊 F1-Score: {self.training_metrics['f1_score']:.4f}")
            
            return self.training_metrics
            
        except Exception as e:
            logger.error(f"❌ Error entrenando modelo: {e}")
            return {}
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Hace predicciones con el modelo
        
        Args:
            X_test: Features de prueba
            
        Returns:
            Array con predicciones
        """
        if not self.is_trained:
            logger.error("❌ El modelo no ha sido entrenado")
            return np.array([])
        
        try:
            predictions = self.model.predict(X_test)
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Error haciendo predicciones: {e}")
            return np.array([])
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Obtiene probabilidades de predicción
        
        Args:
            X_test: Features de prueba
            
        Returns:
            Array con probabilidades para cada clase
        """
        if not self.is_trained:
            logger.error("❌ El modelo no ha sido entrenado")
            return np.array([])
        
        try:
            probabilities = self.model.predict_proba(X_test)
            return probabilities
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo probabilidades: {e}")
            return np.array([])
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evalúa el rendimiento del modelo
        
        Args:
            X_test: Features de prueba
            y_test: Target de prueba
            
        Returns:
            Diccionario con métricas de evaluación
        """
        try:
            logger.info("📊 Evaluando modelo...")
            
            # Hacer predicciones
            y_pred = self.predict(X_test)
            
            # Calcular métricas
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'samples': len(X_test)
            }
            
            # Reporte de clasificación
            report = classification_report(y_test, y_pred, 
                                          target_names=['VENTA', 'HOLD', 'COMPRA'],
                                          zero_division=0)
            
            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            
            logger.info("✅ Evaluación completada")
            logger.info(f"\n{report}")
            logger.info(f"\nMatriz de Confusión:\n{cm}")
            
            metrics['classification_report'] = report
            metrics['confusion_matrix'] = cm
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error evaluando modelo: {e}")
            return {}
    
    def predict_next(self, current_data: pd.DataFrame) -> Dict:
        """
        Predice la próxima señal de trading
        
        Args:
            current_data: Datos actuales del mercado (1 fila con features)
            
        Returns:
            Diccionario con predicción y probabilidades
        """
        if not self.is_trained:
            logger.error("❌ El modelo no ha sido entrenado")
            return {}
        
        try:
            # Hacer predicción
            prediction = self.model.predict(current_data)[0]
            probabilities = self.model.predict_proba(current_data)[0]
            
            # Mapear predicción a señal
            signal_map = {
                -1: 'VENTA',
                0: 'HOLD',
                1: 'COMPRA'
            }
            
            signal = signal_map.get(prediction, 'HOLD')
            confidence = np.max(probabilities)
            
            result = {
                'signal': signal,
                'prediction': int(prediction),
                'confidence': float(confidence),
                'probabilities': {
                    'venta': float(probabilities[0]),
                    'hold': float(probabilities[1]) if len(probabilities) > 2 else 0.0,
                    'compra': float(probabilities[-1])
                },
                'recommendation': 'EJECUTAR' if confidence >= CONFIDENCE_THRESHOLD else 'ESPERAR',
                'timestamp': datetime.now()
            }
            
            logger.info(f"🎯 Predicción: {signal} (Confianza: {confidence:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error prediciendo próxima señal: {e}")
            return {}
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict:
        """
        Realiza validación cruzada
        
        Args:
            X: Features completas
            y: Target completo
            cv: Número de folds
            
        Returns:
            Diccionario con resultados de validación cruzada
        """
        try:
            logger.info(f"🔄 Realizando validación cruzada ({cv} folds)...")
            
            # Realizar validación cruzada
            cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
            
            results = {
                'cv_scores': cv_scores,
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'min_score': cv_scores.min(),
                'max_score': cv_scores.max()
            }
            
            logger.info(f"✅ Validación cruzada completada")
            logger.info(f"📊 Accuracy promedio: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en validación cruzada: {e}")
            return {}
    
    def get_neighbors(self, X_query: pd.DataFrame, n_neighbors: Optional[int] = None) -> Tuple:
        """
        Obtiene los vecinos más cercanos para datos de consulta
        
        Args:
            X_query: Datos de consulta
            n_neighbors: Número de vecinos (usa el del modelo si no se especifica)
            
        Returns:
            Tupla con distancias e índices de los vecinos
        """
        if not self.is_trained:
            logger.error("❌ El modelo no ha sido entrenado")
            return np.array([]), np.array([])
        
        try:
            n = n_neighbors if n_neighbors else self.n_neighbors
            distances, indices = self.model.kneighbors(X_query, n_neighbors=n)
            return distances, indices
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo vecinos: {e}")
            return np.array([]), np.array([])
    
    def save_model(self, filename: Optional[str] = None):
        """
        Guarda el modelo entrenado
        
        Args:
            filename: Nombre del archivo (genera uno automático si no se especifica)
        """
        if not self.is_trained:
            logger.warning("⚠️ Intentando guardar un modelo no entrenado")
        
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"knn_model_{timestamp}.pkl"
            
            filepath = os.path.join(MODELS_DIR, filename)
            
            # Guardar modelo y metadatos
            model_data = {
                'model': self.model,
                'n_neighbors': self.n_neighbors,
                'weights': self.weights,
                'algorithm': self.algorithm,
                'metric': self.metric,
                'training_metrics': self.training_metrics,
                'is_trained': self.is_trained,
                'timestamp': datetime.now()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"💾 Modelo guardado en: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error guardando modelo: {e}")
    
    def load_model(self, filename: str):
        """
        Carga un modelo previamente guardado
        
        Args:
            filename: Nombre del archivo del modelo
        """
        try:
            filepath = os.path.join(MODELS_DIR, filename)
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.n_neighbors = model_data['n_neighbors']
            self.weights = model_data['weights']
            self.algorithm = model_data['algorithm']
            self.metric = model_data['metric']
            self.training_metrics = model_data['training_metrics']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"📂 Modelo cargado desde: {filepath}")
            logger.info(f"🤖 K={self.n_neighbors}, Accuracy={self.training_metrics.get('accuracy', 0):.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")


# Función de prueba
if __name__ == "__main__":
    from data.data_collector import DataCollector
    from data.data_processor import DataProcessor
    
    print("=" * 60)
    print("PROBANDO KNN TRADING MODEL")
    print("=" * 60)
    
    # Obtener y procesar datos
    collector = DataCollector()
    df = collector.get_historical_data()
    
    processor = DataProcessor()
    df_clean = processor.clean_data(df)
    df_features = processor.create_features(df_clean)
    df_target = processor.create_target(df_features)
    
    # Preparar datos
    X, y = processor.prepare_training_data(df_target)
    X_train, X_test, y_train, y_test = processor.split_train_test(X, y)
    
    # Crear y entrenar modelo
    model = KNNTradingModel(n_neighbors=50)
    
    print("\n🎓 Entrenando modelo...")
    train_metrics = model.train(X_train, y_train)
    
    print("\n📊 Evaluando modelo...")
    test_metrics = model.evaluate(X_test, y_test)
    
    print("\n🔄 Validación cruzada...")
    cv_results = model.cross_validate(X, y, cv=5)
    
    print("\n🎯 Predicción con datos actuales...")
    current_data = X_test.iloc[[-1]]
    prediction = model.predict_next(current_data)
    print(f"Señal: {prediction['signal']}")
    print(f"Confianza: {prediction['confidence']:.2%}")
    print(f"Recomendación: {prediction['recommendation']}")
    
    print("\n💾 Guardando modelo...")
    model.save_model()
    
    print("\n✅ Prueba completada exitosamente")