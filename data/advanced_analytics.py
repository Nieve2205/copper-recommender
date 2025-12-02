"""
ANÁLISIS AVANZADO - SISTEMA KNN COBRE
======================================

Módulo de análisis avanzado con técnicas de Business Intelligence:
- Análisis de escenarios (What-If Analysis)
- Backtesting de estrategias
- Optimización de cartera
- Análisis de riesgo (VaR, CVaR)
- Simulación Monte Carlo
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedAnalytics:
    """
    Clase para análisis avanzado de Business Intelligence
    """
    
    def __init__(self):
        logger.info("📊 AdvancedAnalytics inicializado")
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> Dict:
        """
        Calcula Value at Risk (VaR) y Conditional VaR
        
        Args:
            returns: Serie de retornos
            confidence: Nivel de confianza
            
        Returns:
            Diccionario con VaR y CVaR
        """
        try:
            # VaR histórico
            var = np.percentile(returns, (1 - confidence) * 100)
            
            # CVaR (Expected Shortfall)
            cvar = returns[returns <= var].mean()
            
            # VaR paramétrico
            mean = returns.mean()
            std = returns.std()
            var_parametric = stats.norm.ppf(1 - confidence, mean, std)
            
            result = {
                'var_historical': var,
                'cvar': cvar,
                'var_parametric': var_parametric,
                'confidence': confidence
            }
            
            logger.info(f"✅ VaR calculado: {var:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculando VaR: {e}")
            return {}
    
    def monte_carlo_simulation(self, 
                              current_price: float,
                              mean_return: float,
                              volatility: float,
                              days: int = 30,
                              simulations: int = 1000) -> Dict:
        """
        Simulación Monte Carlo de precios futuros
        
        Args:
            current_price: Precio actual
            mean_return: Retorno promedio diario
            volatility: Volatilidad diaria
            days: Días a simular
            simulations: Número de simulaciones
            
        Returns:
            Diccionario con resultados de simulación
        """
        try:
            logger.info(f"🎲 Iniciando {simulations} simulaciones Monte Carlo...")
            
            # Generar simulaciones
            results = np.zeros((simulations, days))
            
            for i in range(simulations):
                prices = [current_price]
                for day in range(days - 1):
                    # Movimiento Browniano Geométrico
                    drift = mean_return - 0.5 * volatility ** 2
                    shock = volatility * np.random.normal()
                    price_change = np.exp(drift + shock)
                    prices.append(prices[-1] * price_change)
                
                results[i] = prices
            
            # Estadísticas finales
            final_prices = results[:, -1]
            
            simulation_results = {
                'mean_final_price': np.mean(final_prices),
                'median_final_price': np.median(final_prices),
                'std_final_price': np.std(final_prices),
                'min_price': np.min(final_prices),
                'max_price': np.max(final_prices),
                'percentile_5': np.percentile(final_prices, 5),
                'percentile_95': np.percentile(final_prices, 95),
                'prob_increase': np.mean(final_prices > current_price),
                'all_simulations': results
            }
            
            logger.info(f"✅ Simulación completada. Precio esperado: ${simulation_results['mean_final_price']:.2f}")
            return simulation_results
            
        except Exception as e:
            logger.error(f"❌ Error en simulación Monte Carlo: {e}")
            return {}
    
    def scenario_analysis(self, 
                         current_price: float,
                         scenarios: Dict[str, Dict]) -> pd.DataFrame:
        """
        Análisis de escenarios (What-If Analysis)
        
        Args:
            current_price: Precio actual
            scenarios: Diccionario con escenarios y sus parámetros
            
        Returns:
            DataFrame con análisis de escenarios
        """
        try:
            logger.info("📈 Ejecutando análisis de escenarios...")
            
            results = []
            
            for scenario_name, params in scenarios.items():
                # Calcular precio esperado bajo cada escenario
                expected_return = params.get('return', 0)
                probability = params.get('probability', 1/len(scenarios))
                
                final_price = current_price * (1 + expected_return)
                impact = final_price - current_price
                
                results.append({
                    'scenario': scenario_name,
                    'probability': probability,
                    'expected_return': expected_return,
                    'final_price': final_price,
                    'price_impact': impact,
                    'description': params.get('description', '')
                })
            
            df_scenarios = pd.DataFrame(results)
            
            # Precio esperado ponderado por probabilidad
            weighted_price = (df_scenarios['final_price'] * df_scenarios['probability']).sum()
            
            logger.info(f"✅ Análisis de escenarios completado. Precio esperado: ${weighted_price:.2f}")
            
            return df_scenarios
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de escenarios: {e}")
            return pd.DataFrame()
    
    def backtest_strategy(self,
                         prices: pd.Series,
                         signals: pd.Series,
                         initial_capital: float = 100000) -> Dict:
        """
        Backtesting de estrategia de trading
        
        Args:
            prices: Serie de precios históricos
            signals: Serie de señales (1=compra, -1=venta, 0=hold)
            initial_capital: Capital inicial
            
        Returns:
            Diccionario con métricas de backtesting
        """
        try:
            logger.info("🔙 Ejecutando backtesting...")
            
            # Calcular retornos
            returns = prices.pct_change()
            
            # Retornos de la estrategia
            strategy_returns = signals.shift(1) * returns
            
            # Curva de equity
            equity_curve = (1 + strategy_returns).cumprod() * initial_capital
            
            # Métricas
            total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
            
            # Sharpe Ratio (asumiendo tasa libre de riesgo = 0)
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            
            # Maximum Drawdown
            cumulative = equity_curve / equity_curve.cummax()
            max_drawdown = (cumulative - 1).min()
            
            # Win Rate
            wins = strategy_returns[strategy_returns > 0]
            losses = strategy_returns[strategy_returns < 0]
            win_rate = len(wins) / (len(wins) + len(losses)) if len(wins) + len(losses) > 0 else 0
            
            # Profit Factor
            profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else 0
            
            backtest_results = {
                'initial_capital': initial_capital,
                'final_capital': equity_curve.iloc[-1],
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
                'win_rate': win_rate,
                'win_rate_pct': win_rate * 100,
                'profit_factor': profit_factor,
                'num_trades': len(signals[signals != 0]),
                'equity_curve': equity_curve
            }
            
            logger.info(f"✅ Backtesting completado. Retorno: {total_return*100:.2f}%")
            return backtest_results
            
        except Exception as e:
            logger.error(f"❌ Error en backtesting: {e}")
            return {}
    
    def calculate_optimal_position_size(self,
                                       capital: float,
                                       win_rate: float,
                                       avg_win: float,
                                       avg_loss: float,
                                       risk_per_trade: float = 0.02) -> Dict:
        """
        Calcula el tamaño óptimo de posición usando Kelly Criterion y gestión de riesgo
        
        Args:
            capital: Capital disponible
            win_rate: Tasa de acierto
            avg_win: Ganancia promedio
            avg_loss: Pérdida promedio
            risk_per_trade: Riesgo máximo por operación (default 2%)
            
        Returns:
            Diccionario con tamaños de posición
        """
        try:
            # Kelly Criterion
            kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_pct = max(0, min(kelly_pct, 0.25))  # Limitar entre 0-25%
            
            # Tamaño de posición conservador (50% Kelly)
            conservative_kelly = kelly_pct * 0.5
            
            # Tamaño basado en riesgo fijo
            risk_based_size = risk_per_trade
            
            # Tamaño recomendado (el menor entre Kelly y riesgo fijo)
            recommended_size = min(conservative_kelly, risk_based_size)
            
            position_sizing = {
                'kelly_criterion': kelly_pct,
                'conservative_kelly': conservative_kelly,
                'risk_based': risk_based_size,
                'recommended_pct': recommended_size,
                'recommended_capital': capital * recommended_size,
                'max_loss_per_trade': capital * risk_per_trade
            }
            
            logger.info(f"✅ Tamaño óptimo: {recommended_size*100:.2f}% del capital")
            return position_sizing
            
        except Exception as e:
            logger.error(f"❌ Error calculando tamaño de posición: {e}")
            return {}
    
    def correlation_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Análisis de correlaciones entre variables
        
        Args:
            df: DataFrame con múltiples series de datos
            
        Returns:
            Diccionario con matriz de correlación y análisis
        """
        try:
            logger.info("🔗 Analizando correlaciones...")
            
            # Matriz de correlación
            corr_matrix = df.corr()
            
            # Encontrar correlaciones fuertes
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # Correlación fuerte
                        strong_correlations.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            result = {
                'correlation_matrix': corr_matrix,
                'strong_correlations': strong_correlations,
                'avg_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()
            }
            
            logger.info(f"✅ {len(strong_correlations)} correlaciones fuertes encontradas")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de correlaciones: {e}")
            return {}
    
    def generate_trading_signals_advanced(self, 
                                         df: pd.DataFrame,
                                         sentiment: float,
                                         supply_demand: float) -> pd.Series:
        """
        Genera señales de trading avanzadas combinando análisis técnico y fundamental
        
        Args:
            df: DataFrame con datos técnicos
            sentiment: Score de sentimiento del mercado
            supply_demand: Balance oferta-demanda
            
        Returns:
            Serie con señales de trading mejoradas
        """
        try:
            signals = pd.Series(0, index=df.index)
            
            # Condiciones técnicas
            ma_condition = df['sma_20'] > df['sma_50']
            rsi_condition = (df['rsi'] > 30) & (df['rsi'] < 70)
            macd_condition = df['macd'] > df['macd_signal']
            
            # Condiciones fundamentales
            sentiment_positive = sentiment > 0.2
            supply_deficit = supply_demand < 0  # Déficit favorece precios altos
            
            # Señal de COMPRA: técnico + fundamental alineados
            buy_condition = ma_condition & rsi_condition & macd_condition & sentiment_positive & supply_deficit
            signals[buy_condition] = 1
            
            # Señal de VENTA: condiciones inversas
            sell_condition = ~ma_condition & (df['rsi'] > 70) & ~macd_condition
            signals[sell_condition] = -1
            
            logger.info(f"✅ Señales generadas: {(signals==1).sum()} compras, {(signals==-1).sum()} ventas")
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error generando señales: {e}")
            return pd.Series(0, index=df.index)


# Función de prueba
if __name__ == "__main__":
    print("=" * 60)
    print("PROBANDO ADVANCED ANALYTICS")
    print("=" * 60)
    
    analytics = AdvancedAnalytics()
    
    # Generar datos de ejemplo
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    
    print("\n📊 Value at Risk...")
    var = analytics.calculate_var(returns)
    print(f"VaR (95%): {var['var_historical']:.4f}")
    print(f"CVaR: {var['cvar']:.4f}")
    
    print("\n🎲 Monte Carlo Simulation...")
    mc = analytics.monte_carlo_simulation(100, 0.001, 0.02, days=30, simulations=1000)
    print(f"Precio esperado (30 días): ${mc['mean_final_price']:.2f}")
    print(f"Probabilidad de subida: {mc['prob_increase']:.1%}")
    
    print("\n📈 Scenario Analysis...")
    scenarios = {
        'Optimista': {'return': 0.15, 'probability': 0.3, 'description': 'Fuerte demanda EV'},
        'Base': {'return': 0.05, 'probability': 0.5, 'description': 'Crecimiento normal'},
        'Pesimista': {'return': -0.10, 'probability': 0.2, 'description': 'Recesión global'}
    }
    scenario_df = analytics.scenario_analysis(100, scenarios)
    print(scenario_df)
    
    print("\n🔙 Backtesting...")
    prices = pd.Series(100 + np.cumsum(np.random.randn(252) * 2), index=pd.date_range('2024-01-01', periods=252))
    signals = pd.Series(np.random.choice([-1, 0, 1], 252), index=prices.index)
    backtest = analytics.backtest_strategy(prices, signals)
    print(f"Retorno total: {backtest['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {backtest['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {backtest['max_drawdown_pct']:.2f}%")
    
    print("\n💰 Position Sizing...")
    position = analytics.calculate_optimal_position_size(100000, 0.6, 0.03, 0.02)
    print(f"Tamaño recomendado: {position['recommended_pct']*100:.2f}%")
    print(f"Capital recomendado: ${position['recommended_capital']:.2f}")
    
    print("\n✅ Prueba completada exitosamente")