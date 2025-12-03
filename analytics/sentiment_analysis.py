"""
ANÁLISIS DE SENTIMIENTO EN TIEMPO REAL - SISTEMA KNN COBRE
===========================================================

Análisis de sentimiento del mercado de cobre usando:
- News API (noticias globales)
- Reddit API (sentimiento de traders)
- Twitter/X (menciones en tiempo real)
- RSS Feeds de medios financieros
- Web Scraping de sitios especializados

El análisis se basa en técnicas de NLP (Natural Language Processing)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
import json
import re
from collections import Counter
import time
import os

# NLP Libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️ TextBlob no disponible. Instalar con: pip install textblob")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("⚠️ VADER no disponible. Instalar con: pip install vaderSentiment")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analizador de sentimiento del mercado de cobre usando múltiples fuentes
    """
    
    def __init__(self, news_api_key: str = None):
        """
        Inicializa el analizador de sentimiento
        
        Args:
            news_api_key: API key de NewsAPI (obtener en https://newsapi.org)
        """
        self.news_api_key = news_api_key or "demo"
        
        # Inicializar analizadores NLP
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        
        # Keywords relacionadas con cobre
        self.copper_keywords = [
            'copper', 'cobre', 'metal', 'mining', 'minería',
            'commodity', 'industrial metal', 'red metal',
            'copper price', 'precio cobre', 'copper demand',
            'copper supply', 'electric vehicle', 'EV',
            'renewable energy', 'construction', 'china demand'
        ]
        
        logger.info("📰 SentimentAnalyzer inicializado")
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """
        Analiza el sentimiento de un texto usando múltiples métodos
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con scores de sentimiento
        """
        if not text or len(text.strip()) == 0:
            return {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 1}
        
        results = {}
        
        # VADER Sentiment (mejor para textos cortos y redes sociales)
        if VADER_AVAILABLE:
            vader_scores = self.vader.polarity_scores(text)
            results['vader'] = vader_scores
        
        # TextBlob Sentiment (análisis general)
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                results['textblob'] = {
                    'polarity': blob.sentiment.polarity,  # -1 a 1
                    'subjectivity': blob.sentiment.subjectivity  # 0 a 1
                }
            except:
                results['textblob'] = {'polarity': 0, 'subjectivity': 0.5}
        
        # Sentiment Score Combinado
        if VADER_AVAILABLE:
            compound = vader_scores['compound']
        elif TEXTBLOB_AVAILABLE:
            compound = results['textblob']['polarity']
        else:
            compound = 0
        
        # Clasificación
        if compound >= 0.05:
            classification = 'positive'
        elif compound <= -0.05:
            classification = 'negative'
        else:
            classification = 'neutral'
        
        results['compound'] = compound
        results['classification'] = classification
        
        return results
    
    def get_news_from_newsapi(self, 
                               query: str = "copper OR metal commodity",
                               days: int = 7,
                               language: str = 'en') -> List[Dict]:
        """
        Obtiene noticias de NewsAPI
        
        Args:
            query: Términos de búsqueda
            days: Días hacia atrás
            language: Idioma de las noticias
            
        Returns:
            Lista de noticias
        """
        try:
            logger.info(f"📰 Obteniendo noticias de NewsAPI (últimos {days} días)...")
            
            # NewsAPI endpoint
            url = "https://newsapi.org/v2/everything"
            
            # Fecha desde
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'language': language,
                'pageSize': 100,
                'apiKey': os.getenv('NEWS_API_KEY')
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                logger.info(f"✅ {len(articles)} noticias obtenidas de NewsAPI")
                return articles
            else:
                logger.warning(f"⚠️ NewsAPI error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error con NewsAPI: {e}")
            return []
    
    def get_reddit_sentiment(self, subreddit: str = 'investing') -> Dict:
        """
        Obtiene sentimiento de Reddit (sin API key usando scraping ligero)
        
        Args:
            subreddit: Subreddit a analizar
            
        Returns:
            Diccionario con análisis de sentimiento
        """
        try:
            logger.info(f"🔴 Obteniendo sentimiento de Reddit r/{subreddit}...")
            
            # Reddit JSON API (público, no requiere auth para lectura básica)
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                'q': 'copper OR metal commodity',
                'limit': 25,
                'sort': 'new',
                't': 'week'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Business Intelligence Project)'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                posts = data.get('data', {}).get('children', [])
                
                sentiments = []
                for post in posts:
                    post_data = post.get('data', {})
                    title = post_data.get('title', '')
                    selftext = post_data.get('selftext', '')
                    
                    text = f"{title} {selftext}"
                    if text.strip():
                        sentiment = self.analyze_text_sentiment(text)
                        sentiments.append(sentiment['compound'])
                
                if sentiments:
                    avg_sentiment = np.mean(sentiments)
                    logger.info(f"✅ Reddit: {len(sentiments)} posts analizados")
                    
                    return {
                        'source': 'reddit',
                        'avg_sentiment': avg_sentiment,
                        'post_count': len(sentiments),
                        'positive_ratio': sum(1 for s in sentiments if s > 0.05) / len(sentiments),
                        'negative_ratio': sum(1 for s in sentiments if s < -0.05) / len(sentiments)
                    }
            
            logger.warning("⚠️ No se pudieron obtener datos de Reddit")
            return {}
                
        except Exception as e:
            logger.error(f"❌ Error con Reddit: {e}")
            return {}
    
    def get_rss_feeds(self) -> List[Dict]:
        """
        Obtiene noticias de RSS feeds de medios financieros
        
        Returns:
            Lista de artículos de RSS
        """
        try:
            import feedparser
            
            logger.info("📡 Obteniendo noticias de RSS feeds...")
            
            # RSS feeds de medios financieros
            feeds = [
                'https://www.reuters.com/markets/commodities/rss',
                'https://www.ft.com/commodities?format=rss',
                'https://www.mining.com/feed/',
            ]
            
            all_articles = []
            
            for feed_url in feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:10]:  # Top 10 de cada feed
                        # Filtrar por keywords de cobre
                        title = entry.get('title', '').lower()
                        summary = entry.get('summary', '').lower()
                        
                        if any(keyword in title or keyword in summary 
                               for keyword in ['copper', 'metal', 'commodity']):
                            all_articles.append({
                                'title': entry.get('title', ''),
                                'summary': entry.get('summary', ''),
                                'link': entry.get('link', ''),
                                'published': entry.get('published', ''),
                                'source': feed_url
                            })
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error con feed {feed_url}: {e}")
                    continue
            
            logger.info(f"✅ {len(all_articles)} artículos obtenidos de RSS")
            return all_articles
            
        except ImportError:
            logger.warning("⚠️ feedparser no disponible. Instalar con: pip install feedparser")
            return []
        except Exception as e:
            logger.error(f"❌ Error con RSS feeds: {e}")
            return []
    
    def scrape_kitco_news(self) -> List[Dict]:
        """
        Scraping ligero de Kitco News (especializado en metales)
        
        Returns:
            Lista de noticias de metales
        """
        try:
            logger.info("⛏️ Obteniendo noticias de Kitco...")
            
            url = "https://www.kitco.com/news/metals.html"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Business Intelligence Project)'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Simple regex para extraer títulos (no usar en producción, usar BeautifulSoup)
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                articles = []
                for headline in soup.find_all(['h3', 'h4'], limit=20):
                    text = headline.get_text().strip()
                    if text and any(kw in text.lower() for kw in ['copper', 'metal', 'commodity']):
                        articles.append({
                            'title': text,
                            'source': 'Kitco News'
                        })
                
                logger.info(f"✅ Kitco: {len(articles)} noticias obtenidas")
                return articles
                
        except ImportError:
            logger.warning("⚠️ BeautifulSoup no disponible. Instalar con: pip install beautifulsoup4")
            return []
        except Exception as e:
            logger.error(f"❌ Error con Kitco: {e}")
            return []
    
    def analyze_market_sentiment(self, days: int = 7) -> Dict:
        """
        Análisis completo de sentimiento del mercado
        
        Args:
            days: Días hacia atrás para analizar
            
        Returns:
            Diccionario completo con análisis de sentimiento
        """
        logger.info("🔍 Iniciando análisis completo de sentimiento...")
        
        all_sentiments = []
        sources_data = {}
        
        # 1. NewsAPI
        news_articles = self.get_news_from_newsapi(days=days)
        if news_articles:
            news_sentiments = []
            for article in news_articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                sentiment = self.analyze_text_sentiment(text)
                news_sentiments.append(sentiment['compound'])
            
            all_sentiments.extend(news_sentiments)
            sources_data['newsapi'] = {
                'count': len(news_sentiments),
                'avg_sentiment': np.mean(news_sentiments) if news_sentiments else 0,
                'positive_ratio': sum(1 for s in news_sentiments if s > 0.05) / len(news_sentiments) if news_sentiments else 0
            }
        
        # 2. Reddit
        reddit_data = self.get_reddit_sentiment()
        if reddit_data and 'avg_sentiment' in reddit_data:
            all_sentiments.extend([reddit_data['avg_sentiment']] * reddit_data.get('post_count', 1))
            sources_data['reddit'] = reddit_data
        
        # 3. RSS Feeds
        rss_articles = self.get_rss_feeds()
        if rss_articles:
            rss_sentiments = []
            for article in rss_articles:
                text = f"{article.get('title', '')} {article.get('summary', '')}"
                sentiment = self.analyze_text_sentiment(text)
                rss_sentiments.append(sentiment['compound'])
            
            all_sentiments.extend(rss_sentiments)
            sources_data['rss_feeds'] = {
                'count': len(rss_sentiments),
                'avg_sentiment': np.mean(rss_sentiments) if rss_sentiments else 0
            }
        
        # 4. Kitco News
        kitco_articles = self.scrape_kitco_news()
        if kitco_articles:
            kitco_sentiments = []
            for article in kitco_articles:
                sentiment = self.analyze_text_sentiment(article.get('title', ''))
                kitco_sentiments.append(sentiment['compound'])
            
            all_sentiments.extend(kitco_sentiments)
            sources_data['kitco'] = {
                'count': len(kitco_sentiments),
                'avg_sentiment': np.mean(kitco_sentiments) if kitco_sentiments else 0
            }
        
        # Análisis consolidado
        if all_sentiments:
            avg_sentiment = np.mean(all_sentiments)
            positive_count = sum(1 for s in all_sentiments if s > 0.05)
            negative_count = sum(1 for s in all_sentiments if s < -0.05)
            neutral_count = len(all_sentiments) - positive_count - negative_count
            
            # Clasificación final
            if avg_sentiment > 0.15:
                classification = 'Very Positive'
            elif avg_sentiment > 0.05:
                classification = 'Positive'
            elif avg_sentiment < -0.15:
                classification = 'Very Negative'
            elif avg_sentiment < -0.05:
                classification = 'Negative'
            else:
                classification = 'Neutral'
            
            result = {
                'timestamp': datetime.now(),
                'overall_sentiment': avg_sentiment,
                'classification': classification,
                'total_articles': len(all_sentiments),
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'positive_ratio': positive_count / len(all_sentiments),
                'negative_ratio': negative_count / len(all_sentiments),
                'sentiment_std': np.std(all_sentiments),
                'sources': sources_data,
                'confidence': self._calculate_confidence(all_sentiments, sources_data)
            }
            
            logger.info(f"✅ Análisis completado: Sentimiento {classification} ({avg_sentiment:.3f})")
            return result
        else:
            logger.warning("⚠️ No se pudo obtener datos de sentimiento")
            return {
                'timestamp': datetime.now(),
                'overall_sentiment': 0,
                'classification': 'Unknown',
                'total_articles': 0,
                'confidence': 0
            }
    
    def _calculate_confidence(self, sentiments: List[float], sources: Dict) -> float:
        """
        Calcula el nivel de confianza del análisis
        
        Args:
            sentiments: Lista de scores de sentimiento
            sources: Datos de fuentes
            
        Returns:
            Score de confianza (0-1)
        """
        if not sentiments:
            return 0
        
        # Factores de confianza
        num_articles = len(sentiments)
        num_sources = len(sources)
        sentiment_consistency = 1 - np.std(sentiments)
        
        # Peso por número de artículos (más artículos = más confianza)
        articles_factor = min(num_articles / 50, 1.0)
        
        # Peso por número de fuentes (más fuentes = más confianza)
        sources_factor = min(num_sources / 4, 1.0)
        
        # Confianza final
        confidence = (articles_factor * 0.4 + 
                     sources_factor * 0.3 + 
                     sentiment_consistency * 0.3)
        
        return min(confidence, 1.0)
    
    def get_trending_topics(self, articles: List[Dict]) -> List[str]:
        """
        Extrae los temas más mencionados
        
        Args:
            articles: Lista de artículos
            
        Returns:
            Lista de temas trending
        """
        try:
            all_text = ' '.join([
                f"{art.get('title', '')} {art.get('description', '')}"
                for art in articles
            ])
            
            # Palabras clave relevantes para el mercado
            market_keywords = [
                'demand', 'supply', 'china', 'electric', 'vehicle',
                'mining', 'production', 'prices', 'shortage', 'surplus',
                'inflation', 'recession', 'growth', 'forecast', 'outlook'
            ]
            
            # Contar menciones
            word_counts = Counter()
            for keyword in market_keywords:
                count = all_text.lower().count(keyword)
                if count > 0:
                    word_counts[keyword] = count
            
            # Top 5 temas
            trending = [word for word, count in word_counts.most_common(5)]
            
            return trending
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo trending topics: {e}")
            return []


# Función de prueba
if __name__ == "__main__":
    print("=" * 60)
    print("PROBANDO SENTIMENT ANALYZER")
    print("=" * 60)
    
    # Verificar dependencias
    if not VADER_AVAILABLE:
        print("\n⚠️ Instala VADER: pip install vaderSentiment")
    if not TEXTBLOB_AVAILABLE:
        print("⚠️ Instala TextBlob: pip install textblob")
    
    if VADER_AVAILABLE or TEXTBLOB_AVAILABLE:
        analyzer = SentimentAnalyzer()
        
        print("\n📝 Probando análisis de texto...")
        test_texts = [
            "Copper prices surge as demand from electric vehicle sector grows",
            "Mining companies face challenges due to supply chain issues",
            "Analysts remain neutral on copper outlook for next quarter"
        ]
        
        for text in test_texts:
            sentiment = analyzer.analyze_text_sentiment(text)
            print(f"\n'{text[:50]}...'")
            print(f"  Sentimiento: {sentiment['compound']:.3f} ({sentiment['classification']})")
        
        print("\n🔍 Ejecutando análisis completo del mercado...")
        print("(Esto puede tomar 30-60 segundos...)")
        
        result = analyzer.analyze_market_sentiment(days=7)
        
        print(f"\n📊 RESULTADOS:")
        print(f"  Sentimiento General: {result['overall_sentiment']:.3f}")
        print(f"  Clasificación: {result['classification']}")
        print(f"  Total de Artículos: {result['total_articles']}")
        print(f"  Ratio Positivo: {result['positive_ratio']:.1%}")
        print(f"  Ratio Negativo: {result['negative_ratio']:.1%}")
        print(f"  Confianza: {result['confidence']:.1%}")
        
        print(f"\n📰 Fuentes analizadas:")
        for source, data in result['sources'].items():
            print(f"  {source}: {data.get('count', 0)} artículos")
        
        print("\n✅ Prueba completada exitosamente")
    else:
        print("\n❌ Instala las dependencias NLP primero")
        print("pip install vaderSentiment textblob beautifulsoup4 feedparser")