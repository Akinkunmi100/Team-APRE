"""
Market Analyzer Module for trend analysis
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """
    Analyze market trends and patterns
    """
    
    def __init__(self):
        """Initialize market analyzer"""
        logger.info("Market Analyzer initialized")
    
    def get_trending_phones(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get trending phones
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of trending phones
        """
        # Simulated data
        phones = [
            {'name': 'iPhone 15 Pro', 'trend_score': 95, 'change': '+15%'},
            {'name': 'Samsung S24 Ultra', 'trend_score': 88, 'change': '+8%'},
            {'name': 'Google Pixel 8', 'trend_score': 82, 'change': '+5%'},
            {'name': 'OnePlus 12', 'trend_score': 76, 'change': '-2%'},
            {'name': 'Xiaomi 14', 'trend_score': 71, 'change': '+12%'}
        ]
        
        return phones[:5]
    
    def get_price_trends(self, phone_name: str, days: int = 30) -> Dict[str, Any]:
        """
        Get price trends for a phone
        
        Args:
            phone_name: Phone model name
            days: Number of days to analyze
            
        Returns:
            Price trend data
        """
        # Generate simulated price data
        base_price = 999
        prices = []
        dates = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days-i)
            price = base_price + random.randint(-50, 50)
            prices.append(price)
            dates.append(date.strftime('%Y-%m-%d'))
        
        return {
            'phone': phone_name,
            'prices': prices,
            'dates': dates,
            'average': sum(prices) / len(prices),
            'trend': 'stable' if abs(prices[-1] - prices[0]) < 50 else ('up' if prices[-1] > prices[0] else 'down')
        }
    
    def get_market_share(self) -> Dict[str, float]:
        """
        Get market share by brand
        
        Returns:
            Market share percentages
        """
        return {
            'Apple': 28.5,
            'Samsung': 24.3,
            'Xiaomi': 12.8,
            'Oppo': 10.5,
            'Vivo': 9.2,
            'Others': 14.7
        }
    
    def get_feature_popularity(self) -> Dict[str, int]:
        """
        Get popularity of phone features
        
        Returns:
            Feature popularity scores
        """
        return {
            'Camera': 92,
            'Battery': 85,
            'Display': 78,
            'Performance': 72,
            '5G': 65,
            'Design': 60,
            'Storage': 55,
            'Price': 50
        }
