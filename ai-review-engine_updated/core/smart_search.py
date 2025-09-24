"""
Smart Search System for Phone Review Analysis
Handles natural language queries and phone model extraction
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
import pandas as pd
from datetime import datetime

@dataclass
class SearchQuery:
    """Structured search query"""
    original_query: str
    phone_model: str
    brand: Optional[str]
    intent: str  # 'reviews', 'sentiment', 'comparison', 'features', 'price'
    aspects: List[str]  # specific aspects user is interested in
    confidence: float

class SmartPhoneSearch:
    """Intelligent phone model search and query understanding"""
    
    def __init__(self):
        # Common phone brands and their variations
        self.brands = {
            'apple': ['iphone', 'ios'],
            'samsung': ['galaxy', 'note', 'fold', 'flip'],
            'google': ['pixel', 'nexus'],
            'oneplus': ['oneplus', '1+'],
            'xiaomi': ['mi', 'redmi', 'poco'],
            'huawei': ['huawei', 'honor'],
            'oppo': ['oppo', 'reno', 'find'],
            'vivo': ['vivo', 'iqoo'],
            'motorola': ['moto', 'motorola', 'razr'],
            'nokia': ['nokia'],
            'sony': ['xperia', 'sony'],
            'lg': ['lg', 'velvet', 'wing'],
            'asus': ['asus', 'rog', 'zenfone'],
            'realme': ['realme'],
            'nothing': ['nothing phone']
        }
        
        # Common phone model patterns
        self.model_patterns = [
            # iPhone patterns
            r'iphone\s*(\d{1,2})\s*(pro\s*max|pro|plus|mini)?',
            r'iphone\s*(se|xr|xs|x)\s*(max)?',
            
            # Samsung patterns
            r'galaxy\s*s(\d{1,2})\s*(ultra|plus|\+)?',
            r'galaxy\s*note\s*(\d{1,2})\s*(ultra|plus)?',
            r'galaxy\s*z\s*(fold|flip)\s*(\d)?',
            r'galaxy\s*a(\d{2})\s*(\d{1}s)?',
            
            # Google patterns
            r'pixel\s*(\d)\s*(pro|a|xl)?',
            r'pixel\s*(fold)',
            
            # OnePlus patterns
            r'oneplus\s*(\d{1,2})\s*(pro|t|r)?',
            r'oneplus\s*(nord|ace)\s*(\d|ce)?',
            
            # Xiaomi patterns
            r'mi\s*(\d{1,2})\s*(pro|ultra|lite)?',
            r'redmi\s*(note\s*)?(\d{1,2})\s*(pro|plus)?',
            r'poco\s*([xfm]\d)\s*(pro)?',
            
            # Generic patterns
            r'([a-z]+)\s*(\d{1,4})\s*(pro|plus|ultra|max|mini|lite)?'
        ]
        
        # Intent keywords
        self.intent_keywords = {
            'reviews': ['review', 'reviews', 'feedback', 'opinion', 'thoughts', 'saying', 'think'],
            'sentiment': ['sentiment', 'positive', 'negative', 'good', 'bad', 'worth', 'recommend'],
            'comparison': ['vs', 'versus', 'compare', 'better', 'difference', 'or'],
            'features': ['camera', 'battery', 'screen', 'display', 'performance', 'storage', 'ram'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'budget', 'worth', 'value']
        }
        
        # Question patterns
        self.question_patterns = [
            r'what\s*(are\s*)?people\s*saying\s*about\s*(.*)',
            r'is\s*(.*)\s*(good|worth|recommended)',
            r'should\s*i\s*buy\s*(.*)',
            r'how\s*is\s*(.*)',
            r'(.*)\s*reviews?',
            r'(.*)\s*opinions?',
            r'tell\s*me\s*about\s*(.*)',
            r'show\s*me\s*(.*)\s*reviews?'
        ]
        
        # Common phone database (for fuzzy matching)
        self.known_phones = [
            # iPhones
            'iPhone 15 Pro Max', 'iPhone 15 Pro', 'iPhone 15 Plus', 'iPhone 15',
            'iPhone 14 Pro Max', 'iPhone 14 Pro', 'iPhone 14 Plus', 'iPhone 14',
            'iPhone 13 Pro Max', 'iPhone 13 Pro', 'iPhone 13', 'iPhone 13 Mini',
            'iPhone 12 Pro Max', 'iPhone 12 Pro', 'iPhone 12', 'iPhone 12 Mini',
            'iPhone 11 Pro Max', 'iPhone 11 Pro', 'iPhone 11',
            'iPhone SE 3rd Gen', 'iPhone SE 2nd Gen',
            
            # Samsung
            'Galaxy S24 Ultra', 'Galaxy S24+', 'Galaxy S24',
            'Galaxy S23 Ultra', 'Galaxy S23+', 'Galaxy S23',
            'Galaxy S22 Ultra', 'Galaxy S22+', 'Galaxy S22',
            'Galaxy Z Fold 5', 'Galaxy Z Flip 5',
            'Galaxy A54', 'Galaxy A34', 'Galaxy A14',
            
            # Google
            'Pixel 8 Pro', 'Pixel 8', 'Pixel 7a',
            'Pixel 7 Pro', 'Pixel 7',
            'Pixel 6 Pro', 'Pixel 6', 'Pixel 6a',
            'Pixel Fold',
            
            # OnePlus
            'OnePlus 12', 'OnePlus 11', 'OnePlus 10 Pro',
            'OnePlus Nord 3', 'OnePlus Nord CE 3',
            
            # Others
            'Xiaomi 14 Pro', 'Xiaomi 13T', 'Redmi Note 13 Pro',
            'Nothing Phone 2', 'ASUS ROG Phone 7',
            'Motorola Edge 40 Pro', 'Motorola Razr 40 Ultra'
        ]
    
    def parse_query(self, query: str) -> SearchQuery:
        """
        Parse user query to extract phone model and intent
        
        Examples:
        - "iPhone 12 Pro Max" -> Direct model search
        - "What are people saying about iPhone 12 Pro Max" -> Review search
        - "iPhone 12 Pro Max reviews" -> Review search
        - "Is Galaxy S24 worth buying" -> Sentiment analysis
        - "iPhone 15 vs Samsung S24" -> Comparison
        """
        
        # Clean and lowercase query
        original_query = query
        query = query.lower().strip()
        
        # Extract phone model
        phone_model, brand, confidence = self._extract_phone_model(query)
        
        # Determine intent
        intent = self._determine_intent(query)
        
        # Extract specific aspects
        aspects = self._extract_aspects(query)
        
        # If no specific intent found but phone model exists, default to reviews
        if phone_model and intent == 'general':
            intent = 'reviews'
        
        return SearchQuery(
            original_query=original_query,
            phone_model=phone_model,
            brand=brand,
            intent=intent,
            aspects=aspects,
            confidence=confidence
        )
    
    def _extract_phone_model(self, query: str) -> Tuple[str, Optional[str], float]:
        """Extract phone model from query"""
        
        best_match = None
        best_confidence = 0
        detected_brand = None
        
        # First, try exact pattern matching
        for pattern in self.model_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                model = match.group(0)
                
                # Detect brand
                for brand, keywords in self.brands.items():
                    if any(kw in model.lower() for kw in keywords):
                        detected_brand = brand
                        break
                
                # Clean up model name
                model = self._normalize_phone_model(model, detected_brand)
                
                # Check against known phones for exact match
                for known_phone in self.known_phones:
                    similarity = SequenceMatcher(None, model.lower(), known_phone.lower()).ratio()
                    if similarity > best_confidence:
                        best_match = known_phone
                        best_confidence = similarity
                
                if best_confidence > 0.8:  # High confidence match:
                    return best_match, detected_brand, best_confidence
        
        # Fuzzy matching against known phones
        query_words = query.split()
        for known_phone in self.known_phones:
            # Check if key parts of phone name are in query
            phone_parts = known_phone.lower().split()
            matches = sum(1 for part in phone_parts if part in query)
            
            if matches >= len(phone_parts) * 0.6:  # At least 60% of parts match:
                similarity = SequenceMatcher(None, query, known_phone.lower()).ratio()
                if similarity > best_confidence:
                    best_match = known_phone
                    best_confidence = similarity
                    
                    # Detect brand from matched phone
                    for brand, keywords in self.brands.items():
                        if any(kw in known_phone.lower() for kw in keywords):
                            detected_brand = brand
                            break
        
        # If still no match, try to extract any phone-like pattern
        if not best_match:
            # Look for brand + number pattern
            for brand, keywords in self.brands.items():
                for keyword in keywords:
                    if keyword in query:
                        # Find numbers near the brand keyword
                        pattern = f"{keyword}\\s*\\S*\\s*\\S*"
                        match = re.search(pattern, query)
                        if match:
                            best_match = self._normalize_phone_model(match.group(0), brand)
                            detected_brand = brand
                            best_confidence = 0.6
                            break
        
        return best_match or "", detected_brand, best_confidence
    
    def _normalize_phone_model(self, model: str, brand: Optional[str]) -> str:
        """Normalize phone model name to standard format"""
        
        model = model.strip()
        
        # Capitalize appropriately
        words = model.split()
        normalized = []
        
        for word in words:
            if word.lower() in ['pro', 'max', 'plus', 'mini', 'ultra', 'lite']:
                normalized.append(word.capitalize())
            elif word.lower() == 'iphone':
                normalized.append('iPhone')
            elif word.lower() == 'galaxy':
                normalized.append('Galaxy')
            elif word.lower() == 'pixel':
                normalized.append('Pixel')
            elif word.lower() in ['se', 'xr', 'xs']:
                normalized.append(word.upper())
            elif word.isdigit():
                normalized.append(word)
            else:
                normalized.append(word.capitalize())
        
        return ' '.join(normalized)
    
    def _determine_intent(self, query: str) -> str:
        """Determine user's intent from query"""
        
        # Check for comparison
        if any(word in query for word in ['vs', 'versus', 'compare', 'or']):
            return 'comparison'
        
        # Check for specific intents
        intent_scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        
        # Check question patterns
        for pattern in self.question_patterns:
            if re.search(pattern, query):
                return 'reviews'
        
        return 'general'
    
    def _extract_aspects(self, query: str) -> List[str]:
        """Extract specific aspects user is interested in"""
        
        aspects = []
        
        aspect_keywords = {
            'camera': ['camera', 'photo', 'picture', 'lens', 'zoom'],
            'battery': ['battery', 'charging', 'battery life', 'power'],
            'display': ['screen', 'display', 'oled', 'amoled', 'brightness'],
            'performance': ['performance', 'speed', 'fast', 'lag', 'processor', 'ram'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'value'],
            'design': ['design', 'build', 'quality', 'color', 'size'],
            'storage': ['storage', 'memory', 'gb', 'space']
        }
        
        for aspect, keywords in aspect_keywords.items():
            if any(keyword in query.lower() for keyword in keywords):
                aspects.append(aspect)
        
        return aspects
    
    def suggest_phones(self, partial_query: str) -> List[Dict[str, Any]]:
        """Provide phone suggestions as user types"""
        
        suggestions = []
        partial = partial_query.lower().strip()
        
        if len(partial) < 2:
            return suggestions
        
        # Find matching phones
        for phone in self.known_phones:
            if partial in phone.lower():
                # Calculate relevance score
                score = 1.0
                if phone.lower().startswith(partial):
                    score = 2.0  # Boost for prefix matches
                
                suggestions.append({
                    'phone': phone,
                    'score': score,
                    'brand': self._get_brand_from_phone(phone)
                })
        
        # Sort by score and return top 10
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        return suggestions[:10]
    
    def _get_brand_from_phone(self, phone: str) -> str:
        """Get brand from phone name"""
        
        phone_lower = phone.lower()
        for brand, keywords in self.brands.items():
            if any(kw in phone_lower for kw in keywords):
                return brand
        return 'unknown'
    
    def generate_search_response(self, search_query: SearchQuery) -> Dict[str, Any]:
        """Generate appropriate response based on search query"""
        
        response = {
            'query': search_query.original_query,
            'phone_model': search_query.phone_model,
            'brand': search_query.brand,
            'intent': search_query.intent,
            'confidence': search_query.confidence,
            'aspects': search_query.aspects,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add intent-specific response data
        if search_query.intent == 'reviews':
            response['action'] = 'show_reviews'
            response['message'] = f"Showing reviews for {search_query.phone_model}"
        
        elif search_query.intent == 'sentiment':
            response['action'] = 'analyze_sentiment'
            response['message'] = f"Analyzing sentiment for {search_query.phone_model}"
        
        elif search_query.intent == 'comparison':
            response['action'] = 'compare_phones'
            response['message'] = "Preparing comparison"
        
        elif search_query.intent == 'features':
            response['action'] = 'show_features'
            response['message'] = f"Showing {', '.join(search_query.aspects)} details for {search_query.phone_model}"
        
        else:
            response['action'] = 'show_overview'
            response['message'] = f"Showing overview for {search_query.phone_model}"
        
        return response

class ReviewAnalyzer:
    """Analyze reviews for extracted phone model"""
    
    def __init__(self):
        self.search_engine = SmartPhoneSearch()
    
    def analyze_phone_query(self, query: str) -> Dict[str, Any]:
        """
        Main entry point for analyzing user queries about phones
        
        Examples:
        - "iPhone 12 Pro Max" -> Show all reviews and analysis
        - "What are people saying about Galaxy S24" -> Show sentiment analysis
        - "iPhone 15 camera reviews" -> Show camera-specific reviews
        """
        
        # Parse the query
        search_query = self.search_engine.parse_query(query)
        
        if not search_query.phone_model:
            return {
                'success': False,
                'message': "Could not identify a phone model. Please specify a phone name like 'iPhone 15 Pro' or 'Galaxy S24'",
                'suggestions': self._get_popular_phones()
            }
        
        # Generate response based on intent
        response = self.search_engine.generate_search_response(search_query)
        
        # Add analysis data (would be fetched from database in production)
        if search_query.phone_model:
            response['analysis'] = self._get_phone_analysis(
                search_query.phone_model,
                search_query.aspects
            )
        
        response['success'] = True
        return response
    
    def _get_phone_analysis(self, phone_model: str, aspects: List[str] = None) -> Dict[str, Any]:
        """Get analysis data for phone (mock data for demo)"""
        
        # In production, this would query the database
        analysis = {
            'phone_model': phone_model,
            'total_reviews': 2456,
            'average_rating': 4.3,
            'sentiment': {
                'positive': 72,
                'neutral': 18,
                'negative': 10
            },
            'top_positive_aspects': ['camera', 'display', 'performance'],
            'top_negative_aspects': ['price', 'battery life'],
            'summary': f"{phone_model} receives mostly positive reviews with users praising the camera and display quality.",
            'recommendation_score': 85
        }
        
        # Filter by aspects if specified
        if aspects:
            analysis['filtered_aspects'] = aspects
            analysis['aspect_sentiments'] = {
                aspect: {
                    'positive': 70 + (hash(aspect) % 20),
                    'negative': 10 + (hash(aspect) % 10)
                }
                for aspect in aspects
            }
        
        return analysis
    
    def _get_popular_phones(self) -> List[str]:
        """Get list of popular phones for suggestions"""
        
        return [
            'iPhone 15 Pro Max',
            'Samsung Galaxy S24 Ultra',
            'Google Pixel 8 Pro',
            'OnePlus 12',
            'Xiaomi 14 Pro'
        ]


# Example usage
if __name__ == "__main__":
    analyzer = ReviewAnalyzer()
    
    # Test queries
    test_queries = [
        "iPhone 12 Pro Max",
        "what are people saying about iPhone 12 Pro Max",
        "iPhone 12 Pro Max reviews",
        "is galaxy s24 worth buying",
        "pixel 8 camera",
        "oneplus 12 battery life reviews",
        "samsung s24 ultra",
        "show me iPhone 15 reviews",
        "iphone15promax"  # No spaces
    ]
    
    print("=" * 60)
    print("SMART PHONE SEARCH DEMONSTRATION")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n📱 Query: '{query}'")
        result = analyzer.analyze_phone_query(query)
        
        if result['success']:
            print(f"✅ Phone Found: {result['phone_model']}")
            print(f"   Brand: {result['brand']}")
            print(f"   Intent: {result['intent']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            if result.get('aspects'):
                print(f"   Aspects: {', '.join(result['aspects'])}")
            print(f"   Action: {result['action']}")
            print(f"   Message: {result['message']}")
        else:
            print(f"❌ {result['message']}")
    
    print("\n" + "=" * 60)
# Alias for backward compatibility
SmartSearchEngine = ReviewAnalyzer
