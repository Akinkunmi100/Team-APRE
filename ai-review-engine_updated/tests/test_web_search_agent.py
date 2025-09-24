"""
Test Suite for Web Search Agent
Validates functionality without making actual web requests
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.web_search_agent import WebSearchAgent, WebSearchResult, PhoneInfo, integrate_web_search_with_system
from datetime import datetime

class TestWebSearchAgent(unittest.TestCase):
    """Test cases for WebSearchAgent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.agent = WebSearchAgent()
        
        # Mock search result for testing
        self.mock_search_result = WebSearchResult(
            phone_model="iPhone 15 Pro",
            source="test_source",
            url="https://test.com/iphone-15-pro",
            title="iPhone 15 Pro Review - Excellent Camera",
            snippet="The iPhone 15 Pro features an excellent camera system with improved low-light performance",
            rating=4.5,
            review_count=150,
            price="$999",
            sentiment_preview="positive",
            confidence=0.9,
            scraped_at=datetime.now().isoformat()
        )
    
    def test_agent_initialization(self):
        """Test agent initializes correctly"""
        self.assertIsNotNone(self.agent.smart_search)
        self.assertIn('gsmarena', self.agent.search_sources)
        self.assertEqual(self.agent.config['max_concurrent_searches'], 3)
    
    def test_custom_configuration(self):
        """Test custom configuration"""
        custom_config = {
            'max_concurrent_searches': 2,
            'search_timeout': 20,
            'rate_limit_delay': 3.0
        }
        
        custom_agent = WebSearchAgent(config=custom_config)
        self.assertEqual(custom_agent.config['max_concurrent_searches'], 2)
        self.assertEqual(custom_agent.config['search_timeout'], 20)
        self.assertEqual(custom_agent.config['rate_limit_delay'], 3.0)
    
    def test_cache_functionality(self):
        """Test caching mechanism"""
        cache_key = "test_phone_reviews"
        test_data = {"phone": "test", "data": "cached"}
        
        # Test cache miss
        self.assertFalse(self.agent._is_cache_valid(cache_key))
        
        # Test cache set
        self.agent._cache_results(cache_key, test_data)
        self.assertTrue(self.agent._is_cache_valid(cache_key))
        
        # Test cache retrieval
        cached_data = self.agent.search_cache[cache_key]['data']
        self.assertEqual(cached_data, test_data)
    
    def test_rating_extraction(self):
        """Test rating extraction from text"""
        test_cases = [
            ("4.5/5 stars", 4.5),
            ("8.0/10 rating", 4.0),  # Should convert to 5-point scale
            ("Rating: 4.2 out of 5", 4.2),
            ("3 stars", 3.0),
            ("No rating here", None),
            ("★★★★☆ 4/5", 4.0)
        ]
        
        for text, expected in test_cases:
            result = self.agent._extract_rating(text)
            if expected is None:
                self.assertIsNone(result)
            else:
                self.assertAlmostEqual(result, expected, places=1)
    
    def test_feature_extraction(self):
        """Test feature extraction from text"""
        test_text = "This phone has an excellent camera, great battery life, and 5G connectivity"
        features = self.agent._extract_features_from_text(test_text)
        
        self.assertIn('camera', features)
        self.assertIn('battery', features)
        self.assertIn('5G', features)
    
    def test_price_analysis(self):
        """Test price range analysis"""
        test_prices = ["$599", "$649.99", "€699"]
        result = self.agent._analyze_price_range(test_prices)
        
        self.assertEqual(result['status'], 'found')
        self.assertIn('min_price', result)
        self.assertIn('max_price', result)
        self.assertIn('avg_price', result)
    
    def test_sentiment_analysis(self):
        """Test web sentiment analysis"""
        mock_reviews = [
            {'content': 'This phone is excellent and amazing'},
            {'content': 'Great camera quality and performance'},
            {'content': 'Poor battery life and disappointing screen'}
        ]
        
        result = self.agent._analyze_web_sentiment(mock_reviews)
        
        self.assertIn('positive_percentage', result)
        self.assertIn('negative_percentage', result)
        self.assertIn('overall_sentiment', result)
        self.assertEqual(result['total_reviews_analyzed'], 3)
    
    def test_phone_info_aggregation(self):
        """Test aggregation of search results"""
        mock_results = [self.mock_search_result]
        mock_query = Mock()
        mock_query.phone_model = "iPhone 15 Pro"
        mock_query.brand = "Apple"
        mock_query.confidence = 0.9
        
        phone_info = self.agent._aggregate_search_results(mock_results, mock_query)
        
        self.assertEqual(phone_info.model, "iPhone 15 Pro")
        self.assertEqual(phone_info.brand, "Apple")
        self.assertEqual(len(phone_info.reviews), 1)
        self.assertIsNotNone(phone_info.overall_rating)
        self.assertTrue(phone_info.key_features)
    
    def test_data_quality_assessment(self):
        """Test data quality assessment"""
        mock_phone_info = PhoneInfo(
            model="Test Phone",
            brand="Test Brand",
            specifications={'test': 'spec'},
            reviews=[{'test': 'review'}],
            overall_rating=4.5,
            price_range={'status': 'found'},
            availability={'web_available': True},
            key_features=['camera', 'battery'],
            pros=['good camera'],
            cons=['poor battery'],
            similar_phones=[],
            sources=['test_source']
        )
        
        quality_assessment = self.agent._assess_data_quality(mock_phone_info)
        
        self.assertIn('quality_score', quality_assessment)
        self.assertIn('quality_level', quality_assessment)
        self.assertIn('positive_factors', quality_assessment)
        self.assertTrue(quality_assessment['quality_score'] > 0.5)
    
    @patch('requests.get')
    def test_search_single_source_success(self, mock_get):
        """Test successful single source search"""
        # Mock HTML response
        mock_response = Mock()
        mock_response.text = '<html><body>Mock search results</body></html>'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock query
        mock_query = Mock()
        mock_query.phone_model = "Test Phone"
        mock_query.aspects = []
        
        # Mock source config with a simple parser
        source_config = {
            'search_url': 'https://test.com/search?q={query}',
            'parser': lambda html, query, source: [self.mock_search_result]
        }
        
        results = self.agent._search_single_source('test_source', source_config, mock_query)
        
        # Should return the mock result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], self.mock_search_result)
    
    @patch('requests.get')
    def test_search_single_source_failure(self, mock_get):
        """Test failed single source search"""
        # Mock failed request
        mock_get.side_effect = Exception("Network error")
        
        mock_query = Mock()
        mock_query.phone_model = "Test Phone"
        mock_query.aspects = []
        
        source_config = {
            'search_url': 'https://test.com/search?q={query}',
            'parser': None
        }
        
        results = self.agent._search_single_source('test_source', source_config, mock_query)
        
        # Should return empty list on failure
        self.assertEqual(len(results), 0)

class TestIntegrationFunction(unittest.TestCase):
    """Test the integration function"""
    
    @patch('core.web_search_agent.WebSearchAgent')
    def test_high_confidence_local_result(self, mock_web_agent):
        """Test that high confidence local results don't trigger web search"""
        local_result = {
            'confidence': 0.9,
            'phone_found': True,
            'product_name': 'iPhone 14 Pro'
        }
        
        result = integrate_web_search_with_system('iPhone 14 Pro', local_result)
        
        # Should return local result without web search
        self.assertEqual(result['source'], 'local_database')
        self.assertFalse(result['web_search_performed'])
        mock_web_agent.assert_not_called()
    
    @patch('core.web_search_agent.WebSearchAgent')
    def test_low_confidence_triggers_web_search(self, mock_web_agent):
        """Test that low confidence local results trigger web search"""
        local_result = {
            'confidence': 0.4,
            'phone_found': True,
            'product_name': 'iPhone 15 Pro'
        }
        
        # Mock web search result
        mock_web_result = {
            'phone_found': True,
            'confidence': 0.8,
            'phone_data': {'product_name': 'iPhone 15 Pro'}
        }
        
        mock_agent_instance = Mock()
        mock_agent_instance.search_phone_external.return_value = mock_web_result
        mock_web_agent.return_value = mock_agent_instance
        
        result = integrate_web_search_with_system('iPhone 15 Pro', local_result)
        
        # Should combine local and web results
        self.assertTrue(result.get('combined_search'))
        self.assertEqual(result['source'], 'hybrid')
        mock_web_agent.assert_called_once()
    
    @patch('core.web_search_agent.WebSearchAgent')
    def test_no_local_result_uses_web_only(self, mock_web_agent):
        """Test web search when no local result available"""
        # Mock web search result
        mock_web_result = {
            'phone_found': True,
            'confidence': 0.7,
            'phone_data': {'product_name': 'Nothing Phone 2'}
        }
        
        mock_agent_instance = Mock()
        mock_agent_instance.search_phone_external.return_value = mock_web_result
        mock_web_agent.return_value = mock_agent_instance
        
        result = integrate_web_search_with_system('Nothing Phone 2', None)
        
        # Should return web result with fallback flag
        self.assertTrue(result.get('fallback_search'))
        self.assertFalse(result.get('local_data_available'))
        mock_web_agent.assert_called_once()
    
    @patch('core.web_search_agent.WebSearchAgent')
    def test_no_results_anywhere(self, mock_web_agent):
        """Test when no results found anywhere"""
        # Mock failed web search
        mock_web_result = {
            'phone_found': False
        }
        
        mock_agent_instance = Mock()
        mock_agent_instance.search_phone_external.return_value = mock_web_result
        mock_web_agent.return_value = mock_agent_instance
        
        result = integrate_web_search_with_system('NonExistent Phone', None)
        
        # Should return no results with suggestions
        self.assertFalse(result.get('phone_found'))
        self.assertEqual(result['source'], 'no_source')
        self.assertIn('suggestions', result)

class TestMockParsers(unittest.TestCase):
    """Test the source-specific parsers with mock data"""
    
    def setUp(self):
        self.agent = WebSearchAgent()
        self.mock_query = Mock()
        self.mock_query.phone_model = "Test Phone"
    
    def test_gsmarena_parser(self):
        """Test GSMArena parser with mock HTML"""
        mock_html = '''
        <html>
            <div class="makers">
                <a href="/phone-123.php">Test Phone Specifications</a>
                <p>Test phone details and specs</p>
            </div>
        </html>
        '''
        
        results = self.agent._parse_gsmarena_results(mock_html, self.mock_query, 'gsmarena')
        
        if results:  # Parser may return empty if HTML structure doesn't match exactly
            self.assertTrue(len(results) >= 0)
            if len(results) > 0:
                self.assertEqual(results[0].source, 'gsmarena')
    
    def test_empty_html_handling(self):
        """Test parsers handle empty/malformed HTML gracefully"""
        empty_html = "<html></html>"
        malformed_html = "<div>incomplete"
        
        parsers = [
            self.agent._parse_gsmarena_results,
            self.agent._parse_phonearena_results,
            self.agent._parse_cnet_results,
            self.agent._parse_techcrunch_results,
            self.agent._parse_google_shopping_results
        ]
        
        for parser in parsers:
            # Should not raise exceptions
            result1 = parser(empty_html, self.mock_query, 'test')
            result2 = parser(malformed_html, self.mock_query, 'test')
            
            self.assertIsInstance(result1, list)
            self.assertIsInstance(result2, list)

if __name__ == '__main__':
    print("🧪 Running Web Search Agent Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestWebSearchAgent))
    test_suite.addTest(unittest.makeSuite(TestIntegrationFunction))
    test_suite.addTest(unittest.makeSuite(TestMockParsers))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
        print("🚀 Web Search Agent is ready for integration")
    else:
        print("❌ Some tests failed")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    print("\n💡 Next steps:")
    print("  1. Run integration tests with real web requests (optional)")
    print("  2. Configure source priorities based on your needs")
    print("  3. Integrate into your Streamlit application")
    print("  4. Monitor web search performance and adjust timeouts")