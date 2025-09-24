# 🌐 Web Search Agent

**AI Phone Review Engine - External Web Search Capability**

The Web Search Agent extends your AI Phone Review Engine to search external web sources when phones are not available in your local dataset. This enables comprehensive coverage of the entire phone market, including the latest releases and regional models.

## 🎯 **Purpose**

- **Fill Data Gaps**: Find phones not in your local database
- **Real-time Information**: Get latest reviews and pricing
- **Comprehensive Coverage**: Access multiple authoritative sources
- **Intelligent Fallback**: Seamlessly integrate with existing system
- **Quality Analysis**: Assess and score web-sourced data

## 🏗️ **Architecture Overview**

```
User Query → Smart Search Parser → Local Database Check
                                          ↓
                                   [Not Found/Low Confidence]
                                          ↓
                          Multi-Source Web Search Agent
                                          ↓
              ┌─────────────┬─────────────┬─────────────┬─────────────┐
              │  GSMArena   │ PhoneArena  │    CNET     │ TechCrunch  │
              │ (Specs)     │ (Reviews)   │(Professional)│(Tech News)  │
              └─────────────┴─────────────┴─────────────┴─────────────┘
                                          ↓
                        Results Aggregation & Analysis
                                          ↓
                      Format for System Integration
                                          ↓
                        Return Enhanced Results
```

## ⚡ **Key Features**

### **🔍 Multi-Source Search**
- **GSMArena**: Detailed specifications and technical data
- **PhoneArena**: User reviews and comparisons
- **CNET**: Professional reviews and ratings
- **TechCrunch**: Latest tech news and announcements
- **Google Shopping**: Pricing information

### **🧠 Intelligent Processing**
- **Natural Language Query Parsing**: Uses existing SmartPhoneSearch
- **Concurrent Search Execution**: Fast multi-source searches
- **Smart Result Aggregation**: Combines data from multiple sources
- **Quality Assessment**: Scores data reliability and completeness

### **🔄 Seamless Integration**
- **Hybrid Search**: Combines local and web data when beneficial
- **Intelligent Fallback**: Only searches web when needed
- **Confidence-Based Decisions**: Uses local data when confidence is high
- **Standardized Output**: Compatible with existing system formats

## 📋 **Quick Start**

### **Basic Usage**

```python
from core.web_search_agent import WebSearchAgent

# Initialize the agent
web_agent = WebSearchAgent()

# Search for a phone
result = web_agent.search_phone_external("iPhone 15 Pro Max")

if result['phone_found']:
    phone_data = result['phone_data']
    print(f"Found: {phone_data['product_name']}")
    print(f"Rating: {phone_data['overall_rating']}")
    print(f"Sources: {phone_data['web_sources']}")
else:
    print("Phone not found")
```

### **Integrated Search (Recommended)**

```python
from core.web_search_agent import integrate_web_search_with_system

# Your existing local search
local_result = search_local_database("Nothing Phone 2")

# Enhanced search with web fallback
final_result = integrate_web_search_with_system("Nothing Phone 2", local_result)

# Handle different result types
if final_result.get('combined_search'):
    print("📊 Combined local and web data")
elif final_result.get('fallback_search'):
    print("🌐 Web search used (not in local database)")
elif final_result.get('source') == 'local_database':
    print("💾 Local database sufficient")
```

### **Streamlit Integration**

```python
import streamlit as st
from core.web_search_agent import integrate_web_search_with_system

def enhanced_phone_search(query):
    """Enhanced search with web fallback"""
    
    # Try local first
    local_result = your_existing_search(query)
    
    # Use web search if needed
    with st.spinner("🔍 Searching..."):
        result = integrate_web_search_with_system(query, local_result)
    
    # Display based on source
    if result.get('combined_search'):
        st.info("📊 Showing combined data from database and web")
        display_combined_results(result)
    elif result.get('fallback_search'):
        st.warning("🌐 Phone not in database. Showing web results")
        display_web_results(result)
    else:
        st.success("💾 Showing local database results")
        display_local_results(result)

# In your Streamlit app
if st.button("Search Phone"):
    enhanced_phone_search(user_query)
```

## ⚙️ **Configuration**

### **Basic Configuration**

```python
custom_config = {
    'max_concurrent_searches': 3,    # Number of sources to search simultaneously
    'search_timeout': 30,            # Maximum time for all searches
    'max_results_per_source': 5,     # Results to collect from each source
    'min_confidence_threshold': 0.6, # Minimum confidence to return results
    'rate_limit_delay': 2.0,         # Delay between requests (seconds)
    'cache_expiry': 3600             # Cache expiry time (seconds)
}

web_agent = WebSearchAgent(config=custom_config)
```

### **Source Configuration**

```python
# Enable/disable specific sources
web_agent.search_sources['gsmarena']['enabled'] = True
web_agent.search_sources['google_shopping']['enabled'] = False

# Adjust source priorities (lower = higher priority)
web_agent.search_sources['cnet']['priority'] = 1  # Make CNET highest priority
```

## 📊 **Output Format**

The web search agent returns data in a format compatible with your existing system:

```python
{
    'phone_found': True,
    'source': 'web_search',  # or 'hybrid', 'local_database', 'no_source'
    'search_query': 'iPhone 15 Pro',
    'confidence': 0.85,
    
    'phone_data': {
        'product_name': 'iPhone 15 Pro',
        'brand': 'Apple',
        'overall_rating': 4.5,
        'review_count': 150,
        'web_sources': ['gsmarena', 'cnet', 'phonearena'],
        'key_features': ['camera', 'performance', '5G'],
        'pros': ['Excellent camera system', 'Fast performance'],
        'cons': ['Expensive', 'Battery life could be better'],
        'price_info': {
            'status': 'found',
            'min_price': '$999',
            'max_price': '$1199',
            'avg_price': '$1099'
        }
    },
    
    'analysis': {
        'sentiment_analysis': {
            'positive_percentage': 75.0,
            'negative_percentage': 15.0,
            'neutral_percentage': 10.0,
            'overall_sentiment': 'positive',
            'confidence': 'medium'
        },
        'recommendation_score': 0.82,
        'data_quality': {
            'quality_score': 0.85,
            'quality_level': 'high',
            'positive_factors': ['Rating available', 'Multiple sources']
        }
    },
    
    'recommendations': {
        'should_buy': True,
        'confidence_level': 'medium',
        'key_considerations': ['Excellent camera', 'Fast performance', 'High price'],
        'next_steps': [
            'Check official website for latest pricing',
            'Read detailed reviews from multiple sources'
        ]
    }
}
```

## 🔧 **Integration Patterns**

### **Pattern 1: Simple Fallback**
```python
def search_phone(query):
    # Try local first
    local_result = search_local_database(query)
    
    if local_result and local_result.get('confidence', 0) > 0.8:
        return local_result
    
    # Use web search as fallback
    return web_agent.search_phone_external(query)
```

### **Pattern 2: Confidence-Based Enhancement**
```python
def enhanced_search(query):
    local_result = search_local_database(query)
    
    # Always get web data for comparison
    web_result = web_agent.search_phone_external(query)
    
    # Combine results intelligently
    return combine_local_and_web_data(local_result, web_result)
```

### **Pattern 3: User Choice**
```python
def user_controlled_search(query, search_web=False):
    local_result = search_local_database(query)
    
    if search_web or not local_result:
        return integrate_web_search_with_system(query, local_result)
    
    return local_result
```

## 📈 **Performance Considerations**

### **Optimization Tips**
- **Caching**: Results are cached for 1 hour by default
- **Concurrent Searches**: Uses ThreadPoolExecutor for parallel requests
- **Rate Limiting**: Built-in delays to respect source limits
- **Timeout Management**: Configurable timeouts prevent hanging
- **Graceful Degradation**: Continues with partial results if some sources fail

### **Monitoring**
```python
# Monitor search performance
search_stats = {
    'cache_hits': len([k for k in web_agent.search_cache.keys()]),
    'active_sources': len([s for s in web_agent.search_sources.values() if s['enabled']]),
    'last_search_time': 'tracked_internally'
}
```

## 🧪 **Testing**

### **Run Tests**
```bash
cd ai-review-engine_updated
python tests/test_web_search_agent.py
```

### **Run Examples**
```bash
python examples/web_search_integration_example.py
```

### **Manual Testing**
```python
# Test with various queries
test_queries = [
    "iPhone 15 Pro Max",
    "Samsung Galaxy S24 Ultra reviews", 
    "Nothing Phone 2 specs",
    "What do people think about Google Pixel 8?"
]

for query in test_queries:
    result = web_agent.search_phone_external(query)
    print(f"{query}: {'✅' if result['phone_found'] else '❌'}")
```

## 🚨 **Error Handling**

The web search agent includes comprehensive error handling:

- **Network Failures**: Continues with available sources
- **Parse Errors**: Logs errors but doesn't crash
- **Rate Limiting**: Respects delays and retries
- **Malformed HTML**: Graceful parsing failures
- **Empty Results**: Returns structured "not found" responses

## 🛡️ **Best Practices**

### **Production Deployment**
1. **Configure Timeouts**: Adjust based on your network
2. **Monitor Rate Limits**: Respect source limitations
3. **Cache Management**: Consider cache size and expiry
4. **Error Logging**: Enable comprehensive logging
5. **Source Selection**: Choose sources relevant to your users

### **Data Quality**
1. **Confidence Thresholds**: Set appropriate minimum confidence
2. **Source Prioritization**: Rank sources by reliability
3. **Result Validation**: Validate data before presenting
4. **User Feedback**: Allow users to report issues

## 🔄 **Integration with Your User-Friendly App**

Add this to your `user_friendly_app.py`:

```python
# Import at the top
from core.web_search_agent import integrate_web_search_with_system

# Modify your search function
def search_phone_with_web_fallback(query):
    """Enhanced phone search with web fallback"""
    
    # Your existing local search
    local_result = None
    if hasattr(st.session_state, 'df') and st.session_state.df is not None:
        # Search in local data
        local_result = search_in_local_data(query, st.session_state.df)
    
    # Use integrated search
    final_result = integrate_web_search_with_system(query, local_result)
    
    # Display results with source indication
    if final_result.get('combined_search'):
        st.info("📊 Combined data from local database and web sources")
    elif final_result.get('fallback_search'):
        st.warning("🌐 Phone not in local database. Showing web search results")
    elif final_result.get('source') == 'local_database':
        st.success("💾 Data from local database")
    
    return final_result
```

## 🚀 **Next Steps**

1. **Test the Implementation**: Run the test suite and examples
2. **Configure Sources**: Enable/disable sources based on your needs
3. **Integrate into Apps**: Add to your Streamlit applications
4. **Monitor Performance**: Track success rates and response times
5. **Gather Feedback**: Collect user feedback on web search results
6. **Optimize Configuration**: Adjust timeouts and limits based on usage

## 📞 **Support**

- **Test Issues**: Run `tests/test_web_search_agent.py`
- **Integration Help**: See `examples/web_search_integration_example.py`
- **Performance Issues**: Adjust configuration parameters
- **Source Problems**: Disable problematic sources temporarily

---

**The Web Search Agent seamlessly extends your AI Phone Review Engine to cover the entire smartphone market! 🚀**