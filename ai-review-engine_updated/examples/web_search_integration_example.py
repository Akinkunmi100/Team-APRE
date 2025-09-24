"""
Web Search Agent Integration Example
Demonstrates how to integrate the web search agent into the AI Phone Review Engine
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.web_search_agent import WebSearchAgent, integrate_web_search_with_system
import json

def example_basic_web_search():
    """Basic example of using the web search agent"""
    
    print("🔍 Basic Web Search Agent Example")
    print("=" * 50)
    
    # Initialize the web search agent
    web_agent = WebSearchAgent()
    
    # Test queries
    test_queries = [
        "iPhone 15 Pro Max",
        "Samsung Galaxy S24 Ultra reviews",
        "What do people think about Google Pixel 8?",
        "OnePlus 12 camera quality",
        "Nothing Phone 2 specs and price"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Searching for: '{query}'")
        print("-" * 30)
        
        try:
            # Perform web search
            result = web_agent.search_phone_external(query, max_sources=2)
            
            if result.get('phone_found'):
                phone_data = result['phone_data']
                print(f"✅ Found: {phone_data['product_name']}")
                print(f"📊 Rating: {phone_data.get('overall_rating', 'N/A')}")
                print(f"💰 Price: {phone_data.get('price_info', {}).get('avg_price', 'N/A')}")
                print(f"📱 Sources: {', '.join(phone_data.get('web_sources', []))}")
                print(f"⭐ Features: {', '.join(phone_data.get('key_features', [])[:3])}")
            else:
                print("❌ Phone not found")
                
        except Exception as e:
            print(f"⚠️ Error: {str(e)}")
        
        print()

def example_integrated_search():
    """Example of integrated search (local + web)"""
    
    print("🔄 Integrated Search Example (Local + Web)")
    print("=" * 50)
    
    # Simulate local search results
    test_cases = [
        # Case 1: High confidence local result (should not trigger web search)
        {
            'query': 'iPhone 14 Pro',
            'local_result': {
                'confidence': 0.9,
                'phone_found': True,
                'product_name': 'iPhone 14 Pro',
                'rating': 4.5,
                'source': 'local_database'
            }
        },
        
        # Case 2: Low confidence local result (should trigger web search)
        {
            'query': 'iPhone 15 Pro Max',
            'local_result': {
                'confidence': 0.4,
                'phone_found': True,
                'product_name': 'iPhone 15 Pro Max',
                'rating': 4.2,
                'source': 'local_database'
            }
        },
        
        # Case 3: No local result (web search only)
        {
            'query': 'Nothing Phone 3',
            'local_result': None
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📱 Case {i}: {case['query']}")
        print("-" * 30)
        
        try:
            # Use the integration function
            result = integrate_web_search_with_system(
                case['query'], 
                case['local_result']
            )
            
            print(f"🎯 Result Type: {result.get('source', 'unknown')}")
            
            if result.get('combined_search'):
                print("🔄 Combined local and web data")
                print(f"   Local confidence: {result['local_data'].get('confidence', 'N/A')}")
                print(f"   Web confidence: {result['web_data'].get('confidence', 'N/A')}")
                
            elif result.get('fallback_search'):
                print("🌐 Web search fallback used")
                print(f"   Web sources: {', '.join(result.get('phone_data', {}).get('web_sources', []))}")
                
            elif result.get('web_search_performed') == False:
                print("💾 Local data sufficient")
                print(f"   Confidence: {result.get('confidence', 'N/A')}")
                
            else:
                print("❌ No data found")
                
        except Exception as e:
            print(f"⚠️ Error: {str(e)}")

def example_streamlit_integration():
    """Example of how to integrate into Streamlit app"""
    
    print("📱 Streamlit Integration Pattern")
    print("=" * 50)
    
    # This is the pattern you would use in your user_friendly_app.py
    integration_code = '''
def enhanced_phone_search(query: str):
    """Enhanced phone search with web fallback"""
    
    # Step 1: Try local database search first
    local_result = search_local_database(query)  # Your existing function
    
    # Step 2: Use web search if needed
    final_result = integrate_web_search_with_system(query, local_result)
    
    # Step 3: Display results based on source
    if final_result.get('combined_search'):
        st.info("📊 Showing combined data from database and web sources")
        display_combined_results(final_result)
        
    elif final_result.get('fallback_search'):
        st.warning("🌐 Phone not in database. Showing web search results")
        display_web_results(final_result)
        
    elif final_result.get('source') == 'local_database':
        st.success("💾 Showing data from local database")
        display_local_results(final_result)
        
    else:
        st.error("❌ Phone not found anywhere")
        display_search_suggestions(final_result.get('suggestions', []))

# Usage in your Streamlit app:
if st.button("Search Phone"):
    enhanced_phone_search(user_query)
    '''
    
    print("Integration code pattern:")
    print(integration_code)

def example_configuration():
    """Example of custom configuration"""
    
    print("⚙️ Custom Configuration Example")
    print("=" * 50)
    
    # Custom configuration for the web search agent
    custom_config = {
        'max_concurrent_searches': 2,  # Reduce for slower connections
        'search_timeout': 20,          # Reduce timeout
        'max_results_per_source': 3,   # Fewer results per source
        'rate_limit_delay': 3.0,       # Longer delay between requests
    }
    
    # Initialize with custom config
    web_agent = WebSearchAgent(config=custom_config)
    
    # You can also disable specific sources
    web_agent.search_sources['google_shopping']['enabled'] = False  # Disable Google Shopping
    web_agent.search_sources['techcrunch']['enabled'] = False       # Disable TechCrunch
    
    print("Custom configuration applied:")
    print(json.dumps(custom_config, indent=2))
    print("\nDisabled sources: google_shopping, techcrunch")

if __name__ == "__main__":
    print("🤖 AI Phone Review Engine - Web Search Agent Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_web_search()
        print("\n" + "=" * 60)
        
        example_integrated_search()
        print("\n" + "=" * 60)
        
        example_streamlit_integration()
        print("\n" + "=" * 60)
        
        example_configuration()
        
    except KeyboardInterrupt:
        print("\n\n👋 Examples interrupted by user")
    except Exception as e:
        print(f"\n⚠️ Error running examples: {str(e)}")
    
    print("\n✅ Web Search Agent examples completed!")
    print("💡 Next steps:")
    print("  1. Test individual components")
    print("  2. Integrate into your main application")
    print("  3. Configure sources based on your needs")
    print("  4. Add error handling for production use")