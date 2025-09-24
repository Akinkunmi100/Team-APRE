# 🔌 Ultimate AI Phone Review Engine - API Reference

## 📖 Table of Contents
- [Authentication](#-authentication)
- [Rate Limits](#-rate-limits)
- [Core API Endpoints](#-core-api-endpoints)
- [Business API Endpoints](#-business-api-endpoints)
- [Error Handling](#-error-handling)
- [Code Examples](#-code-examples)

## 🔐 Authentication

All API endpoints require user authentication via session cookies. Authentication is handled through the web interface login process.

### Login Process
```http
POST /login
Content-Type: application/json

{
  "username": "business_user",
  "password": "business123"
}
```

**Response:**
```json
{
  "success": true,
  "redirect": "/dashboard",
  "user_plan": "business"
}
```

## ⏱️ Rate Limits

Rate limits are enforced based on user plan:

| Plan | Daily Searches | Monthly API Calls | Business Features |
|------|---------------|------------------|-------------------|
| Free | 20 | 0 | ❌ |
| Business | 200 | 1,000 | ✅ |
| Enterprise | 1,000 | 10,000 | ✅ |

### Rate Limit Headers
```http
X-RateLimit-Remaining: 15
X-RateLimit-Reset: 1695369600
```

## 🔍 Core API Endpoints

### Search Phones
Search across phone reviews and models.

```http
GET /api/search?q={query}&type={type}
```

**Parameters:**
- `q` (string, required): Search query
- `type` (string, optional): `single`, `bulk`, `competitor`

**Example Request:**
```bash
curl -X GET "http://localhost:5000/api/search?q=Samsung%20Galaxy%20S24" \
  -H "Content-Type: application/json" \
  --cookie-jar cookies.txt
```

**Response:**
```json
{
  "results": [
    {
      "product": "Samsung Galaxy S24",
      "brand": "Samsung",
      "review_text": "Amazing phone with great camera quality and battery life...",
      "rating": 4.5,
      "date": "2025-01-15",
      "source": "GSM Arena",
      "sentiment": {
        "label": "positive",
        "score": 0.8
      }
    }
  ],
  "total": 25,
  "execution_time": 0.45,
  "search_type": "single",
  "searches_left": 19,
  "user_plan": "business"
}
```

### Phone Analytics
Get detailed analytics for a specific phone model.

```http
GET /api/phone/{phone_name}/analytics
```

**Example Request:**
```bash
curl -X GET "http://localhost:5000/api/phone/Samsung%20Galaxy%20S24/analytics" \
  -H "Content-Type: application/json" \
  --cookie-jar cookies.txt
```

**Response:**
```json
{
  "phone_name": "Samsung Galaxy S24",
  "total_reviews": 156,
  "avg_rating": 4.3,
  "rating_distribution": {
    "5": 45,
    "4": 67,
    "3": 28,
    "2": 12,
    "1": 4
  },
  "sentiment": {
    "positive": 68.5,
    "negative": 12.8,
    "neutral": 18.7
  },
  "recommendation_score": 87.2,
  "key_features": [
    "Battery",
    "Camera", 
    "Display",
    "Performance"
  ],
  "pros_cons": {
    "pros": [
      "Long battery life",
      "Great camera quality", 
      "Smooth performance"
    ],
    "cons": [
      "High price",
      "Limited storage",
      "No headphone jack"
    ]
  },
  "recent_reviews": [
    {
      "product": "Samsung Galaxy S24",
      "review_text": "Great phone overall...",
      "rating": 4.0,
      "sentiment": {
        "label": "positive",
        "score": 0.7
      }
    }
  ]
}
```

## 🏢 Business API Endpoints

These endpoints require Business or Enterprise plan access.

### Competitor Analysis
Compare multiple phone models side-by-side.

```http
GET /api/business/competitor-analysis?models={model1}&models={model2}&models={model3}
```

**Example Request:**
```bash
curl -X GET "http://localhost:5000/api/business/competitor-analysis?models=iPhone%2015&models=Galaxy%20S24&models=Pixel%208" \
  -H "Content-Type: application/json" \
  --cookie-jar cookies.txt
```

**Response:**
```json
{
  "comparison_data": {
    "comparison": {
      "iPhone 15": {
        "phone_name": "iPhone 15",
        "total_reviews": 89,
        "avg_rating": 4.6,
        "sentiment": {
          "positive": 72.3,
          "negative": 8.9,
          "neutral": 18.8
        },
        "recommendation_score": 91.5
      },
      "Galaxy S24": {
        "phone_name": "Galaxy S24", 
        "total_reviews": 156,
        "avg_rating": 4.3,
        "sentiment": {
          "positive": 68.5,
          "negative": 12.8,
          "neutral": 18.7
        },
        "recommendation_score": 87.2
      },
      "Pixel 8": {
        "phone_name": "Pixel 8",
        "total_reviews": 67,
        "avg_rating": 4.1,
        "sentiment": {
          "positive": 65.7,
          "negative": 15.2,
          "neutral": 19.1
        },
        "recommendation_score": 83.4
      }
    },
    "summary": {
      "total_compared": 3,
      "winner": "iPhone 15",
      "best_rating": 4.6,
      "best_positive_sentiment": 72.3
    }
  },
  "generated_at": "2025-09-21T05:20:00Z",
  "models_compared": ["iPhone 15", "Galaxy S24", "Pixel 8"]
}
```

### Market Insights
Get comprehensive market intelligence and trends.

```http
GET /api/business/market-insights
```

**Response:**
```json
{
  "market_insights": {
    "total_reviews": 1456789,
    "total_brands": 62,
    "total_models": 241,
    "avg_rating": 3.8,
    "brand_performance": {
      "Apple": {
        "avg_rating": 4.2,
        "total_reviews": 234567,
        "models_count": 15
      },
      "Samsung": {
        "avg_rating": 4.0,
        "total_reviews": 345678,
        "models_count": 45
      }
    },
    "top_phones": {
      "iPhone 15 Pro Max": 4.7,
      "Galaxy S24 Ultra": 4.5,
      "Pixel 8 Pro": 4.3
    }
  },
  "generated_at": "2025-09-21T05:20:00Z",
  "data_freshness": "Real-time analysis"
}
```

### Usage Analytics
Track your search patterns and performance metrics.

```http
GET /api/business/usage-analytics
```

**Response:**
```json
{
  "total_searches": 45,
  "search_trends": {
    "2025-09-20": 12,
    "2025-09-21": 8
  },
  "search_types": {
    "single": 38,
    "bulk": 5,
    "competitor": 2
  },
  "top_queries": [
    {
      "query": "iPhone 15",
      "results": 23,
      "time": 0.34
    },
    {
      "query": "Samsung Galaxy S24",
      "results": 31,
      "time": 0.42
    }
  ],
  "avg_execution_time": 0.38
}
```

### Custom Reports
Generate custom business intelligence reports.

```http
GET /api/business/custom-report?type={report_type}&phones={phone_filter}&brands={brand_filter}
```

**Parameters:**
- `type`: `summary` or `detailed`
- `phones`: Comma-separated phone models to filter
- `brands`: Comma-separated brands to filter

**Example Request:**
```bash
curl -X GET "http://localhost:5000/api/business/custom-report?type=summary&brands=Apple,Samsung" \
  -H "Content-Type: application/json" \
  --cookie-jar cookies.txt
```

**Response:**
```json
{
  "report_type": "summary",
  "generated_at": "2025-09-21T05:20:00Z",
  "filters_applied": {
    "phones": [],
    "brands": ["Apple", "Samsung"]
  },
  "summary": {
    "total_reviews": 580245,
    "total_brands": 2,
    "total_models": 60,
    "avg_rating": 4.1,
    "brand_performance": {
      "Apple": {
        "avg_rating": 4.2,
        "total_reviews": 234567,
        "models_count": 15
      },
      "Samsung": {
        "avg_rating": 4.0,
        "total_reviews": 345678,
        "models_count": 45
      }
    }
  }
}
```

## ❌ Error Handling

### Error Response Format
```json
{
  "error": "Error description",
  "error_code": "ERROR_CODE",
  "upgrade_needed": false,
  "current_plan": "free"
}
```

### Common Error Codes

#### Rate Limit Exceeded (429)
```json
{
  "error": "Search limit exceeded",
  "searches_left": 0,
  "daily_limit": 20,
  "upgrade_needed": true
}
```

#### Business Feature Required (403)
```json
{
  "error": "Business plan required",
  "upgrade_needed": true,
  "current_plan": "free"
}
```

#### Authentication Required (401)
```json
{
  "error": "Authentication required",
  "login_url": "/login"
}
```

#### Not Found (404)
```json
{
  "error": "Phone not found in database",
  "searched_phone": "iPhone 20"
}
```

#### Validation Error (400)
```json
{
  "error": "Query parameter required",
  "missing_parameter": "q"
}
```

## 💻 Code Examples

### JavaScript (Frontend)
```javascript
// Basic search
async function searchPhones(query) {
    try {
        const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        const data = await response.json();
        
        if (data.error) {
            console.error('Search error:', data.error);
            return;
        }
        
        console.log(`Found ${data.total} results in ${data.execution_time}s`);
        return data.results;
    } catch (error) {
        console.error('Network error:', error);
    }
}

// Competitor analysis (Business+ only)
async function comparePhones(phoneModels) {
    const params = phoneModels.map(model => 
        `models=${encodeURIComponent(model)}`
    ).join('&');
    
    try {
        const response = await fetch(`/api/business/competitor-analysis?${params}`);
        const data = await response.json();
        
        if (data.error) {
            if (data.upgrade_needed) {
                alert('Business plan required for competitor analysis');
            }
            return;
        }
        
        console.log('Comparison winner:', data.comparison_data.summary.winner);
        return data.comparison_data;
    } catch (error) {
        console.error('Comparison error:', error);
    }
}

// Usage example
searchPhones('iPhone 15').then(results => {
    results.forEach(result => {
        console.log(`${result.product}: ${result.sentiment.label} sentiment`);
    });
});
```

### Python (Server-side)
```python
import requests
import json

class PhoneReviewAPI:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def login(self, username, password):
        """Login to get session cookie"""
        response = self.session.post(f"{self.base_url}/login", json={
            "username": username,
            "password": password
        })
        return response.json()
    
    def search_phones(self, query, search_type="single"):
        """Search for phones"""
        response = self.session.get(f"{self.base_url}/api/search", params={
            "q": query,
            "type": search_type
        })
        return response.json()
    
    def get_phone_analytics(self, phone_name):
        """Get analytics for specific phone"""
        response = self.session.get(
            f"{self.base_url}/api/phone/{phone_name}/analytics"
        )
        return response.json()
    
    def competitor_analysis(self, phone_models):
        """Compare multiple phones (Business+ only)"""
        params = [("models", model) for model in phone_models]
        response = self.session.get(
            f"{self.base_url}/api/business/competitor-analysis",
            params=params
        )
        return response.json()

# Usage example
api = PhoneReviewAPI()

# Login
login_result = api.login("business_user", "business123")
print(f"Logged in as {login_result['user_plan']} user")

# Search phones
results = api.search_phones("Samsung Galaxy S24")
print(f"Found {results['total']} results")

# Get analytics
analytics = api.get_phone_analytics("Samsung Galaxy S24")
print(f"Average rating: {analytics['avg_rating']}")
print(f"Recommendation score: {analytics['recommendation_score']}%")

# Compare phones (Business feature)
comparison = api.competitor_analysis(["iPhone 15", "Galaxy S24", "Pixel 8"])
print(f"Winner: {comparison['comparison_data']['summary']['winner']}")
```

### cURL Examples
```bash
# Login
curl -X POST "http://localhost:5000/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "business_user", "password": "business123"}' \
  -c cookies.txt

# Basic search
curl -X GET "http://localhost:5000/api/search?q=iPhone%2015" \
  -H "Content-Type: application/json" \
  -b cookies.txt

# Bulk search (Business+ only)
curl -X GET "http://localhost:5000/api/search?q=iPhone%2015,Galaxy%20S24,Pixel%208&type=bulk" \
  -H "Content-Type: application/json" \
  -b cookies.txt

# Phone analytics
curl -X GET "http://localhost:5000/api/phone/Samsung%20Galaxy%20S24/analytics" \
  -H "Content-Type: application/json" \
  -b cookies.txt

# Competitor analysis (Business+ only)
curl -X GET "http://localhost:5000/api/business/competitor-analysis?models=iPhone%2015&models=Galaxy%20S24" \
  -H "Content-Type: application/json" \
  -b cookies.txt

# Usage analytics (Business+ only)
curl -X GET "http://localhost:5000/api/business/usage-analytics" \
  -H "Content-Type: application/json" \
  -b cookies.txt
```

## 📊 Response Times & Performance

| Endpoint | Avg Response Time | Data Points |
|----------|------------------|-------------|
| `/api/search` | 0.3-0.8s | 1.4M+ reviews |
| `/api/phone/*/analytics` | 0.2-0.5s | Cached results |
| `/api/business/competitor-analysis` | 0.5-1.2s | Multiple phones |
| `/api/business/market-insights` | 0.4-0.9s | Full dataset |
| `/api/business/usage-analytics` | 0.1-0.3s | User history |

## 🔄 SDKs & Libraries

### JavaScript/TypeScript SDK
```typescript
interface SearchResult {
  product: string;
  brand: string;
  review_text: string;
  rating: number;
  sentiment: {
    label: 'positive' | 'negative' | 'neutral';
    score: number;
  };
}

class UltimatePhoneReviewSDK {
  constructor(private baseUrl: string = 'http://localhost:5000') {}
  
  async search(query: string): Promise<SearchResult[]> {
    const response = await fetch(`${this.baseUrl}/api/search?q=${query}`);
    const data = await response.json();
    return data.results;
  }
}
```

---

**API Version:** 1.0  
**Last Updated:** September 21, 2025  
**Base URL:** `http://localhost:5000/api`