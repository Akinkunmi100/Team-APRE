# AI Review Engine API Examples

This document provides examples for common API operations using various tools and programming languages.

## Authentication

### Register a New User

```bash
# Using curl
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "securepass123"
  }'
```

```python
# Using Python requests
import requests

response = requests.post(
    "http://localhost:8000/api/v1/auth/register",
    json={
        "username": "testuser",
        "email": "test@example.com",
        "password": "securepass123"
    }
)
print(response.json())
```

### Login

```bash
# Using curl
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "securepass123"
  }'
```

```python
# Using Python requests
response = requests.post(
    "http://localhost:8000/api/v1/auth/login",
    json={
        "email": "test@example.com",
        "password": "securepass123"
    }
)
token = response.json()["access_token"]
```

## Reviews

### List Reviews

```bash
# Using curl
curl -X GET http://localhost:8000/api/v1/reviews \
  -H "Authorization: Bearer your_token_here" \
  -H "Content-Type: application/json"

# With filters
curl -X GET "http://localhost:8000/api/v1/reviews?page=1&per_page=20&sentiment=positive" \
  -H "Authorization: Bearer your_token_here"
```

```python
# Using Python requests
headers = {"Authorization": f"Bearer {token}"}
response = requests.get(
    "http://localhost:8000/api/v1/reviews",
    headers=headers,
    params={
        "page": 1,
        "per_page": 20,
        "sentiment": "positive"
    }
)
reviews = response.json()
```

### Create a Review

```bash
# Using curl
curl -X POST http://localhost:8000/api/v1/reviews \
  -H "Authorization: Bearer your_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "12345",
    "rating": 5,
    "text": "Great product! Very satisfied with the quality."
  }'
```

```python
# Using Python requests
response = requests.post(
    "http://localhost:8000/api/v1/reviews",
    headers=headers,
    json={
        "product_id": "12345",
        "rating": 5,
        "text": "Great product! Very satisfied with the quality."
    }
)
new_review = response.json()
```

### Update a Review

```bash
# Using curl
curl -X PUT http://localhost:8000/api/v1/reviews/review_id_here \
  -H "Authorization: Bearer your_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "rating": 4,
    "text": "Updated review text here"
  }'
```

```python
# Using Python requests
response = requests.put(
    f"http://localhost:8000/api/v1/reviews/{review_id}",
    headers=headers,
    json={
        "rating": 4,
        "text": "Updated review text here"
    }
)
updated_review = response.json()
```

### Delete a Review

```bash
# Using curl
curl -X DELETE http://localhost:8000/api/v1/reviews/review_id_here \
  -H "Authorization: Bearer your_token_here"
```

```python
# Using Python requests
response = requests.delete(
    f"http://localhost:8000/api/v1/reviews/{review_id}",
    headers=headers
)
```

## Analytics

### Get Review Analytics

```bash
# Using curl
curl -X GET "http://localhost:8000/api/v1/analytics/reviews?start_date=2025-01-01&end_date=2025-12-31" \
  -H "Authorization: Bearer your_token_here"
```

```python
# Using Python requests
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=30)

response = requests.get(
    "http://localhost:8000/api/v1/analytics/reviews",
    headers=headers,
    params={
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d")
    }
)
analytics = response.json()
```

## System Health

### Check System Health

```bash
# Using curl
curl -X GET http://localhost:8000/api/v1/health
```

```python
# Using Python requests
response = requests.get("http://localhost:8000/api/v1/health")
health_status = response.json()
```

## Error Handling

### Common Error Responses

1. **Unauthorized Access (401)**
```json
{
    "code": 401,
    "message": "Invalid or expired token",
    "details": {
        "error": "token_invalid"
    }
}
```

2. **Invalid Input (400)**
```json
{
    "code": 400,
    "message": "Invalid input",
    "details": {
        "rating": ["Rating must be between 1 and 5"],
        "text": ["Review text must be at least 10 characters long"]
    }
}
```

3. **Resource Not Found (404)**
```json
{
    "code": 404,
    "message": "Review not found",
    "details": {
        "review_id": "requested_id_here"
    }
}
```

### Error Handling in Code

```python
# Using Python requests
try:
    response = requests.get(
        f"http://localhost:8000/api/v1/reviews/{review_id}",
        headers=headers
    )
    response.raise_for_status()  # Raises exception for 4XX/5XX status codes
    review = response.json()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
        print("Review not found")
    elif e.response.status_code == 401:
        print("Authentication failed")
    else:
        print(f"Request failed: {e.response.json()['message']}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {str(e)}")
```

## Best Practices

1. **Authentication**
   - Always store tokens securely
   - Refresh tokens before they expire
   - Include tokens in Authorization header

2. **Error Handling**
   - Always handle potential errors
   - Check response status codes
   - Parse error messages for details

3. **Rate Limiting**
   - Implement exponential backoff for retries
   - Respect rate limit headers
   - Cache responses when appropriate

4. **Request Optimization**
   - Use pagination for large datasets
   - Include only necessary fields
   - Batch operations when possible

## SDK Example

```python
# Example SDK class for the API
class AIReviewEngineClient:
    def __init__(self, base_url, api_key=None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def authenticate(self, email, password):
        response = self.session.post(
            f"{self.base_url}/auth/login",
            json={"email": email, "password": password}
        )
        response.raise_for_status()
        token = response.json()["access_token"]
        self.session.headers.update({"Authorization": f"Bearer {token}"})
        return token
    
    def list_reviews(self, page=1, per_page=20, **filters):
        response = self.session.get(
            f"{self.base_url}/reviews",
            params={"page": page, "per_page": per_page, **filters}
        )
        response.raise_for_status()
        return response.json()
    
    def create_review(self, product_id, rating, text):
        response = self.session.post(
            f"{self.base_url}/reviews",
            json={
                "product_id": product_id,
                "rating": rating,
                "text": text
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_analytics(self, start_date=None, end_date=None):
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
            
        response = self.session.get(
            f"{self.base_url}/analytics/reviews",
            params=params
        )
        response.raise_for_status()
        return response.json()

# Usage example
client = AIReviewEngineClient("http://localhost:8000/api/v1")
client.authenticate("test@example.com", "securepass123")

# List reviews
reviews = client.list_reviews(sentiment="positive")

# Create review
new_review = client.create_review(
    product_id="12345",
    rating=5,
    text="Excellent product!"
)

# Get analytics
analytics = client.get_analytics(
    start_date="2025-01-01",
    end_date="2025-12-31"
)
```