"""
Unit tests for the API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock

from api.main import app
from api.models import User, Review
from api.database import get_db

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.parametrize("endpoint", [
    "/api/v1/reviews",
    "/api/v1/users",
    "/api/v1/analytics"
])
def test_endpoints_require_authentication(client, endpoint):
    """Test that endpoints require authentication."""
    response = client.get(endpoint)
    assert response.status_code == 401

def test_create_review(client, db_session, mock_redis):
    """Test creating a new review."""
    # Create test user
    user = User(
        username="testuser",
        email="test@example.com"
    )
    db_session.add(user)
    db_session.commit()
    
    # Create authentication token
    token = create_test_token(user)
    
    review_data = {
        "product_id": "12345",
        "rating": 5,
        "text": "Great product!",
        "sentiment": "positive"
    }
    
    response = client.post(
        "/api/v1/reviews",
        json=review_data,
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["rating"] == 5
    assert data["text"] == "Great product!"

def test_get_review(client, db_session):
    """Test retrieving a review."""
    # Create test review
    review = Review(
        product_id="12345",
        user_id=1,
        rating=5,
        text="Great product!",
        sentiment="positive"
    )
    db_session.add(review)
    db_session.commit()
    
    response = client.get(f"/api/v1/reviews/{review.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == review.id
    assert data["rating"] == 5

def test_update_review(client, db_session):
    """Test updating a review."""
    # Create test review
    review = Review(
        product_id="12345",
        user_id=1,
        rating=3,
        text="Okay product",
        sentiment="neutral"
    )
    db_session.add(review)
    db_session.commit()
    
    update_data = {
        "rating": 4,
        "text": "Better than I thought!"
    }
    
    response = client.put(
        f"/api/v1/reviews/{review.id}",
        json=update_data
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["rating"] == 4
    assert data["text"] == "Better than I thought!"

def test_delete_review(client, db_session):
    """Test deleting a review."""
    # Create test review
    review = Review(
        product_id="12345",
        user_id=1,
        rating=1,
        text="Bad product",
        sentiment="negative"
    )
    db_session.add(review)
    db_session.commit()
    
    response = client.delete(f"/api/v1/reviews/{review.id}")
    assert response.status_code == 204
    
    # Verify review was deleted
    deleted_review = db_session.query(Review).filter_by(id=review.id).first()
    assert deleted_review is None

def test_review_validation(client):
    """Test review input validation."""
    invalid_data = {
        "product_id": "12345",
        "rating": 6,  # Invalid rating (should be 1-5)
        "text": ""    # Empty text
    }
    
    response = client.post("/api/v1/reviews", json=invalid_data)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(error["field"] == "rating" for error in errors)
    assert any(error["field"] == "text" for error in errors)

@pytest.mark.parametrize("rating,sentiment", [
    (5, "positive"),
    (4, "positive"),
    (3, "neutral"),
    (2, "negative"),
    (1, "negative")
])
def test_sentiment_analysis(client, db_session, rating, sentiment):
    """Test sentiment analysis for different ratings."""
    review_data = {
        "product_id": "12345",
        "rating": rating,
        "text": "Test review"
    }
    
    response = client.post("/api/v1/reviews", json=review_data)
    assert response.status_code == 201
    data = response.json()
    assert data["sentiment"] == sentiment

def test_cache_middleware(client, mock_redis):
    """Test that caching middleware works."""
    mock_redis.get.return_value = None
    
    # First request - should cache
    response1 = client.get("/api/v1/products/12345")
    assert response1.status_code == 200
    mock_redis.set.assert_called_once()
    
    # Second request - should use cache
    mock_redis.get.return_value = response1.content
    response2 = client.get("/api/v1/products/12345")
    assert response2.status_code == 200
    assert response2.content == response1.content

def test_rate_limiting(client):
    """Test rate limiting middleware."""
    # Make multiple requests quickly
    for _ in range(5):
        response = client.get("/api/v1/products/12345")
        assert response.status_code == 200
    
    # Next request should be rate limited
    response = client.get("/api/v1/products/12345")
    assert response.status_code == 429

# Helper functions
def create_test_token(user: User) -> str:
    """Create a test JWT token."""
    from api.auth import create_access_token
    return create_access_token({"sub": user.email})