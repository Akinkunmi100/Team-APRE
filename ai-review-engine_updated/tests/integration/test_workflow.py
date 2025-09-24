"""
Integration tests for complete workflows
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app
from web.app import app as web_app
from worker.tasks import process_review

@pytest.fixture
def api_client():
    return TestClient(app)

@pytest.fixture
def web_client(web_app):
    return web_app.test_client()

@pytest.mark.integration
def test_complete_review_workflow(api_client, web_client, db_session, redis_client):
    """Test the complete review workflow from submission to processing."""
    
    # 1. User registration
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "securepass123"
    }
    
    response = web_client.post("/auth/register", json=user_data)
    assert response.status_code == 201
    user_id = response.json()["id"]
    
    # 2. User login
    login_data = {
        "username": user_data["email"],
        "password": user_data["password"]
    }
    
    response = web_client.post("/auth/login", json=login_data)
    assert response.status_code == 200
    token = response.json()["access_token"]
    
    # 3. Submit review
    review_data = {
        "product_id": "12345",
        "rating": 5,
        "text": "This is an excellent product! Really satisfied with the quality."
    }
    
    response = api_client.post(
        "/api/v1/reviews",
        json=review_data,
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 201
    review_id = response.json()["id"]
    
    # 4. Verify review processing
    task_result = process_review.delay(review_id)
    processed_review = task_result.get(timeout=10)
    
    assert processed_review["sentiment"] == "positive"
    assert processed_review["spam_probability"] < 0.1
    
    # 5. Check review in database
    db_review = db_session.query(Review).filter_by(id=review_id).first()
    assert db_review is not None
    assert db_review.processed == True
    assert db_review.sentiment == "positive"
    
    # 6. Verify review appears in user's history
    response = web_client.get(
        f"/users/{user_id}/reviews",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    reviews = response.json()
    assert len(reviews) == 1
    assert reviews[0]["id"] == review_id
    
    # 7. Check review analytics
    response = api_client.get(
        f"/api/v1/analytics/reviews",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    analytics = response.json()
    assert analytics["total_reviews"] == 1
    assert analytics["average_rating"] == 5.0
    
    # 8. Test review update
    update_data = {
        "rating": 4,
        "text": "Updated: Still a great product, but could be better."
    }
    
    response = api_client.put(
        f"/api/v1/reviews/{review_id}",
        json=update_data,
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    
    # 9. Verify review cache is updated
    cached_review = redis_client.get(f"review:{review_id}")
    assert cached_review is not None
    
    # 10. Test review deletion
    response = api_client.delete(
        f"/api/v1/reviews/{review_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 204
    
    # Verify review is deleted
    db_review = db_session.query(Review).filter_by(id=review_id).first()
    assert db_review is None
    
    # Verify cache is cleared
    cached_review = redis_client.get(f"review:{review_id}")
    assert cached_review is None

@pytest.mark.integration
def test_concurrent_review_processing(api_client, db_session, redis_client):
    """Test processing multiple reviews concurrently."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    # Create multiple reviews
    review_data_list = [
        {
            "product_id": "12345",
            "rating": 5,
            "text": f"Review number {i}"
        }
        for i in range(10)
    ]
    
    async def process_reviews():
        review_ids = []
        
        # Submit reviews concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for review_data in review_data_list:
                future = executor.submit(
                    api_client.post,
                    "/api/v1/reviews",
                    json=review_data
                )
                futures.append(future)
            
            for future in futures:
                response = future.result()
                assert response.status_code == 201
                review_ids.append(response.json()["id"])
        
        # Process reviews concurrently
        tasks = []
        for review_id in review_ids:
            task = process_review.delay(review_id)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = [task.get(timeout=30) for task in tasks]
        
        # Verify all reviews were processed
        for result in results:
            assert "sentiment" in result
            assert "spam_probability" in result
        
        return review_ids
    
    review_ids = asyncio.run(process_reviews())
    
    # Verify all reviews in database
    reviews = db_session.query(Review).filter(Review.id.in_(review_ids)).all()
    assert len(reviews) == len(review_data_list)
    
    # Verify all reviews are cached
    for review_id in review_ids:
        cached_review = redis_client.get(f"review:{review_id}")
        assert cached_review is not None

@pytest.mark.integration
def test_error_handling(api_client, db_session, redis_client):
    """Test error handling and recovery in the review workflow."""
    
    # 1. Test invalid review data
    invalid_data = {
        "product_id": "12345",
        "rating": 10,  # Invalid rating
        "text": ""     # Empty text
    }
    
    response = api_client.post("/api/v1/reviews", json=invalid_data)
    assert response.status_code == 422
    
    # 2. Test database transaction rollback
    original_review_count = db_session.query(Review).count()
    
    try:
        # Simulate database error
        with db_session.begin():
            review = Review(
                product_id="12345",
                rating=5,
                text="Test review"
            )
            db_session.add(review)
            raise Exception("Simulated database error")
    except:
        pass
    
    current_review_count = db_session.query(Review).count()
    assert current_review_count == original_review_count
    
    # 3. Test cache recovery
    redis_client.flushall()
    
    # Submit a review
    review_data = {
        "product_id": "12345",
        "rating": 5,
        "text": "Test review"
    }
    
    response = api_client.post("/api/v1/reviews", json=review_data)
    assert response.status_code == 201
    review_id = response.json()["id"]
    
    # Verify cache is populated
    cached_review = redis_client.get(f"review:{review_id}")
    assert cached_review is not None