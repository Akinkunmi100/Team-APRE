"""
Load testing configuration using Locust
"""
import json
import random
from locust import HttpUser, task, between
from faker import Faker

fake = Faker()

class ReviewUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup user before starting tasks."""
        # Register user
        user_data = {
            "username": fake.user_name(),
            "email": fake.email(),
            "password": "testpass123"
        }
        response = self.client.post("/auth/register", json=user_data)
        assert response.status_code in [201, 409]  # 409 if user exists
        
        # Login
        login_data = {
            "username": user_data["email"],
            "password": "testpass123"
        }
        response = self.client.post("/auth/login", json=login_data)
        assert response.status_code == 200
        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(3)
    def view_reviews(self):
        """Simulate viewing reviews."""
        self.client.get(
            "/api/v1/reviews",
            headers=self.headers,
            name="/api/v1/reviews - GET"
        )
    
    @task(2)
    def submit_review(self):
        """Simulate submitting a review."""
        review_data = {
            "product_id": str(random.randint(1000, 9999)),
            "rating": random.randint(1, 5),
            "text": fake.paragraph()
        }
        
        response = self.client.post(
            "/api/v1/reviews",
            json=review_data,
            headers=self.headers,
            name="/api/v1/reviews - POST"
        )
        
        if response.status_code == 201:
            self.review_ids = getattr(self, "review_ids", [])
            self.review_ids.append(response.json()["id"])
    
    @task(1)
    def update_review(self):
        """Simulate updating a review."""
        if hasattr(self, "review_ids") and self.review_ids:
            review_id = random.choice(self.review_ids)
            update_data = {
                "rating": random.randint(1, 5),
                "text": fake.paragraph()
            }
            
            self.client.put(
                f"/api/v1/reviews/{review_id}",
                json=update_data,
                headers=self.headers,
                name="/api/v1/reviews/{id} - PUT"
            )
    
    @task(1)
    def view_analytics(self):
        """Simulate viewing analytics."""
        self.client.get(
            "/api/v1/analytics/reviews",
            headers=self.headers,
            name="/api/v1/analytics/reviews - GET"
        )

class AdminUser(HttpUser):
    wait_time = between(2, 5)
    weight = 1  # Less frequent than regular users
    
    def on_start(self):
        """Setup admin user."""
        # Login as admin
        login_data = {
            "username": "admin@example.com",
            "password": "admin123"
        }
        response = self.client.post("/auth/login", json=login_data)
        assert response.status_code == 200
        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(2)
    def view_all_users(self):
        """Simulate viewing all users."""
        self.client.get(
            "/api/v1/users",
            headers=self.headers,
            name="/api/v1/users - GET"
        )
    
    @task(1)
    def view_system_metrics(self):
        """Simulate viewing system metrics."""
        self.client.get(
            "/api/v1/metrics",
            headers=self.headers,
            name="/api/v1/metrics - GET"
        )
    
    @task
    def generate_report(self):
        """Simulate generating a report."""
        report_config = {
            "report_type": "user_activity",
            "time_range": "last_7_days",
            "format": "pdf"
        }
        
        self.client.post(
            "/api/v1/reports/generate",
            json=report_config,
            headers=self.headers,
            name="/api/v1/reports/generate - POST"
        )

class SearchUser(HttpUser):
    wait_time = between(1, 2)
    
    def on_start(self):
        """Setup search user."""
        self.client.headers = {"Content-Type": "application/json"}
    
    @task(3)
    def search_products(self):
        """Simulate product search."""
        search_terms = [
            "phone",
            "laptop",
            "camera",
            "headphones",
            "smartwatch"
        ]
        term = random.choice(search_terms)
        
        self.client.get(
            f"/api/v1/search?q={term}",
            name="/api/v1/search - GET"
        )
    
    @task(1)
    def advanced_search(self):
        """Simulate advanced search."""
        search_params = {
            "q": random.choice(["phone", "laptop"]),
            "min_rating": random.randint(1, 5),
            "max_price": random.randint(500, 2000),
            "category": random.choice(["electronics", "accessories"]),
            "sort": random.choice(["price_asc", "rating_desc"])
        }
        
        self.client.get(
            "/api/v1/search/advanced",
            params=search_params,
            name="/api/v1/search/advanced - GET"
        )

# Load test configuration
class LoadTestConfig:
    """Load test configuration parameters."""
    
    TARGET_RPS = 50  # Target requests per second
    TEST_TIME = "30m"  # Test duration
    SPAWN_RATE = 10  # Users to spawn per second
    
    USER_DISTRIBUTION = {
        ReviewUser: 70,   # 70% regular users
        SearchUser: 25,   # 25% search users
        AdminUser: 5      # 5% admin users
    }
    
    PERCENTILES = [50, 95, 99]  # Response time percentiles to track
    
    THRESHOLDS = {
        "response_time_95": 500,    # 95th percentile response time (ms)
        "error_rate": 1,            # Maximum error rate (%)
        "rps": 45                   # Minimum requests per second
    }