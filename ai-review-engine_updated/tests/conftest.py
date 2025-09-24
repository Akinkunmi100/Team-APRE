"""
Test configuration and fixtures for AI Review Engine
"""
import os
import pytest
import tempfile
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from redis import Redis

# Import your application modules
from api.database import Base
from api.config import settings

# Test database URL
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/ai_review_engine_test"
)

# Test Redis URL
TEST_REDIS_URL = os.getenv(
    "TEST_REDIS_URL",
    "redis://localhost:6379/1"
)

@pytest.fixture(scope="session")
def db_engine():
    """Create a test database engine."""
    engine = create_engine(TEST_DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def db_session(db_engine) -> Generator[Session, None, None]:
    """Create a test database session."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()

@pytest.fixture(scope="session")
def redis_client() -> Generator[Redis, None, None]:
    """Create a test Redis client."""
    client = Redis.from_url(TEST_REDIS_URL)
    try:
        yield client
    finally:
        client.flushdb()
        client.close()

@pytest.fixture(scope="function")
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture(scope="session")
def api_client():
    """Create a test API client."""
    from fastapi.testclient import TestClient
    from api.main import app
    
    client = TestClient(app)
    return client

@pytest.fixture(scope="session")
def web_client():
    """Create a test web client."""
    from web.app import app
    
    client = app.test_client()
    return client

@pytest.fixture(scope="function")
def test_data(db_session: Session):
    """Create test data in the database."""
    # Add your test data creation logic here
    yield

@pytest.fixture(scope="function")
def mock_redis(mocker):
    """Mock Redis for unit tests."""
    return mocker.patch("redis.Redis")

@pytest.fixture(scope="function")
def mock_celery(mocker):
    """Mock Celery for unit tests."""
    return mocker.patch("celery.Celery")

@pytest.fixture(scope="session")
def load_test_config():
    """Load test configuration for locust load tests."""
    return {
        "host": "http://localhost:8000",
        "num_users": 100,
        "spawn_rate": 10,
        "run_time": "1m"
    }

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "api: mark test as API test")
    config.addinivalue_line("markers", "web: mark test as web interface test")
    config.addinivalue_line("markers", "worker: mark test as worker test")
    config.addinivalue_line("markers", "security: mark test as security test")

def pytest_collection_modifyitems(items):
    """Modify test items to add marks based on location."""
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "load" in str(item.fspath):
            item.add_marker(pytest.mark.load)