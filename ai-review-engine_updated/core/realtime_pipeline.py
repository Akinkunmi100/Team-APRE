"""
Real-Time Data Pipeline & Live Updates System
Optional module for streaming data, live notifications, and real-time analytics
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import deque, defaultdict
import time
import hashlib
from abc import ABC, abstractmethod

# Kafka imports (optional)
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logging.warning("Kafka not available. Real-time features will use fallback methods.")

# Redis imports (optional)
try:
    import redis
    from redis.client import PubSub
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. Using in-memory pub/sub.")

# WebSocket imports
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logging.warning("WebSockets not available. Live updates disabled.")

# Apache Spark imports (optional)
try:
    from pyspark.sql import SparkSession
    from pyspark.streaming import StreamingContext
    from pyspark.sql.functions import window, col, count, avg
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    logging.warning("Apache Spark not available. Stream processing will use basic methods.")

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of real-time events"""
    NEW_REVIEW = "new_review"
    PRICE_CHANGE = "price_change"
    SENTIMENT_SHIFT = "sentiment_shift"
    TRENDING_PHONE = "trending_phone"
    STOCK_ALERT = "stock_alert"
    COMPARISON_REQUEST = "comparison_request"
    USER_QUERY = "user_query"
    SYSTEM_ALERT = "system_alert"
    ANALYSIS_COMPLETE = "analysis_complete"
    RECOMMENDATION_UPDATE = "recommendation_update"


class StreamProcessor(Enum):
    """Available stream processors"""
    KAFKA = "kafka"
    REDIS = "redis"
    SPARK = "spark"
    MEMORY = "memory"  # Fallback


@dataclass
class StreamEvent:
    """Represents a streaming event"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    source: str = "system"
    priority: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False
    retry_count: int = 0


@dataclass
class StreamConfig:
    """Configuration for stream processing"""
    processor_type: StreamProcessor = StreamProcessor.MEMORY
    batch_size: int = 100
    window_size: int = 60  # seconds
    checkpoint_interval: int = 300  # seconds
    max_retries: int = 3
    enable_persistence: bool = True
    enable_notifications: bool = True
    kafka_config: Optional[Dict] = None
    redis_config: Optional[Dict] = None
    spark_config: Optional[Dict] = None


class BaseStreamProcessor(ABC):
    """Abstract base class for stream processors"""
    
    @abstractmethod
    async def publish(self, event: StreamEvent) -> bool:
        """Publish an event to the stream"""
        pass
    
    @abstractmethod
    async def consume(self, callback: Callable) -> None:
        """Consume events from the stream"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the stream processor"""
        pass


class InMemoryStreamProcessor(BaseStreamProcessor):
    """In-memory stream processor (fallback)"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.event_queue = asyncio.Queue(maxsize=10000)
        self.subscribers = []
        self.running = False
        logger.info("Initialized in-memory stream processor")
    
    async def publish(self, event: StreamEvent) -> bool:
        """Publish event to in-memory queue"""
        try:
            await self.event_queue.put(event)
            # Notify subscribers
            for subscriber in self.subscribers:
                await subscriber(event)
            return True
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False
    
    async def consume(self, callback: Callable) -> None:
        """Consume events from queue"""
        self.running = True
        while self.running:
            try:
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )
                await callback(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error consuming event: {e}")
    
    def subscribe(self, callback: Callable):
        """Subscribe to events"""
        self.subscribers.append(callback)
    
    async def close(self) -> None:
        """Close the processor"""
        self.running = False


class KafkaStreamProcessor(BaseStreamProcessor):
    """Kafka-based stream processor"""
    
    def __init__(self, config: StreamConfig):
        if not KAFKA_AVAILABLE:
            raise ImportError("Kafka is not available")
        
        self.config = config
        kafka_config = config.kafka_config or {}
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_config.get('bootstrap_servers', 'localhost:9092'),
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        
        # Initialize Kafka consumer
        self.consumer = KafkaConsumer(
            kafka_config.get('topic', 'phone-reviews'),
            bootstrap_servers=kafka_config.get('bootstrap_servers', 'localhost:9092'),
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id=kafka_config.get('group_id', 'review-engine'),
            auto_offset_reset='latest'
        )
        
        logger.info("Initialized Kafka stream processor")
    
    async def publish(self, event: StreamEvent) -> bool:
        """Publish event to Kafka"""
        try:
            future = self.producer.send(
                self.config.kafka_config.get('topic', 'phone-reviews'),
                key=event.event_id,
                value=event.__dict__
            )
            result = future.get(timeout=10)
            return True
        except KafkaError as e:
            logger.error(f"Kafka publish error: {e}")
            return False
    
    async def consume(self, callback: Callable) -> None:
        """Consume events from Kafka"""
        for message in self.consumer:
            try:
                event_data = message.value
                event = StreamEvent(**event_data)
                await callback(event)
            except Exception as e:
                logger.error(f"Error processing Kafka message: {e}")
    
    async def close(self) -> None:
        """Close Kafka connections"""
        self.producer.close()
        self.consumer.close()


class RedisStreamProcessor(BaseStreamProcessor):
    """Redis-based stream processor with pub/sub"""
    
    def __init__(self, config: StreamConfig):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available")
        
        self.config = config
        redis_config = config.redis_config or {}
        
        # Initialize Redis clients
        self.redis_client = redis.Redis(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0),
            decode_responses=True
        )
        
        self.pubsub = self.redis_client.pubsub()
        self.channel = redis_config.get('channel', 'phone-reviews')
        
        logger.info("Initialized Redis stream processor")
    
    async def publish(self, event: StreamEvent) -> bool:
        """Publish event to Redis"""
        try:
            event_json = json.dumps({
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'timestamp': event.timestamp.isoformat(),
                'data': event.data,
                'source': event.source,
                'priority': event.priority
            })
            
            # Publish to channel
            self.redis_client.publish(self.channel, event_json)
            
            # Store in Redis stream for persistence
            self.redis_client.xadd(
                f"stream:{self.channel}",
                {"event": event_json},
                maxlen=10000  # Keep last 10k events
            )
            
            return True
        except Exception as e:
            logger.error(f"Redis publish error: {e}")
            return False
    
    async def consume(self, callback: Callable) -> None:
        """Consume events from Redis"""
        self.pubsub.subscribe(self.channel)
        
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                try:
                    event_data = json.loads(message['data'])
                    event = StreamEvent(
                        event_id=event_data['event_id'],
                        event_type=EventType(event_data['event_type']),
                        timestamp=datetime.fromisoformat(event_data['timestamp']),
                        data=event_data['data'],
                        source=event_data['source'],
                        priority=event_data['priority']
                    )
                    await callback(event)
                except Exception as e:
                    logger.error(f"Error processing Redis message: {e}")
    
    async def close(self) -> None:
        """Close Redis connections"""
        self.pubsub.close()
        self.redis_client.close()


class RealTimeAnalytics:
    """Real-time analytics engine"""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: defaultdict(float))
        self.time_series = defaultdict(lambda: deque(maxlen=1000))
        self.aggregations = {}
        self.alerts = []
        
        # Spark session for advanced analytics (optional)
        if SPARK_AVAILABLE:
            self.spark = SparkSession.builder \
                .appName("PhoneReviewAnalytics") \
                .config("spark.sql.adaptive.enabled", "true") \
                .getOrCreate()
        else:
            self.spark = None
    
    def update_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Update a real-time metric"""
        key = self._generate_key(metric_name, tags)
        
        # Update running statistics
        self.metrics[key]['count'] += 1
        self.metrics[key]['sum'] += value
        self.metrics[key]['avg'] = self.metrics[key]['sum'] / self.metrics[key]['count']
        self.metrics[key]['min'] = min(self.metrics[key].get('min', float('inf')), value)
        self.metrics[key]['max'] = max(self.metrics[key].get('max', float('-inf')), value)
        
        # Add to time series
        self.time_series[key].append({
            'timestamp': datetime.now(),
            'value': value
        })
    
    def _generate_key(self, metric_name: str, tags: Optional[Dict] = None) -> str:
        """Generate unique key for metric"""
        if tags:
            tag_str = "_".join([f"{k}:{v}" for k, v in sorted(tags.items())])
            return f"{metric_name}_{tag_str}"
        return metric_name
    
    def calculate_trend(self, metric_name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Calculate trend for a metric"""
        data = self.time_series.get(metric_name, deque())
        if len(data) < 2:
            return {'trend': 'insufficient_data'}
        
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent_data = [d for d in data if d['timestamp'] > cutoff]
        
        if len(recent_data) < 2:
            return {'trend': 'insufficient_data'}
        
        # Simple linear regression for trend
        x = [(d['timestamp'] - recent_data[0]['timestamp']).total_seconds() for d in recent_data]
        y = [d['value'] for d in recent_data]
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            
            return {
                'trend': 'increasing' if slope > 0 else 'decreasing',
                'slope': slope,
                'current_value': y[-1],
                'change_rate': slope * 60  # Change per minute
            }
        
        return {'trend': 'stable'}
    
    def detect_anomalies(self, metric_name: str, threshold: float = 3.0) -> List[Dict]:
        """Detect anomalies using statistical methods"""
        data = self.time_series.get(metric_name, deque())
        if len(data) < 10:
            return []
        
        values = [d['value'] for d in data]
        mean = np.mean(values)
        std = np.std(values)
        
        anomalies = []
        for d in data:
            z_score = abs((d['value'] - mean) / std) if std > 0 else 0
            if z_score > threshold:
                anomalies.append({
                    'timestamp': d['timestamp'],
                    'value': d['value'],
                    'z_score': z_score,
                    'severity': 'high' if z_score > 4 else 'medium'
                })
        
        return anomalies
    
    def generate_alert(self, alert_type: str, message: str, severity: str = "info", data: Dict = None):
        """Generate an alert"""
        alert = {
            'id': hashlib.md5(f"{alert_type}{message}{datetime.now()}".encode()).hexdigest(),
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now(),
            'data': data or {},
            'acknowledged': False
        }
        
        self.alerts.append(alert)
        logger.info(f"Alert generated: {alert_type} - {message}")
        
        return alert
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for dashboard display"""
        return {
            'real_time_metrics': {
                name: {
                    'current': metrics.get('avg', 0),
                    'min': metrics.get('min', 0),
                    'max': metrics.get('max', 0),
                    'count': metrics.get('count', 0)
                }
                for name, metrics in self.metrics.items()
            },
            'trends': {
                name: self.calculate_trend(name)
                for name in self.time_series.keys()
            },
            'recent_alerts': self.alerts[-10:],
            'timestamp': datetime.now().isoformat()
        }


class LiveNotificationService:
    """Service for sending live notifications"""
    
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.notification_queue = asyncio.Queue()
        self.websocket_clients = set()
        
        if WEBSOCKET_AVAILABLE:
            self.websocket_server = None
    
    async def subscribe(self, user_id: str, event_types: List[EventType], callback: Callable):
        """Subscribe to notifications"""
        for event_type in event_types:
            self.subscribers[event_type].append({
                'user_id': user_id,
                'callback': callback
            })
    
    async def notify(self, event: StreamEvent):
        """Send notification for an event"""
        # Get subscribers for this event type
        subscribers = self.subscribers.get(event.event_type, [])
        
        for subscriber in subscribers:
            try:
                await subscriber['callback'](event)
            except Exception as e:
                logger.error(f"Error notifying subscriber {subscriber['user_id']}: {e}")
        
        # Send to WebSocket clients
        if self.websocket_clients:
            await self._broadcast_websocket(event)
    
    async def _broadcast_websocket(self, event: StreamEvent):
        """Broadcast event to WebSocket clients"""
        if not WEBSOCKET_AVAILABLE:
            return
        
        message = json.dumps({
            'type': 'notification',
            'event': {
                'id': event.event_id,
                'type': event.event_type.value,
                'timestamp': event.timestamp.isoformat(),
                'data': event.data
            }
        })
        
        # Send to all connected clients
        disconnected = set()
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except:
                disconnected.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected
    
    async def start_websocket_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server for live updates"""
        if not WEBSOCKET_AVAILABLE:
            logger.warning("WebSocket not available")
            return
        
        async def handle_client(websocket: WebSocketServerProtocol, path: str):
            """Handle WebSocket client connection"""
            self.websocket_clients.add(websocket)
            logger.info(f"WebSocket client connected: {websocket.remote_address}")
            
            try:
                async for message in websocket:
                    # Handle client messages (subscriptions, etc.)
                    data = json.loads(message)
                    if data.get('action') == 'subscribe':
                        # Handle subscription
                        pass
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.websocket_clients.remove(websocket)
                logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
        
        self.websocket_server = await websockets.serve(handle_client, host, port)
        logger.info(f"WebSocket server started on {host}:{port}")


class RealTimePipeline:
    """Main real-time data pipeline orchestrator"""
    
    def __init__(self, config: Optional[StreamConfig] = None, enable: bool = True):
        """
        Initialize real-time pipeline
        
        Args:
            config: Stream configuration
            enable: Whether to enable real-time features
        """
        self.enabled = enable
        
        if not self.enabled:
            logger.info("Real-time pipeline disabled")
            return
        
        self.config = config or StreamConfig()
        
        # Initialize components based on availability
        self.stream_processor = self._initialize_processor()
        self.analytics = RealTimeAnalytics()
        self.notification_service = LiveNotificationService()
        
        # Event handlers
        self.event_handlers = defaultdict(list)
        
        # Background tasks
        self.tasks = []
        
        # Statistics
        self.stats = {
            'events_processed': 0,
            'events_failed': 0,
            'start_time': datetime.now()
        }
        
        logger.info("Real-time pipeline initialized")
    
    def _initialize_processor(self) -> BaseStreamProcessor:
        """Initialize the appropriate stream processor"""
        if self.config.processor_type == StreamProcessor.KAFKA and KAFKA_AVAILABLE:
            return KafkaStreamProcessor(self.config)
        elif self.config.processor_type == StreamProcessor.REDIS and REDIS_AVAILABLE:
            return RedisStreamProcessor(self.config)
        else:
            return InMemoryStreamProcessor(self.config)
    
    async def publish_event(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: str = "system",
        priority: int = 5
    ) -> bool:
        """
        Publish an event to the pipeline
        
        Args:
            event_type: Type of event
            data: Event data
            source: Event source
            priority: Event priority (1-10)
            
        Returns:
            Success status
        """
        if not self.enabled:
            return True  # Silently succeed if disabled
        
        event = StreamEvent(
            event_id=hashlib.md5(f"{event_type}{data}{datetime.now()}".encode()).hexdigest()[:16],
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            source=source,
            priority=priority
        )
        
        # Publish to stream
        success = await self.stream_processor.publish(event)
        
        # Process immediately if in-memory
        if isinstance(self.stream_processor, InMemoryStreamProcessor):
            await self._process_event(event)
        
        return success
    
    async def _process_event(self, event: StreamEvent):
        """Process a single event"""
        try:
            # Update analytics
            self.analytics.update_metric(
                f"events.{event.event_type.value}",
                1,
                tags={'source': event.source}
            )
            
            # Call registered handlers
            handlers = self.event_handlers.get(event.event_type, [])
            for handler in handlers:
                await handler(event)
            
            # Send notifications
            if self.config.enable_notifications:
                await self.notification_service.notify(event)
            
            # Check for alerts
            await self._check_alerts(event)
            
            self.stats['events_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")
            self.stats['events_failed'] += 1
    
    async def _check_alerts(self, event: StreamEvent):
        """Check if event triggers any alerts"""
        # Price drop alert
        if event.event_type == EventType.PRICE_CHANGE:
            change = event.data.get('change_percent', 0)
            if change < -10:
                self.analytics.generate_alert(
                    'price_drop',
                    f"Significant price drop: {event.data.get('phone_model')} down {abs(change):.1f}%",
                    severity='high',
                    data=event.data
                )
        
        # Sentiment shift alert
        elif event.event_type == EventType.SENTIMENT_SHIFT:
            shift = event.data.get('shift_magnitude', 0)
            if abs(shift) > 20:
                self.analytics.generate_alert(
                    'sentiment_shift',
                    f"Major sentiment shift detected for {event.data.get('phone_model')}",
                    severity='medium',
                    data=event.data
                )
        
        # System alert
        elif event.event_type == EventType.SYSTEM_ALERT:
            self.analytics.generate_alert(
                'system',
                event.data.get('message', 'System alert'),
                severity=event.data.get('severity', 'info'),
                data=event.data
            )
    
    def register_handler(self, event_type: EventType, handler: Callable):
        """Register an event handler"""
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value}")
    
    async def start(self):
        """Start the real-time pipeline"""
        if not self.enabled:
            return
        
        logger.info("Starting real-time pipeline")
        
        # Start consumer task
        consumer_task = asyncio.create_task(
            self.stream_processor.consume(self._process_event)
        )
        self.tasks.append(consumer_task)
        
        # Start WebSocket server
        if self.config.enable_notifications:
            ws_task = asyncio.create_task(
                self.notification_service.start_websocket_server()
            )
            self.tasks.append(ws_task)
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_pipeline())
        self.tasks.append(monitor_task)
        
        logger.info("Real-time pipeline started")
    
    async def _monitor_pipeline(self):
        """Monitor pipeline health"""
        while True:
            try:
                # Calculate metrics
                uptime = (datetime.now() - self.stats['start_time']).total_seconds()
                events_per_second = self.stats['events_processed'] / max(uptime, 1)
                
                # Log statistics
                logger.debug(f"Pipeline stats: {self.stats['events_processed']} processed, "
                           f"{self.stats['events_failed']} failed, "
                           f"{events_per_second:.2f} events/sec")
                
                # Check for anomalies
                anomalies = self.analytics.detect_anomalies('events.total')
                if anomalies:
                    logger.warning(f"Detected {len(anomalies)} anomalies in event stream")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in pipeline monitor: {e}")
                await asyncio.sleep(60)
    
    async def stop(self):
        """Stop the real-time pipeline"""
        if not self.enabled:
            return
        
        logger.info("Stopping real-time pipeline")
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Close stream processor
        await self.stream_processor.close()
        
        logger.info("Real-time pipeline stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        if not self.enabled:
            return {'status': 'disabled'}
        
        return {
            'status': 'running',
            'processor': self.config.processor_type.value,
            'stats': self.stats,
            'analytics': self.analytics.get_dashboard_metrics(),
            'handlers_registered': {
                event_type.value: len(handlers)
                for event_type, handlers in self.event_handlers.items():
            }
        }


# Singleton instance (optional, can be disabled)
_pipeline_instance = None


def get_pipeline(config: Optional[StreamConfig] = None, enable: bool = True) -> RealTimePipeline:
    """Get or create pipeline instance"""
    global _pipeline_instance
    
    if _pipeline_instance is None:
        _pipeline_instance = RealTimePipeline(config, enable)
    
    return _pipeline_instance


# Example usage
async def example_usage():
    """Example of using the real-time pipeline"""
    
    # Create pipeline (can be disabled)
    config = StreamConfig(
        processor_type=StreamProcessor.MEMORY,  # Use in-memory for demo
        enable_notifications=True
    )
    
    pipeline = RealTimePipeline(config, enable=True)  # Set enable=False to disable
    
    # Register event handlers
    async def handle_new_review(event: StreamEvent):
        print(f"New review: {event.data}")
        # Process review in real-time
    
    async def handle_price_change(event: StreamEvent):
        print(f"Price changed: {event.data}")
        # Update price tracking
    
    pipeline.register_handler(EventType.NEW_REVIEW, handle_new_review)
    pipeline.register_handler(EventType.PRICE_CHANGE, handle_price_change)
    
    # Start pipeline
    await pipeline.start()
    
    # Publish events
    await pipeline.publish_event(
        EventType.NEW_REVIEW,
        {
            'phone_model': 'iPhone 15 Pro',
            'rating': 5,
            'text': 'Amazing phone!',
            'user': 'user123'
        }
    )
    
    await pipeline.publish_event(
        EventType.PRICE_CHANGE,
        {
            'phone_model': 'Samsung Galaxy S24',
            'old_price': 999,
            'new_price': 899,
            'change_percent': -10
        },
        priority=8
    )
    
    # Get status
    status = pipeline.get_status()
    print(f"Pipeline status: {json.dumps(status, indent=2, default=str)}")
    
    # Wait a bit
    await asyncio.sleep(5)
    
    # Stop pipeline
    await pipeline.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())
