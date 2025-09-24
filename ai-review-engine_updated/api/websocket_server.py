"""
WebSocket Server for Real-time Updates
Handles live sentiment analysis, notifications, and real-time data streaming
"""

from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import json
import asyncio
from typing import Dict, List, Set, Optional
from datetime import datetime
import logging
import redis.asyncio as redis
from dataclasses import dataclass, asdict
import jwt
import hashlib

from models.advanced_ai_model import AdvancedAIEngine
from database.models import db_manager, Review, Product, User

logger = logging.getLogger(__name__)

# Initialize components
ai_engine = AdvancedAIEngine()
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
security = HTTPBearer()

# Connection manager for WebSocket clients
class ConnectionManager:
    """Manages WebSocket connections and broadcasting"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.user_connections: Dict[str, WebSocket] = {}
        self.room_connections: Dict[str, Set[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str, room: Optional[str] = None):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        
        # Add to active connections
        if client_id not in self.active_connections:
            self.active_connections[client_id] = []
        self.active_connections[client_id].append(websocket)
        
        # Add to room if specified
        if room:
            if room not in self.room_connections:
                self.room_connections[room] = set()
            self.room_connections[room].add(websocket)
        
        logger.info(f"Client {client_id} connected to room {room}")
        
        # Send welcome message
        await self.send_personal_message(
            {
                "type": "connection",
                "status": "connected",
                "client_id": client_id,
                "room": room,
                "timestamp": datetime.now().isoformat()
            },
            websocket
        )
    
    def disconnect(self, websocket: WebSocket, client_id: str):
        """Remove a WebSocket connection"""
        if client_id in self.active_connections:
            if websocket in self.active_connections[client_id]:
                self.active_connections[client_id].remove(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]
        
        # Remove from all rooms
        for room in self.room_connections:
            if websocket in self.room_connections[room]:
                self.room_connections[room].remove(websocket)
        
        logger.info(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific WebSocket"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def broadcast_to_room(self, message: dict, room: str):
        """Broadcast message to all connections in a room"""
        if room in self.room_connections:
            disconnected = []
            for connection in self.room_connections[room]:
                try:
                    await connection.send_json(message)
                except:
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for conn in disconnected:
                self.room_connections[room].remove(conn)
    
    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for client_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_json(message)
                except:
                    disconnected.append((client_id, connection))
        
        # Remove disconnected clients
        for client_id, conn in disconnected:
            if client_id in self.active_connections:
                if conn in self.active_connections[client_id]:
                    self.active_connections[client_id].remove(conn)

# Create connection manager instance
manager = ConnectionManager()

# WebSocket message handlers
@dataclass
class WSMessage:
    """WebSocket message structure"""
    type: str
    data: dict
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class RealtimeAnalyzer:
    """Handles real-time analysis requests"""
    
    def __init__(self):
        self.ai_engine = ai_engine
        self.analysis_cache = {}
    
    async def analyze_text_stream(self, text: str, request_id: str) -> dict:
        """Analyze text in real-time with caching"""
        
        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.analysis_cache:
            return self.analysis_cache[text_hash]
        
        # Perform analysis
        analysis = self.ai_engine.advanced_sentiment_analysis(text)
        aspects = self.ai_engine.extract_advanced_aspects(text)
        
        result = {
            "request_id": request_id,
            "text": text[:200] + "..." if len(text) > 200 else text,
            "sentiment": analysis.get('ensemble_prediction', 'neutral'),
            "confidence": analysis.get('confidence', 0),
            "emotions": analysis.get('emotions', [])[:3] if analysis.get('emotions') else [],
            "entities": analysis.get('entities', [])[:5] if analysis.get('entities') else [],
            "aspects": {
                "technical": aspects.get('technical_aspects', [])[:3],
                "quantitative": aspects.get('quantitative_aspects', [])[:3]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache result
        self.analysis_cache[text_hash] = result
        
        # Clean cache if too large
        if len(self.analysis_cache) > 1000:
            # Remove oldest entries
            keys_to_remove = list(self.analysis_cache.keys())[:500]
            for key in keys_to_remove:
                del self.analysis_cache[key]
        
        return result
    
    async def analyze_product_trend(self, product_id: int) -> dict:
        """Analyze product sentiment trend"""
        session = db_manager.get_session()
        
        # Get recent reviews
        recent_reviews = session.query(Review).filter_by(
            product_id=product_id
        ).order_by(Review.review_date.desc()).limit(50).all()
        
        if not recent_reviews:
            session.close()
            return {"error": "No reviews found"}
        
        # Analyze trend
        sentiments = []
        for review in recent_reviews:
            if review.sentiment:
                sentiments.append({
                    "date": review.review_date.isoformat() if review.review_date else None,
                    "sentiment": review.sentiment,
                    "confidence": review.sentiment_confidence or 0
                })
        
        # Calculate trend
        positive_count = sum(1 for s in sentiments if s['sentiment'] == 'positive')
        negative_count = sum(1 for s in sentiments if s['sentiment'] == 'negative')
        neutral_count = sum(1 for s in sentiments if s['sentiment'] == 'neutral')
        
        total = len(sentiments)
        
        trend_data = {
            "product_id": product_id,
            "total_analyzed": total,
            "positive_percentage": (positive_count / total * 100) if total > 0 else 0,
            "negative_percentage": (negative_count / total * 100) if total > 0 else 0,
            "neutral_percentage": (neutral_count / total * 100) if total > 0 else 0,
            "recent_sentiments": sentiments[:10],
            "trend": self._calculate_trend_direction(sentiments),
            "timestamp": datetime.now().isoformat()
        }
        
        session.close()
        return trend_data
    
    def _calculate_trend_direction(self, sentiments: List[dict]) -> str:
        """Calculate if sentiment is improving or declining"""
        if len(sentiments) < 2:
            return "stable"
        
        # Compare first half with second half
        mid = len(sentiments) // 2
        first_half = sentiments[:mid]
        second_half = sentiments[mid:]
        
        first_positive = sum(1 for s in first_half if s['sentiment'] == 'positive')
        second_positive = sum(1 for s in second_half if s['sentiment'] == 'positive')
        
        if second_positive > first_positive:
            return "improving"
        elif second_positive < first_positive:
            return "declining"
        else:
            return "stable"

# Create analyzer instance
realtime_analyzer = RealtimeAnalyzer()

# WebSocket endpoints
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
    room: Optional[str] = None
):
    """Main WebSocket endpoint"""
    await manager.connect(websocket, client_id, room)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            await handle_websocket_message(websocket, client_id, message, room)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
        if room:
            await manager.broadcast_to_room(
                {
                    "type": "user_left",
                    "client_id": client_id,
                    "timestamp": datetime.now().isoformat()
                },
                room
            )
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, client_id)

async def handle_websocket_message(
    websocket: WebSocket,
    client_id: str,
    message: dict,
    room: Optional[str]
):
    """Handle incoming WebSocket messages"""
    msg_type = message.get('type')
    data = message.get('data', {})
    
    if msg_type == 'ping':
        # Heartbeat
        await manager.send_personal_message(
            {"type": "pong", "timestamp": datetime.now().isoformat()},
            websocket
        )
    
    elif msg_type == 'analyze':
        # Real-time text analysis
        text = data.get('text', '')
        request_id = data.get('request_id', 'unknown')
        
        if text:
            # Start analysis
            await manager.send_personal_message(
                {
                    "type": "analysis_started",
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                },
                websocket
            )
            
            # Perform analysis
            result = await realtime_analyzer.analyze_text_stream(text, request_id)
            
            # Send result
            await manager.send_personal_message(
                {
                    "type": "analysis_result",
                    "data": result,
                    "timestamp": datetime.now().isoformat()
                },
                websocket
            )
            
            # Broadcast to room if specified
            if room:
                await manager.broadcast_to_room(
                    {
                        "type": "room_analysis",
                        "client_id": client_id,
                        "sentiment": result['sentiment'],
                        "confidence": result['confidence'],
                        "timestamp": datetime.now().isoformat()
                    },
                    room
                )
    
    elif msg_type == 'subscribe_product':
        # Subscribe to product updates
        product_id = data.get('product_id')
        if product_id:
            room_name = f"product_{product_id}"
            if room_name not in manager.room_connections:
                manager.room_connections[room_name] = set()
            manager.room_connections[room_name].add(websocket)
            
            await manager.send_personal_message(
                {
                    "type": "subscribed",
                    "product_id": product_id,
                    "timestamp": datetime.now().isoformat()
                },
                websocket
            )
            
            # Send initial trend data
            trend_data = await realtime_analyzer.analyze_product_trend(product_id)
            await manager.send_personal_message(
                {
                    "type": "product_trend",
                    "data": trend_data,
                    "timestamp": datetime.now().isoformat()
                },
                websocket
            )
    
    elif msg_type == 'chat':
        # Chat message in room
        if room:
            await manager.broadcast_to_room(
                {
                    "type": "chat_message",
                    "client_id": client_id,
                    "message": data.get('message', ''),
                    "timestamp": datetime.now().isoformat()
                },
                room
            )
    
    elif msg_type == 'typing':
        # Typing indicator
        if room:
            await manager.broadcast_to_room(
                {
                    "type": "typing",
                    "client_id": client_id,
                    "is_typing": data.get('is_typing', False),
                    "timestamp": datetime.now().isoformat()
                },
                room
            )

# Background tasks for real-time updates
async def redis_listener():
    """Listen to Redis pub/sub for real-time updates"""
    pubsub = redis_client.pubsub()
    await pubsub.subscribe('review_updates', 'analysis_updates', 'scraping_updates')
    
    async for message in pubsub.listen():
        if message['type'] == 'message':
            channel = message['channel']
            data = json.loads(message['data'])
            
            if channel == 'review_updates':
                # Broadcast new review to relevant product room
                product_id = data.get('product_id')
                if product_id:
                    await manager.broadcast_to_room(
                        {
                            "type": "new_review",
                            "data": data,
                            "timestamp": datetime.now().isoformat()
                        },
                        f"product_{product_id}"
                    )
            
            elif channel == 'analysis_updates':
                # Broadcast analysis updates to all
                await manager.broadcast_to_all(
                    {
                        "type": "analysis_update",
                        "data": data,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            elif channel == 'scraping_updates':
                # Broadcast scraping status
                await manager.broadcast_to_all(
                    {
                        "type": "scraping_update",
                        "data": data,
                        "timestamp": datetime.now().isoformat()
                    }
                )

async def periodic_stats_broadcaster():
    """Periodically broadcast system stats"""
    while True:
        await asyncio.sleep(30)  # Every 30 seconds
        
        # Get system stats
        stats = {
            "connected_clients": len(manager.active_connections),
            "active_rooms": len(manager.room_connections),
            "model_version": ai_engine.model_version,
            "timestamp": datetime.now().isoformat()
        }
        
        # Broadcast to all clients
        await manager.broadcast_to_all(
            {
                "type": "system_stats",
                "data": stats,
                "timestamp": datetime.now().isoformat()
            }
        )

# Start background tasks
def start_websocket_background_tasks():
    """Start all background tasks for WebSocket server"""
    asyncio.create_task(redis_listener())
    asyncio.create_task(periodic_stats_broadcaster())
    logger.info("WebSocket background tasks started")

# WebSocket authentication
async def get_current_user_ws(token: str) -> Optional[str]:
    """Verify WebSocket connection token"""
    try:
        from api.main import SECRET_KEY, ALGORITHM
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except jwt.JWTError:
        return None

# Export WebSocket app for FastAPI
def create_websocket_app(app):
    """Add WebSocket routes to FastAPI app"""
    
    @app.websocket("/ws/{client_id}")
    async def websocket_route(websocket: WebSocket, client_id: str, room: Optional[str] = None):
        await websocket_endpoint(websocket, client_id, room)
    
    @app.websocket("/ws/auth/{token}")
    async def authenticated_websocket(websocket: WebSocket, token: str):
        username = await get_current_user_ws(token)
        if not username:
            await websocket.close(code=1008, reason="Unauthorized")
            return
        
        await websocket_endpoint(websocket, username, "authenticated")
    
    # Start background tasks on startup
    @app.on_event("startup")
    async def startup_event():
        start_websocket_background_tasks()
    
    return app
