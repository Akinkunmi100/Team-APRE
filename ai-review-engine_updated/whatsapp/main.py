"""
WhatsApp Integration API for AI Review Engine
"""
import os
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from .whatsapp_service import WhatsAppService, WhatsAppMessage
import hmac
import hashlib
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET")

app = FastAPI(title="WhatsApp Integration API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize WhatsApp service
whatsapp_service = WhatsAppService()

def verify_webhook_signature(request: Request) -> bool:
    """Verify webhook signature from WhatsApp."""
    signature = request.headers.get("X-Hub-Signature-256")
    if not signature or not signature.startswith("sha256="):
        return False

    expected_signature = signature.split("sha256=")[1]
    body = await request.body()
    
    hmac_obj = hmac.new(
        WHATSAPP_APP_SECRET.encode(),
        msg=body,
        digestmod=hashlib.sha256
    )
    calculated_signature = hmac_obj.hexdigest()
    
    return hmac.compare_digest(calculated_signature, expected_signature)

@app.get("/webhook")
async def verify_webhook(
    mode: str = None,
    token: str = None,
    challenge: str = None
):
    """Verify webhook endpoint for WhatsApp API setup."""
    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        return int(challenge)
    raise HTTPException(status_code=403, detail="Invalid verification token")

@app.post("/webhook")
async def webhook(request: Request):
    """Handle incoming webhooks from WhatsApp."""
    # Verify signature
    if not verify_webhook_signature(request):
        raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        data = await request.json()
        await whatsapp_service.handle_incoming_message(data)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/send")
async def send_message(message: WhatsAppMessage):
    """Send a message via WhatsApp."""
    try:
        result = await whatsapp_service.send_message(message)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to send message")

@app.get("/health")
async def health_check():
    """Check service health."""
    return {
        "status": "healthy",
        "service": "whatsapp-integration",
        "version": "1.0.0"
    }