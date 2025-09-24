"""
WhatsApp Integration Service for AI Review Engine
"""
import os
from typing import Dict, Any, List, Optional
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
WHATSAPP_API_URL = os.getenv("WHATSAPP_API_URL", "https://graph.facebook.com/v17.0")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")

class WhatsAppMessage(BaseModel):
    phone_number: str
    message: str
    message_type: str = "text"
    template_name: Optional[str] = None
    template_params: Optional[Dict[str, Any]] = None

class ReviewRequest(BaseModel):
    phone_number: str
    product_id: str
    rating: Optional[int] = None
    review_text: Optional[str] = None

class WhatsAppService:
    def __init__(self):
        self.api_url = WHATSAPP_API_URL
        self.phone_id = WHATSAPP_PHONE_ID
        self.access_token = WHATSAPP_ACCESS_TOKEN
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

    async def send_message(self, message: WhatsAppMessage) -> Dict[str, Any]:
        """Send a message via WhatsApp."""
        endpoint = f"{self.api_url}/{self.phone_id}/messages"

        if message.message_type == "template":
            payload = self._create_template_payload(message)
        else:
            payload = self._create_text_payload(message)

        try:
            response = requests.post(endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send WhatsApp message: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to send WhatsApp message")

    def _create_text_payload(self, message: WhatsAppMessage) -> Dict[str, Any]:
        """Create payload for text message."""
        return {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": message.phone_number,
            "type": "text",
            "text": {"body": message.message}
        }

    def _create_template_payload(self, message: WhatsAppMessage) -> Dict[str, Any]:
        """Create payload for template message."""
        return {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": message.phone_number,
            "type": "template",
            "template": {
                "name": message.template_name,
                "language": {"code": "en"},
                "components": [
                    {
                        "type": "body",
                        "parameters": [
                            {"type": "text", "text": param}
                            for param in message.template_params.values()
                        ]
                    }
                ]
            }
        }

    async def handle_incoming_message(self, data: Dict[str, Any]) -> None:
        """Handle incoming WhatsApp messages."""
        try:
            if "messages" in data["entry"][0]["changes"][0]["value"]:
                message = data["entry"][0]["changes"][0]["value"]["messages"][0]
                sender = message["from"]
                message_type = message["type"]
                
                if message_type == "text":
                    text = message["text"]["body"]
                    await self._process_text_message(sender, text)
                elif message_type == "interactive":
                    await self._process_interactive_message(sender, message)

        except KeyError as e:
            logger.error(f"Invalid message format: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid message format")

    async def _process_text_message(self, sender: str, text: str) -> None:
        """Process incoming text messages."""
        # Check for commands or keywords
        text_lower = text.lower()
        
        if text_lower.startswith("review"):
            # Format: review <product_id> <rating> <text>
            parts = text.split(" ", 3)
            if len(parts) >= 4:
                review_request = ReviewRequest(
                    phone_number=sender,
                    product_id=parts[1],
                    rating=int(parts[2]),
                    review_text=parts[3]
                )
                await self._handle_review_submission(review_request)
        
        elif text_lower.startswith("help"):
            await self.send_help_message(sender)
        
        elif text_lower.startswith("status"):
            await self.send_status_message(sender)
        
        else:
            await self.send_message(WhatsAppMessage(
                phone_number=sender,
                message="I don't understand that command. Type 'help' for available commands."
            ))

    async def _process_interactive_message(self, sender: str, message: Dict[str, Any]) -> None:
        """Process interactive messages (buttons, lists)."""
        try:
            if "button_reply" in message["interactive"]:
                button_id = message["interactive"]["button_reply"]["id"]
                await self._handle_button_response(sender, button_id)
            elif "list_reply" in message["interactive"]:
                list_id = message["interactive"]["list_reply"]["id"]
                await self._handle_list_response(sender, list_id)
        except KeyError as e:
            logger.error(f"Invalid interactive message format: {str(e)}")

    async def _handle_review_submission(self, review: ReviewRequest) -> None:
        """Handle review submission from WhatsApp."""
        try:
            # Submit review to main API
            response = requests.post(
                "http://localhost:8000/api/v1/reviews",
                json={
                    "product_id": review.product_id,
                    "rating": review.rating,
                    "text": review.review_text,
                    "source": "whatsapp",
                    "user_contact": review.phone_number
                }
            )
            response.raise_for_status()

            # Send confirmation
            await self.send_message(WhatsAppMessage(
                phone_number=review.phone_number,
                message="Thank you for your review! It has been submitted successfully."
            ))

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to submit review: {str(e)}")
            await self.send_message(WhatsAppMessage(
                phone_number=review.phone_number,
                message="Sorry, there was an error submitting your review. Please try again later."
            ))

    async def send_help_message(self, phone_number: str) -> None:
        """Send help message with available commands."""
        help_text = """
Available commands:
- review <product_id> <rating> <text>: Submit a review
- status: Check system status
- help: Show this help message

Example:
review 12345 5 Great product! Very satisfied with the quality.
        """.strip()

        await self.send_message(WhatsAppMessage(
            phone_number=phone_number,
            message=help_text
        ))

    async def send_status_message(self, phone_number: str) -> None:
        """Send system status message."""
        try:
            # Check main API status
            response = requests.get("http://localhost:8000/health")
            status = "🟢 Online" if response.status_code == 200 else "🔴 Offline"

            status_text = f"""
System Status:
API: {status}
Last Check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()

            await self.send_message(WhatsAppMessage(
                phone_number=phone_number,
                message=status_text
            ))

        except requests.exceptions.RequestException:
            await self.send_message(WhatsAppMessage(
                phone_number=phone_number,
                message="Unable to check system status at the moment."
            ))

    async def _handle_button_response(self, sender: str, button_id: str) -> None:
        """Handle button response."""
        # Implement button response logic
        pass

    async def _handle_list_response(self, sender: str, list_id: str) -> None:
        """Handle list response."""
        # Implement list response logic
        pass