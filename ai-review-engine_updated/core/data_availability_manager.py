"""
Data Availability Manager

Replaces all synthetic data generation with proper error handling,
data availability checks, and user-friendly messaging when data is unavailable.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class DataUnavailableReason(Enum):
    """Reasons why data might be unavailable"""
    NO_MATCHING_PHONE = "no_matching_phone"
    API_TIMEOUT = "api_timeout"
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    INSUFFICIENT_DATA = "insufficient_data"
    DATA_SOURCE_OFFLINE = "data_source_offline"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_QUERY = "invalid_query"
    DATABASE_ERROR = "database_error"


class DataAvailabilityStatus:
    """Represents the availability status of requested data"""
    
    def __init__(
        self, 
        available: bool, 
        reason: Optional[DataUnavailableReason] = None,
        message: str = "",
        suggestions: List[str] = None,
        retry_after: Optional[int] = None,
        partial_data: Optional[Dict[str, Any]] = None
    ):
        self.available = available
        self.reason = reason
        self.message = message
        self.suggestions = suggestions or []
        self.retry_after = retry_after  # seconds
        self.partial_data = partial_data or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason.value if self.reason else None,
            "message": self.message,
            "suggestions": self.suggestions,
            "retry_after": self.retry_after,
            "partial_data": self.partial_data,
            "timestamp": self.timestamp.isoformat()
        }


class DataAvailabilityManager:
    """
    Manages data availability and provides proper error handling
    instead of generating synthetic data.
    """
    
    def __init__(self):
        self.availability_cache = {}
        self.error_messages = {
            DataUnavailableReason.NO_MATCHING_PHONE: {
                "message": "No information found for the requested phone model.",
                "suggestions": [
                    "Check the spelling of the phone model",
                    "Try searching with just the brand name",
                    "Use the full official product name",
                    "Browse available phones by brand"
                ]
            },
            DataUnavailableReason.API_TIMEOUT: {
                "message": "Data sources are currently slow to respond.",
                "suggestions": [
                    "Please try again in a few moments",
                    "Check your internet connection",
                    "Try a more specific search query"
                ]
            },
            DataUnavailableReason.API_ERROR: {
                "message": "External data sources are currently unavailable.",
                "suggestions": [
                    "Please try again later",
                    "Contact support if the issue persists",
                    "Try browsing recently cached results"
                ]
            },
            DataUnavailableReason.NETWORK_ERROR: {
                "message": "Unable to connect to data sources.",
                "suggestions": [
                    "Check your internet connection",
                    "Try again in a few moments",
                    "Contact your network administrator if the issue persists"
                ]
            },
            DataUnavailableReason.INSUFFICIENT_DATA: {
                "message": "Insufficient data available for this phone.",
                "suggestions": [
                    "Try a more popular phone model",
                    "Check back later as we regularly update our database",
                    "Contact support to request this phone be added"
                ]
            },
            DataUnavailableReason.DATA_SOURCE_OFFLINE: {
                "message": "Primary data sources are temporarily offline.",
                "suggestions": [
                    "Please try again in a few minutes",
                    "Check our status page for updates",
                    "Contact support if the issue persists"
                ]
            },
            DataUnavailableReason.RATE_LIMIT_EXCEEDED: {
                "message": "Too many requests. Please wait before trying again.",
                "suggestions": [
                    "Wait a few minutes before your next request",
                    "Consider upgrading to a premium plan for higher limits",
                    "Try refining your search to be more specific"
                ]
            },
            DataUnavailableReason.INVALID_QUERY: {
                "message": "The search query is not valid or too broad.",
                "suggestions": [
                    "Include a specific phone model in your search",
                    "Use the format 'Brand Model' (e.g., 'iPhone 15 Pro')",
                    "Avoid using special characters or very short queries"
                ]
            },
            DataUnavailableReason.DATABASE_ERROR: {
                "message": "Database temporarily unavailable.",
                "suggestions": [
                    "Please try again in a few moments",
                    "Contact support if the issue persists",
                    "Check our status page for updates"
                ]
            }
        }
    
    def check_phone_data_availability(self, phone_query: str) -> DataAvailabilityStatus:
        """
        Check if phone data is available instead of generating synthetic data.
        
        Args:
            phone_query: The phone model query
            
        Returns:
            DataAvailabilityStatus indicating availability and reason if not available
        """
        # Cache key for this query
        cache_key = f"phone_{phone_query.lower().strip()}"
        
        # Check cache first (avoid repeated API calls)
        if cache_key in self.availability_cache:
            cached = self.availability_cache[cache_key]
            if datetime.now() - cached.timestamp < timedelta(minutes=5):
                return cached
        
        # Validate query format
        if not phone_query or len(phone_query.strip()) < 3:
            status = DataAvailabilityStatus(
                available=False,
                reason=DataUnavailableReason.INVALID_QUERY,
                **self.error_messages[DataUnavailableReason.INVALID_QUERY]
            )
            self.availability_cache[cache_key] = status
            return status
        
        # Check if query looks like a valid phone model
        if not self._is_valid_phone_query(phone_query):
            status = DataAvailabilityStatus(
                available=False,
                reason=DataUnavailableReason.INVALID_QUERY,
                **self.error_messages[DataUnavailableReason.INVALID_QUERY]
            )
            self.availability_cache[cache_key] = status
            return status
        
        # For now, return unavailable with suggestion to check database
        # In a real implementation, this would check actual data sources
        status = DataAvailabilityStatus(
            available=False,
            reason=DataUnavailableReason.NO_MATCHING_PHONE,
            **self.error_messages[DataUnavailableReason.NO_MATCHING_PHONE]
        )
        
        self.availability_cache[cache_key] = status
        return status
    
    def check_api_availability(self, api_name: str) -> DataAvailabilityStatus:
        """
        Check if an API is available instead of using mock data.
        
        Args:
            api_name: Name of the API to check
            
        Returns:
            DataAvailabilityStatus indicating API availability
        """
        # In a real implementation, this would ping the actual APIs
        # For now, return unavailable to prevent mock data usage
        
        return DataAvailabilityStatus(
            available=False,
            reason=DataUnavailableReason.API_ERROR,
            **self.error_messages[DataUnavailableReason.API_ERROR],
            retry_after=300  # 5 minutes
        )
    
    def get_alternative_suggestions(self, failed_query: str) -> List[str]:
        """
        Get alternative suggestions when data is not available.
        
        Args:
            failed_query: The query that failed
            
        Returns:
            List of alternative suggestions
        """
        suggestions = [
            "Browse phones by brand",
            "Check our most popular phones list",
            "Try a different phone model from the same brand",
            "Contact support to request this phone be added"
        ]
        
        # Add query-specific suggestions
        if " " not in failed_query.strip():
            suggestions.insert(0, "Try adding the brand name to your search")
        
        return suggestions
    
    def create_data_unavailable_response(
        self, 
        query: str, 
        reason: DataUnavailableReason,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a proper response for when data is unavailable.
        
        Args:
            query: The original query
            reason: Reason for unavailability
            additional_context: Additional context information
            
        Returns:
            Structured response for data unavailability
        """
        error_info = self.error_messages[reason]
        
        response = {
            "success": False,
            "data_available": False,
            "query": query,
            "error": {
                "code": reason.value,
                "message": error_info["message"],
                "suggestions": error_info["suggestions"],
                "timestamp": datetime.now().isoformat()
            },
            "alternatives": self.get_alternative_suggestions(query)
        }
        
        if additional_context:
            response["context"] = additional_context
        
        return response
    
    def _is_valid_phone_query(self, query: str) -> bool:
        """
        Check if a query looks like a valid phone model query.
        
        Args:
            query: Query string to validate
            
        Returns:
            True if query appears valid for phone search
        """
        query_lower = query.lower().strip()
        
        # Common phone brands
        phone_brands = [
            'iphone', 'samsung', 'galaxy', 'pixel', 'oneplus', 'xiaomi', 
            'huawei', 'oppo', 'vivo', 'realme', 'motorola', 'nokia', 
            'sony', 'lg', 'nothing'
        ]
        
        # Check if query contains a known brand
        has_brand = any(brand in query_lower for brand in phone_brands)
        
        # Check if query has reasonable length
        reasonable_length = 3 <= len(query) <= 50
        
        # Check if query doesn't contain obvious non-phone terms
        non_phone_terms = ['laptop', 'computer', 'tablet', 'tv', 'watch', 'car']
        has_non_phone_terms = any(term in query_lower for term in non_phone_terms)
        
        return has_brand and reasonable_length and not has_non_phone_terms
    
    def log_data_unavailability(
        self, 
        query: str, 
        reason: DataUnavailableReason,
        source: str = "unknown"
    ):
        """
        Log data unavailability for monitoring and improvement.
        
        Args:
            query: The query that failed
            reason: Reason for unavailability  
            source: Source component that requested the data
        """
        logger.info(
            f"Data unavailable: query='{query}', reason={reason.value}, source={source}"
        )


# Global instance
data_availability_manager = DataAvailabilityManager()


def check_data_availability(query: str) -> DataAvailabilityStatus:
    """
    Convenience function to check data availability.
    
    Args:
        query: Query to check availability for
        
    Returns:
        DataAvailabilityStatus indicating availability
    """
    return data_availability_manager.check_phone_data_availability(query)


def create_unavailable_response(
    query: str, 
    reason: DataUnavailableReason,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to create unavailable data response.
    
    Args:
        query: Original query
        reason: Reason for unavailability
        context: Additional context
        
    Returns:
        Structured unavailable response
    """
    return data_availability_manager.create_data_unavailable_response(
        query, reason, context
    )