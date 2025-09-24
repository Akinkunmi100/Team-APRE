"""
Conversational AI Assistant for Phone Review Engine
Provides natural language interface for phone discovery and queries
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class ConversationContext:
    """Stores conversation context and history"""
    user_preferences: Dict[str, Any] = None
    conversation_history: List[Dict[str, str]] = None
    current_intent: str = None
    extracted_entities: Dict[str, Any] = None
    session_id: str = None
    
    def __post_init__(self):
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.conversation_history is None:
            self.conversation_history = []
        if self.extracted_entities is None:
            self.extracted_entities = {}

class ConversationalAI:
    """Advanced conversational AI for phone recommendations and queries"""
    
    def __init__(self):
        """Initialize the conversational AI system"""
        self.intent_patterns = self._initialize_intent_patterns()
        self.entity_patterns = self._initialize_entity_patterns()
        self.response_templates = self._initialize_response_templates()
        self.phone_features = self._initialize_phone_features()
        self.conversation_contexts = {}  # Store user sessions
        
        logger.info("Conversational AI initialized")
    
    def _initialize_intent_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for intent recognition"""
        return {
            'find_phone': [
                r'find.*phone|recommend.*phone|suggest.*phone|looking for.*phone',
                r'want.*phone|need.*phone|buy.*phone|purchase.*phone',
                r'best phone|good phone|phone recommendation'
            ],
            'compare_phones': [
                r'compare.*phone|vs|versus|difference between',
                r'which.*better|better than|which.*choose',
                r'compare.*and.*|.*vs.*|.*versus.*'
            ],
            'ask_feature': [
                r'how.*camera|camera.*quality|photo.*quality',
                r'battery.*life|battery.*last|how long.*battery',
                r'performance|speed|fast|processor|gaming',
                r'price|cost|expensive|cheap|budget'
            ],
            'ask_specific': [
                r'tell me about|what about|information about',
                r'review.*of|opinion.*on|thoughts.*on',
                r'is.*good|how.*is'
            ],
            'greeting': [
                r'hello|hi|hey|good morning|good afternoon',
                r'start|begin|help me'
            ],
            'goodbye': [
                r'bye|goodbye|thanks|thank you|that\'s all',
                r'done|finished|no more questions'
            ]
        }
    
    def _initialize_entity_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for entity extraction"""
        return {
            'brands': [
                r'apple|iphone', r'samsung|galaxy', r'google|pixel', 
                r'oneplus|1\+', r'xiaomi|redmi', r'huawei', r'oppo', 
                r'vivo', r'motorola', r'nokia', r'sony'
            ],
            'features': [
                r'camera|photo|picture|selfie', r'battery|power|charge',
                r'screen|display|size', r'performance|speed|fast|processor',
                r'storage|memory|gb|tb', r'price|cost|budget|cheap|expensive',
                r'gaming|game', r'design|look|appearance', r'5g|network'
            ],
            'budget_ranges': [
                r'under \$?(\d+)', r'less than \$?(\d+)', r'below \$?(\d+)',
                r'\$?(\d+)-\$?(\d+)', r'\$?(\d+) to \$?(\d+)',
                r'budget|cheap|affordable|expensive|premium'
            ],
            'phone_models': [
                r'iphone \d+|iphone \d+ pro|iphone \d+ plus',
                r'galaxy s\d+|galaxy note \d+|galaxy a\d+',
                r'pixel \d+|pixel \d+ pro|pixel \d+a',
                r'oneplus \d+|oneplus \d+ pro'
            ]
        }
    
    def _initialize_response_templates(self) -> Dict[str, List[str]]:
        """Initialize response templates for different intents"""
        return {
            'find_phone': [
                "I'd be happy to help you find the perfect phone! {context}",
                "Let me help you discover the ideal phone for your needs. {context}",
                "Great! I can recommend some excellent phones based on what you're looking for. {context}"
            ],
            'compare_phones': [
                "I'll help you compare these phones side by side. {context}",
                "Let me analyze the differences between these phones for you. {context}",
                "Here's a detailed comparison of the phones you mentioned. {context}"
            ],
            'ask_feature': [
                "Here's what I found about {feature} in the phones we have data for: {context}",
                "Based on user reviews, here's the {feature} analysis: {context}",
                "Let me break down the {feature} performance for you: {context}"
            ],
            'greeting': [
                "Hello! I'm your AI phone assistant. I can help you find phones, compare options, or answer questions about any device. What would you like to know?",
                "Hi there! I'm here to help you make the best phone choice. You can ask me to recommend phones, compare models, or get insights from real user reviews. How can I assist you?",
                "Welcome! I'm your intelligent phone shopping assistant. I can analyze thousands of reviews to help you find the perfect phone. What are you looking for today?"
            ],
            'goodbye': [
                "Thank you for using the AI Phone Assistant! Feel free to ask if you need more help finding the perfect phone.",
                "You're welcome! I hope I helped you make a great phone choice. Come back anytime for more recommendations!",
                "Goodbye! Remember, I'm always here to help with your phone questions and recommendations."
            ],
            'clarification': [
                "Could you tell me more about what you're looking for? For example, your budget, preferred features, or how you'll use the phone?",
                "I'd like to give you the best recommendations. What's most important to you - camera quality, battery life, performance, or something else?",
                "To help you better, could you share your budget range and what features matter most to you?"
            ]
        }
    
    def _initialize_phone_features(self) -> Dict[str, List[str]]:
        """Initialize feature synonyms for better understanding"""
        return {
            'camera': ['camera', 'photo', 'picture', 'selfie', 'photography', 'video', 'recording'],
            'battery': ['battery', 'power', 'charge', 'charging', 'life', 'lasting'],
            'performance': ['performance', 'speed', 'fast', 'processor', 'cpu', 'lag', 'smooth', 'gaming'],
            'display': ['screen', 'display', 'size', 'resolution', 'bright', 'colors'],
            'storage': ['storage', 'memory', 'space', 'gb', 'tb', 'capacity'],
            'price': ['price', 'cost', 'budget', 'cheap', 'expensive', 'affordable', 'value'],
            'design': ['design', 'look', 'appearance', 'style', 'build', 'quality'],
            'connectivity': ['5g', '4g', 'wifi', 'network', 'internet', 'connection']
        }
    
    def process_query(self, user_input: str, session_id: str = "default", 
                     context_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process user query and generate appropriate response
        
        Args:
            user_input: User's natural language input
            session_id: Unique session identifier
            context_data: Additional context (like available phone data)
            
        Returns:
            Response dictionary with answer, intent, and entities
        """
        # Get or create conversation context
        if session_id not in self.conversation_contexts:
            self.conversation_contexts[session_id] = ConversationContext(session_id=session_id)
        
        context = self.conversation_contexts[session_id]
        
        # Add to conversation history
        context.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'type': 'user'
        })
        
        # Extract intent and entities
        intent = self._extract_intent(user_input)
        entities = self._extract_entities(user_input)
        
        # Update context
        context.current_intent = intent
        context.extracted_entities.update(entities)
        
        # Generate response based on intent
        response = self._generate_response(intent, entities, context, context_data)
        
        # Add response to history
        context.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'ai_response': response['message'],
            'type': 'assistant'
        })
        
        return response
    
    def _extract_intent(self, user_input: str) -> str:
        """Extract user intent from input"""
        user_input_lower = user_input.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    return intent
        
        return 'unknown'
    
    def _extract_entities(self, user_input: str) -> Dict[str, Any]:
        """Extract entities from user input"""
        entities = {}
        user_input_lower = user_input.lower()
        
        # Extract brands
        for pattern in self.entity_patterns['brands']:
            if re.search(pattern, user_input_lower):
                entities['brand'] = re.search(pattern, user_input_lower).group()
                break
        
        # Extract features
        mentioned_features = []
        for pattern in self.entity_patterns['features']:
            if re.search(pattern, user_input_lower):
                mentioned_features.append(re.search(pattern, user_input_lower).group())
        
        if mentioned_features:
            entities['features'] = mentioned_features
        
        # Extract budget
        for pattern in self.entity_patterns['budget_ranges']:
            match = re.search(pattern, user_input_lower)
            if match:
                entities['budget'] = match.group()
                break
        
        # Extract specific phone models
        for pattern in self.entity_patterns['phone_models']:
            match = re.search(pattern, user_input_lower)
            if match:
                entities['phone_model'] = match.group()
                break
        
        return entities
    
    def _generate_response(self, intent: str, entities: Dict[str, Any], 
                          context: ConversationContext, context_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate appropriate response based on intent and entities"""
        
        if intent == 'greeting':
            return {
                'message': self._get_random_template('greeting'),
                'intent': intent,
                'entities': entities,
                'action': 'show_welcome',
                'suggestions': [
                    "Find me a phone under $500",
                    "Compare iPhone vs Samsung",
                    "What's the best camera phone?",
                    "Show me gaming phones"
                ]
            }
        
        elif intent == 'find_phone':
            return self._handle_find_phone(entities, context, context_data)
        
        elif intent == 'compare_phones':
            return self._handle_compare_phones(entities, context, context_data)
        
        elif intent == 'ask_feature':
            return self._handle_feature_question(entities, context, context_data)
        
        elif intent == 'ask_specific':
            return self._handle_specific_question(entities, context, context_data)
        
        elif intent == 'goodbye':
            return {
                'message': self._get_random_template('goodbye'),
                'intent': intent,
                'entities': entities,
                'action': 'end_conversation'
            }
        
        else:
            # Unknown intent - ask for clarification
            return {
                'message': self._get_random_template('clarification'),
                'intent': 'clarification',
                'entities': entities,
                'action': 'ask_clarification',
                'suggestions': [
                    "Find me a phone",
                    "Compare two phones", 
                    "Tell me about camera quality",
                    "What's the best budget phone?"
                ]
            }
    
    def _handle_find_phone(self, entities: Dict[str, Any], context: ConversationContext, 
                          context_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Handle phone finding requests"""
        
        # Build search criteria from entities and context
        search_criteria = {}
        
        if 'brand' in entities:
            search_criteria['brand'] = entities['brand']
        
        if 'budget' in entities:
            search_criteria['budget'] = entities['budget']
        
        if 'features' in entities:
            search_criteria['preferred_features'] = entities['features']
        
        # Update user preferences in context
        context.user_preferences.update(search_criteria)
        
        message = f"I'll help you find the perfect phone! "
        
        if search_criteria:
            criteria_text = []
            if 'brand' in search_criteria:
                criteria_text.append(f"Brand: {search_criteria['brand']}")
            if 'budget' in search_criteria:
                criteria_text.append(f"Budget: {search_criteria['budget']}")
            if 'preferred_features' in search_criteria:
                features = ', '.join(search_criteria['preferred_features'])
                criteria_text.append(f"Important features: {features}")
            
            message += f"Based on your preferences ({', '.join(criteria_text)}), here are my top recommendations:"
        else:
            message += "Let me show you some of the most popular phones based on user reviews:"
        
        return {
            'message': message,
            'intent': 'find_phone',
            'entities': entities,
            'action': 'show_recommendations',
            'search_criteria': search_criteria,
            'suggestions': [
                "Tell me more about the top choice",
                "Compare the top 2 phones",
                "Show me alternatives",
                "What about budget options?"
            ]
        }
    
    def _handle_compare_phones(self, entities: Dict[str, Any], context: ConversationContext,
                              context_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Handle phone comparison requests"""
        
        message = "I'll help you compare phones! "
        
        if 'phone_model' in entities:
            message += f"I can see you mentioned {entities['phone_model']}. "
        
        if 'brand' in entities:
            message += f"You're interested in {entities['brand']} phones. "
        
        message += "Let me analyze the differences in features, performance, and user satisfaction based on real reviews."
        
        return {
            'message': message,
            'intent': 'compare_phones',
            'entities': entities,
            'action': 'show_comparison',
            'comparison_criteria': {
                'phones': entities.get('phone_model', []),
                'brands': entities.get('brand', []),
                'features': entities.get('features', ['overall', 'camera', 'battery', 'performance'])
            },
            'suggestions': [
                "Which has better camera?",
                "Compare battery life",
                "What about value for money?",
                "Show detailed specs"
            ]
        }
    
    def _handle_feature_question(self, entities: Dict[str, Any], context: ConversationContext,
                                context_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Handle specific feature questions"""
        
        feature = entities.get('features', ['general'])[0] if 'features' in entities else 'general'
        
        # Map feature to standard categories
        feature_mapping = {
            'camera': 'camera quality',
            'battery': 'battery life', 
            'performance': 'performance',
            'screen': 'display quality',
            'price': 'value for money'
        }
        
        mapped_feature = feature_mapping.get(feature, feature)
        
        message = f"Great question about {mapped_feature}! Based on thousands of user reviews, I can provide insights about {mapped_feature} across different phone models. Let me analyze the data for you."
        
        return {
            'message': message,
            'intent': 'ask_feature',
            'entities': entities,
            'action': 'show_feature_analysis',
            'feature': mapped_feature,
            'suggestions': [
                f"Best phones for {mapped_feature}",
                f"Compare {mapped_feature} across brands",
                f"Budget phones with good {mapped_feature}",
                "Show me overall rankings"
            ]
        }
    
    def _handle_specific_question(self, entities: Dict[str, Any], context: ConversationContext,
                                 context_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Handle questions about specific phones"""
        
        phone_model = entities.get('phone_model', 'the phone you mentioned')
        
        message = f"I'd be happy to tell you about {phone_model}! I'll analyze user reviews to give you insights about its strengths, weaknesses, and overall user satisfaction."
        
        return {
            'message': message,
            'intent': 'ask_specific',
            'entities': entities,
            'action': 'show_phone_details',
            'phone_model': phone_model,
            'suggestions': [
                "What do users like most?",
                "Any common complaints?",
                "How does it compare to competitors?",
                "Is it worth buying?"
            ]
        }
    
    def _get_random_template(self, template_type: str) -> str:
        """Get a random response template"""
        templates = self.response_templates.get(template_type, ["I understand."])
        return np.random.choice(templates)
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session"""
        if session_id in self.conversation_contexts:
            return self.conversation_contexts[session_id].conversation_history
        return []
    
    def clear_conversation(self, session_id: str):
        """Clear conversation context for a session"""
        if session_id in self.conversation_contexts:
            del self.conversation_contexts[session_id]
    
    def get_user_preferences(self, session_id: str) -> Dict[str, Any]:
        """Get accumulated user preferences from conversation"""
        if session_id in self.conversation_contexts:
            return self.conversation_contexts[session_id].user_preferences
        return {}

# Example usage and testing
if __name__ == "__main__":
    # Test the conversational AI
    ai = ConversationalAI()
    
    # Test queries
    test_queries = [
        "Hello, I need help finding a phone",
        "I want an iPhone with good camera under $800",
        "Compare iPhone 15 vs Samsung Galaxy S24",
        "What's the best camera phone?",
        "Tell me about battery life in Pixel phones"
    ]
    
    for query in test_queries:
        print(f"User: {query}")
        response = ai.process_query(query, "test_session")
        print(f"AI: {response['message']}")
        print(f"Action: {response.get('action', 'none')}")
        print("-" * 50)