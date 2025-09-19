"""
AI Chat Assistant with RAG (Retrieval Augmented Generation)
Provides intelligent, context-aware responses about phones using review data
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import re
import numpy as np
from collections import deque
import asyncio
import threading
from queue import Queue

# LangChain imports
from langchain import LLMChain, PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma, Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.schema import Document

# Model imports
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
import torch
from sentence_transformers import SentenceTransformer, util

# Database and storage
import chromadb
from chromadb.config import Settings
import faiss
import pickle
import sqlite3

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a chat message"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatContext:
    """Maintains conversation context"""
    messages: List[ChatMessage] = field(default_factory=list)
    current_phone: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))


class RAGChatAssistant:
    """
    Advanced Chat Assistant with RAG capabilities
    - Context-aware responses
    - Review data retrieval
    - Multi-turn conversations
    - Personalized recommendations
    """
    
    def __init__(self,
        
        model_name: str = "microsoft/DialoGPT-medium",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_gpu: bool = True,
        max_context_length: int = 2048
    ):
        """
        Initialize the chat assistant
        
        Args:
            model_name: Name of the conversational model
            embedding_model: Name of the embedding model
            use_gpu: Whether to use GPU
            max_context_length: Maximum context window
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.max_context_length = max_context_length
        
        logger.info(f"Initializing RAG Chat Assistant on {self.device}")
        
        # Initialize models
        self._initialize_models(model_name, embedding_model)
        
        # Initialize vector stores
        self._initialize_vector_stores()
        
        # Initialize memory and context
        self.conversation_memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            return_messages=True
        )
        
        # Response templates
        self._initialize_templates()
        
        # Knowledge base
        self.knowledge_base = {}
        self.review_database = None
        
    def _initialize_models(self, model_name: str, embedding_model: str):
        """Initialize language and embedding models"""
        try:
            # Load conversational model
            logger.info(f"Loading conversational model: {model_name}")
            
            if "gpt" in model_name.lower():
                # For GPT-based models
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            else:
                # For other models (T5, BART, etc.)
                self.chat_pipeline = pipeline(
                    "text-generation",
                    model=model_name,
                    device=0 if self.device == "cuda" else -1,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True
                )
            
            # Load embedding model
            logger.info(f"Loading embedding model: {embedding_model}")
            self.embedder = SentenceTransformer(embedding_model)
            
            # Additional models for specific tasks
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                device=0 if self.device == "cuda" else -1
            )
            
            self.qa_model = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        """Load lighter fallback models"""
        try:
            logger.info("Loading fallback models")
            self.chat_pipeline = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",
                device=-1
            )
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load fallback models: {e}")
    
    def _initialize_vector_stores(self):
        """Initialize vector stores for RAG"""
        try:
            # FAISS for fast similarity search
            self.faiss_index = None
            self.faiss_documents = []
            
            # ChromaDB for persistent storage
            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db"
            ))
            
            # Create or get collection
            try:
                self.chroma_collection = self.chroma_client.create_collection(
                    name="phone_reviews",
                    metadata={"hnsw:space": "cosine"}
                )
            except:
                self.chroma_collection = self.chroma_client.get_collection("phone_reviews")
            
            logger.info("Vector stores initialized")
            
        except Exception as e:
            logger.error(f"Error initializing vector stores: {e}")
            # Use simple in-memory storage as fallback
            self.simple_storage = {}
    
    def _initialize_templates(self):
        """Initialize response templates"""
        self.templates = {
            'greeting': [
                "Hello! I'm your AI phone assistant. I can help you find the perfect phone, "
                "answer questions about specific models, compare devices, and analyze reviews. "
                "What would you like to know?",
                "Hi there! 👋 I'm here to help you with phone decisions. "
                "Ask me anything about phone reviews, comparisons, or recommendations!"
            ],
            
            'clarification': [
                "Could you provide more details about what you're looking for?",
                "I'd be happy to help! Can you be more specific about your needs?",
                "Let me understand better - are you looking for {context}?"
            ],
            
            'recommendation': [
                "Based on your requirements, I recommend {phone} because {reason}",
                "For your needs, {phone} would be an excellent choice. {reason}",
                "Considering what you've told me, {phone} stands out because {reason}"
            ],
            
            'comparison': [
                "Comparing {phone1} and {phone2}:\n{comparison}",
                "Here's how {phone1} stacks up against {phone2}:\n{comparison}",
                "Let me break down the differences between {phone1} and {phone2}:\n{comparison}"
            ],
            
            'no_data': [
                "I don't have enough data about {phone} to answer that question accurately.",
                "Unfortunately, I lack sufficient reviews for {phone} to provide a detailed answer.",
                "I need more information about {phone} to give you a reliable answer."
            ]
        }
    
    def index_reviews(self, reviews: List[Dict[str, Any]], phone_model: str):
        """
        Index reviews for RAG retrieval
        
        Args:
            reviews: List of review dictionaries
            phone_model: Name of the phone model
        """
        logger.info(f"Indexing {len(reviews)} reviews for {phone_model}")
        
        # Prepare documents
        documents = []
        for review in reviews:
            text = review.get('text', '')
            rating = review.get('rating', 0)
            date = review.get('date', '')
            
            # Create structured document
            doc_text = f"Phone: {phone_model}\nRating: {rating}/5\nDate: {date}\nReview: {text}"
            documents.append(doc_text)
        
        # Create embeddings
        embeddings = self.embedder.encode(documents)
        
        # Store in FAISS
        if self.faiss_index is None:
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
        
        self.faiss_index.add(embeddings.astype('float32'))
        self.faiss_documents.extend(documents)
        
        # Store in ChromaDB
        try:
            self.chroma_collection.add(
                documents=documents,
                embeddings=embeddings.tolist(),
                metadatas=[{"phone": phone_model, "rating": r.get('rating', 0)} for r in reviews],
                ids=[f"{phone_model}_{i}" for i in range(len(reviews))]
            )
        except Exception as e:
            logger.error(f"Error adding to ChromaDB: {e}")
        
        # Update knowledge base
        if phone_model not in self.knowledge_base:
            self.knowledge_base[phone_model] = {
                'reviews': [],
                'summary': None,
                'stats': {}
            }
        
        self.knowledge_base[phone_model]['reviews'].extend(reviews)
        self._update_phone_stats(phone_model)
        
        logger.info(f"Successfully indexed reviews for {phone_model}")
    
    def _update_phone_stats(self, phone_model: str):
        """Update statistics for a phone model"""
        if phone_model not in self.knowledge_base:
            return
        
        reviews = self.knowledge_base[phone_model]['reviews']
        if not reviews:
            return
        
        ratings = [r.get('rating', 0) for r in reviews if r.get('rating')]
        
        stats = {
            'total_reviews': len(reviews),
            'average_rating': np.mean(ratings) if ratings else 0,
            'rating_distribution': {
                i: ratings.count(i) for i in range(1, 6)
            } if ratings else {},
            'last_updated': datetime.now().isoformat()
        }
        
        self.knowledge_base[phone_model]['stats'] = stats
    
    def retrieve_relevant_context(self,
        
        query: str,
        phone_model: Optional[str] = None,
        k: int = 5
    ) -> List[str]:
        """
        Retrieve relevant context using RAG
        
        Args:
            query: User query
            phone_model: Specific phone model (optional)
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        # Encode query
        query_embedding = self.embedder.encode([query])
        
        relevant_docs = []
        
        # Search in FAISS
        if self.faiss_index and self.faiss_index.ntotal > 0:
            distances, indices = self.faiss_index.search(
                query_embedding.astype('float32'), k
            )
            
            for idx in indices[0]:
    # Skip None or negative indices returned by FAISS (-1 indicates missing).
    if idx is None:
        continue
    if not isinstance(idx, int) or idx < 0:
        continue
    # Ensure index is within bounds of stored documents.
    if idx >= 0 and idx < len(self.faiss_documents):
        doc = self.faiss_documents[idx]
        if phone_model is None or phone_model.lower() in doc.lower():
            relevant_docs.append(doc)
# Search in ChromaDB as backup
        if len(relevant_docs) < k:
            try:
                results = self.chroma_collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=k - len(relevant_docs),
                    where={"phone": phone_model} if phone_model else None
                )
                
                if results['documents']:
                    relevant_docs.extend(results['documents'][0])
            except Exception as e:
                logger.error(f"Error querying ChromaDB: {e}")
        
        return relevant_docs[:k]
    
    def generate_response(self,
        
        query: str,
        context: ChatContext,
        use_rag: bool = True
    ) -> str:
        """
        Generate response to user query
        
        Args:
            query: User question
            context: Chat context
            use_rag: Whether to use RAG
            
        Returns:
            Generated response
        """
        # Detect intent
        intent = self._detect_intent(query)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Retrieve relevant context if using RAG
        relevant_context = []
        if use_rag:
            phone_model = entities.get('phone_model') or context.current_phone
            relevant_context = self.retrieve_relevant_context(query, phone_model)
        
        # Generate response based on intent
        if intent == 'greeting':
            response = self._handle_greeting()
        elif intent == 'comparison':
            response = self._handle_comparison(query, entities, relevant_context)
        elif intent == 'recommendation':
            response = self._handle_recommendation(query, context, relevant_context)
        elif intent == 'question':
            response = self._handle_question(query, entities, relevant_context)
        elif intent == 'review_analysis':
            response = self._handle_review_analysis(query, entities, relevant_context)
        else:
            response = self._handle_general(query, relevant_context)
        
        # Update context
        if entities.get('phone_model'):
            context.current_phone = entities['phone_model']
        
        return response
    
    def _detect_intent(self, query: str) -> str:
        """Detect user intent from query"""
        query_lower = query.lower()
        
        # Intent patterns
        intents = {
            'greeting': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon'],
            'comparison': ['compare', 'versus', 'vs', 'difference between', 'better than', 'or'],
            'recommendation': ['recommend', 'suggest', 'best', 'should i buy', 'what phone', 'which phone'],
            'review_analysis': ['reviews say', 'people think', 'user opinion', 'sentiment', 'feedback'],
            'question': ['what', 'how', 'when', 'where', 'why', 'is', 'does', 'can'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'budget'],
            'technical': ['specs', 'specifications', 'processor', 'ram', 'camera', 'battery', 'display']
        }
        
        for intent, keywords in intents.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        
        return 'general'
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from query"""
        entities = {}
        
        # Phone model patterns
        phone_patterns = [
            r'(iphone\s*\d+[\s\w]*)',
            r'(galaxy\s*[sazfn]\d+[\s\w]*)',
            r'(pixel\s*\d+[\s\w]*)',
            r'(oneplus\s*\d+[\s\w]*)',
            r'(xiaomi\s*[\w\s]+)',
            r'(huawei\s*[\w\s]+)',
            r'(oppo\s*[\w\s]+)',
            r'(vivo\s*[\w\s]+)'
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entities['phone_model'] = match.group(1).strip()
                break
        
        # Price range
        price_match = re.search(r'\$?(\d+)[\s-]*(?:to|\-)[\s-]*\$?(\d+)', query)
        if price_match:
            entities['price_range'] = (int(price_match.group(1)), int(price_match.group(2)))
        
        # Features
        features = ['camera', 'battery', 'display', 'performance', 'storage', 'design']
        mentioned_features = [f for f in features if f in query.lower()]
        if mentioned_features:
            entities['features'] = mentioned_features
        
        return entities
    
    def _handle_greeting(self) -> str:
        """Handle greeting intent"""
        import random
        return random.choice(self.templates['greeting'])
    
    def _handle_comparison(self, query: str, entities: Dict[str, Any], context: List[str]) -> str:
        """
        Improved comparison handler that specifically parses "A vs B" / "A versus B" patterns
        and falls back to splitting on common separators. Strips leading verbs like "compare".
        """
        # Normalize and remove leading comparison verbs/prefixes
        q = query.strip()
        q = re.sub(r'^(compare|compare:|compare\sof)\s+', '', q, flags=re.IGNORECASE)

        # Try explicit "A vs B" or "A versus B" pattern
        m = re.search(r'([\w0-9\s+\-\._]+?)\s+(?:vs|versus)\s+([\w0-9\s+\-\._]+)', q, re.IGNORECASE)
        if m:
            phone1 = m.group(1).strip()
            phone2 = m.group(2).strip()
        else:
            # Fallback: split on common separators
            parts = re.split(r'\s+(?:vs|versus|or|and)\s+', q, flags=re.IGNORECASE)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) >= 2:
                phone1, phone2 = parts[0], parts[1]
            else:
                return "Please specify two phone models to compare. For example: 'Compare iPhone 14 vs Galaxy S23'"

        # basic normalization
        phone1 = phone1.strip(' .,:;')
        phone2 = phone2.strip(' .,:;')

        # attempt to retrieve context for both phones
        ctx1 = self.retrieve_relevant_context(phone1)
        ctx2 = self.retrieve_relevant_context(phone2)

        # Simple textual comparison: collect top facts for each phone from context
        def summarize_ctx(ctx):
            if not ctx:
                return "No data available."
            joined = "\n".join(ctx[:3])
            return joined if joined else "No data available."

        summary1 = summarize_ctx(ctx1)
        summary2 = summarize_ctx(ctx2)

        response = (f"Comparison between '{phone1}' and '{phone2}':\n\n"
                    f"{phone1} summary:\n{summary1}\n\n"
                    f"{phone2} summary:\n{summary2}\n\n"
                    "If you want a feature-by-feature comparison (battery, camera, price), "
                    "ask for that specifically.")
        return response
    def _handle_recommendation(self,
        
        query: str,
        context: ChatContext,
        relevant_docs: List[str]
    ) -> str:
        """Handle recommendation queries"""
        # Analyze user preferences
        preferences = self._analyze_preferences(query, context)
        
        # Get candidate phones
        candidates = self._get_recommendation_candidates(preferences)
        
        if not candidates:
            return "I need more information about your preferences to make a recommendation. What's most important to you in a phone?"
        
        # Rank candidates
        ranked = self._rank_candidates(candidates, preferences, relevant_docs)
        
        if ranked:
            top_choice = ranked[0]
            reason = self._generate_recommendation_reason(top_choice, preferences)
            
            response = self.templates['recommendation'][0].format(
                phone=top_choice['model'],
                reason=reason
            )
            
            # Add alternatives
            if len(ranked) > 1:
                response += f"\n\nAlternatives to consider:\n"
                for alt in ranked[1:3]:
                    response += f"• {alt['model']}: {alt['key_strength']}\n"
            
            return response
        
        return "Based on your requirements, I'd need more specific information to make a recommendation."
    
    def _handle_question(self,
        
        query: str,
        entities: Dict[str, Any],
        context: List[str]
    ) -> str:
        """Handle question queries"""
        # Use QA model with context
        if context:
            combined_context = "\n".join(context[:3])
            
            try:
                answer = self.qa_model(
                    question=query,
                    context=combined_context
                )
                
                if answer['score'] > 0.5:
                    return answer['answer']
            except Exception as e:
                logger.error(f"QA model error: {e}")
        
        # Fallback to generation
        if entities.get('phone_model'):
            phone_data = self._get_phone_summary(entities['phone_model'])
            if phone_data:
                return self._generate_answer_from_data(query, phone_data)
        
        return "I don't have enough information to answer that question. Could you provide more context?"
    
    def _handle_review_analysis(self,
        
        query: str,
        entities: Dict[str, Any],
        context: List[str]
    ) -> str:
        """Handle review analysis queries"""
        phone_model = entities.get('phone_model')
        
        if not phone_model:
            return "Please specify which phone's reviews you'd like me to analyze."
        
        # Get review data
        if phone_model in self.knowledge_base:
            reviews = self.knowledge_base[phone_model]['reviews']
            stats = self.knowledge_base[phone_model]['stats']
            
            # Analyze sentiment
            sentiments = self._analyze_review_sentiments(reviews[:20])
            
            response = f"📊 Review Analysis for {phone_model}:\n\n"
            response += f"📈 Overall Rating: {stats['average_rating']:.1f}/5 "
            response += f"({stats['total_reviews']} reviews)\n\n"
            
            response += f"😊 Positive Sentiment: {sentiments['positive']:.0%}\n"
            response += f"😐 Neutral Sentiment: {sentiments['neutral']:.0%}\n"
            response += f"😔 Negative Sentiment: {sentiments['negative']:.0%}\n\n"
            
            # Add common themes
            themes = self._extract_review_themes(reviews[:10])
            if themes:
                response += "🎯 Common Themes:\n"
                for theme in themes[:5]:
                    response += f"• {theme}\n"
            
            return response
        
        return self.templates['no_data'][0].format(phone=phone_model)
    
    def _handle_general(self, query: str, context: List[str]) -> str:
        """Handle general queries"""
        # Generate response using language model
        if context:
            context_text = "\n".join(context[:2])
            prompt = f"Context: {context_text}\n\nQuestion: {query}\n\nAnswer:"
        else:
            prompt = f"Question: {query}\n\nAnswer:"
        
        try:
            if hasattr(self, 'chat_pipeline'):
                response = self.chat_pipeline(prompt, max_length=200)[0]['generated_text']
                # Extract just the answer part
                if "Answer:" in response:
                    response = response.split("Answer:")[-1].strip()
                return response
            else:
                # Use tokenizer and model directly
                inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=200,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "Answer:" in response:
                    response = response.split("Answer:")[-1].strip()
                return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm having trouble generating a response. Could you rephrase your question?"
    
    def _get_phone_summary(self, phone_model: str) -> Optional[Dict[str, Any]]:
        """Get summary data for a phone"""
        if phone_model in self.knowledge_base:
            return self.knowledge_base[phone_model]
        return None
    
    def _generate_comparison_text(self,
        
        phone1: str,
        data1: Optional[Dict],
        phone2: str,
        data2: Optional[Dict]
    ) -> str:
        """Generate comparison text between two phones"""
        comparison = ""
        
        if data1 and data2:
            # Compare ratings
            rating1 = data1['stats'].get('average_rating', 0)
            rating2 = data2['stats'].get('average_rating', 0)
            
            comparison += f"📊 Ratings:\n"
            comparison += f"• {phone1}: {rating1:.1f}/5\n"
            comparison += f"• {phone2}: {rating2:.1f}/5\n"
            comparison += f"Winner: {'⭐ ' + phone1 if rating1 > rating2 else '⭐ ' + phone2 if rating2 > rating1 else 'Tie'}\n\n"
            
            # Compare review counts
            comparison += f"📝 Review Count:\n"
            comparison += f"• {phone1}: {data1['stats']['total_reviews']} reviews\n"
            comparison += f"• {phone2}: {data2['stats']['total_reviews']} reviews\n\n"
            
            # Add recommendation
            if rating1 > rating2 + 0.3:
                comparison += f"💡 Recommendation: {phone1} has significantly better user satisfaction"
            elif rating2 > rating1 + 0.3:
                comparison += f"💡 Recommendation: {phone2} has significantly better user satisfaction"
            else:
                comparison += f"💡 Recommendation: Both phones have similar user satisfaction. Choose based on specific features you need."
        else:
            comparison = "Limited data available for comparison. Please ensure both phone models are in our database."
        
        return comparison
    
    def _analyze_preferences(self, query: str, context: ChatContext) -> Dict[str, Any]:
        """Analyze user preferences from query and context"""
        preferences = context.user_preferences.copy()
        
        # Extract preferences from current query
        query_lower = query.lower()
        
        # Budget preferences
        if 'budget' in query_lower or 'cheap' in query_lower:
            preferences['budget'] = 'low'
        elif 'premium' in query_lower or 'flagship' in query_lower:
            preferences['budget'] = 'high'
        
        # Use case preferences
        if 'gaming' in query_lower:
            preferences['use_case'] = 'gaming'
        elif 'photo' in query_lower or 'camera' in query_lower:
            preferences['use_case'] = 'photography'
        elif 'business' in query_lower or 'work' in query_lower:
            preferences['use_case'] = 'business'
        
        # Feature preferences
        for feature in ['camera', 'battery', 'display', 'performance']:
            if feature in query_lower:
                preferences['priority_feature'] = feature
        
        return preferences
    
    def _get_recommendation_candidates(self, preferences: Dict[str, Any]) -> List[Dict]:
        """Get candidate phones based on preferences"""
        candidates = []
        
        for phone_model, data in self.knowledge_base.items():
            if data['stats'].get('average_rating', 0) >= 3.5:
                candidate = {
                    'model': phone_model,
                    'rating': data['stats']['average_rating'],
                    'reviews': data['stats']['total_reviews']
                }
                candidates.append(candidate)
        
        return candidates
    
    def _rank_candidates(self,
        
        candidates: List[Dict],
        preferences: Dict[str, Any],
        context: List[str]
    ) -> List[Dict]:
        """Rank candidate phones based on preferences"""
        # Simple ranking by rating and review count
        for candidate in candidates:
            score = candidate['rating'] * 0.7 + min(candidate['reviews'] / 100, 1) * 0.3
            candidate['score'] = score
            
            # Adjust based on preferences
            if preferences.get('budget') == 'low' and 'pro' not in candidate['model'].lower():
                candidate['score'] *= 1.2
            elif preferences.get('budget') == 'high' and 'pro' in candidate['model'].lower():
                candidate['score'] *= 1.2
            
            # Add key strength
            if candidate['rating'] >= 4.5:
                candidate['key_strength'] = "Exceptional user satisfaction"
            elif candidate['reviews'] > 50:
                candidate['key_strength'] = "Well-tested by many users"
            else:
                candidate['key_strength'] = "Solid overall performer"
        
        # Sort by score
        return sorted(candidates, key=lambda x: x['score'], reverse=True)
    
    def _generate_recommendation_reason(self, phone: Dict, preferences: Dict) -> str:
        """Generate reason for recommendation"""
        reasons = []
        
        if phone['rating'] >= 4.5:
            reasons.append(f"it has an outstanding {phone['rating']:.1f}/5 rating")
        
        if phone['reviews'] > 50:
            reasons.append(f"it's been thoroughly tested by {phone['reviews']} users")
        
        if preferences.get('use_case'):
            reasons.append(f"it's excellent for {preferences['use_case']}")
        
        if not reasons:
            reasons.append("it offers great overall value")
        
        return " and ".join(reasons)
    
    def _generate_answer_from_data(self, query: str, phone_data: Dict) -> str:
        """Generate answer from phone data"""
        stats = phone_data['stats']
        
        # Simple keyword-based response
        query_lower = query.lower()
        
        if 'rating' in query_lower or 'score' in query_lower:
            return f"The average rating is {stats['average_rating']:.1f}/5 based on {stats['total_reviews']} reviews."
        elif 'popular' in query_lower:
            return f"With {stats['total_reviews']} reviews, this phone has gained significant user attention."
        elif 'recommend' in query_lower:
            if stats['average_rating'] >= 4:
                return f"Yes, with a {stats['average_rating']:.1f}/5 rating, users generally recommend this phone."
            else:
                return f"The {stats['average_rating']:.1f}/5 rating suggests mixed user opinions. Consider your specific needs."
        
        return "Based on the available data, this phone has received varied user feedback. Would you like specific details?"
    
    def _analyze_review_sentiments(self, reviews: List[Dict]) -> Dict[str, float]:
        """Analyze sentiments from reviews"""
        sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for review in reviews:
            try:
                text = review.get('text', '')
                if text:
                    result = self.sentiment_analyzer(text[:512])[0]
                    label = result['label']
                    
                    if 'pos' in label.lower() or '5' in label or '4' in label:
                        sentiments['positive'] += 1
                    elif 'neg' in label.lower() or '1' in label or '2' in label:
                        sentiments['negative'] += 1
                    else:
                        sentiments['neutral'] += 1
            except:
                continue
        
        total = sum(sentiments.values())
        if total > 0:
            for key in sentiments:
                sentiments[key] /= total
        
        return sentiments
    
    def _extract_review_themes(self, reviews: List[Dict]) -> List[str]:
        """Extract common themes from reviews"""
        themes = []
        all_text = " ".join([r.get('text', '') for r in reviews])
        
        # Simple keyword extraction
        keywords = {
            'camera': ['camera', 'photo', 'picture', 'selfie'],
            'battery': ['battery', 'charge', 'charging', 'power'],
            'performance': ['fast', 'slow', 'lag', 'smooth', 'performance'],
            'display': ['screen', 'display', 'bright', 'color'],
            'value': ['price', 'value', 'worth', 'expensive', 'cheap']
        }
        
        for theme, words in keywords.items():
            if any(word in all_text.lower() for word in words):
                themes.append(theme.capitalize())
        
        return themes
    
    async def chat(self,
        
        message: str,
        context: Optional[ChatContext] = None,
        stream: bool = False
    ) -> str:
        """
        Main chat interface
        
        Args:
            message: User message
            context: Chat context
            stream: Whether to stream response
            
        Returns:
            Assistant response
        """
        if context is None:
            context = ChatContext()
        
        # Add user message to context
        context.messages.append(ChatMessage(role="user", content=message))
        
        # Generate response
        response = self.generate_response(message, context)
        
        # Add assistant response to context
        context.messages.append(ChatMessage(role="assistant", content=response))
        
        # Handle streaming if requested
        if stream:
            return self._stream_response(response)
        
        return response
    
    def _stream_response(self, response: str) -> str:
        """Stream response character by character"""
        # This would typically yield characters for real-time display
        # For now, return the full response
        return response
    
    def save_conversation(self, context: ChatContext, filepath: str):
        """Save conversation history"""
        conversation_data = {
            'session_id': context.session_id,
            'messages': [
                {
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp.isoformat()
                }
            for msg in context.messages:
            ],
            'current_phone': context.current_phone,
            'user_preferences': context.user_preferences
        }
        
        with open(filepath, 'w') as f:
            json.dump(conversation_data, f, indent=2)
    
    def load_conversation(self, filepath: str) -> ChatContext:
        """Load conversation history"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        context = ChatContext(
            session_id=data['session_id'],
            current_phone=data.get('current_phone'),
            user_preferences=data.get('user_preferences', {})
        )
        
        for msg_data in data['messages']:
            msg = ChatMessage(
                role=msg_data['role'],
                content=msg_data['content'],
                timestamp=datetime.fromisoformat(msg_data['timestamp'])
            )
            context.messages.append(msg)
        
        return context


# Convenience functions
def create_chat_assistant() -> RAGChatAssistant:
    """Create a chat assistant instance"""
    return RAGChatAssistant()


async def chat_with_assistant(
    assistant: RAGChatAssistant,
    message: str,
    context: Optional[ChatContext] = None
) -> Tuple[str, ChatContext]:
    """
    Chat with the assistant
    
    Returns:
        Tuple of (response, updated_context)
    """
    if context is None:
        context = ChatContext()
    
    response = await assistant.chat(message, context)
    return response, context


# Example usage
if __name__ == "__main__":
    # Initialize assistant
    assistant = RAGChatAssistant()
    
    # Sample reviews for indexing
    sample_reviews = [
        {
            'text': 'Amazing phone with great camera quality! Battery lasts all day.',
            'rating': 5,
            'date': '2024-01-15'
        },
        {
            'text': 'Good value for money but the display could be brighter.',
            'rating': 4,
            'date': '2024-01-20'
        },
        {
            'text': 'Disappointing performance, lags during gaming.',
            'rating': 2,
            'date': '2024-01-25'
        }
    ]
    
    # Index reviews
    assistant.index_reviews(sample_reviews, "iPhone 14 Pro")
    
    # Create chat context
    context = ChatContext()
    
    # Example conversations
    test_queries = [
        "Hello! I'm looking for a new phone",
        "What do people think about the iPhone 14 Pro?",
        "Compare iPhone 14 Pro vs Galaxy S23",
        "I need a phone with great camera for photography",
        "What's the average rating for iPhone 14 Pro?",
        "Should I buy the iPhone 14 Pro?"
    ]
    
    print("=" * 80)
    print("AI CHAT ASSISTANT DEMO")
    print("=" * 80)
    
    import asyncio

    async def demo():
        for query in test_queries:
            print(f"\n👤 User: {query}")
            response = await assistant.chat(query, context)
            print(f"🤖 Assistant: {response}")
            print("-" * 40)
    
    # Run demo
    asyncio.run(demo())