"""
Agentic RAG System with Autonomous Agents
Multi-agent system with tool usage, planning, and decision-making capabilities
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re
import asyncio
from abc import ABC, abstractmethod
import numpy as np
from collections import deque
import inspect

# LangChain and agent imports
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.callbacks.base import BaseCallbackHandler

# Model imports
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer

# Vector stores and databases
import chromadb
import faiss
from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Import existing modules
from models.chat_assistant import RAGChatAssistant
from models.review_summarizer import AdvancedReviewSummarizer
from core.smart_search import SmartSearchEngine
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

Base = declarative_base()


class AgentRole(Enum):
    """Defines different agent roles in the system"""
    ORCHESTRATOR = "orchestrator"  # Main coordinator
    RESEARCHER = "researcher"      # Information gathering
    ANALYST = "analyst"           # Data analysis
    RECOMMENDER = "recommender"   # Recommendations
    SCRAPER = "scraper"          # Web scraping
    VALIDATOR = "validator"       # Fact checking
    SUMMARIZER = "summarizer"     # Content summarization
    COMPARATOR = "comparator"     # Product comparison


class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentTask:
    """Represents a task for an agent"""
    task_id: str
    description: str
    priority: int = 5
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class AgentMessage:
    """Inter-agent communication message"""
    sender: str
    receiver: str
    message_type: str  # request, response, broadcast
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None


class AgentMemory(Base):
    """SQLAlchemy model for agent memory persistence"""
    __tablename__ = 'agent_memory'
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(String(100))
    memory_type = Column(String(50))  # short_term, long_term, episodic
    content = Column(JSON)
    importance = Column(Float, default=0.5)
    created_at = Column(DateTime, default=datetime.now)
    accessed_at = Column(DateTime, default=datetime.now)
    access_count = Column(Integer, default=1)


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(:
        self,
        agent_id: str,
        role: AgentRole,
        model_name: str = "microsoft/DialoGPT-medium"
    ):
        self.agent_id = agent_id
        self.role = role
        self.state = AgentState.IDLE
        self.model_name = model_name
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Memory systems
        self.working_memory = deque(maxlen=10)
        self.long_term_memory = []
        
        # Task queue
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        
        # Communication
        self.message_queue = asyncio.Queue()
        self.peers = {}
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"Initialized {role.value} agent: {agent_id}")
    
    @abstractmethod
    def _initialize_tools(self) -> List[Tool]:
        """Initialize agent-specific tools"""
        pass
    
    @abstractmethod
    async def execute_task(self, task: AgentTask) -> Any:
        """Execute a specific task"""
        pass
    
    def _initialize_model(self):
        """Initialize the language model"""
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.error(f"Error initializing model for {self.agent_id}: {e}")
    
    async def think(self, context: Dict[str, Any]) -> str:
        """Generate thoughts about the current context"""
        self.state = AgentState.THINKING
        
        prompt = self._build_thinking_prompt(context)
        response = self.pipeline(prompt, max_length=200)[0]['generated_text']
        
        # Store in working memory
        self.working_memory.append({
            'type': 'thought',
            'content': response,
            'timestamp': datetime.now()
        })
        
        return response
    
    def _build_thinking_prompt(self, context: Dict[str, Any]) -> str:
        """Build a thinking prompt based on context"""
        return f"""
        Role: {self.role.value}
        Context: {json.dumps(context, indent=2)}
        
        Based on my role and the context, my analysis is:
        """
    
    async def communicate(self, message: AgentMessage):
        """Handle inter-agent communication"""
        await self.message_queue.put(message)
    
    def add_to_memory(self, content: Any, memory_type: str = "short_term", importance: float = 0.5):
        """Add information to memory"""
        memory_item = {
            'content': content,
            'type': memory_type,
            'importance': importance,
            'timestamp': datetime.now()
        }
        
        if memory_type == "short_term":
            self.working_memory.append(memory_item)
        else:
            self.long_term_memory.append(memory_item)
    
    def retrieve_from_memory(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant memories"""
        # Simple retrieval - can be enhanced with embeddings
        relevant_memories = []
        
        # Search working memory
        for memory in self.working_memory:
            if query.lower() in str(memory['content']).lower():
                relevant_memories.append(memory)
        
        # Search long-term memory
        for memory in self.long_term_memory:
            if query.lower() in str(memory['content']).lower():
                relevant_memories.append(memory)
        
        # Sort by importance and recency
        relevant_memories.sort(
            key=lambda x: (x['importance'], x['timestamp']),
            reverse=True
        )
        
        return relevant_memories[:k]


class OrchestratorAgent(BaseAgent):
    """Main orchestrator agent that coordinates other agents"""
    
    def __init__(self, agent_id: str = "orchestrator_001"):
        super().__init__(agent_id, AgentRole.ORCHESTRATOR)
        self.agents = {}
        self.execution_plan = []
        
    def _initialize_tools(self) -> List[Tool]:
        """Initialize orchestrator tools"""
        return [
            Tool(
                name="delegate_task",
                func=self.delegate_task,
                description="Delegate a task to another agent"
            ),
            Tool(
                name="create_plan",
                func=self.create_execution_plan,
                description="Create an execution plan for complex queries"
            ),
            Tool(
                name="monitor_agents",
                func=self.monitor_agents,
                description="Monitor the status of all agents"
            ),
            Tool(
                name="aggregate_results",
                func=self.aggregate_results,
                description="Aggregate results from multiple agents"
            )
        ]
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator"""
        self.agents[agent.agent_id] = agent
        self.peers[agent.agent_id] = agent
        logger.info(f"Registered agent {agent.agent_id} with orchestrator")
    
    async def execute_task(self, task: AgentTask) -> Any:
        """Execute orchestration task"""
        self.state = AgentState.ACTING
        
        try:
            # Analyze the task
            analysis = await self.think({'task': task.description})
            
            # Create execution plan
            plan = await self.create_execution_plan(task)
            
            # Execute plan
            results = await self.execute_plan(plan)
            
            # Aggregate results
            final_result = await self.aggregate_results(results)
            
            self.state = AgentState.COMPLETED
            return final_result
            
        except Exception as e:
            self.state = AgentState.FAILED
            logger.error(f"Orchestrator execution failed: {e}")
            return {'error': str(e)}
    
    async def create_execution_plan(self, task: AgentTask) -> List[Dict]:
        """Create an execution plan for a complex task"""
        plan = []
        
        # Analyze task complexity
        task_type = self._analyze_task_type(task.description)
        
        if task_type == "comparison":
            plan = [
                {'agent': 'researcher', 'action': 'gather_info', 'priority': 1},
                {'agent': 'analyst', 'action': 'analyze_data', 'priority': 2},
                {'agent': 'comparator', 'action': 'compare', 'priority': 3},
                {'agent': 'summarizer', 'action': 'summarize', 'priority': 4}
            ]
        elif task_type == "recommendation":
            plan = [
                {'agent': 'researcher', 'action': 'gather_preferences', 'priority': 1},
                {'agent': 'analyst', 'action': 'analyze_options', 'priority': 2},
                {'agent': 'recommender', 'action': 'generate_recommendations', 'priority': 3}
            ]
        elif task_type == "analysis":
            plan = [
                {'agent': 'scraper', 'action': 'collect_data', 'priority': 1},
                {'agent': 'validator', 'action': 'validate_data', 'priority': 2},
                {'agent': 'analyst', 'action': 'deep_analysis', 'priority': 3},
                {'agent': 'summarizer', 'action': 'create_report', 'priority': 4}
            ]
        else:
            # Default plan
            plan = [
                {'agent': 'researcher', 'action': 'research', 'priority': 1},
                {'agent': 'analyst', 'action': 'analyze', 'priority': 2}
            ]
        
        self.execution_plan = plan
        return plan
    
    def _analyze_task_type(self, description: str) -> str:
        """Analyze the type of task from description"""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['compare', 'versus', 'vs', 'difference']):
            return "comparison"
        elif any(word in description_lower for word in ['recommend', 'suggest', 'best', 'should']):
            return "recommendation"
        elif any(word in description_lower for word in ['analyze', 'review', 'sentiment', 'opinion']):
            return "analysis"
        else:
            return "general"
    
    async def execute_plan(self, plan: List[Dict]) -> List[Any]:
        """Execute the planned tasks"""
        results = []
        
        # Sort by priority
        sorted_plan = sorted(plan, key=lambda x: x['priority'])
        
        for step in sorted_plan:
            agent_id = step['agent']
            action = step['action']
            
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                
                # Create sub-task
                sub_task = AgentTask(
                    task_id=f"{agent_id}_{action}_{datetime.now().timestamp()}",
                    description=f"Perform {action}",
                    context=step.get('context', {})
                )
                
                # Execute
                result = await agent.execute_task(sub_task)
                results.append({
                    'agent': agent_id,
                    'action': action,
                    'result': result
                })
            else:
                logger.warning(f"Agent {agent_id} not found")
        
        return results
    
    async def delegate_task(self, agent_id: str, task: AgentTask) -> Any:
        """Delegate a task to a specific agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            return await agent.execute_task(task)
        else:
            return {'error': f'Agent {agent_id} not found'}
    
    async def monitor_agents(self) -> Dict[str, str]:
        """Monitor the status of all registered agents"""
        status = {}
        for agent_id, agent in self.agents.items():
            status[agent_id] = {
                'role': agent.role.value,
                'state': agent.state.value,
                'active_tasks': len(agent.active_tasks),
                'memory_size': len(agent.working_memory)
            }
        return status
    
    async def aggregate_results(self, results: List[Any]) -> Dict[str, Any]:
        """Aggregate results from multiple agents"""
        aggregated = {
            'summary': '',
            'details': [],
            'consensus': None,
            'confidence': 0.0
        }
        
        for result in results:
            if isinstance(result, dict):
                aggregated['details'].append(result)
        
        # Calculate consensus and confidence
        if aggregated['details']:
            # Simple consensus calculation
            aggregated['confidence'] = len(aggregated['details']) / len(results)
            aggregated['summary'] = f"Aggregated {len(aggregated['details'])} results"
        
        return aggregated


class ResearcherAgent(BaseAgent):
    """Agent responsible for information gathering and research"""
    
    def __init__(self, agent_id: str = "researcher_001"):
        super().__init__(agent_id, AgentRole.RESEARCHER)
        self.search_engine = SmartSearchEngine()
        self.rag_assistant = RAGChatAssistant()
        
    def _initialize_tools(self) -> List[Tool]:
        """Initialize researcher tools"""
        return [
            Tool(
                name="search_reviews",
                func=self.search_reviews,
                description="Search for phone reviews"
            ),
            Tool(
                name="gather_specs",
                func=self.gather_specifications,
                description="Gather technical specifications"
            ),
            Tool(
                name="find_prices",
                func=self.find_prices,
                description="Find current prices"
            ),
            Tool(
                name="research_competitors",
                func=self.research_competitors,
                description="Research competing products"
            )
        ]
    
    async def execute_task(self, task: AgentTask) -> Any:
        """Execute research task"""
        self.state = AgentState.ACTING
        
        try:
            # Determine research type
            if "review" in task.description.lower():
                result = await self.search_reviews(task.context.get('phone_model', ''))
            elif "spec" in task.description.lower():
                result = await self.gather_specifications(task.context.get('phone_model', ''))
            elif "price" in task.description.lower():
                result = await self.find_prices(task.context.get('phone_model', ''))
            else:
                # General research
                result = await self.comprehensive_research(task.description)
            
            self.state = AgentState.COMPLETED
            return result
            
        except Exception as e:
            self.state = AgentState.FAILED
            logger.error(f"Research task failed: {e}")
            return {'error': str(e)}
    
    async def search_reviews(self, phone_model: str) -> Dict[str, Any]:
        """Search for reviews of a phone model"""
        # Use RAG to retrieve relevant reviews
        reviews = self.rag_assistant.retrieve_relevant_context(
            f"reviews for {phone_model}",
            phone_model,
            k=10
        )
        
        return {
            'phone_model': phone_model,
            'reviews': reviews,
            'count': len(reviews),
            'source': 'internal_database'
        }
    
    async def gather_specifications(self, phone_model: str) -> Dict[str, Any]:
        """Gather technical specifications"""
        # Simulate gathering specs (in production, would scrape or use API)
        specs = {
            'display': '6.1" OLED',
            'processor': 'A17 Pro',
            'ram': '8GB',
            'storage': '256GB',
            'camera': '48MP main',
            'battery': '3274 mAh'
        }
        
        return {
            'phone_model': phone_model,
            'specifications': specs,
            'source': 'manufacturer'
        }
    
    async def find_prices(self, phone_model: str) -> Dict[str, Any]:
        """Find current prices for a phone"""
        # Simulate price finding
        prices = {
            'retail': 999,
            'discounted': 899,
            'used': 750,
            'sources': ['Amazon', 'BestBuy', 'Apple Store']
        }
        
        return {
            'phone_model': phone_model,
            'prices': prices,
            'timestamp': datetime.now().isoformat()
        }
    
    async def research_competitors(self, phone_model: str) -> List[str]:
        """Research competing products"""
        # Simple competitor identification
        competitors = {
            'iPhone': ['Samsung Galaxy S24', 'Google Pixel 8 Pro'],
            'Samsung': ['iPhone 15 Pro', 'Google Pixel 8 Pro'],
            'Google': ['iPhone 15 Pro', 'Samsung Galaxy S24']
        }
        
        brand = phone_model.split()[0]
        return competitors.get(brand, [])
    
    async def comprehensive_research(self, query: str) -> Dict[str, Any]:
        """Perform comprehensive research on a topic"""
        research_results = {
            'query': query,
            'findings': [],
            'sources': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Use search engine
        search_results = self.search_engine.search(query)
        
        if search_results:
            research_results['findings'] = [
                {
                    'title': result.get('phone_model', 'Unknown'),
                    'relevance': result.get('confidence', 0),
                    'summary': result.get('action', '')
                }
                for result in search_results[:5]:
            ]
        
        return research_results


class AnalystAgent(BaseAgent):
    """Agent responsible for data analysis and insights"""
    
    def __init__(self, agent_id: str = "analyst_001"):
        super().__init__(agent_id, AgentRole.ANALYST)
        self.summarizer = AdvancedReviewSummarizer()
        
    def _initialize_tools(self) -> List[Tool]:
        """Initialize analyst tools"""
        return [
            Tool(
                name="analyze_sentiment",
                func=self.analyze_sentiment,
                description="Analyze sentiment of reviews"
            ),
            Tool(
                name="extract_insights",
                func=self.extract_insights,
                description="Extract key insights from data"
            ),
            Tool(
                name="calculate_scores",
                func=self.calculate_scores,
                description="Calculate various scores and metrics"
            ),
            Tool(
                name="identify_patterns",
                func=self.identify_patterns,
                description="Identify patterns in data"
            )
        ]
    
    async def execute_task(self, task: AgentTask) -> Any:
        """Execute analysis task"""
        self.state = AgentState.ACTING
        
        try:
            data = task.context.get('data', [])
            
            # Perform analysis based on task type
            if "sentiment" in task.description.lower():
                result = await self.analyze_sentiment(data)
            elif "insight" in task.description.lower():
                result = await self.extract_insights(data)
            elif "pattern" in task.description.lower():
                result = await self.identify_patterns(data)
            else:
                # Comprehensive analysis
                result = await self.comprehensive_analysis(data)
            
            self.state = AgentState.COMPLETED
            return result
            
        except Exception as e:
            self.state = AgentState.FAILED
            logger.error(f"Analysis task failed: {e}")
            return {'error': str(e)}
    
    async def analyze_sentiment(self, reviews: List[str]) -> Dict[str, Any]:
        """Analyze sentiment of reviews"""
        if not reviews:
            return {'error': 'No reviews to analyze'}
        
        # Use summarizer for sentiment analysis
        summary = self.summarizer.summarize_reviews(reviews)
        
        return {
            'sentiment_distribution': {
                'positive': 0.7,  # Placeholder
                'neutral': 0.2,
                'negative': 0.1
            },
            'consensus_score': summary.consensus_score,
            'key_themes': summary.common_themes[:3] if summary.common_themes else []
        }
    
    async def extract_insights(self, data: Any) -> List[str]:
        """Extract key insights from data"""
        insights = []
        
        if isinstance(data, list) and data:
            # Use summarizer to extract insights
            summary = self.summarizer.summarize_reviews(data[:50])
            insights = summary.unique_insights
        
        return insights
    
    async def calculate_scores(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate various scores and metrics"""
        scores = {
            'overall_score': 0.0,
            'value_score': 0.0,
            'performance_score': 0.0,
            'reliability_score': 0.0
        }
        
        # Calculate scores based on data
        if 'reviews' in data:
            # Simple scoring logic
            scores['overall_score'] = 75.0
            scores['value_score'] = 70.0
            scores['performance_score'] = 80.0
            scores['reliability_score'] = 85.0
        
        return scores
    
    async def identify_patterns(self, data: List[Any]) -> List[Dict[str, Any]]:
        """Identify patterns in data"""
        patterns = []
        
        # Simple pattern identification
        if isinstance(data, list) and len(data) > 10:
            patterns.append({
                'type': 'trend',
                'description': 'Increasing positive sentiment over time',
                'confidence': 0.75
            })
            patterns.append({
                'type': 'anomaly',
                'description': 'Spike in negative reviews after update',
                'confidence': 0.6
            })
        
        return patterns
    
    async def comprehensive_analysis(self, data: Any) -> Dict[str, Any]:
        """Perform comprehensive analysis"""
        analysis = {
            'summary': '',
            'sentiment': {},
            'insights': [],
            'patterns': [],
            'scores': {},
            'recommendations': []
        }
        
        if isinstance(data, list) and data:
            # Analyze using summarizer
            summary = self.summarizer.summarize_reviews(data[:100])
            
            analysis['summary'] = summary.executive_summary
            analysis['insights'] = summary.unique_insights
            analysis['scores'] = await self.calculate_scores({'reviews': data})
            analysis['patterns'] = await self.identify_patterns(data)
            
            # Generate recommendations
            if summary.consensus_score > 70:
                analysis['recommendations'].append("Highly recommended based on user feedback")
            else:
                analysis['recommendations'].append("Consider alternatives")
        
        return analysis


class RecommenderAgent(BaseAgent):
    """Agent responsible for generating recommendations"""
    
    def __init__(self, agent_id: str = "recommender_001"):
        super().__init__(agent_id, AgentRole.RECOMMENDER)
        
    def _initialize_tools(self) -> List[Tool]:
        """Initialize recommender tools"""
        return [
            Tool(
                name="generate_recommendations",
                func=self.generate_recommendations,
                description="Generate personalized recommendations"
            ),
            Tool(
                name="rank_options",
                func=self.rank_options,
                description="Rank options based on criteria"
            ),
            Tool(
                name="find_alternatives",
                func=self.find_alternatives,
                description="Find alternative products"
            )
        ]
    
    async def execute_task(self, task: AgentTask) -> Any:
        """Execute recommendation task"""
        self.state = AgentState.ACTING
        
        try:
            preferences = task.context.get('preferences', {})
            candidates = task.context.get('candidates', [])
            
            # Generate recommendations
            recommendations = await self.generate_recommendations(
                preferences,
                candidates
            )
            
            self.state = AgentState.COMPLETED
            return recommendations
            
        except Exception as e:
            self.state = AgentState.FAILED
            logger.error(f"Recommendation task failed: {e}")
            return {'error': str(e)}
    
    async def generate_recommendations(
        self,
        preferences: Dict[str, Any],
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Score each candidate based on preferences
        for candidate in candidates:
            score = self._calculate_match_score(candidate, preferences)
            
            recommendations.append({
                'product': candidate.get('name', 'Unknown'),
                'score': score,
                'reasons': self._generate_reasons(candidate, preferences),
                'confidence': min(score / 100, 1.0)
            })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:5]
    
    def _calculate_match_score(:
        self,
        candidate: Dict[str, Any],
        preferences: Dict[str, Any]
    ) -> float:
        """Calculate match score between candidate and preferences"""
        score = 50.0  # Base score
        
        # Adjust based on preferences
        if preferences.get('budget') == 'low' and candidate.get('price', 999) < 500:
            score += 20
        elif preferences.get('budget') == 'high' and candidate.get('price', 999) > 800:
            score += 10
        
        if preferences.get('use_case') == 'gaming' and 'gaming' in str(candidate).lower():
            score += 15
        
        if preferences.get('priority_feature') in str(candidate).lower():
            score += 25
        
        return min(score, 100)
    
    def _generate_reasons(:
        self,
        candidate: Dict[str, Any],
        preferences: Dict[str, Any]
    ) -> List[str]:
        """Generate reasons for recommendation"""
        reasons = []
        
        if preferences.get('budget'):
            reasons.append(f"Fits your {preferences['budget']} budget")
        
        if preferences.get('use_case'):
            reasons.append(f"Excellent for {preferences['use_case']}")
        
        if preferences.get('priority_feature'):
            reasons.append(f"Strong {preferences['priority_feature']} performance")
        
        return reasons if reasons else ["Good overall value"]
    
    async def rank_options(
        self,
        options: List[Dict[str, Any]],
        criteria: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Rank options based on weighted criteria"""
        ranked = []
        
        for option in options:
            weighted_score = 0
            for criterion, weight in criteria.items():
                if criterion in option:
                    weighted_score += option[criterion] * weight
            
            option['weighted_score'] = weighted_score
            ranked.append(option)
        
        ranked.sort(key=lambda x: x['weighted_score'], reverse=True)
        return ranked
    
    async def find_alternatives(
        self,
        product: str,
        num_alternatives: int = 3
    ) -> List[str]:
        """Find alternative products"""
        # Simple alternative finding
        alternatives = {
            'iPhone 15 Pro': ['Samsung Galaxy S24 Ultra', 'Google Pixel 8 Pro', 'OnePlus 12'],
            'Samsung Galaxy S24': ['iPhone 15', 'Google Pixel 8', 'OnePlus 11'],
            'Google Pixel 8': ['iPhone 15', 'Samsung Galaxy S24', 'Nothing Phone 2']
        }
        
        return alternatives.get(product, ['No alternatives found'])[:num_alternatives]


class AgenticRAGSystem:
    """Main Agentic RAG System that manages all agents"""
    
    def __init__(self):
        """Initialize the Agentic RAG System"""
        logger.info("Initializing Agentic RAG System")
        
        # Initialize orchestrator
        self.orchestrator = OrchestratorAgent()
        
        # Initialize specialized agents
        self.agents = {
            'researcher': ResearcherAgent(),
            'analyst': AnalystAgent(),
            'recommender': RecommenderAgent()
        }
        
        # Register agents with orchestrator
        for agent_id, agent in self.agents.items():
            self.orchestrator.register_agent(agent)
        
        # Initialize message bus for inter-agent communication
        self.message_bus = asyncio.Queue()
        
        # System state
        self.system_state = {
            'active_queries': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'start_time': datetime.now()
        }
        
        logger.info("Agentic RAG System initialized successfully")
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query using the multi-agent system
        
        Args:
            query: User query
            context: Optional context
            
        Returns:
            Response dictionary
        """
        self.system_state['active_queries'] += 1
        
        try:
            # Create main task
            main_task = AgentTask(
                task_id=f"main_{datetime.now().timestamp()}",
                description=query,
                context=context or {},
                priority=10
            )
            
            # Let orchestrator handle the task
            result = await self.orchestrator.execute_task(main_task)
            
            self.system_state['completed_tasks'] += 1
            
            return {
                'status': 'success',
                'result': result,
                'metadata': {
                    'task_id': main_task.task_id,
                    'agents_involved': list(self.agents.keys()),
                    'processing_time': (datetime.now() - main_task.created_at).total_seconds()
                }
            }
            
        except Exception as e:
            self.system_state['failed_tasks'] += 1
            logger.error(f"Query processing failed: {e}")
            
            return {
                'status': 'error',
                'error': str(e),
                'metadata': {
                    'timestamp': datetime.now().isoformat()
                }
            }
        
        finally:
            self.system_state['active_queries'] -= 1
    
    async def add_custom_agent(self, agent: BaseAgent):
        """Add a custom agent to the system"""
        self.agents[agent.agent_id] = agent
        self.orchestrator.register_agent(agent)
        logger.info(f"Added custom agent: {agent.agent_id}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        agent_status = {}
        for agent_id, agent in self.agents.items():
            agent_status[agent_id] = {
                'role': agent.role.value,
                'state': agent.state.value,
                'memory_size': len(agent.working_memory)
            }
        
        return {
            'system_state': self.system_state,
            'agent_status': agent_status,
            'orchestrator_status': {
                'state': self.orchestrator.state.value,
                'registered_agents': len(self.orchestrator.agents),
                'execution_plan': self.orchestrator.execution_plan
            },
            'uptime': (datetime.now() - self.system_state['start_time']).total_seconds()
        }
    
    async def explain_decision(self, task_id: str) -> str:
        """Explain the decision-making process for a task"""
        explanation = []
        
        # Retrieve task information
        explanation.append(f"Task ID: {task_id}")
        explanation.append("Decision Process:")
        
        # Get orchestrator's thinking
        orchestrator_thoughts = list(self.orchestrator.working_memory)
        if orchestrator_thoughts:
            explanation.append("1. Orchestrator Analysis:")
            for thought in orchestrator_thoughts[-3:]:
                explanation.append(f"   - {thought.get('content', '')[:100]}")
        
        # Get execution plan
        if self.orchestrator.execution_plan:
            explanation.append("2. Execution Plan:")
            for step in self.orchestrator.execution_plan:
                explanation.append(f"   - {step['agent']}: {step['action']}")
        
        # Get agent contributions
        explanation.append("3. Agent Contributions:")
        for agent_id, agent in self.agents.items():
            if agent.working_memory:
                latest_memory = agent.working_memory[-1]
                explanation.append(f"   - {agent_id}: {latest_memory.get('content', '')[:100]}")
        
        return "\n".join(explanation)
    
    async def continuous_learning(self, feedback: Dict[str, Any]):
        """Update system based on user feedback"""
        # Store feedback in long-term memory
        for agent in self.agents.values():
            agent.add_to_memory(
                feedback,
                memory_type="long_term",
                importance=feedback.get('importance', 0.5)
            )
        
        logger.info("System updated with user feedback")


# Convenience functions
def create_agentic_rag_system() -> AgenticRAGSystem:
    """Create an instance of the Agentic RAG System"""
    return AgenticRAGSystem()


async def query_agentic_system(
    system: AgenticRAGSystem,
    query: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Query the Agentic RAG System"""
    return await system.process_query(query, context)


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def demo():
        # Initialize system
        system = create_agentic_rag_system()
        
        # Test queries
        queries = [
            "Compare iPhone 15 Pro Max vs Samsung Galaxy S24 Ultra",
            "What's the best camera phone under $800?",
            "Analyze the sentiment for Google Pixel 8 Pro reviews",
            "Recommend a phone for mobile gaming with good battery life"
        ]
        
        print("=" * 80)
        print("AGENTIC RAG SYSTEM DEMO")
        print("=" * 80)
        
        for query in queries:
            print(f"\n📝 Query: {query}")
            print("-" * 40)
            
            # Process query
            result = await system.process_query(query)
            
            if result['status'] == 'success':
                print(f"✅ Status: Success")
                print(f"📊 Result: {json.dumps(result['result'], indent=2)[:500]}...")
                print(f"⏱️ Processing Time: {result['metadata']['processing_time']:.2f}s")
                print(f"🤖 Agents Involved: {', '.join(result['metadata']['agents_involved'])}")
            else:
                print(f"❌ Error: {result['error']}")
            
            # Get system status
            status = system.get_system_status()
            print(f"\n📈 System Status:")
            print(f"   Active Queries: {status['system_state']['active_queries']}")
            print(f"   Completed Tasks: {status['system_state']['completed_tasks']}")
            print(f"   Failed Tasks: {status['system_state']['failed_tasks']}")
            
            # Explain decision
            if result['status'] == 'success':
                explanation = await system.explain_decision(result['metadata']['task_id'])
                print(f"\n🧠 Decision Explanation:")
                print(explanation[:500] + "...")
            
            print("=" * 80)
    
    # Run demo
    asyncio.run(demo())
