"""
Streamlit Chat Interface for AI Phone Assistant
Interactive chat application with RAG-powered responses
"""

import streamlit as st
import asyncio
from datetime import datetime
import json
import os
from typing import Optional, List, Dict, Any
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our modules
from models.chat_assistant import RAGChatAssistant, ChatContext, ChatMessage, create_chat_assistant
from models.review_summarizer import AdvancedReviewSummarizer
from core.smart_search import SmartSearchEngine
from visualization.decision_charts import DecisionVisualizer
from utils.unified_data_access import (
    get_primary_dataset,      # Full dataset
    create_sample_data,        # Sample data
    get_products_for_comparison,  # Product list
    get_brands_list,          # Brand list
    generate_fake_realtime_data  # Simulated events
)

# Page configuration
st.set_page_config(
    page_title="AI Phone Assistant Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e3f2fd;
        flex-direction: row-reverse;
    }
    .chat-message.assistant {
        background-color: #f5f5f5;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin: 0 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        background: white;
        border: 2px solid #ddd;
    }
    .chat-message .content {
        flex: 1;
        padding: 0 10px;
    }
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        border-top: 1px solid #ddd;
        z-index: 100;
    }
    .suggested-query {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        background: #e8f5e9;
        border-radius: 15px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .suggested-query:hover {
        background: #4caf50;
        color: white;
        transform: scale(1.05);
    }
    .typing-indicator {
        display: inline-block;
        padding: 10px;
    }
    .typing-indicator span {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #888;
        margin: 0 2px;
        animation: typing 1.4s infinite;
    }
    .typing-indicator span:nth-child(2) {
        animation-delay: 0.2s;
    }
    .typing-indicator span:nth-child(3) {
        animation-delay: 0.4s;
    }
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
        }
        30% {
            transform: translateY(-10px);
        }
    }
</style>
""", unsafe_allow_html=True)


class ChatInterface:
    """Manages the chat interface and interactions"""
    
    def __init__(self):
        """Initialize the chat interface"""
        self.initialize_session_state()
        self.assistant = self.load_assistant()
        self.visualizer = DecisionVisualizer()
        self.search_engine = SmartSearchEngine()
        
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'chat_context' not in st.session_state:
            st.session_state.chat_context = ChatContext()
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'assistant_typing' not in st.session_state:
            st.session_state.assistant_typing = False
        
        if 'indexed_phones' not in st.session_state:
            st.session_state.indexed_phones = set()
        
        if 'conversation_saved' not in st.session_state:
            st.session_state.conversation_saved = False
    
    @st.cache_resource
    def load_assistant():
        """Load the chat assistant (cached)"""
        return create_chat_assistant()
    
    def render_sidebar(self):
        """Render the sidebar with controls and information"""
        with st.sidebar:
            st.title("🤖 AI Phone Assistant")
            st.markdown("---")
            
            # Chat controls
            st.subheader("💬 Chat Controls")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 New Chat", use_container_width=True):
                    self.clear_chat()
            
            with col2:
                if st.button("💾 Save Chat", use_container_width=True):
                    self.save_conversation()
            
            # Load conversation
            uploaded_file = st.file_uploader(
                "Load Previous Chat",
                type=['json'],
                help="Upload a saved conversation file"
            )
            if uploaded_file:
                self.load_conversation(uploaded_file)
            
            st.markdown("---")
            
            # Quick actions
            st.subheader("⚡ Quick Actions")
            
            # Suggested queries
            st.markdown("**Suggested Queries:**")
            suggestions = [
                "Compare iPhone 15 vs Galaxy S24",
                "Best camera phone under $800",
                "Is the Pixel 8 Pro worth it?",
                "Gaming phone recommendations",
                "Battery life champions"
            ]
            
            for suggestion in suggestions:
                if st.button(f"💡 {suggestion}", key=f"sug_{suggestion}"):
                    self.send_message(suggestion)
            
            st.markdown("---")
            
            # Data management
            st.subheader("📊 Data Management")
            
            # Index sample data from cleaned dataset
            if st.button("📥 Load Dataset", use_container_width=True):
                self.load_sample_data()
            
            # Upload reviews
            reviews_file = st.file_uploader(
                "Upload Reviews (CSV)",
                type=['csv'],
                help="CSV with columns: phone_model, text, rating, date"
            )
            if reviews_file:
                self.load_reviews_from_csv(reviews_file)
            
            st.markdown("---")
            
            # Chat statistics
            st.subheader("📈 Chat Statistics")
            stats = self.get_chat_statistics()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Messages", stats['total_messages'])
                st.metric("Phones Discussed", stats['phones_discussed'])
            
            with col2:
                st.metric("Session Time", stats['session_time'])
                st.metric("Indexed Phones", len(st.session_state.indexed_phones))
            
            st.markdown("---")
            
            # Settings
            with st.expander("⚙️ Settings"):
                st.checkbox("Enable RAG", value=True, key="use_rag")
                st.checkbox("Show Debug Info", value=False, key="debug_mode")
                st.slider("Response Length", 50, 500, 200, key="max_response_length")
                st.selectbox(
                    "Chat Style",
                    ["Professional", "Friendly", "Technical"],
                    key="chat_style"
                )
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        st.title("💬 AI Phone Assistant Chat")
        
        # Welcome message if no messages
        if not st.session_state.messages:
            self.show_welcome_message()
        
        # Chat messages container
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.messages:
                self.render_message(message)
            
            # Typing indicator
            if st.session_state.assistant_typing:
                self.show_typing_indicator()
        
        # Input area (fixed at bottom)
        self.render_input_area()
    
    def show_welcome_message(self):
        """Show welcome message for new chat"""
        welcome_html = """
        <div style="text-align: center; padding: 2rem;">
            <h1 style="color: #1e88e5;">👋 Welcome to AI Phone Assistant!</h1>
            <p style="font-size: 1.2rem; color: #666;">
                I can help you with:
            </p>
            <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 1rem; margin: 2rem 0;">
                <div style="background: #e3f2fd; padding: 1rem; border-radius: 10px; flex: 1; min-width: 200px;">
                    <h3>🔍 Phone Research</h3>
                    <p>Find the perfect phone for your needs</p>
                </div>
                <div style="background: #f3e5f5; padding: 1rem; border-radius: 10px; flex: 1; min-width: 200px;">
                    <h3>📊 Review Analysis</h3>
                    <p>Understand what users really think</p>
                </div>
                <div style="background: #e8f5e9; padding: 1rem; border-radius: 10px; flex: 1; min-width: 200px;">
                    <h3>⚖️ Comparisons</h3>
                    <p>Compare phones side by side</p>
                </div>
                <div style="background: #fff3e0; padding: 1rem; border-radius: 10px; flex: 1; min-width: 200px;">
                    <h3>💡 Recommendations</h3>
                    <p>Get personalized suggestions</p>
                </div>
            </div>
            <p style="color: #888;">
                Start by asking a question or selecting a suggested query from the sidebar!
            </p>
        </div>
        """
        st.markdown(welcome_html, unsafe_allow_html=True)
    
    def render_message(self, message: Dict[str, Any]):
        """Render a single chat message"""
        role = message['role']
        content = message['content']
        timestamp = message.get('timestamp', '')
        
        if role == 'user':
            avatar = "👤"
            css_class = "user"
        else:
            avatar = "🤖"
            css_class = "assistant"
        
        message_html = f"""
        <div class="chat-message {css_class}">
            <div class="avatar">{avatar}</div>
            <div class="content">
                <div style="font-size: 0.8rem; color: #888; margin-bottom: 0.5rem;">
                    {timestamp}
                </div>
                <div>{content}</div>
            </div>
        </div>
        """
        st.markdown(message_html, unsafe_allow_html=True)
        
        # Add action buttons for assistant messages
        if role == 'assistant' and 'phone' in content.lower():
            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button("📊 Visualize", key=f"viz_{len(st.session_state.messages)}"):
                    self.show_visualization(content)
            with col2:
                if st.button("📋 Summary", key=f"sum_{len(st.session_state.messages)}"):
                    self.show_summary(content)
    
    def show_typing_indicator(self):
        """Show typing indicator"""
        typing_html = """
        <div class="chat-message assistant">
            <div class="avatar">🤖</div>
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
        """
        st.markdown(typing_html, unsafe_allow_html=True)
    
    def render_input_area(self):
        """Render the input area"""
        col1, col2 = st.columns([10, 1])
        
        with col1:
            user_input = st.chat_input(
                "Ask me anything about phones...",
                key="chat_input"
            )
        
        with col2:
            send_button = st.button("📤", key="send_button")
        
        if user_input or send_button:
            if user_input:
                self.send_message(user_input)
    
    def send_message(self, message: str):
        """Send a message to the assistant"""
        # Add user message
        timestamp = datetime.now().strftime("%I:%M %p")
        st.session_state.messages.append({
            'role': 'user',
            'content': message,
            'timestamp': timestamp
        })
        
        # Show typing indicator
        st.session_state.assistant_typing = True
        
        # Get assistant response
        response = self.get_assistant_response(message)
        
        # Add assistant message
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().strftime("%I:%M %p")
        })
        
        st.session_state.assistant_typing = False
        st.rerun()
    
    def get_assistant_response(self, message: str) -> str:
        """Get response from the assistant"""
        try:
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(
                self.assistant.chat(
                    message,
                    st.session_state.chat_context,
                    stream=False
                )
            )
            return response
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
    
    def clear_chat(self):
        """Clear the chat history"""
        st.session_state.messages = []
        st.session_state.chat_context = ChatContext()
        st.session_state.conversation_saved = False
        st.rerun()
    
    def save_conversation(self):
        """Save the current conversation"""
        if not st.session_state.messages:
            st.warning("No messages to save!")
            return
        
        # Create conversation data
        conversation_data = {
            'session_id': st.session_state.chat_context.session_id,
            'timestamp': datetime.now().isoformat(),
            'messages': st.session_state.messages,
            'context': {
                'current_phone': st.session_state.chat_context.current_phone,
                'user_preferences': st.session_state.chat_context.user_preferences
            }
        }
        
        # Convert to JSON
        json_str = json.dumps(conversation_data, indent=2)
        
        # Offer download
        st.download_button(
            label="📥 Download Conversation",
            data=json_str,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        st.session_state.conversation_saved = True
        st.success("Conversation saved!")
    
    def load_conversation(self, uploaded_file):
        """Load a saved conversation"""
        try:
            data = json.load(uploaded_file)
            
            # Restore messages
            st.session_state.messages = data.get('messages', [])
            
            # Restore context
            context_data = data.get('context', {})
            st.session_state.chat_context.current_phone = context_data.get('current_phone')
            st.session_state.chat_context.user_preferences = context_data.get('user_preferences', {})
            
            st.success("Conversation loaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading conversation: {str(e)}")
    
    def load_sample_data(self):
        """Load sample review data from cleaned dataset"""
        with st.spinner("Loading review data..."):
            # Get sample data from the cleaned dataset
            sample_df = create_sample_data(n_samples=100)
            
            if sample_df is not None and not sample_df.empty:
                # Group by product/phone model
                grouped = sample_df.groupby('product')
                
                phone_count = 0
                for phone_model, group in grouped:
                    # Convert to format expected by assistant
                    reviews = []
                    for _, row in group.iterrows():
                        review_dict = {
                            "text": row['text'],
                            "rating": row.get('rating', 0) if pd.notna(row.get('rating', 0)) else 0
                        }
                        reviews.append(review_dict)
                    
                    if reviews:  # Only index if there are reviews
                        self.assistant.index_reviews(reviews, phone_model)
                        st.session_state.indexed_phones.add(phone_model)
                        phone_count += 1
                
                st.success(f"Loaded data for {phone_count} phones from cleaned dataset!")
            else:
                st.warning("No data available. Please check the dataset.")
    
    def load_reviews_from_csv(self, csv_file):
        """Load reviews from CSV file"""
        try:
            df = pd.read_csv(csv_file)
            
            # Group by phone model
            grouped = df.groupby('phone_model')
            
            with st.spinner("Indexing reviews..."):
                for phone_model, group in grouped:
                    reviews = group.to_dict('records')
                    self.assistant.index_reviews(reviews, phone_model)
                    st.session_state.indexed_phones.add(phone_model)
            
            st.success(f"Indexed {len(df)} reviews for {len(grouped)} phones!")
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
    
    def get_chat_statistics(self) -> Dict[str, Any]:
        """Get chat statistics"""
        messages = st.session_state.messages
        
        # Calculate statistics
        total_messages = len(messages)
        user_messages = sum(1 for m in messages if m['role'] == 'user')
        assistant_messages = total_messages - user_messages
        
        # Extract mentioned phones
        phones = set()
        for msg in messages:
            content = msg['content'].lower()
            # Simple extraction (can be improved)
            if 'iphone' in content:
                phones.add('iPhone')
            if 'galaxy' in content:
                phones.add('Galaxy')
            if 'pixel' in content:
                phones.add('Pixel')
        
        # Session time
        if messages:
            first_time = messages[0].get('timestamp', '')
            last_time = messages[-1].get('timestamp', '')
            session_time = f"{first_time} - {last_time}"
        else:
            session_time = "No messages yet"
        
        return {
            'total_messages': total_messages,
            'user_messages': user_messages,
            'assistant_messages': assistant_messages,
            'phones_discussed': len(phones),
            'session_time': session_time
        }
    
    def show_visualization(self, content: str):
        """Show visualization based on content"""
        # Extract phone model from content (simplified)
        phone_model = "Phone"  # Default
        
        # Create sample visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Positive', 'Neutral', 'Negative'],
            y=[70, 20, 10],
            marker_color=['green', 'gray', 'red']
        ))
        fig.update_layout(
            title=f"Sentiment Analysis for {phone_model}",
            yaxis_title="Percentage (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_summary(self, content: str):
        """Show summary of the content"""
        with st.expander("📋 Summary", expanded=True):
            st.markdown("""
            **Key Points:**
            - High user satisfaction rating
            - Excellent camera performance
            - Good battery life
            - Premium build quality
            
            **Recommendation:** Recommended for users who prioritize camera quality and performance.
            """)
    
    def run(self):
        """Run the chat interface"""
        # Render sidebar
        self.render_sidebar()
        
        # Render main chat interface
        self.render_chat_interface()
        
        # Show debug info if enabled
        if st.session_state.get('debug_mode', False):
            with st.expander("🐛 Debug Information"):
                st.json({
                    'context': {
                        'current_phone': st.session_state.chat_context.current_phone,
                        'preferences': st.session_state.chat_context.user_preferences
                    },
                    'indexed_phones': list(st.session_state.indexed_phones),
                    'message_count': len(st.session_state.messages)
                })


def main():
    """Main application entry point"""
    chat_interface = ChatInterface()
    chat_interface.run()


if __name__ == "__main__":
    main()
