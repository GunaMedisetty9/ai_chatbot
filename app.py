import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import uuid

from chatbot.conversation_manager import ConversationManager
from utils.database import ChatDatabase

# Page configuration
st.set_page_config(
    page_title="AI Chatbot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def init_chatbot():
    return ConversationManager()

@st.cache_resource
def init_database():
    return ChatDatabase()

chatbot = init_chatbot()
database = init_database()

# Session state initialization
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
        border-left: 4px solid #2196F3;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: 20%;
        border-left: 4px solid #4CAF50;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🤖 AI Chatbot")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "📍 Navigate to:",
        ["💬 Chat", "📊 Analytics Dashboard", "⚙️ Settings", "📝 Training"]
    )
    
    st.markdown("---")
    
    # Quick Stats
    st.subheader("📈 Quick Stats")
    
    analytics_data = database.get_analytics_data(days=7)
    if analytics_data:
        total_messages = sum(row[1] for row in analytics_data)
        total_sessions = sum(row[2] for row in analytics_data)
        st.metric("Messages (7d)", total_messages)
        st.metric("Sessions (7d)", total_sessions)
    else:
        st.info("No data yet")
    
    st.markdown("---")
    
    # Session Info
    st.subheader("🔑 Session")
    st.code(st.session_state.session_id[:8] + "...")
    
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        chatbot.clear_context(st.session_state.session_id)
        st.rerun()

# Main Content
if page == "💬 Chat":
    st.title("💬 Chat with AI Assistant")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            if not st.session_state.chat_history:
                st.info("👋 Start a conversation by typing a message below!")
            
            for chat in st.session_state.chat_history:
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong><br>{chat['user']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Bot:</strong><br>{chat['bot']}<br>
                    <small style="color: gray;">Intent: {chat['intent']} | Confidence: {chat['confidence']:.0%}</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Input area
        with st.form(key='chat_form', clear_on_submit=True):
            col_input, col_btn = st.columns([4, 1])
            with col_input:
                user_input = st.text_input(
                    "Message:",
                    placeholder="Type your message here...",
                    label_visibility="collapsed"
                )
            with col_btn:
                submit = st.form_submit_button("Send 📤", use_container_width=True)
            
            if submit and user_input.strip():
                response_data = chatbot.process_message(
                    st.session_state.session_id,
                    user_input
                )
                
                st.session_state.chat_history.append({
                    'user': user_input,
                    'bot': response_data['response'],
                    'intent': response_data['intent'],
                    'confidence': response_data['confidence'],
                    'sentiment': response_data['sentiment'],
                    'entities': response_data['entities']
                })
                st.rerun()
    
    with col2:
        st.subheader("📊 Live Analysis")
        
        if st.session_state.chat_history:
            latest = st.session_state.chat_history[-1]
            
            # Sentiment gauge
            sentiment = latest['sentiment']
            polarity = sentiment['polarity']
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=polarity,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, -0.3], 'color': "#ff6b6b"},
                        {'range': [-0.3, 0.3], 'color': "#ffd93d"},
                        {'range': [0.3, 1], 'color': "#6bcb77"}
                    ]
                },
                title={'text': "Sentiment"}
            ))
            fig_gauge.update_layout(height=200, margin=dict(t=50, b=0, l=0, r=0))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Sentiment label
            sentiment_emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
            st.markdown(f"**Sentiment:** {sentiment_emoji.get(sentiment['sentiment'], '😐')} {sentiment['sentiment'].capitalize()}")
            
            # Emotions
            st.markdown("**Emotions:**")
            for emotion in sentiment.get('emotions', ['neutral']):
                st.markdown(f"• {emotion.capitalize()}")
            
            # Entities
            st.markdown("---")
            st.markdown("**🏷️ Entities:**")
            entities = latest['entities']
            if entities:
                for entity_type, values in entities.items():
                    st.markdown(f"• **{entity_type}:** {', '.join(values[:3])}")
            else:
                st.caption("No entities detected")
        else:
            st.info("Send a message to see analysis!")

elif page == "📊 Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    
    # Time range selector
    days_range = st.selectbox("📅 Time Range", [7, 14, 30, 60], index=0)
    
    # Get data
    analytics_data = database.get_analytics_data(days=days_range)
    intent_data = database.get_intent_distribution()
    recent_conversations = database.get_recent_conversations(limit=100)
    
    # Key Metrics
    st.subheader("📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if analytics_data:
        total_messages = sum(row[1] for row in analytics_data)
        total_sessions = sum(row[2] for row in analytics_data)
        avg_conf = sum(row[3] or 0 for row in analytics_data) / len(analytics_data) if analytics_data else 0
        positive = sum(row[4] or 0 for row in analytics_data)
        negative = sum(row[5] or 0 for row in analytics_data)
        
        col1.metric("💬 Total Messages", total_messages)
        col2.metric("👥 Unique Sessions", total_sessions)
        col3.metric("🎯 Avg Confidence", f"{avg_conf:.0%}")
        
        pos_rate = positive / (positive + negative) * 100 if (positive + negative) > 0 else 0
        col4.metric("😊 Positive Rate", f"{pos_rate:.1f}%")
    else:
        st.info("📭 No analytics data yet. Start chatting to generate data!")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Messages Over Time")
        if analytics_data:
            df = pd.DataFrame(analytics_data, columns=[
                'date', 'messages', 'sessions', 'avg_conf', 'pos', 'neg', 'neu'
            ])
            fig = px.line(df, x='date', y='messages', markers=True,
                         title='Daily Messages')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available")
    
    with col2:
        st.subheader("🎯 Intent Distribution")
        if intent_data:
            df = pd.DataFrame(intent_data, columns=['intent', 'count'])
            fig = px.pie(df, values='count', names='intent',
                        title='User Intents')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available")
    
    # Sentiment
    st.subheader("😊 Sentiment Distribution")
    if analytics_data:
        sentiment_data = {
            'Sentiment': ['Positive', 'Negative', 'Neutral'],
            'Count': [
                sum(row[4] or 0 for row in analytics_data),
                sum(row[5] or 0 for row in analytics_data),
                sum(row[6] or 0 for row in analytics_data)
            ]
        }
        df = pd.DataFrame(sentiment_data)
        fig = px.bar(df, x='Sentiment', y='Count', color='Sentiment',
                    color_discrete_map={'Positive': '#6bcb77', 'Negative': '#ff6b6b', 'Neutral': '#ffd93d'})
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Conversations
    st.subheader("💬 Recent Conversations")
    if recent_conversations:
        df = pd.DataFrame(recent_conversations, columns=[
            'Session', 'User Message', 'Bot Response', 'Intent', 
            'Confidence', 'Sentiment', 'Timestamp'
        ])
        df['Session'] = df['Session'].str[:8] + '...'
        df['User Message'] = df['User Message'].str[:40] + '...'
        df['Bot Response'] = df['Bot Response'].str[:40] + '...'
        df['Confidence'] = df['Confidence'].apply(lambda x: f"{x:.0%}" if x else "N/A")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No conversations yet")

elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Chatbot Settings")
        
        confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.1)
        max_context = st.number_input("Max Context Length", 1, 20, 5)
        
        st.checkbox("Enable Sentiment Analysis", value=True)
        st.checkbox("Enable Entity Extraction", value=True)
    
    with col2:
        st.subheader("📊 Display Settings")
        
        st.checkbox("Show Confidence Scores", value=True)
        st.checkbox("Show Entities", value=True)
        st.checkbox("Show Sentiment", value=True)
    
    st.markdown("---")
    
    st.subheader("🗄️ Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.success("✅ Chat cleared!")
    
    with col2:
        if st.button("📥 Export Data", use_container_width=True):
            data = database.get_recent_conversations(1000)
            if data:
                df = pd.DataFrame(data, columns=[
                    'Session', 'User', 'Bot', 'Intent', 'Confidence', 'Sentiment', 'Time'
                ])
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, "chatbot_data.csv", "text/csv")
            else:
                st.warning("No data to export")
    
    with col3:
        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.success("✅ Session reset!")

elif page == "📝 Training":
    st.title("📝 Training & Intent Management")
    
    # Add new intent
    st.subheader("➕ Add New Intent")
    
    with st.form("add_intent"):
        tag = st.text_input("Intent Tag", placeholder="e.g., product_inquiry")
        patterns = st.text_area("Patterns (one per line)", 
                               placeholder="what products do you have\nshow me products")
        responses = st.text_area("Responses (one per line)",
                                placeholder="We have many products!\nHow can I help with products?")
        
        if st.form_submit_button("➕ Add Intent", use_container_width=True):
            if tag and patterns and responses:
                patterns_list = [p.strip() for p in patterns.split('\n') if p.strip()]
                responses_list = [r.strip() for r in responses.split('\n') if r.strip()]
                chatbot.intent_classifier.add_intent(tag, patterns_list, responses_list)
                st.success(f"✅ Intent '{tag}' added!")
                st.rerun()
            else:
                st.error("Please fill all fields")
    
    st.markdown("---")
    
    # View intents
    st.subheader("📋 Existing Intents")
    
    intents = chatbot.intent_classifier.intents_data['intents']
    
    for intent in intents:
        with st.expander(f"🏷️ {intent['tag']} ({len(intent.get('patterns', []))} patterns)"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Patterns:**")
                for p in intent.get('patterns', [])[:5]:
                    st.markdown(f"• {p}")
                if len(intent.get('patterns', [])) > 5:
                    st.caption(f"... and {len(intent['patterns']) - 5} more")
            with col2:
                st.markdown("**Responses:**")
                for r in intent.get('responses', [])[:3]:
                    st.markdown(f"• {r[:50]}...")
    
    st.markdown("---")
    
    # Test intent
    st.subheader("🧪 Test Classification")
    
    test_msg = st.text_input("Enter test message:")
    if test_msg:
        intent, conf = chatbot.intent_classifier.predict(test_msg)
        col1, col2 = st.columns(2)
        col1.metric("Predicted Intent", intent)
        col2.metric("Confidence", f"{conf:.0%}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🤖 Dynamic AI Chatbot | Built with Streamlit | "
    "<a href='https://github.com/YOUR_USERNAME/ai_chatbot'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)