import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid
import random
import json
import sqlite3
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Chatbot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== DATABASE ====================
class ChatDatabase:
    def __init__(self):
        self.db_path = "data/chat_history.db"
        os.makedirs("data", exist_ok=True)
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_message TEXT,
            bot_response TEXT,
            intent TEXT,
            confidence REAL,
            sentiment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()
        conn.close()

    def save(self, session_id, user_msg, bot_resp, intent, confidence, sentiment):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''INSERT INTO conversations 
                (session_id, user_message, bot_response, intent, confidence, sentiment)
                VALUES (?, ?, ?, ?, ?, ?)''',
                (session_id, user_msg, bot_resp, intent, confidence, sentiment))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def get_analytics(self, days=30):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''SELECT DATE(timestamp) as date,
                COUNT(*) as messages,
                COUNT(DISTINCT session_id) as sessions,
                AVG(confidence) as avg_conf,
                SUM(CASE WHEN sentiment='positive' THEN 1 ELSE 0 END) as pos,
                SUM(CASE WHEN sentiment='negative' THEN 1 ELSE 0 END) as neg,
                SUM(CASE WHEN sentiment='neutral' THEN 1 ELSE 0 END) as neu
                FROM conversations
                WHERE timestamp >= DATE('now', ?)
                GROUP BY DATE(timestamp)
                ORDER BY date DESC''', (f'-{days} days',))
            results = c.fetchall()
            conn.close()
            return results
        except Exception:
            return []

    def get_intent_dist(self):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''SELECT intent, COUNT(*) as count
                FROM conversations WHERE intent IS NOT NULL
                GROUP BY intent ORDER BY count DESC''')
            results = c.fetchall()
            conn.close()
            return results
        except Exception:
            return []

    def get_recent(self, limit=50):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''SELECT session_id, user_message, bot_response, 
                intent, confidence, sentiment, timestamp
                FROM conversations ORDER BY timestamp DESC LIMIT ?''', (limit,))
            results = c.fetchall()
            conn.close()
            return results
        except Exception:
            return []


# ==================== NLP ENGINE ====================
import re

class NLPEngine:
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'shall', 'to',
            'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
            'into', 'through', 'during', 'before', 'after', 'and', 'but',
            'or', 'so', 'if', 'then', 'than', 'that', 'this', 'these',
            'those', 'it', 'its', 'also', 'just', 'very', 'really'
        }

    def preprocess(self, text):
        text = text.lower().strip()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = text.split()
        result = [t for t in tokens if t not in self.stop_words and len(t) > 0]
        if not result:
            result = text.split()
        return result


# ==================== INTENT CLASSIFIER ====================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class IntentClassifier:
    def __init__(self, intents_data):
        self.nlp = NLPEngine()
        self.intents_data = intents_data
        self.model = None
        self.classes = []
        self.raw_patterns = {}
        self.train()

    def train(self):
        sentences = []
        labels = []
        self.classes = []
        self.raw_patterns = {}

        for intent in self.intents_data.get('intents', []):
            tag = intent['tag']
            if tag not in self.classes:
                self.classes.append(tag)
            self.raw_patterns[tag] = []

            for pattern in intent.get('patterns', []):
                self.raw_patterns[tag].append(pattern.lower().strip())
                processed = ' '.join(self.nlp.preprocess(pattern))
                if processed.strip():
                    sentences.append(processed)
                    labels.append(tag)

        if not sentences:
            return

        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=5000, sublinear_tf=True)),
            ('clf', LogisticRegression(max_iter=1000, C=10, solver='lbfgs'))
        ])
        self.model.fit(sentences, labels)

    def predict(self, text):
        text_lower = text.lower().strip()

        # Keyword matching first
        kw_result = self.keyword_match(text_lower)
        if kw_result:
            return kw_result

        # ML model
        if not self.model:
            return "unknown", 0.0

        processed = ' '.join(self.nlp.preprocess(text))
        if not processed.strip():
            return "unknown", 0.0

        pred = self.model.predict([processed])[0]
        probs = self.model.predict_proba([processed])[0]
        conf = max(probs)
        return pred, conf

    def keyword_match(self, text):
        text_words = set(text.split())
        best_match = None
        best_score = 0

        for tag, patterns in self.raw_patterns.items():
            for pattern in patterns:
                pattern_words = set(pattern.split())

                if text == pattern:
                    return tag, 0.99

                if pattern_words and pattern_words.issubset(text_words):
                    score = len(pattern_words) / len(text_words)
                    if score > best_score:
                        best_score = score
                        best_match = tag

                if pattern in text or text in pattern:
                    score = min(len(text), len(pattern)) / max(len(text), len(pattern))
                    if score > best_score:
                        best_score = score
                        best_match = tag

                if text_words and pattern_words:
                    overlap = text_words.intersection(pattern_words)
                    if overlap:
                        score = len(overlap) / max(len(text_words), len(pattern_words))
                        if score > best_score:
                            best_score = score
                            best_match = tag

        if best_match and best_score >= 0.3:
            return best_match, min(best_score + 0.3, 0.99)
        return None


# ==================== SENTIMENT ANALYZER ====================
from textblob import TextBlob


class SentimentAnalyzer:
    def __init__(self):
        self.emotion_keywords = {
            'happy': ['happy', 'glad', 'excited', 'great', 'awesome', 'amazing', 'love', 'fantastic'],
            'sad': ['sad', 'unhappy', 'disappointed', 'upset', 'sorry'],
            'angry': ['angry', 'furious', 'mad', 'annoyed', 'frustrated', 'hate', 'terrible'],
            'neutral': ['okay', 'fine', 'alright', 'normal']
        }

    def analyze(self, text):
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            emotions = self.detect_emotions(text.lower())

            return {
                'sentiment': sentiment,
                'polarity': round(polarity, 3),
                'subjectivity': round(subjectivity, 3),
                'confidence': round(abs(polarity), 3),
                'emotions': emotions,
                'has_intensifier': False
            }
        except Exception:
            return {
                'sentiment': 'neutral',
                'polarity': 0.0,
                'subjectivity': 0.0,
                'confidence': 0.0,
                'emotions': ['neutral'],
                'has_intensifier': False
            }

    def detect_emotions(self, text):
        detected = []
        for emotion, keywords in self.emotion_keywords.items():
            for kw in keywords:
                if kw in text:
                    detected.append(emotion)
                    break
        return detected if detected else ['neutral']


# ==================== ENTITY EXTRACTOR ====================
class EntityExtractor:
    def __init__(self):
        self.patterns = {
            'email': r'[\w\.-]+@[\w\.-]+\.\w+',
            'phone': r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}',
            'order_id': r'[A-Z]{2,3}[-]?\d{6,10}',
            'url': r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
            'date': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'money': r'\$\d+(?:\.\d{2})?',
            'number': r'\b\d+\b'
        }

    def extract(self, text):
        entities = {}
        for etype, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[etype] = list(set(matches))
        return entities


# ==================== CHATBOT ====================
class Chatbot:
    def __init__(self):
        self.intents_data = self.load_intents()
        self.classifier = IntentClassifier(self.intents_data)
        self.sentiment = SentimentAnalyzer()
        self.entities = EntityExtractor()
        self.database = ChatDatabase()
        self.context = {}
        self.fallback = [
            "I am not sure I understand. Could you rephrase?",
            "I did not catch that. Try saying it differently?",
            "I am still learning! Could you explain?",
            "Sorry, can you provide more details?"
        ]

    def load_intents(self):
        try:
            with open('data/intents.json', 'r') as f:
                return json.load(f)
        except Exception:
            return {"intents": []}

    def get_response(self, intent, confidence):
        if confidence < 0.15:
            return random.choice(self.fallback)
        try:
            for item in self.intents_data.get('intents', []):
                if item.get('tag', '') == intent:
                    responses = item.get('responses', [])
                    if len(responses) > 0:
                        return str(random.choice(responses))
            return random.choice(self.fallback)
        except Exception:
            return random.choice(self.fallback)

    def process_message(self, session_id, user_message):
        try:
            if session_id not in self.context:
                self.context[session_id] = {'history': [], 'last_intent': None}

            try:
                intent, confidence = self.classifier.predict(user_message)
            except Exception:
                intent = "unknown"
                confidence = 0.0

            try:
                extracted_entities = self.entities.extract(user_message)
            except Exception:
                extracted_entities = {}

            try:
                sentiment_result = self.sentiment.analyze(user_message)
            except Exception:
                sentiment_result = {
                    'sentiment': 'neutral', 'polarity': 0.0,
                    'subjectivity': 0.0, 'confidence': 0.0,
                    'emotions': ['neutral'], 'has_intensifier': False
                }

            response = self.get_response(intent, confidence)

            try:
                self.database.save(
                    session_id, user_message, response,
                    str(intent), float(confidence),
                    str(sentiment_result.get('sentiment', 'neutral'))
                )
            except Exception:
                pass

            try:
                self.context[session_id]['history'].append({
                    'message': user_message, 'intent': intent
                })
                self.context[session_id]['last_intent'] = intent
                if len(self.context[session_id]['history']) > 5:
                    self.context[session_id]['history'] = self.context[session_id]['history'][-5:]
            except Exception:
                pass

            return {
                'response': response,
                'intent': str(intent),
                'confidence': round(float(confidence), 3),
                'sentiment': sentiment_result,
                'entities': extracted_entities,
                'session_id': session_id
            }

        except Exception:
            return {
                'response': 'I am here to help! Please try again.',
                'intent': 'unknown', 'confidence': 0.0,
                'sentiment': {'sentiment': 'neutral', 'polarity': 0.0,
                    'subjectivity': 0.0, 'confidence': 0.0,
                    'emotions': ['neutral'], 'has_intensifier': False},
                'entities': {}, 'session_id': session_id
            }

    def clear_context(self, session_id):
        if session_id in self.context:
            del self.context[session_id]


# ==================== INITIALIZE ====================
@st.cache_resource
def init_chatbot():
    return Chatbot()

chatbot = init_chatbot()
database = chatbot.database

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ==================== CSS ====================
st.markdown("""
<style>
    .user-msg {
        padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;
        background-color: #e3f2fd; margin-left: 20%; border-left: 4px solid #2196F3;
    }
    .bot-msg {
        padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;
        background-color: #f5f5f5; margin-right: 20%; border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("🤖 AI Chatbot")
    st.markdown("---")

    page = st.radio("📍 Navigate:", ["💬 Chat", "📊 Analytics", "📝 Training"])

    st.markdown("---")
    st.subheader("📈 Quick Stats")

    analytics_data = database.get_analytics()
    if analytics_data:
        total_msg = sum(row[1] for row in analytics_data)
        total_sess = sum(row[2] for row in analytics_data)
        st.metric("Messages (7d)", total_msg)
        st.metric("Sessions (7d)", total_sess)
    else:
        st.info("No data yet")

    st.markdown("---")
    st.code(st.session_state.session_id[:8] + "...")

    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        chatbot.clear_context(st.session_state.session_id)
        st.rerun()

# ==================== CHAT PAGE ====================
if page == "💬 Chat":
    st.title("💬 Chat with AI Assistant")

    col1, col2 = st.columns([2, 1])

    with col1:
        chat_container = st.container()

        with chat_container:
            if not st.session_state.chat_history:
                st.info("👋 Start a conversation by typing a message below!")

            for chat in st.session_state.chat_history:
                st.markdown(f'<div class="user-msg"><strong>👤 You:</strong><br>{chat["user"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-msg"><strong>🤖 Bot:</strong><br>{chat["bot"]}<br><small style="color:gray;">Intent: {chat["intent"]} | Confidence: {chat["confidence"]:.0%}</small></div>', unsafe_allow_html=True)

        st.markdown("---")

        with st.form(key='chat_form', clear_on_submit=True):
            col_input, col_btn = st.columns([4, 1])
            with col_input:
                user_input = st.text_input("Message:", placeholder="Type your message here...", label_visibility="collapsed")
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
            sentiment = latest['sentiment']
            polarity = sentiment['polarity']

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=polarity,
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

            emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
            st.markdown(f"**Sentiment:** {emoji.get(sentiment['sentiment'], '😐')} {sentiment['sentiment'].capitalize()}")

            st.markdown("**Emotions:**")
            for em in sentiment.get('emotions', ['neutral']):
                st.markdown(f"• {em.capitalize()}")

            st.markdown("---")
            st.markdown("**🏷️ Entities:**")
            entities = latest['entities']
            if entities:
                for etype, vals in entities.items():
                    st.markdown(f"• **{etype}:** {', '.join(vals[:3])}")
            else:
                st.caption("No entities detected")
        else:
            st.info("Send a message to see analysis!")

# ==================== ANALYTICS PAGE ====================
elif page == "📊 Analytics":
    st.title("📊 Analytics Dashboard")

    days_range = st.selectbox("📅 Time Range", [7, 14, 30, 60], index=0)

    analytics_data = database.get_analytics(days=days_range)
    intent_data = database.get_intent_dist()
    recent = database.get_recent(limit=100)

    st.subheader("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    if analytics_data:
        total_msg = sum(row[1] for row in analytics_data)
        total_sess = sum(row[2] for row in analytics_data)
        avg_conf = sum(row[3] or 0 for row in analytics_data) / len(analytics_data) if analytics_data else 0
        pos = sum(row[4] or 0 for row in analytics_data)
        neg = sum(row[5] or 0 for row in analytics_data)

        col1.metric("💬 Messages", total_msg)
        col2.metric("👥 Sessions", total_sess)
        col3.metric("🎯 Avg Confidence", f"{avg_conf:.0%}")
        pos_rate = pos / (pos + neg) * 100 if (pos + neg) > 0 else 0
        col4.metric("😊 Positive Rate", f"{pos_rate:.1f}%")
    else:
        st.info("📭 No data yet. Start chatting!")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Messages Over Time")
        if analytics_data:
            df = pd.DataFrame(analytics_data, columns=['date', 'messages', 'sessions', 'avg_conf', 'pos', 'neg', 'neu'])
            fig = px.line(df, x='date', y='messages', markers=True, title='Daily Messages')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data")

    with col2:
        st.subheader("🎯 Intent Distribution")
        if intent_data:
            df = pd.DataFrame(intent_data, columns=['intent', 'count'])
            fig = px.pie(df, values='count', names='intent', title='User Intents')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data")

    st.subheader("😊 Sentiment Distribution")
    if analytics_data:
        sent_data = {
            'Sentiment': ['Positive', 'Negative', 'Neutral'],
            'Count': [
                sum(row[4] or 0 for row in analytics_data),
                sum(row[5] or 0 for row in analytics_data),
                sum(row[6] or 0 for row in analytics_data)
            ]
        }
        df = pd.DataFrame(sent_data)
        fig = px.bar(df, x='Sentiment', y='Count', color='Sentiment',
                    color_discrete_map={'Positive': '#6bcb77', 'Negative': '#ff6b6b', 'Neutral': '#ffd93d'})
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("💬 Recent Conversations")
    if recent:
        df = pd.DataFrame(recent, columns=['Session', 'User', 'Bot', 'Intent', 'Confidence', 'Sentiment', 'Time'])
        df['Session'] = df['Session'].str[:8] + '...'
        df['User'] = df['User'].str[:40]
        df['Bot'] = df['Bot'].str[:40]
        df['Confidence'] = df['Confidence'].apply(lambda x: f"{x:.0%}" if x else "N/A")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No conversations yet")

# ==================== TRAINING PAGE ====================
elif page == "📝 Training":
    st.title("📝 Training & Intent Management")

    st.subheader("➕ Add New Intent")
    with st.form("add_intent"):
        tag = st.text_input("Intent Tag", placeholder="e.g., product_inquiry")
        patterns = st.text_area("Patterns (one per line)", placeholder="what products do you have\nshow me products")
        responses = st.text_area("Responses (one per line)", placeholder="We have many products!")

        if st.form_submit_button("➕ Add Intent", use_container_width=True):
            if tag and patterns and responses:
                p_list = [p.strip() for p in patterns.split('\n') if p.strip()]
                r_list = [r.strip() for r in responses.split('\n') if r.strip()]
                new_intent = {"tag": tag, "patterns": p_list, "responses": r_list}
                chatbot.intents_data['intents'].append(new_intent)
                try:
                    with open('data/intents.json', 'w') as f:
                        json.dump(chatbot.intents_data, f, indent=4)
                    chatbot.classifier = IntentClassifier(chatbot.intents_data)
                    st.success(f"Intent '{tag}' added!")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Fill all fields")

    st.markdown("---")

    st.subheader("📋 Existing Intents")
    for intent in chatbot.intents_data.get('intents', []):
        with st.expander(f"🏷️ {intent['tag']} ({len(intent.get('patterns', []))} patterns)"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Patterns:**")
                for p in intent.get('patterns', [])[:5]:
                    st.markdown(f"• {p}")
                if len(intent.get('patterns', [])) > 5:
                    st.caption(f"...and {len(intent['patterns']) - 5} more")
            with col2:
                st.markdown("**Responses:**")
                for r in intent.get('responses', [])[:3]:
                    st.markdown(f"• {r[:50]}...")

    st.markdown("---")

    st.subheader("🧪 Test Classification")
    test_msg = st.text_input("Enter test message:")
    if test_msg:
        intent, conf = chatbot.classifier.predict(test_msg)
        col1, col2 = st.columns(2)
        col1.metric("Predicted Intent", intent)
        col2.metric("Confidence", f"{conf:.0%}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;'>🤖 Dynamic AI Chatbot | Built with Streamlit</div>",
    unsafe_allow_html=True
)
