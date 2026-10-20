import streamlit as st
import json
import os
import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK resources are available
for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# Page Configuration
st.set_page_config(
    page_title="AI FAQ Chatbot | CodeAlpha",
    page_icon="🤖",
    layout="centered"
)

# Load Knowledge Base
FAQ_FILE = os.path.join(os.path.dirname(__file__), "faqs.json")

@st.cache_data
def load_faq_database():
    if os.path.exists(FAQ_FILE):
        with open(FAQ_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return [
        {"question": "What is CodeAlpha internship?", "answer": "CodeAlpha provides virtual internships across AI and software engineering.", "category": "General"},
        {"question": "How many tasks should I complete?", "answer": "Complete 2 or 3 tasks to earn the verified certificate.", "category": "Tasks"},
        {"question": "How to contact support?", "answer": "Contact services@codealpha.tech or WhatsApp +91 9336576683.", "category": "Support"}
    ]

faqs_data = load_faq_database()

# NLP Preprocessing Pipeline
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = {"a", "an", "the", "in", "on", "is", "are", "of", "to", "for", "with", "at", "by"}

def preprocess_text(text: str) -> str:
    """Cleans, tokenizes, removes stop words and lemmatizes input text."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = text.split()
    processed_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]
    return " ".join(processed_tokens) if processed_tokens else text

# Build TF-IDF Model
corpus = [faq["question"] for faq in faqs_data]
preprocessed_corpus = [preprocess_text(q) for q in corpus]

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(preprocessed_corpus)

def match_query(query: str, threshold: float = 0.22):
    """Matches user query against FAQ corpus using Cosine Similarity."""
    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score >= threshold:
        matched_faq = faqs_data[best_idx]
        return {
            "answer": matched_faq["answer"],
            "matched_question": matched_faq["question"],
            "category": matched_faq.get("category", "General"),
            "confidence": round(best_score * 100, 1),
            "found": True
        }
    return {
        "answer": "I'm sorry, I couldn't find a direct answer to that question in the knowledge base. You can reach out directly to the CodeAlpha support team at **services@codealpha.tech** or on WhatsApp at **+91 9336576683**.",
        "matched_question": None,
        "category": None,
        "confidence": round(best_score * 100, 1),
        "found": False
    }

# UI Layout
st.markdown("""
<style>
    .chat-header {
        text-align: center;
        padding-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        background: #E8F5E9;
        color: #2E7D32;
        margin-bottom: 0.5rem;
    }
    .confidence-badge {
        font-size: 0.8rem;
        color: #666;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-header"><span class="badge">CodeAlpha AI Internship — Task 2</span><h2>🤖 Intelligent FAQ Assistant</h2><p style="color: #666;">Ask any question about the AI internship, task requirements, submission rules, or technical architecture.</p></div>', unsafe_allow_html=True)

# Quick Suggestions
st.write("**💡 Suggested questions to try:**")
col1, col2, col3 = st.columns(3)
quick_q = None

if col1.button("📌 How many tasks needed?"):
    quick_q = "How many tasks are required to get the internship certificate?"
if col2.button("📤 How do I submit?"):
    quick_q = "How do I submit my completed internship projects?"
if col3.button("📜 How does FAQ matching work?"):
    quick_q = "How does the FAQ Chatbot algorithm work?"

# Session State for Conversation History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi there! I am your CodeAlpha AI FAQ Assistant. How can I assist you with your internship tasks today?"}
    ]

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "confidence" in msg and msg["confidence"]:
            st.caption(f"🎯 Match Confidence: {msg['confidence']}% | Category: {msg.get('category', 'General')}")

# Handle User Input
prompt = st.chat_input("Ask a question about the internship...") or quick_q

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Analyzing query with NLP..."):
        match_result = match_query(prompt)

    assistant_msg = {
        "role": "assistant",
        "content": match_result["answer"],
        "confidence": match_result["confidence"] if match_result["found"] else None,
        "category": match_result["category"]
    }
    st.session_state.messages.append(assistant_msg)
    
    with st.chat_message("assistant"):
        st.markdown(assistant_msg["content"])
        if assistant_msg["confidence"]:
            st.caption(f"🎯 Matched FAQ: *\"{match_result['matched_question']}\"* ({assistant_msg['confidence']}%)")

# Sidebar
with st.sidebar:
    st.title("⚙️ Chatbot Options")
    st.info(f"📚 Knowledge Base: **{len(faqs_data)}** FAQs loaded across domains.")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat history cleared. How can I help you?"}
        ]
        st.rerun()
        
    st.markdown("---")
    st.markdown("### 🔍 Model Architecture")
    st.markdown("- **Tokenization & Lemmatization**: NLTK")
    st.markdown("- **Feature Extraction**: TF-IDF (Unigrams + Bigrams)")
    st.markdown("- **Similarity Measure**: Cosine Similarity")
    st.markdown("---")
    st.caption("CodeAlpha AI Internship | Author: Abhi B Sheth")
