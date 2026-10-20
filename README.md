# CodeAlpha_FAQChatbot
### Task 2: Intelligent FAQ Chatbot — CodeAlpha Artificial Intelligence Internship

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Chat-FF4B4B.svg)](https://streamlit.io)
[![NLTK](https://img.shields.io/badge/NLP-NLTK-green.svg)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, NLP-driven Question Answering chatbot built to answer inquiries about the **CodeAlpha Artificial Intelligence Internship** using **Natural Language Processing (NLTK)**, **TF-IDF Vectorization**, and **Cosine Similarity**.

---

## 📌 Features

- **NLP Preprocessing Pipeline**: Tokenizes queries, strips punctuation, eliminates English stop words, and applies WordNet lemmatization for normalized text matching.
- **Semantic Similarity Matching**: Uses TF-IDF (Unigram + Bigram) matrix representations and computes Cosine Similarity against the knowledge base.
- **Match Confidence Indicator**: Shows the user exact percentage confidence of the retrieved answer.
- **Graceful Fallback**: Automatically informs the user when a question is outside the knowledge threshold and provides direct official support channels.
- **Interactive Modern Chat UI**: Built with Streamlit's native chat components (`st.chat_message`, `st.chat_input`).
- **One-Click Suggestions**: Interactive quick-prompt chips to try popular questions with a single click.
- **Session History & Reset**: Remembers the back-and-forth conversational turns, with an instant clear history toggle.

---

## 🛠️ Tech Stack

- **NLP Pipeline**: [NLTK](https://www.nltk.org/) (Tokenization, Stopwords, WordNet Lemmatization)
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/) (TfidfVectorizer, Cosine Similarity)
- **Frontend / UI**: [Streamlit](https://streamlit.io/)
- **Data Format**: JSON (`faqs.json`)

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/a-b-infotech/CodeAlpha_FAQChatbot.git
cd CodeAlpha_FAQChatbot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Chatbot
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📖 Project Structure
```
CodeAlpha_FAQChatbot/
│
├── app.py              # Streamlit chatbot application & NLP engine
├── faqs.json           # Categorized FAQ knowledge base
├── requirements.txt    # Project dependencies
└── README.md           # Documentation & instructions
```

---

## 👨‍💻 Author
- **Name**: Abhi B Sheth
- **Internship**: CodeAlpha Artificial Intelligence Internship
- **GitHub**: [@a-b-infotech](https://github.com/a-b-infotech)
