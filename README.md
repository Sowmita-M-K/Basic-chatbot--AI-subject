# AI & DS Department Chatbot 🤖

A foundational Natural Language Processing (NLP) chatbot designed for 3rd-year Artificial Intelligence & Data Science (AI & DS) engineering students.

---

## 📁 Project Structure

```text
aids_chatbot/
├── data/
│   └── intents.json          # Knowledge base containing department FAQs, tags, and patterns
├── src/
│   ├── __init__.py           # Package initializer
│   └── nlp_engine.py         # NLP vectorizer, intent matcher & cosine similarity logic
├── app.py                    # Interactive web UI built with Streamlit
├── cli_chat.py               # Interactive terminal chatbot (runs with zero extra libraries!)
├── requirements.txt          # Python library dependencies
└── README.md                 # Project documentation and setup guide
```

---

## ⚙️ Core Concepts (for College Viva & Presentation)

1. **Text Preprocessing**: Normalization, lowercasing, and special character removal using regular expressions (`re`).
2. **Feature Extraction (TF-IDF)**:
   - Converts department queries and FAQ patterns into numerical vectors.
   - Uses `ngram_range=(1, 2)` (unigrams + bigrams) to capture phrases like "3rd year" or "mini project".
3. **Similarity Matching**:
   - Calculates **Cosine Similarity** between the user's vector and known dataset vectors.
   - Triggers fallback when similarity confidence is below the threshold (`0.25`).

---

## 🚀 How to Run

### Option A: Instant CLI Mode (No installation required!)
Run directly in your command prompt:
```bash
cd C:\Users\sowmi\.gemini\antigravity\scratch\aids_chatbot
python cli_chat.py
```

### Option B: Interactive Web UI (Streamlit)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the web application:
   ```bash
   streamlit run app.py
   ```
   The browser will automatically open at `http://localhost:8501`.

---

## 🔮 Future Enhancements (Level-Up Ideas)
- **RAG (Retrieval-Augmented Generation)**: Ingest syllabus PDFs using LangChain and query with Google Gemini API.
- **Speech-to-Text / Text-to-Speech**: Add voice input with Python `speech_recognition`.
- **Database Integration**: Connect SQLite or MongoDB to log unanswered questions for continuous improvement.
