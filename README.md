# 📚 Doc_Q-A: AI-Powered Document Learning & Quiz Assistant

A simple, modular Retrieval-Augmented Generation (RAG) platform with a **FastAPI backend**, **Vanilla HTML/CSS frontend**, and **LangChain pipeline**.

---

## 📂 Project Structure

```
Doc_Q-A/
│
├── data/                 # All PDF documents and datasets
│   └── sample.pdf
│
├── frontend/             # All frontend UI files (HTML, CSS, JS)
│   └── index.html
│
├── pipeline/             # Clean LangChain RAG pipeline
│   ├── __init__.py
│   ├── ingestion.py      # PDF Loader -> Recursive Splitter -> Embeddings -> FAISS
│   └── rag_pipeline.py   # FAISS Retriever -> Gemini 3.5 Flash LCEL -> Answer/Quiz
│
├── faiss_index/          # Local vector index generated from data/
├── main.py               # FastAPI server connecting frontend and pipeline
├── requirements.txt      # Minimal project dependencies
├── .env.example          # Environment variables template
└── README.md
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Gemini API Key
In `.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
```

### 3. Start the Server
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **`http://localhost:8000`**!

---

## 🌟 Features

- **💬 Ask Queries Tab:** Chat with your document and view retrieved context chunks.
- **📝 Take Quiz Tab:** Generate multiple-choice quizzes on any topic, submit answers, and receive instant scores and explanations.
- **📄 Sidebar Upload:** Upload any new PDF into the `data/` folder and re-index the FAISS vector database with one click.
