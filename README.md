# 📚 Doc_Q-A: AI-Powered Document Learning & Quiz Assistant

A full-stack Retrieval-Augmented Generation (RAG) platform with a **FastAPI backend** and a clean **HTML & CSS frontend**.

---

## 📂 Project Structure

```
Doc_Q-A/
│
├── main.py               # FastAPI backend with REST API routes (/api/chat, /api/quiz, /api/upload)
├── templates/
│   └── index.html        # HTML5 & CSS3 frontend UI (Dark mode, responsive)
├── ingestion.py          # PDF Ingestion: loads PDF, chunks text, creates FAISS index
├── rag_pipeline.py       # RAG logic: retriever, Gemini 3.5 Flash LLM chain, query & quiz generation
├── sample.pdf            # Sample test PDF
├── faiss_index/          # Local FAISS vector store
├── requirements.txt      # Dependencies (FastAPI, Uvicorn, LangChain, etc.)
├── .env.example          # Environment variables template
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/TejasKabotula/Doc_Q-A.git
cd Doc_Q-A
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Gemini API Key
In `.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
```

### 4. Start the Application
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **`http://localhost:8000`**!

---

## 🌟 Key Features

- **💬 Ask Queries Tab:** Chat with your PDF and view retrieved context chunks with a single click.
- **📝 Take Quiz Tab:** Generate multiple-choice quizzes, select your options, and see automatic grading with explanations.
- **📄 Sidebar Upload:** Drag-and-drop or select any PDF to re-index the knowledge base into FAISS.
- **🧠 Zero-Hallucination RAG:** Gemini 3.5 Flash responses are strictly grounded in retrieved document chunks.
