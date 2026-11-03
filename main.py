import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ingestion import run_ingestion
from rag_pipeline import answer_query, generate_quiz

app = FastAPI(title="PDF Learning Assistant RAG API")

# Ensure static and templates exist
BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "templates" / "index.html"

# Pydantic request models
class QueryRequest(BaseModel):
    question: str

class QuizRequest(BaseModel):
    topic: str = "Main concepts"
    num_questions: int = 5

@app.get("/")
async def serve_home():
    """Serve the single-page HTML frontend."""
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="Frontend template not found.")
    return FileResponse(INDEX_HTML)

@app.get("/api/status")
async def get_status():
    """Return vector index readiness."""
    has_index = os.path.exists("faiss_index")
    return {"indexed": has_index}

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, extract text, chunk, and index into FAISS."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = f"uploaded_{file.filename}"
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run ingestion pipeline
        vectorstore = run_ingestion(save_path, index_dir="faiss_index")
        return {
            "status": "success",
            "filename": file.filename,
            "message": f"Successfully indexed '{file.filename}'"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()

@app.post("/api/chat")
async def chat_query(req: QueryRequest):
    """Retrieve top FAISS chunks and generate Gemini RAG answer."""
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer, docs = answer_query(req.question.strip())
        contexts = [d.page_content for d in docs]
        return {
            "answer": answer,
            "contexts": contexts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/quiz")
async def make_quiz(req: QuizRequest):
    """Generate multiple-choice quiz grounded in FAISS retrieved chunks."""
    try:
        questions, docs = generate_quiz(
            topic=req.topic or "Main concepts",
            num_questions=req.num_questions
        )
        if not questions:
            raise HTTPException(status_code=500, detail="No questions could be generated from document context.")
        return {"questions": questions}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

