import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline import run_ingestion, answer_query, generate_quiz

app = FastAPI(title="PDF Learning Assistant RAG API")

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_HTML = BASE_DIR / "frontend" / "index.html"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

class QueryRequest(BaseModel):
    question: str

class QuizRequest(BaseModel):
    topic: str = "Main concepts"
    num_questions: int = 5

@app.get("/")
async def serve_frontend():
    """Serves the single-page HTML frontend from the frontend folder."""
    if not FRONTEND_HTML.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found.")
    return FileResponse(FRONTEND_HTML)

@app.get("/api/status")
async def get_status():
    """Returns vector index status."""
    return {"indexed": os.path.exists("faiss_index")}

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Saves uploaded PDF to data/ folder and runs the ingestion pipeline."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = DATA_DIR / file.filename
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        run_ingestion(str(save_path), index_dir="faiss_index")
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
    """Answers a question using RAG from the FAISS vector store."""
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer, docs = answer_query(req.question.strip())
        contexts = [d.page_content for d in docs]
        return {"answer": answer, "contexts": contexts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/quiz")
async def make_quiz(req: QuizRequest):
    """Generates an MCQ quiz based on retrieved document chunks."""
    try:
        questions, _ = generate_quiz(topic=req.topic, num_questions=req.num_questions)
        if not questions:
            raise HTTPException(status_code=500, detail="Unable to generate questions from document context.")
        return {"questions": questions}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
