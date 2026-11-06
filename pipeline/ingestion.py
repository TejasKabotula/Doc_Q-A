import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def run_ingestion(pdf_path: str = "data/sample.pdf", index_dir: str = "faiss_index"):
    """Loads PDF, splits into chunks, creates embeddings, and saves FAISS vector store."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # 3. Create Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Create and save FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_dir)
    return vectorstore

if __name__ == "__main__":
    import sys
    file_to_index = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"
    run_ingestion(file_to_index)
    print("Ingestion complete. FAISS index created.")

