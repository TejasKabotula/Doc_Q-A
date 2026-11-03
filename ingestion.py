import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def run_ingestion(pdf_path: str = "sample.pdf", index_dir: str = "faiss_index"):
    """
    Ingestion Pipeline:
    1. Loads PDF document
    2. Splits into chunks using RecursiveCharacterTextSplitter
    3. Generates embeddings using sentence-transformers/all-MiniLM-L6-v2
    4. Creates and saves FAISS vector store
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found!")
        sys.exit(1)

    print(f"Loading PDF from: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} page(s).")

    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print("Number of chunks created:", len(chunks))

    print("Creating HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embeddings model loaded successfully.")

    print("Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_dir)
    print(f"FAISS vector store saved successfully to '{index_dir}'.")
    return vectorstore

if __name__ == "__main__":
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    run_ingestion(pdf_file)

