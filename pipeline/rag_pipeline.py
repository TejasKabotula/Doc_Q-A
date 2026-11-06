import os
import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

INDEX_DIR = "faiss_index"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def get_vectorstore(index_dir: str = INDEX_DIR):
    if not os.path.exists(index_dir):
        from .ingestion import run_ingestion
        run_ingestion("data/sample.pdf", index_dir)

    embeddings = get_embeddings()
    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )

def get_retriever(k: int = 5, index_dir: str = INDEX_DIR):
    vectorstore = get_vectorstore(index_dir)
    return vectorstore.as_retriever(search_kwargs={"k": k})

def get_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        temperature=0.2
    )

def answer_query(query: str, retriever=None):
    if retriever is None:
        retriever = get_retriever()

    llm = get_llm()
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template(
        """Answer the question using only the given context. If the answer is not in the context, say that the document does not contain this information.

Context:
{context}

Question:
{question}

Answer:"""
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})
    return answer, docs

def generate_quiz(topic: str = "Main concepts", num_questions: int = 5, retriever=None):
    if retriever is None:
        retriever = get_retriever(k=min(num_questions * 2, 10))

    llm = get_llm()
    docs = retriever.invoke(topic)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template(
        """Generate a {num_questions}-question multiple choice quiz on the topic '{topic}' using only the context below.

Each question must contain:
- "id": integer (1, 2, 3...)
- "question": clear question text
- "options": object with keys "A", "B", "C", "D"
- "correct_answer": "A", "B", "C", or "D"
- "explanation": brief explanation from the context

Return valid JSON adhering to this structure:
{{
  "questions": [
    {{
      "id": 1,
      "question": "Question text?",
      "options": {{"A": "Choice A", "B": "Choice B", "C": "Choice C", "D": "Choice D"}},
      "correct_answer": "A",
      "explanation": "Explanation here."
    }}
  ]
}}

Context:
{context}
"""
    )

    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({
        "num_questions": num_questions,
        "topic": topic,
        "context": context
    })

    # Clean code fences if present and parse JSON
    cleaned = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
        return data.get("questions", []), docs
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data.get("questions", []), docs
        return [], docs

if __name__ == "__main__":
    query = input("Enter your question: ")
    ans, _ = answer_query(query)
    print("\n========== WITH RAG ==========\n", ans)

