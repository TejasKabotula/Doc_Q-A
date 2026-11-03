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
_cached_embeddings = None

def clean_text_output(output) -> str:
    """Normalize output whether it's a string or a list of content blocks."""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = []
        for item in output:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(output)

def get_embeddings():
    global _cached_embeddings
    if _cached_embeddings is None:
        _cached_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _cached_embeddings

def get_vectorstore(index_dir: str = INDEX_DIR):
    if not os.path.exists(index_dir):
        from ingestion import run_ingestion
        run_ingestion("sample.pdf", index_dir)

    embeddings = get_embeddings()
    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )

def get_retriever(k: int = 5):
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})

def get_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        temperature=0.2
    )

def answer_query(query: str, retriever=None, llm=None):
    if retriever is None:
        retriever = get_retriever()
    if llm is None:
        llm = get_llm()

    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template(
        """Answer the question using only the given context. If the answer is not contained in the context, say so clearly.

Context:
{context}

Question:
{question}

Answer:"""
    )

    chain = prompt | llm | StrOutputParser()
    raw_answer = chain.invoke({"context": context, "question": query})
    return clean_text_output(raw_answer), docs

def generate_quiz(topic: str = "main concepts", num_questions: int = 5):
    retriever = get_retriever(k=min(num_questions * 2, 10))
    llm = get_llm()

    docs = retriever.invoke(topic)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template(
        """Generate a {num_questions}-question multiple choice quiz on the topic '{topic}' using ONLY the context below.

Each question must have:
- "id": integer (1, 2, 3...)
- "question": question text
- "options": dictionary with keys "A", "B", "C", "D"
- "correct_answer": "A", "B", "C", or "D"
- "explanation": short reason why this is correct according to the context

Return ONLY valid JSON format like:
{{
  "questions": [
    {{
      "id": 1,
      "question": "Sample question?",
      "options": {{"A": "text", "B": "text", "C": "text", "D": "text"}},
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

    cleaned = clean_text_output(raw).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        data = json.loads(cleaned)
        return data.get("questions", []), docs
    except Exception as e:
        print("Quiz JSON parse error:", e)
        return [], docs

if __name__ == "__main__":
    query = input("Enter your question: ")
    ans, _ = answer_query(query)
    print("\n\n\n========== WITH RAG ==========")
    print(ans)
