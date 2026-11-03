import os
import streamlit as st
from ingestion import run_ingestion
from rag_pipeline import answer_query, generate_quiz

st.set_page_config(page_title="PDF Learning Assistant", page_icon="📚", layout="centered")

st.title("📚 PDF Learning Assistant")
st.caption("Ask questions and take quizzes directly from your PDF documents using RAG.")

# Sidebar for PDF Upload and Setup
with st.sidebar:
    st.header("📄 Document Setup")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        save_path = f"uploaded_{uploaded_file.name}"
        if st.button("⚡ Process & Index PDF", type="primary"):
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("Chunking text & generating vector index..."):
                run_ingestion(save_path)
                st.session_state["indexed"] = True
                st.session_state["current_pdf"] = uploaded_file.name
                st.success(f"Indexed '{uploaded_file.name}'!")

    if os.path.exists("faiss_index"):
        current_doc = st.session_state.get("current_pdf", "sample.pdf")
        st.success(f"🟢 Active Index: {current_doc}")
    else:
        st.info("⚪ No index found. Upload a PDF or run ingestion on sample.pdf.")

# Session state setup
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

# Navigation Tabs
tab_chat, tab_quiz = st.tabs(["💬 Ask Queries", "📝 Take Quiz"])

# -------------------------------------------------------------
# TAB 1: ASK QUERIES
# -------------------------------------------------------------
with tab_chat:
    st.subheader("Chat with PDF")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("🔍 View Retrieved Context"):
                    for i, doc in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk {i} (Page {doc.metadata.get('page', '?')})**")
                        st.text(doc.page_content)

    user_query = st.chat_input("Ask a question about the document...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching document & generating answer..."):
                try:
                    answer, docs = answer_query(user_query)
                    st.markdown(answer)
                    with st.expander("🔍 View Retrieved Context"):
                        for i, doc in enumerate(docs, 1):
                            st.markdown(f"**Chunk {i} (Page {doc.metadata.get('page', '?')})**")
                            st.text(doc.page_content)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": docs
                    })
                except Exception as e:
                    st.error(f"Error: {e}")

# -------------------------------------------------------------
# TAB 2: TAKE QUIZ
# -------------------------------------------------------------
with tab_quiz:
    st.subheader("Interactive Quiz Generator")

    c1, c2 = st.columns([3, 1])
    with c1:
        topic = st.text_input("Quiz Topic", value="Main concepts", help="Focus area for quiz generation.")
    with c2:
        num_q = st.selectbox("Questions", [3, 5, 10], index=1)

    if st.button("🎯 Generate Quiz", type="primary"):
        with st.spinner("Generating quiz from retrieved document chunks..."):
            questions, _ = generate_quiz(topic=topic, num_questions=num_q)
            if questions:
                st.session_state.quiz_questions = questions
                st.session_state.quiz_submitted = False
                st.session_state.user_answers = {}
                st.success(f"Generated {len(questions)} questions on '{topic}'!")
            else:
                st.error("Failed to generate quiz. Ensure your document is indexed and Gemini API key is configured.")

    if st.session_state.quiz_questions:
        st.markdown("---")
        user_answers = {}

        with st.form("quiz_form"):
            for q in st.session_state.quiz_questions:
                st.markdown(f"**Question {q['id']}:** {q['question']}")
                opts = q.get("options", {})
                choices = [f"{k}: {opts[k]}" for k in ["A", "B", "C", "D"] if k in opts]
                
                selected = st.radio(
                    f"Select answer for Q{q['id']}:",
                    options=choices,
                    index=None,
                    key=f"q_choice_{q['id']}",
                    label_visibility="collapsed"
                )
                if selected:
                    user_answers[q["id"]] = selected.split(":")[0].strip()
                st.markdown("")

            submitted = st.form_submit_button("Submit Quiz", type="primary")

        if submitted:
            st.session_state.quiz_submitted = True
            st.session_state.user_answers = user_answers

        if st.session_state.quiz_submitted:
            correct_count = 0
            total = len(st.session_state.quiz_questions)

            for q in st.session_state.quiz_questions:
                user_ans = st.session_state.user_answers.get(q["id"], "")
                if user_ans == q["correct_answer"]:
                    correct_count += 1

            percentage = round((correct_count / total) * 100, 1)
            st.markdown(f"### 🎯 Final Score: {correct_count}/{total} ({percentage}%)")

            for q in st.session_state.quiz_questions:
                user_ans = st.session_state.user_answers.get(q["id"], "None")
                is_correct = (user_ans == q["correct_answer"])

                if is_correct:
                    st.success(f"**Q{q['id']}: Correct!** (Your answer: {user_ans})")
                else:
                    st.error(f"**Q{q['id']}: Incorrect.** Your answer: `{user_ans}`, Correct: `{q['correct_answer']}`")

                st.info(f"💡 **Explanation:** {q.get('explanation', 'N/A')}")
                st.markdown("---")

