import streamlit as st
from config import VECTOR_DIR, OPENAI_API_KEY
from retriever import answer_question

st.set_page_config(page_title="Tamil Nadu Agriculture Schemes Assistant",
                   page_icon="🌾", layout="wide")
st.title("🌾 Tamil Nadu Agriculture Schemes Assistant")
st.write(
    "Ask questions about agriculture schemes using the indexed official documents. "
    "Answers are grounded in the saved FAISS knowledge base; this screen does not browse the web."
)
st.caption("தமிழிலும் கேள்விகள் கேட்கலாம். பதில்கள் குறியிடப்பட்ட ஆவணங்களின் அடிப்படையில் வழங்கப்படும்.")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing. Add your key to the .env file and restart Streamlit.")
    st.stop()

if not (VECTOR_DIR / "faiss_index.faiss").exists():
    st.warning("The knowledge base has not been built yet. Open PowerShell in the project folder and run: python ingest.py")
    st.stop()

with st.sidebar:
    st.subheader("Example questions")
    st.markdown(
        "- What agriculture subsidies are indexed?\n"
        "- What documents are required for crop insurance?\n"
        "- How can I apply for drip irrigation assistance?\n"
        "- Who is eligible for a specific scheme?\n"
        "- இந்தத் திட்டத்திற்கு விண்ணப்பிக்க என்னென்ன ஆவணங்கள் தேவை?"
    )
    st.info("For official confirmation, use the source links shown with each answer.")
    if st.button("Clear question and answer"):
        st.session_state.question = ""
        st.session_state.result = None
        st.rerun()

if "question" not in st.session_state:
    st.session_state.question = ""
if "result" not in st.session_state:
    st.session_state.result = None

with st.form("question_form"):
    question = st.text_area("Enter your question / உங்கள் கேள்வியை உள்ளிடவும்",
                            value=st.session_state.question, height=100,
                            placeholder="Example: What documents are needed for crop insurance?")
    submitted = st.form_submit_button("Ask the agriculture assistant", type="primary")

if submitted:
    st.session_state.question = question
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching the local FAISS knowledge base and preparing an answer..."):
            try:
                st.session_state.result = answer_question(question)
            except FileNotFoundError as exc:
                st.session_state.result = {"answer": str(exc), "sources": []}
            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")
                st.info("Check your .env settings, API quota, and logs. The app does not fall back to web search.")
                st.session_state.result = None

result = st.session_state.result
if result:
    st.subheader("Answer")
    st.write(result["answer"])
    sources = result.get("sources", [])
    if sources:
        st.subheader("Sources and supporting excerpts")
        for i, source in enumerate(sources, start=1):
            label = f"[Source {i}] {source['title']}"
            if source.get("page"):
                label += f" — page {source['page']}"
            with st.expander(label, expanded=(i == 1)):
                st.write(source["excerpt"])
                if source.get("url"):
                    st.markdown(f"**Source URL:** [{source['url']}]({source['url']})")
                st.caption(f"FAISS distance: {source.get('distance', 'n/a')}")
    else:
        st.caption("No source excerpts passed the configured relevance threshold.")