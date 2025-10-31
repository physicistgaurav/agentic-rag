import os
import shutil
from typing import List

import streamlit as st
from dotenv import load_dotenv

import google.generativeai as genai
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# ----------------------------------------------------------------------
# 1. Load env + basic Streamlit config
# ----------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="Resume RAG — Gemini Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Resume RAG — Gemini Research Assistant")
st.caption("Analyze and extract insights from uploaded resumes")

# ----------------------------------------------------------------------
# 2. Sidebar – API key, model, upload
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    if not api_key:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get one at https://aistudio.google.com/app/apikey",
        )
    if not api_key:
        st.warning("API key required")
        st.stop()

    genai.configure(api_key=api_key)

    model_choice = st.selectbox(
        "Gemini model",
        ["gemini-2.5-flash", "gemini-2.5-pro"],
        index=1,
    )

    st.subheader("Upload Resumes")
    uploaded_files = st.file_uploader(
        "PDFs or TXT resumes",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

# ----------------------------------------------------------------------
# 3. Helper: build / retrieve LLM + embedder (cached)
# ----------------------------------------------------------------------


@st.cache_resource
def get_llm_and_embedder(_api_key: str, _model: str):
    llm = GoogleGenAI(model=_model, api_key=_api_key)
    embed = GoogleGenAIEmbedding(
        model="models/embedding-001", api_key=_api_key)
    Settings.llm = llm
    Settings.embed_model = embed
    return llm, embed


# ----------------------------------------------------------------------
# 4. Index building (cached per upload set)
# ----------------------------------------------------------------------
TEMP_DIR = "uploaded_resumes"


def _save_uploads(files):
    os.makedirs(TEMP_DIR, exist_ok=True)
    for f in files:
        path = os.path.join(TEMP_DIR, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
    return len(files)


@st.cache_resource(show_spinner="Building vector index…")
def build_index(_uploaded_files):
    if not _uploaded_files:
        return None

    cnt = _save_uploads(_uploaded_files)
    st.success(f"Uploaded {cnt} resume(s)")

    reader = SimpleDirectoryReader(TEMP_DIR)
    docs = reader.load_data()

    get_llm_and_embedder(api_key, model_choice)

    index = VectorStoreIndex.from_documents(docs)
    return index


# ----------------------------------------------------------------------
# 5. Refine output and prompt for resume context
# ----------------------------------------------------------------------
RESUME_PROMPT = """
You are a resume analysis assistant. Your tasks:
- Extract key skills, experiences, and education from resumes
- Summarize candidate strengths
- Compare candidates if multiple resumes
- Provide concise insights for hiring decision making

User Query: {query}
"""


def refine_output(text: str):
    text = text.strip()
    return f"🧩 **Refined Resume Insights:**\n\n{text}"


# ----------------------------------------------------------------------
# 6. Suggested template questions for resumes
# ----------------------------------------------------------------------
EXAMPLE_QUERIES = [
    "Summarize key skills and experience of the candidate.",
    "Highlight education and certifications.",
    "Compare all uploaded resumes and rank candidates.",
    "Identify potential gaps or weaknesses in the resume."
]

# ----------------------------------------------------------------------
# 7. Helper function to process any query (template or input)
# ----------------------------------------------------------------------


def handle_query(query_text):
    """Process a query: add to chat, run index query, and store assistant response"""
    st.session_state.messages.append({"role": "user", "content": query_text})

    with st.chat_message("user"):
        st.markdown(query_text)

    # Optional reasoning trace toggle
    show_trace = st.checkbox("🔍 Show reasoning trace",
                             value=False, key=f"trace_{query_text}")

    with st.chat_message("assistant"):
        with st.spinner("Analyzing resumes…"):
            engine = st.session_state.index.as_query_engine(
                similarity_top_k=5,
                response_mode="tree_summarize",
            )
            resp = engine.query(RESUME_PROMPT.format(query=query_text))
            answer = refine_output(resp.response)
            st.markdown(answer)

            if show_trace:
                st.markdown("### 🧠 Reasoning Trace")
                st.code(getattr(resp, "source_nodes",
                        "No reasoning trace available."))

            st.session_state.messages.append(
                {"role": "assistant", "content": answer})


# ----------------------------------------------------------------------
# 8. Main UI
# ----------------------------------------------------------------------
if uploaded_files:
    index = build_index(uploaded_files)
    if index:
        st.session_state.index = index  # store globally for queries
        st.success("Index ready – start querying resumes!")

        # ---- chat history ----
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Template questions
        st.subheader("💡 Suggested Resume Queries")
        cols = st.columns(len(EXAMPLE_QUERIES))
        for i, q in enumerate(EXAMPLE_QUERIES):
            if cols[i].button(q):
                handle_query(q)

        # Query input
        if prompt := st.chat_input("Ask about the resumes…"):
            handle_query(prompt)

        # Clear & Download chat
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear Chat"):
                st.session_state.messages = []
                st.success("Chat cleared!")

        with col2:
            if st.download_button(
                "💾 Download Chat",
                data="\n\n".join(
                    [f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages]
                ),
                file_name="resume_chat_history.txt",
                mime="text/plain",
            ):
                st.success("Chat downloaded!")

        # Clear documents & index
        st.divider()
        if st.button("🗑️ Clear resumes & chat"):
            if os.path.isdir(TEMP_DIR):
                shutil.rmtree(TEMP_DIR)
            st.cache_resource.clear()
            st.session_state.messages = []
            if "index" in st.session_state:
                del st.session_state.index
            st.success("Everything cleared!")

else:
    st.info("Upload one or more resumes to start.")
    st.markdown(
        """
        ### Features
        - **Multi-step reasoning** (`tree_summarize`)  
        - **Gemini 1.5 Pro / Flash**  
        - **Gemini embeddings** (`embedding-001`)  
        - **Full source trace for each resume**  
        - **Template queries** for instant resume insights  
        - **Clear & download chat**
        """
    )
