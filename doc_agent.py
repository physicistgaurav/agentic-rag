import os
import shutil
from typing import List

import streamlit as st
from dotenv import load_dotenv

import google.generativeai as genai
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="Resume RAG — Gemini Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Resume RAG — Gemini Research Assistant")
st.caption("Analyze and extract insights from uploaded resumes")

# 2. Sidebar – API key, model, upload
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
        ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
        index=0,
    )

    st.subheader("Upload Resumes")
    uploaded_files = st.file_uploader(
        "PDFs or TXT resumes",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    # Add show trace toggle in sidebar
    show_trace = st.checkbox("🔍 Show reasoning trace", value=False)

# 3. Helper: build / retrieve LLM + embedder (cached)


@st.cache_resource
def get_llm_and_embedder(_api_key: str, _model: str):
    llm = GoogleGenAI(model=_model, api_key=_api_key)
    embed = GoogleGenAIEmbedding(
        model="models/text-embedding-004", api_key=_api_key)
    Settings.llm = llm
    Settings.embed_model = embed
    return llm, embed


# 4. Index building (cached per upload set)
TEMP_DIR = "uploaded_resumes"


def _save_uploads(files):
    os.makedirs(TEMP_DIR, exist_ok=True)
    for f in files:
        path = os.path.join(TEMP_DIR, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
    return len(files)


@st.cache_resource(show_spinner="Building vector index…")
def build_index(_uploaded_files, _api_key, _model):
    if not _uploaded_files:
        return None

    cnt = _save_uploads(_uploaded_files)

    reader = SimpleDirectoryReader(TEMP_DIR)
    docs = reader.load_data()

    get_llm_and_embedder(_api_key, _model)

    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    return index


# 5. Refine output and prompt for resume context
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


# 6. Suggested template questions for resumes
EXAMPLE_QUERIES = [
    "Summarize key skills and experience of the candidate.",
    "Highlight education and certifications.",
    "Compare all uploaded resumes and rank candidates.",
    "Identify potential gaps or weaknesses in the resume."
]

# 7. Helper function to process any query (template or input)


def handle_query(query_text, show_trace_flag):
    """Process a query: add to chat, run index query, and store assistant response"""
    st.session_state.messages.append({"role": "user", "content": query_text})

    with st.chat_message("user"):
        st.markdown(query_text)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing resumes…"):
            engine = st.session_state.index.as_query_engine(
                similarity_top_k=5,
                response_mode="tree_summarize",
            )
            resp = engine.query(RESUME_PROMPT.format(query=query_text))
            answer = refine_output(resp.response)
            st.markdown(answer)

            # Show trace if enabled
            if show_trace_flag:
                st.markdown("---")
                st.markdown("### 🧠 Reasoning Trace")
                if hasattr(resp, "source_nodes") and resp.source_nodes:
                    for i, node in enumerate(resp.source_nodes, 1):
                        with st.expander(f"Source {i} (Score: {node.score:.3f})"):
                            st.markdown(
                                f"**File:** {node.node.metadata.get('file_name', 'Unknown')}")
                            st.markdown(f"**Text:**")
                            st.text(
                                node.node.text[:500] + "..." if len(node.node.text) > 500 else node.node.text)
                else:
                    st.info("No source nodes available.")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer})


# 8. Main UI
if uploaded_files:
    # Build index with proper cache invalidation
    index = build_index(uploaded_files, api_key, model_choice)
    if index:
        st.session_state.index = index
        st.success(
            f"✅ Index ready with {len(uploaded_files)} resume(s) – start querying!")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Template questions
        st.subheader("💡 Suggested Resume Queries")
        cols = st.columns(len(EXAMPLE_QUERIES))
        for i, q in enumerate(EXAMPLE_QUERIES):
            if cols[i].button(q, key=f"btn_{i}"):
                handle_query(q, show_trace)

        # Query input
        if prompt := st.chat_input("Ask about the resumes…"):
            handle_query(prompt, show_trace)

        # Action buttons
        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🧹 Clear Chat"):
                st.session_state.messages = []
                st.rerun()

        with col2:
            if st.session_state.messages:
                chat_text = "\n\n".join(
                    [f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages]
                )
                st.download_button(
                    "💾 Download Chat",
                    data=chat_text,
                    file_name="resume_chat_history.txt",
                    mime="text/plain",
                )

        with col3:
            if st.button("🗑️ Clear Everything"):
                if os.path.isdir(TEMP_DIR):
                    shutil.rmtree(TEMP_DIR)
                st.cache_resource.clear()
                st.session_state.messages = []
                if "index" in st.session_state:
                    del st.session_state.index
                st.rerun()

else:
    st.info("📤 Upload one or more resumes to start.")
    st.markdown(
        """
        ### Features
        - **Multi-step reasoning** with `tree_summarize` mode
        - **Gemini 2.0 Flash** for faster processing
        - **Latest Gemini embeddings** (`text-embedding-004`)
        - **Source tracing** - see which resume parts were used
        - **Template queries** for instant insights
        - **Chat history** with download option
        - **Clean interface** with optimized caching
        
        ### How to Use
        1. Add your Gemini API key (if not in `.env`)
        2. Upload resume files (PDF or TXT)
        3. Click suggested queries or ask your own
        4. Toggle "Show reasoning trace" to see sources
        """
    )
