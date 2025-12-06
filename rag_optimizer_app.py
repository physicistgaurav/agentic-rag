from sentence_transformers import SentenceTransformer
import numpy as np
import os
import time
import json
import concurrent.futures
from typing import List, Tuple, Dict, Any
import requests

import streamlit as st
import pdfplumber
import nltk

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# Try FAISS, fallback to sklearn
USE_FAISS = True
try:
    import faiss
except Exception:
    USE_FAISS = False
    from sklearn.neighbors import NearestNeighbors

# UI: page config

st.set_page_config(
    page_title="RAG Pipeline Optimizer (Claude API)", layout="wide")
st.title("RAG Pipeline Optimizer — Claude API Edition")
st.caption(
    "Runs 4 RAG pipelines, uses Claude Sonnet 4 for generation & judging")

# API Key Input
with st.sidebar:
    st.header("API Configuration")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Get free API key at: https://console.anthropic.com/"
    )

    if api_key:
        st.success("API Key configured!")
        st.info("Free tier includes generous usage limits.")
    else:
        st.warning("Enter API key to use Claude")
        st.markdown("""
        **Get FREE API Key:**
        1. Go to [console.anthropic.com](https://console.anthropic.com/)
        2. Sign up (free)
        3. Create API key
        4. Paste it above
        
        """)

    st.write("---")
    st.header("Settings")
    max_workers = st.slider("Parallel pipelines",
                            min_value=1, max_value=4, value=2)
    top_k = st.slider("Retriever top_k", min_value=1, max_value=10, value=5)

    st.write("---")
    st.markdown("**Tech Stack**")
    st.markdown("- Embedding: `all-MiniLM-L6-v2` (local)")
    st.markdown("- Generator: Claude Sonnet 4 (API)")
    st.markdown("- Judge: Claude Sonnet 4 (API)")

# Claude API Helper - call Anthropic's Claude API with a prompt


def call_claude_api(prompt: str, api_key: str, max_tokens: int = 1024) -> str:
    """Call Claude API with error handling"""
    if not api_key:
        return "ERROR: No API key provided"

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            return data['content'][0]['text']
        else:
            error_msg = response.json().get('error', {}).get('message', 'Unknown error')
            return f"API Error ({response.status_code}): {error_msg}"

    except requests.exceptions.Timeout:
        return "ERROR: Request timed out"
    except Exception as e:
        return f"ERROR: {str(e)}"

# Model Loading
# Load local sentence embedding model using Streamlit's caching.


@st.cache_resource
def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Extract text from PDF or plain text files.
def extract_text_from_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        with pdfplumber.open(path) as pdf:
            pages = []
            for p in pdf.pages:
                text = p.extract_text()
                if text:
                    pages.append(text)
        return "\n".join(pages)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

# Chunking
# split text into chunks with sentence boundaries and overlap


def simple_sentence_chunker(text: str, max_chars: int = 1000, overlap_chars: int = 200):
    """Chunk text by sentences with overlap"""
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Add overlap
    if overlap_chars > 0 and len(chunks) > 1:
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                prev_tail = chunks[i-1][-overlap_chars:]
                overlapped.append(prev_tail + " " + chunk)
        chunks = overlapped

    return chunks

# Vector Index - to build and search vector indices using FAISS or sklearn


def build_index_with_faiss(embs: np.ndarray):
    d = embs.shape[1]
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    return index


def search_faiss(index, query_emb: np.ndarray, k: int):
    faiss.normalize_L2(query_emb)
    D, I = index.search(query_emb, k)
    return D, I


def build_index_with_sklearn(embs: np.ndarray):
    nbrs = NearestNeighbors(
        n_neighbors=min(embs.shape[0], 10),
        algorithm="auto",
        metric="cosine"
    ).fit(embs)
    return nbrs


def search_sklearn(nbrs, query_emb: np.ndarray, k: int):
    D, I = nbrs.kneighbors(query_emb, n_neighbors=k, return_distance=True)
    sim = 1 - D
    return sim, I

# RAG Pipeline with Claude
# - build knowledge system (chunking + embedding + index)
# - query with retrieval + generation(claude)


class ClaudeRAGPipeline:
    def __init__(self, name: str, chunk_max_chars: int, chunk_overlap: int, top_k: int, api_key: str):
        self.name = name
        self.chunk_max_chars = chunk_max_chars
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k  # Number of top documents to retrieve
        self.api_key = api_key
        self.embedder = get_embedder()

        self._index = None
        self._metadatas = []
        self._embeddings = None
        self._index_info = {}

    def build(self, docs: List[Tuple[str, str]]):
        chunks = []
        metas = []

        for fname, text in docs:
            pieces = simple_sentence_chunker(
                text,
                max_chars=self.chunk_max_chars,
                overlap_chars=self.chunk_overlap
            )
            for p in pieces:
                chunks.append(p)
                metas.append({"file": fname, "text": p})

        # Compute embeddings
        t0 = time.time()
        embs = self.embedder.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        embed_time = time.time() - t0

        self._embeddings = embs.astype("float32")
        self._metadatas = metas

        # Build index
        if USE_FAISS:
            self._index = build_index_with_faiss(self._embeddings)
            self._index_info = {
                "type": "faiss",
                "n_chunks": len(chunks),
                "embed_time": embed_time,
                "chunk_config": f"{self.chunk_max_chars} chars, {self.chunk_overlap} overlap"
            }
        else:
            self._index = build_index_with_sklearn(self._embeddings)
            self._index_info = {
                "type": "sklearn",
                "n_chunks": len(chunks),
                "embed_time": embed_time,
                "chunk_config": f"{self.chunk_max_chars} chars, {self.chunk_overlap} overlap"
            }

        return self._index_info

    def query(self, question: str) -> Dict[str, Any]:
        # Retrieve relevant chunks
        q_emb = self.embedder.encode(
            [question], convert_to_numpy=True).astype("float32")
        t0 = time.time()

        if USE_FAISS:
            D, I = search_faiss(self._index, q_emb, self.top_k)
            hits = [{"score": float(score), "meta": self._metadatas[int(idx)]}
                    for score, idx in zip(D[0], I[0])]
        else:
            D, I = search_sklearn(self._index, q_emb, self.top_k)
            hits = [{"score": float(score), "meta": self._metadatas[int(idx)]}
                    for score, idx in zip(D[0], I[0])]

        search_time = time.time() - t0

        # Generate answer with Claude
        contexts = "\n\n---\n\n".join([
            f"Source: {h['meta']['file']}\n{h['meta']['text']}"
            for h in hits
        ])

        prompt = f"""You are a helpful assistant that answers questions based on provided context.

Question: {question}

Context from documents:
{contexts}

Instructions:
- Answer the question directly and concisely
- Cite specific source files when making claims
- If the context doesn't contain enough information, say so
- Be specific with numbers, dates, or facts when available

Answer:"""

        t0 = time.time()
        generated = call_claude_api(prompt, self.api_key, max_tokens=800)
        gen_time = time.time() - t0

        return {
            "answer": generated,
            "hits": hits,
            "search_time": search_time,
            "gen_time": gen_time,
            "index_info": self._index_info,
        }

# Run 4 Pipelines


def run_4_pipelines(docs, question, api_key, top_k=5, max_workers=2):
    configs = [
        ("A_tiny_300", 300, 50),
        ("B_small_600", 600, 100),
        ("C_medium_1200", 1200, 200),
        ("D_large_2400", 2400, 400),
    ]

    pipelines = [
        ClaudeRAGPipeline(
            name=c[0],
            chunk_max_chars=c[1],
            chunk_overlap=c[2],
            top_k=top_k,
            api_key=api_key
        )
        for c in configs
    ]

    results = {}

    # Build indices in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
        futs = {exe.submit(p.build, docs): p for p in pipelines}
        for fut in concurrent.futures.as_completed(futs):
            p = futs[fut]
            try:
                idx_info = fut.result()
                results[p.name] = {"index_info": idx_info}
            except Exception as e:
                results[p.name] = {"index_info": {"error": str(e)}}

    # Query in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
        futs = {exe.submit(p.query, question): p for p in pipelines}
        for fut in concurrent.futures.as_completed(futs):
            p = futs[fut]
            try:
                out = fut.result()
                results[p.name].update(out)
            except Exception as e:
                results[p.name].update({"error": str(e)})

    return results

# Claude as Judge ( prompt engineering )


def evaluate_with_claude_judge(question: str, results: Dict[str, Any], api_key: str) -> Dict:
    """Use Claude to evaluate pipeline outputs"""

    prompt = f"""You are an expert evaluator of RAG (Retrieval-Augmented Generation) systems. 

Your task is to score 4 different RAG pipeline configurations based on their performance.

Question asked: "{question}"

Pipeline Results:
"""

    for name, out in results.items():
        prompt += f"\n{'='*60}\n"
        prompt += f"Pipeline: {name}\n"
        prompt += f"Configuration: {out.get('index_info', {}).get('chunk_config', 'N/A')}\n"
        prompt += f"Number of chunks: {out.get('index_info', {}).get('n_chunks', 'N/A')}\n"
        prompt += f"Search time: {out.get('search_time', 0):.3f}s\n"
        prompt += f"Generation time: {out.get('gen_time', 0):.3f}s\n"

        if out.get('error'):
            prompt += f"ERROR: {out['error']}\n"
        else:
            prompt += f"\nAnswer:\n{out.get('answer', 'N/A')}\n"
            prompt += f"\nTop retrieved sources:\n"
            for i, hit in enumerate(out.get('hits', [])[:3], 1):
                prompt += f"  {i}. {hit['meta']['file']} (similarity: {hit['score']:.3f})\n"

    prompt += f"""

{'='*60}

Evaluation Criteria:

1. **Accuracy** (0-10): How factually correct and complete is the answer?
   - Does it answer the question directly?
   - Are facts/numbers accurate based on the sources?
   - Is it specific rather than vague?

2. **Relevance** (0-10): How well does the answer match the question?
   - Does it address what was asked?
   - Are the retrieved sources relevant?
   - Does it cite appropriate sources?

3. **Cost** (0-10): How efficient is this pipeline?
   - Faster is better (search + gen time)
   - Fewer chunks is more efficient
   - Consider speed vs quality trade-off

Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
{{
  "A_tiny_300": {{
    "accuracy": 8,
    "relevance": 7,
    "cost": 9,
    "rationale": "Brief explanation of scores"
  }},
  "B_small_600": {{ ... }},
  "C_medium_1200": {{ ... }},
  "D_large_2400": {{ ... }}
}}

Be critical and differentiate between pipelines. Not all should get high scores."""

    response = call_claude_api(prompt, api_key, max_tokens=2000)

    # Parse JSON from response
    try:
        # Find JSON in response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            parsed = json.loads(json_str)
            return parsed
        else:
            return {"error": "No JSON found in response", "raw": response}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {str(e)}", "raw": response}

# Streamlit UI


if not api_key:
    st.warning(" Please enter your Anthropic API key in the sidebar to continue")
    st.stop()

st.header("1) Upload Documents")
uploaded = st.file_uploader(
    "Upload resumes, reports, or text files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded:
    # Extract text
    tmpdir = "uploaded_docs"
    os.makedirs(tmpdir, exist_ok=True)
    docs = []

    with st.spinner("Extracting text from documents..."):
        for f in uploaded:
            path = os.path.join(tmpdir, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            txt = extract_text_from_file(path)
            docs.append((f.name, txt))

    st.success(f"Loaded {len(docs)} file(s)")

    # Show document preview
    with st.expander("Document Preview"):
        for fname, txt in docs:
            st.markdown(f"**{fname}** ({len(txt)} chars)")
            st.text(txt[:300] + "..." if len(txt) > 300 else txt)

    st.markdown("### 2)Enter Your Query")
    question = st.text_input(
        "What do you want to know?",
        value="Briefly summarize the file in 3 sentences."
    )

    if st.button("Run RAG Optimizer with Claude", type="primary"):
        st.info("Running 4 RAG configurations with Claude Sonnet 4...")

        start_time = time.time()

        # Run pipelines
        with st.spinner("Building indices and generating answers with Claude..."):
            results = run_4_pipelines(
                docs, question, api_key, top_k=top_k, max_workers=max_workers)

        total_time = time.time() - start_time
        st.success(f"Completed in {total_time:.1f}s")

        # Display results
        st.subheader("Pipeline Results")

        cols = st.columns(2)
        for idx, (name, out) in enumerate(results.items()):
            with cols[idx % 2]:
                with st.expander(f"**{name}**", expanded=True):
                    if out.get("error"):
                        st.error(f" {out['error']}")
                        continue

                    # Config info
                    config = out.get('index_info', {}).get(
                        'chunk_config', 'N/A')
                    chunks = out.get('index_info', {}).get('n_chunks', 'N/A')
                    st.caption(f"⚙️ {config} | {chunks} chunks")
                    st.caption(
                        f"{out.get('search_time', 0):.3f}s search + {out.get('gen_time', 0):.3f}s gen")

                    # Answer
                    st.markdown("**Answer:**")
                    answer = out.get("answer", "No answer generated")
                    if answer.startswith("ERROR"):
                        st.error(answer)
                    else:
                        st.info(answer)

                    # Top sources
                    st.markdown("**Top Sources:**")
                    for i, hit in enumerate(out.get("hits", [])[:3], 1):
                        st.write(
                            f"{i}. {hit['meta']['file']} (similarity: {hit['score']:.3f})")

        # Claude Judge Evaluation
        st.subheader(" Claude AI Judge Evaluation")

        with st.spinner(" Claude is evaluating all pipelines..."):
            judge_scores = evaluate_with_claude_judge(
                question, results, api_key)

        if "error" in judge_scores:
            st.error(f"Judge error: {judge_scores['error']}")
            with st.expander("Raw response"):
                st.code(judge_scores.get('raw', ''))
        else:
            # Create comparison table
            import pandas as pd

            score_data = []
            for pipeline, scores in judge_scores.items():
                if isinstance(scores, dict) and 'accuracy' in scores:
                    score_data.append({
                        "Pipeline": pipeline,
                        "Accuracy": scores.get('accuracy', 0),
                        "Relevance": scores.get('relevance', 0),
                        "Cost": scores.get('cost', 0),
                        "Total": scores.get('accuracy', 0) + scores.get('relevance', 0) + scores.get('cost', 0),
                        "Rationale": scores.get('rationale', '')
                    })

            if score_data:
                df = pd.DataFrame(score_data).sort_values(
                    'Total', ascending=False)

                # Display table
                st.dataframe(df, use_container_width=True)

                # Winner
                winner = df.iloc[0]
                st.success(
                    f"**Winner:** {winner['Pipeline']} with total score {winner['Total']}/30")
                st.write(f"**Claude's Reasoning:** {winner['Rationale']}")

                # Insights
                st.markdown("###  Detailed Insights")
                col1, col2, col3 = st.columns(3)

                with col1:
                    best_acc = df.nlargest(1, 'Accuracy').iloc[0]
                    st.metric("Best Accuracy",
                              best_acc['Pipeline'], f"{best_acc['Accuracy']}/10")

                with col2:
                    best_rel = df.nlargest(1, 'Relevance').iloc[0]
                    st.metric(" Best Relevance",
                              best_rel['Pipeline'], f"{best_rel['Relevance']}/10")

                with col3:
                    best_cost = df.nlargest(1, 'Cost').iloc[0]
                    st.metric(" Most Efficient",
                              best_cost['Pipeline'], f"{best_cost['Cost']}/10")

            # Full JSON view
            with st.expander("Full Judge Scores (JSON)"):
                st.json(judge_scores)

        # Download
        all_results = {
            "question": question,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
            "claude_evaluation": judge_scores,
            "winner": winner['Pipeline'] if score_data else None
        }

        st.download_button(
            "Download Results (JSON)",
            data=json.dumps(all_results, indent=2),
            file_name=f"rag_results_{int(time.time())}.json",
            mime="application/json"
        )

else:
    st.info(" Upload documents to get started!")

    st.markdown("""
    ###  How It Works
    
    This tool finds the **optimal RAG configuration** for your documents using Claude AI:
    
    1. **Tests 4 chunking strategies** (tiny/small/medium/large)
    2. **Uses Claude Sonnet 4** for high-quality answer generation
    3. **Claude judges each pipeline** on accuracy, relevance, and cost
    4. **Recommends the best** configuration for your use case
    
    ### Tech Stack
    
    - **Embedding**: `all-MiniLM-L6-v2` (local, free)
    - **Generator**: Claude Sonnet 4 (API, free tier)
    - **Vector Search**: FAISS or sklearn (local, free)
    - **Judge**: Claude Sonnet 4 (API, free tier)
    
    
    ###  Example Questions to Try
    
    - "List my GPA and my semester"
    - "What are the candidate's top 3 skills?"
    - "Summarize work experience"
    - "What programming languages does the candidate know?"
    """)
