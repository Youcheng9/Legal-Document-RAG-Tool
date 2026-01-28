# ui/app.py
import os
import requests
import streamlit as st
from pathlib import Path

# Config: point to your running FastAPI server
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Legal Document RAG", page_icon="⚖️", layout="wide")

st.title("⚖️ Legal Document RAG")
st.markdown("Upload a PDF, ingest it, and ask questions about its contents.")

# Sidebar controls
st.sidebar.header("Settings")
top_k = st.sidebar.number_input("Top K chunks to retrieve", min_value=1, max_value=30, value=10)
model_name = st.sidebar.text_input("Model (for info only)", value=os.getenv("OLLAMA_MODEL", "configured on server"), disabled=True)

# Session state
if "file_id" not in st.session_state:
    st.session_state.file_id = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "ingest_status" not in st.session_state:
    st.session_state.ingest_status = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

# Upload box
st.header("1) Upload PDF")
uploaded = st.file_uploader("Select a PDF file", type=["pdf"], accept_multiple_files=False)

col1, col2 = st.columns([1, 3])
with col1:
    if uploaded:
        st.write("Filename:")
        st.write(uploaded.name)
        if st.button("Upload"):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                resp = requests.post(f"{API_URL}/upload", files=files, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                st.session_state.file_id = data["file_id"]
                st.session_state.uploaded_filename = data["filename"]
                st.success(f"Uploaded {uploaded.name} → file_id: {st.session_state.file_id}")
                # reset ingest state
                st.session_state.ingest_status = None
                st.session_state.last_answer = None
            except Exception as e:
                st.error(f"Upload failed: {e}")

with col2:
    st.info("Upload a PDF. After upload, click 'Ingest Document' to chunk + index it into the vector store.")

# Ingest
st.header("2) Ingest document")
if st.session_state.file_id:
    st.write("File ID:", st.session_state.file_id)
    if st.button("Ingest Document"):
        try:
            with st.spinner("Ingesting (chunking, embedding, indexing) — this can take a few seconds..."):
                r = requests.post(f"{API_URL}/ingest/{st.session_state.file_id}", timeout=120)
                r.raise_for_status()
                data = r.json()
                st.session_state.ingest_status = data.get("status", "ingested")
                st.success(f"Ingest finished: {st.session_state.ingest_status} — chunks: {data.get('chunks')}")
                # store entities optionally
                if data.get("entities"):
                    st.write("Extracted Entities (sample):")
                    st.json({k: data["entities"].get(k)[:10] for k in data["entities"]})
        except Exception as e:
            st.error(f"Ingest failed: {e}")
else:
    st.info("Upload a PDF first to ingest it.")

st.write("---")

# Querying UI
st.header("3) Ask a question")
query = st.text_input("Enter your question here", value="", placeholder="e.g. What are the parties and the effective date?")
ask_button = st.button("Ask")

if ask_button:
    if not st.session_state.file_id:
        st.error("No file uploaded/ingested — upload and ingest first.")
    elif query.strip() == "":
        st.error("Please enter a question.")
    else:
        try:
            with st.spinner("Running retrieval and LLM..."):
                params = {"question": query, "file_id": st.session_state.file_id, "top_k": top_k}
                r = requests.post(f"{API_URL}/query", params=params, timeout=120)
                r.raise_for_status()
                result = r.json()
                st.session_state.last_answer = result
        except Exception as e:
            st.error(f"Query failed: {e}")
            try:
                st.json(r.json())
            except Exception:
                pass

# Display results
if st.session_state.last_answer:
    res = st.session_state.last_answer
    st.subheader("Answer")
    st.write(res.get("answer", ""))

    st.subheader("Retrieved sources")
    sources = res.get("sources", []) or []
    st.write(f"Retrieved: {len(sources)}")
    for idx, s in enumerate(sources):
        with st.expander(f"Source {idx+1} — {s.get('source')} page:{s.get('page')} (score: {s.get('score')})"):
            st.write(s.get("text", ""))
            if st.button(f"Copy excerpt #{idx+1} to clipboard", key=f"copy_{idx}"):
                # streams can't access clipboard in server; provide as text to copy
                st.write("Excerpt (select and copy):")
                st.code(s.get("text", ""))

    st.write("---")
    st.download_button("Download Answer (txt)", res.get("answer", ""), file_name="answer.txt")

st.write("---")
st.caption("Streamlit UI talking to FastAPI. Make sure the API is running at RAG_API_URL (env) or default http://127.0.0.1:8000")
