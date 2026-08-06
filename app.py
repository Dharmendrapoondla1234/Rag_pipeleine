from pathlib import Path

import faiss
import numpy as np
import streamlit as st

from utils.chucking import chunk_text, read_text_file
from utils.completiton import generate_answer
from utils.embedding import embed_query, generate_embeddings, get_client
from utils.prompt import build_prompt
from utils.retrieval import (
    load_chunk_data,
    load_faiss_index,
    save_chunk_data,
    save_faiss_index,
    retrieve_context,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAISS_DIR = BASE_DIR / "faiss_store"
DATA_FILE = DATA_DIR / "founder.txt"
INDEX_FILE = FAISS_DIR / "index.faiss"
CHUNK_DATA_FILE = FAISS_DIR / "chunk_data.pkl"


def build_index(file_path, index_path=None, chunk_data_path=None, chunk_size=500, overlap=100):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    text = read_text_file(file_path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    if not chunks:
        raise ValueError("No text found in the source file.")

    client = get_client()
    embedded_chunks = generate_embeddings(chunks, client)

    if not embedded_chunks:
        raise ValueError("Embedding generation returned no chunks.")

    dimension = len(embedded_chunks[0]["embedding"])
    index = faiss.IndexFlatL2(dimension)
    vectors = np.array([chunk["embedding"] for chunk in embedded_chunks], dtype=np.float32)
    index.add(vectors)

    index_path = Path(index_path or INDEX_FILE)
    chunk_data_path = Path(chunk_data_path or CHUNK_DATA_FILE)

    save_faiss_index(index, index_path)

    chunk_data = [
        {"chunk_id": chunk["chunk_id"], "text": chunk["text"]}
        for chunk in embedded_chunks
    ]
    save_chunk_data(chunk_data, chunk_data_path)

    return index_path, chunk_data_path


def answer_query(query, index_path=None, chunk_data_path=None):
    client = get_client()
    index_path = Path(index_path or INDEX_FILE)
    chunk_data_path = Path(chunk_data_path or CHUNK_DATA_FILE)

    if not index_path.exists() or not chunk_data_path.exists():
        raise FileNotFoundError("Index files are missing. Build the index first.")

    index = load_faiss_index(index_path)
    chunk_data = load_chunk_data(chunk_data_path)
    query_embedding = embed_query(query, client)

    retrieved_text, matched_chunks, _ = retrieve_context(index, chunk_data, query_embedding, k=3)
    prompt = build_prompt(query, retrieved_text)
    answer = generate_answer(prompt, client)

    return {
        "answer": answer,
        "retrieved_text": retrieved_text,
        "matched_chunks": matched_chunks,
    }


def ensure_data_file():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Source data file not found: {DATA_FILE}")


def ensure_faiss_store():
    FAISS_DIR.mkdir(exist_ok=True)


st.set_page_config(page_title="RAG App", page_icon="🤖")
st.title("RAG Application")
st.write("Ask questions about the uploaded or local document content.")

ensure_data_file()
ensure_faiss_store()

if "last_result" not in st.session_state:
    st.session_state.last_result = None

index_exists = INDEX_FILE.exists() and CHUNK_DATA_FILE.exists()
if index_exists:
    st.success("Existing index found. Ready to ask questions.")
else:
    st.info("No saved index found. The index will be built automatically on first question.")

query = st.text_input("Enter your question")

if st.button("Ask"):
    if not query:
        st.warning("Enter a question before asking.")
    else:
        with st.spinner("Preparing answer..."):
            try:
                if not index_exists:
                    build_index(DATA_FILE)
                    st.success("Index built successfully.")
                st.session_state.last_result = answer_query(query)
            except Exception as exc:
                st.error(f"Error: {exc}")

if st.session_state.last_result:
    st.subheader("Answer")
    st.write(st.session_state.last_result["answer"])

    if st.checkbox("Show retrieved context"):
        st.subheader("Retrieved Context")
        st.text_area("Context", st.session_state.last_result["retrieved_text"], height=220)

with st.expander("Advanced options"):
    if st.button("Rebuild index"):
        with st.spinner("Rebuilding index..."):
            try:
                build_index(DATA_FILE)
                st.success("Index rebuilt successfully")
                st.experimental_rerun()
            except Exception as exc:
                st.error(f"Error: {exc}")
