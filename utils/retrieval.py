import pickle
from pathlib import Path

import faiss


def save_faiss_index(index, index_path):
    index_path = str(index_path)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)


def load_faiss_index(index_path):
    index_path = str(index_path)
    if not Path(index_path).exists():
        raise FileNotFoundError(f"FAISS index not found at: {index_path}")
    return faiss.read_index(index_path)


def save_chunk_data(chunk_data, chunk_data_path):
    Path(chunk_data_path).parent.mkdir(parents=True, exist_ok=True)
    with open(chunk_data_path, "wb") as file:
        pickle.dump(chunk_data, file)


def load_chunk_data(chunk_data_path):
    with open(chunk_data_path, "rb") as file:
        return pickle.load(file)


def retrieve_context(index, chunk_data, query_embedding, k=3):
    distances, indices = index.search(query_embedding, k)

    retrieved_text = ""
    matched_chunks = []

    for idx in indices[0]:
        if 0 <= int(idx) < len(chunk_data):
            chunk = chunk_data[int(idx)]
            matched_chunks.append(chunk)
            retrieved_text += chunk["text"] + "\n\n"

    return retrieved_text.strip(), matched_chunks, distances[0]
