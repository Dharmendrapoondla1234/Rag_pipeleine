import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def get_client(api_key=None):
    key = (
        api_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("API_KEY")
    )
    if not key:
        raise ValueError(
            "Set GEMINI_API_KEY, GOOGLE_API_KEY, or API_KEY in your .env file or environment before running the app."
        )

    return genai.Client(api_key=key)


def generate_embeddings(chunks, client, model="models/gemini-embedding-001"):
    embeddings = []

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        response = client.models.embed_content(model=model, contents=chunk)
        embedding = response.embeddings[0].values

        embeddings.append({
            "chunk_id": i,
            "text": chunk,
            "embedding": np.array(embedding, dtype=np.float32)
        })

    return embeddings


def embed_query(query, client, model="models/gemini-embedding-001"):
    response = client.models.embed_content(model=model, contents=query)
    embedding = response.embeddings[0].values
    return np.array(embedding, dtype=np.float32).reshape(1, -1)
