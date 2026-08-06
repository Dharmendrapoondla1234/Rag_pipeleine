def build_prompt(query, retrieved_text):
    return f"""
Answer the question using only the context below.

Context:
{retrieved_text}

Question:
{query}

Answer:
"""