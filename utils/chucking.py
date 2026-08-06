from pathlib import Path


def read_text_file(file_path):
    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as file:
        return file.read()


def chunk_text(text, chunk_size=500, overlap=100):
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = max(0, end - overlap)

    return chunks