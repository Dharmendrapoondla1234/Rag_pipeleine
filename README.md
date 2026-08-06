# Euron RAG Application

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using:
- text chunking
- Gemini embeddings
- FAISS vector search
- prompt construction
- Gemini answer generation

## Folder structure

- app.py: main entry point
- data/founder.txt: source text document
- faiss_store/: saved FAISS index and chunk metadata
- utils/: modular helpers for chunking, embeddings, retrieval, prompting, and completion

## Run

1. Install dependencies:
   ```bash
   pip install -r faiss_store/requirements.txt
   ```

2. Set your API key:
   ```bash
   set GEMINI_API_KEY=your_key_here
   ```

3. Build the index:
   ```bash
   python app.py --build --file data/founder.txt
   ```

4. Ask a question:
   ```bash
   python app.py --query "What is this document about?"
   ```
