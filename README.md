# Differentbyte Machine Test

A simple RAG (Retrieval-Augmented Generation) pipeline built for the Differentbyte Machine Test Interview. It takes a Resume(PDF), chunks and embeds it into a local vector store, and answers questions using relevant context.

## Tech Stack

- **Framework**: LangChain
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **Vector Store**: ChromaDB (stored locally in `chroma_db/`)
- **LLM**: Groq (`ChatGroq`)

## Quick Setup

### 1. Create & activate virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add API Key
Create a `.env` file with your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key
```

### 4. Run
```bash
python rag.py
```

## How It Works

1. **Load**: Loads the target document (`.pdf` or `.txt`).
2. **Split**: Chunks text with overlap (`chunk_size=1000`, `chunk_overlap=200`).
3. **Store**: Generates vector embeddings and stores them in ChromaDB.
4. **Retrieve & Answer**: Fetches top matching chunks and prompts the LLM to answer strictly from the retrieved context.
