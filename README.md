# Arco Papers API

AI-powered backend for Arco Papers, a paper manufacturer 
in Islamabad, Pakistan.

## Tech Stack
- FastAPI
- LangChain + OpenAI GPT-4o-mini
- RAG (Retrieval Augmented Generation)
- ChromaDB vector database
- Python 3.12

## Features
- AI sales assistant with product knowledge
- Automatic pricing tier calculation
- Multi-turn conversation memory
- Tool-calling agent for complex queries

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Architecture
Flutter frontend → FastAPI backend → LangChain agent 
→ ChromaDB (product knowledge) + OpenAI GPT-4o-mini
