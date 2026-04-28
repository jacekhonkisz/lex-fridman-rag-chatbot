# Lex Fridman Podcast RAG Chatbot

This Streamlit app implements a Retrieval-Augmented Generation chatbot for the Lex Fridman Podcast transcript dataset.

## Architecture

- Podcast transcripts are loaded from Kaggle.
- Text is cleaned and split into overlapping chunks.
- Chunks are embedded with sentence-transformers/all-MiniLM-L6-v2.
- Embeddings and metadata are stored in Chroma.
- User questions are embedded and matched against the Chroma vector database.
- Retrieved transcript excerpts are inserted into a RAG prompt.
- A local Ollama model generates the answer.
- Streamlit provides the chat interface and displays retrieved transcript sources.

## Model choice

The app includes a model selector:

- mistral: default and most assessment-aligned model, but slower in Colab or low-resource environments.
- llama3.2:1b: faster testing option while still using local Ollama.
- tinyllama: fallback for very weak environments.

The RAG architecture remains the same regardless of selected model.

## Local run

1. ollama pull mistral
2. ollama serve
3. streamlit run "Reassessment Part2 Jacek Honkisz.py"

For faster Colab-style testing:

1. ollama pull llama3.2:1b
2. OLLAMA_MODEL=llama3.2:1b streamlit run "Reassessment Part2 Jacek Honkisz.py"

## Note on Streamlit Cloud

Streamlit Cloud can host the interface, but full local Ollama generation requires a reachable Ollama server.
Local execution gives the complete RAG + Ollama behaviour.