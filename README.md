# Lex Fridman Podcast RAG Chatbot

Published app:
https://lex-fridman-rag-chatbot-85qnuxvdwcd43ginzbjs96.streamlit.app

This project implements a Retrieval-Augmented Generation chatbot for the Lex Fridman Podcast transcript dataset.

## Architecture

- Podcast transcripts are loaded from Kaggle.
- Text is cleaned and split into overlapping chunks.
- Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2`.
- Embeddings and metadata are stored in Chroma.
- User questions are embedded and matched against the Chroma vector database.
- Retrieved transcript excerpts are inserted into a RAG prompt.
- A local Ollama model generates the answer.
- Streamlit provides the chat interface and displays retrieved transcript sources.

## Model choice

The app includes two Ollama model options:

- `mistral` — default assessment-aligned model. Better quality, but slower in Colab or low-resource environments.
- `llama3.2:1b` — faster model for live testing and Streamlit Cloud/Colab demonstrations.

Both models use the same Chroma retrieval pipeline. The selector changes only the generation model.

## Streamlit Cloud and Ollama

The Streamlit Cloud app can generate answers when `OLLAMA_URL` points to a reachable Ollama endpoint.

For this submission, Ollama can be run locally or in Colab and exposed through a temporary Cloudflare tunnel. For local execution:

1. `ollama pull mistral`
2. `ollama serve`
3. `streamlit run "Reassessment Part2 Jacek Honkisz.py"`

For faster testing:

1. `ollama pull llama3.2:1b`
2. Set `OLLAMA_MODEL=llama3.2:1b`
3. Run the Streamlit app