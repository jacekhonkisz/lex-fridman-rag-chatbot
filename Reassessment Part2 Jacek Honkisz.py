# =========================================================
# Assessment/runtime conclusion:
#
# This app implements a complete Retrieval-Augmented Generation pipeline:
# transcript loading, text chunking, sentence-transformer embeddings,
# Chroma vector storage, semantic retrieval, RAG prompt construction,
# local Ollama generation, Streamlit chat UI, and retrieved source display.
#
# Two Ollama models are available:
# - mistral: the default assessment-aligned model, stronger but slower.
# - llama3.2:1b: faster model used for live testing and constrained environments.
#
# Both models use the same RAG pipeline. The model selector only changes the
# local Ollama generation model, not the retrieval architecture.
#
# Streamlit Cloud does not run Ollama by default. To make the public app generate
# answers, OLLAMA_URL must point to a reachable Ollama endpoint. For testing,
# Ollama was run in Colab and exposed through a temporary Cloudflare tunnel.
# For local assessment execution, run:
#   ollama pull mistral
#   ollama serve
#   streamlit run "Reassessment Part2 Jacek Honkisz.py"
# =========================================================


import os
import re
import requests
import pandas as pd
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import kagglehub

# Runtime note:
# The default model is Mistral because it is directly aligned with the reassessment brief.
# In low-resource environments such as Google Colab CPU, Mistral may be slow.
# The app allows selecting smaller Ollama models such as llama3.2:1b or tinyllama
# while preserving the same RAG architecture: Chroma retrieval + local Ollama generation.

# =========================================================
# Reassessment Part 2 — Lex Fridman Podcast RAG Chatbot
# Author: Jacek Honkisz
#
# Streamlit app link:
# https://lex-fridman-rag-chatbot-85qnuxvdwcd43ginzbjs96.streamlit.app
#
# Local run instructions:
# 1. Install Ollama
# 2. Run: ollama pull mistral
# 3. Run: ollama serve
# 4. Run: streamlit run "Reassessment Part2 Jacek Honkisz.py"
# =========================================================

st.set_page_config(
    page_title="Lex Fridman Podcast RAG Chatbot",
    page_icon="🎙️",
    layout="wide"
)

FULL_NAME = "Jacek Honkisz"
STREAMLIT_APP_LINK = "https://lex-fridman-rag-chatbot-85qnuxvdwcd43ginzbjs96.streamlit.app"

COLLECTION_NAME = "lex_fridman_podcast"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_EPISODES = 20
CHUNK_SIZE_WORDS = 700
CHUNK_OVERLAP_WORDS = 100
TOP_K = 3

try:
    OLLAMA_URL = st.secrets.get("OLLAMA_URL", os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"))
except Exception:
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
try:
    DEFAULT_OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "mistral"))
except Exception:
    DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

AVAILABLE_OLLAMA_MODELS = [
    "mistral",
    "llama3.2:1b"
]

MODEL_NOTES = {
    "mistral": "Default assessment-aligned Ollama model. Better quality, but slower in Colab or low-resource environments.",
    "llama3.2:1b": "Faster Ollama model for live testing and Streamlit Cloud/Colab demonstrations."
}

PROJECT_DIR = Path.cwd()
CHROMA_DIR = PROJECT_DIR / "chroma_db"


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, chunk_size=700, overlap=100):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if len(chunk.split()) > 50:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_or_create_collection():
    embedding_model = load_embedding_model()
    CHROMA_DIR.mkdir(exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() > 0:
            return collection
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    with st.spinner("Downloading and indexing podcast transcripts. This may take a few minutes on first run..."):
        dataset_path = kagglehub.dataset_download("rajneesh231/lex-fridman-podcast-transcript")
        dataset_path = Path(dataset_path)

        candidate_files = []
        for ext in ["*.csv", "*.json", "*.jsonl", "*.txt"]:
            candidate_files.extend(list(dataset_path.rglob(ext)))

        if not candidate_files:
            raise FileNotFoundError("No transcript files found in the Kaggle dataset.")

        file_path = max(candidate_files, key=lambda f: f.stat().st_size)
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(file_path)
        elif suffix == ".json":
            df = pd.read_json(file_path)
        elif suffix == ".jsonl":
            df = pd.read_json(file_path, lines=True)
        elif suffix == ".txt":
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            df = pd.DataFrame([{
                "title": file_path.stem,
                "guest": "Unknown guest",
                "transcript": text
            }])
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        object_cols = df.select_dtypes(include=["object"]).columns.tolist()

        length_stats = []
        for col in object_cols:
            avg_len = df[col].astype(str).str.len().mean()
            length_stats.append((col, avg_len))

        length_stats = sorted(length_stats, key=lambda x: x[1], reverse=True)
        text_col = length_stats[0][0]

        possible_title_cols = ["title", "episode_title", "Title", "name", "episode_name", "video_title"]
        possible_guest_cols = ["guest", "Guest", "guests", "Guests", "guest_name"]
        possible_episode_cols = ["episode", "episode_id", "episode_number", "id", "Number"]

        title_col = next((c for c in possible_title_cols if c in df.columns), None)
        guest_col = next((c for c in possible_guest_cols if c in df.columns), None)
        episode_col = next((c for c in possible_episode_cols if c in df.columns), None)

        records = []

        for idx, row in df.iterrows():
            transcript = clean_text(row[text_col])

            if len(transcript.split()) < 100:
                continue

            title = str(row[title_col]) if title_col else f"Episode {idx}"
            guest = str(row[guest_col]) if guest_col else "Unknown guest"
            episode_id = str(row[episode_col]) if episode_col else str(idx)

            records.append({
                "episode_id": episode_id,
                "title": title,
                "guest": guest,
                "transcript": transcript
            })

        clean_df = pd.DataFrame(records)

        if MAX_EPISODES is not None:
            clean_df = clean_df.head(MAX_EPISODES)

        documents = []
        metadatas = []
        ids = []

        global_chunk_counter = 0

        for row_idx, row in clean_df.reset_index(drop=True).iterrows():
            chunks = chunk_text(
                row["transcript"],
                chunk_size=CHUNK_SIZE_WORDS,
                overlap=CHUNK_OVERLAP_WORDS
            )

            for chunk_idx, chunk in enumerate(chunks):
                doc_id = f"row{row_idx}_episode{row['episode_id']}_chunk{chunk_idx}_global{global_chunk_counter}"

                documents.append(chunk)
                metadatas.append({
                    "row_index": row_idx,
                    "episode_id": str(row["episode_id"]),
                    "title": str(row["title"]),
                    "guest": str(row["guest"]),
                    "chunk_id": chunk_idx,
                    "global_chunk_id": global_chunk_counter
                })
                ids.append(doc_id)

                global_chunk_counter += 1

        batch_size = 64

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]

            batch_embeddings = embedding_model.encode(batch_docs).tolist()

            collection.add(
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_metas,
                ids=batch_ids
            )

    return collection


embedding_model = load_embedding_model()
collection = load_or_create_collection()


def retrieve_context(query, top_k=5):
    query_embedding = embedding_model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved = []

    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved.append({
            "text": doc,
            "metadata": meta,
            "distance": distance
        })

    return retrieved


def build_prompt(query, retrieved_chunks):
    context_blocks = []

    for i, item in enumerate(retrieved_chunks, start=1):
        meta = item["metadata"]

        source_header = (
            f"Source {i}: "
            f"Title: {meta.get('title', 'Unknown title')} | "
            f"Guest: {meta.get('guest', 'Unknown guest')} | "
            f"Chunk: {meta.get('chunk_id', 'Unknown chunk')}"
        )

        context_blocks.append(source_header + "\n" + item["text"])

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a helpful chatbot answering questions about the Lex Fridman Podcast.

Use only the transcript context provided below.
Do not invent information.
If the context does not contain enough evidence, say that the available transcript context is insufficient.

When useful, mention the podcast guest or episode title.

Transcript context:
{context}

User question:
{query}

Answer:
"""
    return prompt


def generate_with_ollama(prompt, model_name):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 220,
                "num_ctx": 2048,
                "temperature": 0.2
            }
        },
        timeout=300
    )
    response.raise_for_status()
    return response.json()["response"]

def answer_question(query, top_k=3, model_name=DEFAULT_OLLAMA_MODEL):
    retrieved_chunks = retrieve_context(query, top_k=top_k)
    prompt = build_prompt(query, retrieved_chunks)

    try:
        answer = generate_with_ollama(prompt, model_name=model_name)
    except Exception as e:
        answer = (
            "The retrieval pipeline is working and relevant transcript sources were found. "
            "However, full answer generation requires a local Ollama server. "
            "On Streamlit Cloud, localhost does not connect to a local laptop or Colab Ollama instance. "
            "To run full generation locally, use: ollama pull mistral, ollama serve, then streamlit run this file. "
            f"Selected model: {model_name}."
        )

    return answer, retrieved_chunks

st.title("🎙️ Lex Fridman Podcast RAG Chatbot")

st.write(
    "This chatbot uses Lex Fridman Podcast transcripts, Chroma vector search, "
    "sentence-transformer embeddings and a local Ollama language model. Mistral is the default option, with smaller models available for low-resource testing."
)

with st.sidebar:
    st.header("Project details")
    st.write(f"Author: {FULL_NAME}")
    st.write(f"Published app: {STREAMLIT_APP_LINK}")
    st.write("Vector database: Chroma")
    st.write(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    default_index = (
        AVAILABLE_OLLAMA_MODELS.index(DEFAULT_OLLAMA_MODEL)
        if DEFAULT_OLLAMA_MODEL in AVAILABLE_OLLAMA_MODELS
        else 0
    )

    selected_model = st.selectbox(
        "Ollama model",
        AVAILABLE_OLLAMA_MODELS,
        index=default_index,
        help="Choose the local Ollama model used for answer generation."
    )

    st.write(f"Selected local LLM: {selected_model} via Ollama")
    st.caption(MODEL_NOTES.get(selected_model, ""))
    st.write(f"Indexed chunks: {collection.count()}")

    st.info(
        "For full generation, run Ollama locally with `ollama pull mistral` and `ollama serve`."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("Ask a question about the Lex Fridman Podcast")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving transcript context and generating answer..."):
            answer, retrieved_chunks = answer_question(user_question, top_k=TOP_K, model_name=selected_model)
            st.markdown(answer)

            with st.expander("Retrieved transcript sources"):
                for i, item in enumerate(retrieved_chunks, start=1):
                    meta = item["metadata"]
                    st.markdown(
                        f"**Source {i}: {meta.get('title')} — {meta.get('guest')}**"
                    )
                    st.caption(f"Chunk: {meta.get('chunk_id')} | Distance: {item['distance']}")
                    st.write(item["text"][:1200] + "...")

            st.session_state.messages.append({"role": "assistant", "content": answer})
