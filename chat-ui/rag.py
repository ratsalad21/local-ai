import hashlib
import os
from typing import List, Optional

import chromadb
import torch
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

client: Optional[chromadb.PersistentClient] = None
collection: Optional[chromadb.Collection] = None
embedder: Optional[SentenceTransformer] = None

# ================================
# Configuration
# ================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
COLLECTION_NAME = "documents"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DB_PATH = os.getenv("CHROMA_DB_PATH", "/chroma_db")
EMBED_BATCH_SIZE = 32

# ================================
# Initialization
# ================================

def init_rag(path: str = DEFAULT_DB_PATH) -> None:
    global client, collection, embedder

    if client is not None and collection is not None and embedder is not None:
        return

    client = chromadb.PersistentClient(
        path=path,
        settings=Settings(allow_reset=False),
    )

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding model using: {device}")

    embedder = SentenceTransformer(EMBED_MODEL, device=device)


# ================================
# Helpers
# ================================

def _stable_chunk_id(doc_id: str, chunk_index: int, chunk_text: str) -> str:
    digest = hashlib.sha1(chunk_text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    safe_doc_id = doc_id.replace(" ", "_")
    return f"{safe_doc_id}_{chunk_index}_{digest}"
# ================================
# Chunking
# ================================

def chunk_text(text: str) -> List[str]:
    """Split document into overlapping chunks by words."""
    words = text.split()

    if not words:
        return []

    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    chunks: List[str] = []

    for i in range(0, len(words), step):
        chunk_words = words[i : i + CHUNK_SIZE]
        chunk = " ".join(chunk_words).strip()

        if len(chunk) > 50:
            chunks.append(chunk)

    return chunks


# ================================
# Add / Replace Document
# ================================

def remove_document(doc_id: str) -> None:
    """Remove all chunks associated with a document."""
    init_rag()
    assert collection is not None

    existing = collection.get(
        where={"source": doc_id},
        include=[],
    )

    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)


def add_document(text: str, doc_id: str) -> int:
    """
    Add a document to the vector store.
    If the same doc_id already exists, replace its chunks.
    Returns the number of chunks stored.
    """
    init_rag()
    assert embedder is not None and collection is not None

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Document produced no valid chunks")

    # Replace existing chunks for this document so re-uploads do not duplicate data
    remove_document(doc_id)

    embeddings = embedder.encode(
        chunks,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    ids = [
        _stable_chunk_id(doc_id, i, chunk)
        for i, chunk in enumerate(chunks)
    ]

    metadatas = [
        {
            "source": doc_id,
            "chunk": i,
            "total_chunks": len(chunks),
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


# ================================
# Query
# ================================

def query_documents(query: str, k: int = 5, include_sources: bool = True) -> str:
    """Query the vector store and return a formatted context string."""
    init_rag()
    assert embedder is not None and collection is not None

    if not query.strip():
        return ""

    q_emb = embedder.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not docs:
        return ""

    formatted_chunks: List[str] = []

    for i, doc in enumerate(docs):
        metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
        source = metadata.get("source", "unknown")
        chunk_num = metadata.get("chunk", "?")

        if include_sources:
            formatted_chunks.append(
                f"[Source: {source} | Chunk: {chunk_num}]\n{doc}"
            )
        else:
            formatted_chunks.append(doc)

    return "\n\n".join(formatted_chunks)
