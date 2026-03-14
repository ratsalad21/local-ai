import hashlib
import os
import re
from typing import Any, List, Optional

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
DEFAULT_QUERY_K = 5
QUERY_CANDIDATE_MULTIPLIER = 3
MAX_RESULTS_PER_SOURCE = 2
MAX_DISTANCE = 1.1


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


def _similarity_from_distance(distance: Optional[float]) -> Optional[float]:
    if distance is None:
        return None
    return max(0.0, min(1.0, 1.0 - (distance / 2.0)))


def _normalize_lookup_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _source_match_score(query: str, source: str) -> int:
    normalized_query = _normalize_lookup_text(query)
    normalized_source = _normalize_lookup_text(source)

    if not normalized_query or not normalized_source:
        return 0

    if normalized_source in normalized_query:
        return 3

    source_tokens = [token for token in normalized_source.split() if len(token) > 2]
    if not source_tokens:
        return 0

    matched_tokens = sum(1 for token in source_tokens if token in normalized_query)
    if matched_tokens == len(source_tokens):
        return 2
    if matched_tokens > 0:
        return 1
    return 0


def _filename_matches(
    query: str,
    max_per_source: int,
) -> List[dict[str, Any]]:
    assert collection is not None

    results = collection.get(include=["documents", "metadatas"])
    docs = results.get("documents", []) or []
    metadatas = results.get("metadatas", []) or []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, doc in enumerate(docs):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        source = metadata.get("source")
        if not source:
            continue

        grouped.setdefault(source, []).append(
            {
                "source": source,
                "chunk": metadata.get("chunk", -1),
                "total_chunks": metadata.get("total_chunks"),
                "text": doc,
                "distance": None,
                "similarity": None,
            }
        )

    matched_sources = sorted(
        (
            (source, _source_match_score(query, source))
            for source in grouped
        ),
        key=lambda item: (-item[1], item[0].lower()),
    )

    matches: List[dict[str, Any]] = []
    for source, score in matched_sources:
        if score <= 0:
            continue

        source_chunks = sorted(grouped[source], key=lambda item: item.get("chunk", -1))
        matches.extend(source_chunks[:max_per_source])

    return matches


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
# Document Management
# ================================

def remove_document(doc_id: str) -> bool:
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
        return True

    return False


def list_indexed_documents() -> List[dict[str, Any]]:
    """Return indexed documents with chunk counts."""
    init_rag()
    assert collection is not None

    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas", []) or []

    docs: dict[str, dict[str, Any]] = {}
    for metadata in metadatas:
        if not metadata:
            continue

        source = metadata.get("source")
        if not source:
            continue

        doc_entry = docs.setdefault(
            source,
            {
                "source": source,
                "chunks": 0,
                "total_chunks": metadata.get("total_chunks"),
            },
        )
        doc_entry["chunks"] += 1
        if metadata.get("total_chunks") is not None:
            doc_entry["total_chunks"] = metadata.get("total_chunks")

    return sorted(docs.values(), key=lambda item: item["source"].lower())


def clear_documents() -> int:
    """Remove all indexed documents and return the number of deleted chunks."""
    init_rag()
    assert collection is not None

    results = collection.get(include=[])
    ids = results.get("ids", []) or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)


# ================================
# Add / Replace Document
# ================================

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

    remove_document(doc_id)

    embeddings = embedder.encode(
        chunks,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    ids = [_stable_chunk_id(doc_id, i, chunk) for i, chunk in enumerate(chunks)]

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

def search_documents(
    query: str,
    k: int = DEFAULT_QUERY_K,
    max_distance: float = MAX_DISTANCE,
    max_per_source: int = MAX_RESULTS_PER_SOURCE,
) -> List[dict[str, Any]]:
    """Return filtered retrieval matches with metadata for UI and prompt building."""
    init_rag()
    assert embedder is not None and collection is not None

    if not query.strip():
        return []

    filename_matches = _filename_matches(query, max_per_source=max_per_source)

    q_emb = embedder.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=max(k * QUERY_CANDIDATE_MULTIPLIER, k),
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        return []

    matches: List[dict[str, Any]] = []
    seen_keys: set[tuple[str, int, str]] = set()
    per_source_counts: dict[str, int] = {}

    for match in filename_matches:
        dedupe_key = (match["source"], match["chunk"], match["text"])
        if dedupe_key in seen_keys:
            continue

        if per_source_counts.get(match["source"], 0) >= max_per_source:
            continue

        seen_keys.add(dedupe_key)
        per_source_counts[match["source"]] = per_source_counts.get(match["source"], 0) + 1
        matches.append(match)

        if len(matches) >= k:
            return matches

    for i, doc in enumerate(docs):
        metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
        distance = distances[i] if i < len(distances) else None
        source = metadata.get("source", "unknown")
        chunk_num = metadata.get("chunk", -1)
        total_chunks = metadata.get("total_chunks")

        if distance is not None and distance > max_distance:
            continue

        dedupe_key = (source, chunk_num, doc)
        if dedupe_key in seen_keys:
            continue

        if per_source_counts.get(source, 0) >= max_per_source:
            continue

        seen_keys.add(dedupe_key)
        per_source_counts[source] = per_source_counts.get(source, 0) + 1

        matches.append(
            {
                "source": source,
                "chunk": chunk_num,
                "total_chunks": total_chunks,
                "text": doc,
                "distance": distance,
                "similarity": _similarity_from_distance(distance),
            }
        )

        if len(matches) >= k:
            break

    return matches


def format_retrieval_context(
    matches: List[dict[str, Any]],
    include_sources: bool = True,
) -> str:
    formatted_chunks: List[str] = []

    for match in matches:
        doc = match["text"]
        source = match.get("source", "unknown")
        chunk_num = match.get("chunk", "?")

        if include_sources:
            formatted_chunks.append(f"[Source: {source} | Chunk: {chunk_num}]\n{doc}")
        else:
            formatted_chunks.append(doc)

    return "\n\n".join(formatted_chunks)


def list_sources(matches: List[dict[str, Any]]) -> List[str]:
    seen: set[str] = set()
    sources: List[str] = []

    for match in matches:
        source = match.get("source")
        if source and source not in seen:
            seen.add(source)
            sources.append(source)

    return sources


def query_documents(query: str, k: int = DEFAULT_QUERY_K, include_sources: bool = True) -> str:
    """Query the vector store and return a formatted context string."""
    matches = search_documents(query, k=k)
    return format_retrieval_context(matches, include_sources=include_sources)
