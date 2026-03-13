from pathlib import Path

content = '''# rag.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import Optional

client: Optional[chromadb.PersistentClient] = None
collection: Optional[chromadb.Collection] = None
embedder: Optional[SentenceTransformer] = None


def init_rag(path: str = "/chroma_db"):
    """Lazily initialize the Chroma client, collection, and embedding model."""
    global client, collection, embedder
    if client is not None and collection is not None and embedder is not None:
        return

    client = chromadb.PersistentClient(
        path=path,
        settings=Settings(allow_reset=True)
    )

    collection = client.get_or_create_collection("documents")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def add_document(text: str, doc_id: str):
    if embedder is None or collection is None:
        init_rag()

    embedding = embedder.encode([text])[0].tolist()
    collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding],
    )


def query_documents(query: str, k: int = 3):
    if embedder is None or collection is None:
        init_rag()

    q_emb = embedder.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
    )
    docs = results.get("documents", [[]])[0]
    return "\n\n".join(docs)
'''

path = Path(r"g:\local-ai\chat-ui\rag.py")
path.write_text(content)
print(f"Wrote {path}")
