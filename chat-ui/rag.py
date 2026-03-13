# rag.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(
    path="/chroma_db",
    settings=Settings(allow_reset=True)
)

collection = client.get_or_create_collection("documents")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def add_document(text: str, doc_id: str):
    embedding = embedder.encode([text])[0].tolist()
    collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding],
    )

def query_documents(query: str, k: int = 3):
    q_emb = embedder.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
    )
    docs = results.get("documents", [[]])[0]
    return "\n\n".join(docs)