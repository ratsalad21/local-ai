import pytest

import rag


class DummyCollection:
    def __init__(self):
        self.added = []

    def add(self, ids, documents, embeddings):
        self.added.append({"ids": ids, "documents": documents, "embeddings": embeddings})

    def query(self, query_embeddings, n_results):
        # Return documents corresponding to query_embeddings length for test
        docs = [f"doc_{i}" for i in range(n_results)]
        return {"documents": [docs]}


class DummyEmbedder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(texts)
        # Return a fixed embedding vector for each input
        return [[0.123, 0.456, 0.789] for _ in texts]


@pytest.fixture(autouse=True)
def patch_rag_module(monkeypatch):
    """Patch the rag module to use dummy embedder and collection for deterministic tests."""
    dummy_embedder = DummyEmbedder()
    dummy_collection = DummyCollection()
    monkeypatch.setattr(rag, "embedder", dummy_embedder)
    monkeypatch.setattr(rag, "collection", dummy_collection)
    return dummy_embedder, dummy_collection


def test_add_document_stores_document_and_embedding(patch_rag_module):
    embedder, collection = patch_rag_module

    rag.add_document("hello world", doc_id="test1")

    assert len(collection.added) == 1
    entry = collection.added[0]
    assert entry["ids"] == ["test1"]
    assert entry["documents"] == ["hello world"]
    assert entry["embeddings"] == [[0.123, 0.456, 0.789]]

    assert embedder.calls == [["hello world"]]


def test_query_documents_returns_joined_docs(patch_rag_module):
    _, _ = patch_rag_module

    result = rag.query_documents("what is test?", k=2)

    assert result == "doc_0\n\ndoc_1"
