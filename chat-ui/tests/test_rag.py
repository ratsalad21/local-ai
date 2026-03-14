from types import SimpleNamespace

import pytest

import rag


class DummyEmbeddings:
    def __init__(self, values):
        self._values = values

    def __getitem__(self, index):
        return DummyEmbeddingRow(self._values[index])

    def tolist(self):
        return self._values


class DummyEmbeddingRow:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class DummyCollection:
    def __init__(self):
        self.add_calls = []
        self.delete_calls = []
        self.query_calls = []
        self.get_calls = []
        self.ids_to_return = []
        self.query_result = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    def add(self, ids, documents, embeddings, metadatas):
        self.add_calls.append(
            {
                "ids": ids,
                "documents": documents,
                "embeddings": embeddings,
                "metadatas": metadatas,
            }
        )

    def get(self, where, include):
        self.get_calls.append({"where": where, "include": include})
        return {"ids": self.ids_to_return}

    def delete(self, ids):
        self.delete_calls.append(ids)

    def query(self, query_embeddings, n_results, include):
        self.query_calls.append(
            {
                "query_embeddings": query_embeddings,
                "n_results": n_results,
                "include": include,
            }
        )
        return self.query_result


class DummyEmbedder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": texts, "kwargs": kwargs})
        return DummyEmbeddings([[0.1, 0.2, 0.3] for _ in texts])


@pytest.fixture(autouse=True)
def patch_rag_module(monkeypatch):
    dummy_embedder = DummyEmbedder()
    dummy_collection = DummyCollection()

    monkeypatch.setattr(rag, "embedder", dummy_embedder)
    monkeypatch.setattr(rag, "collection", dummy_collection)
    monkeypatch.setattr(rag, "client", SimpleNamespace())

    return dummy_embedder, dummy_collection


def test_add_document_chunks_and_replaces_existing_entries(patch_rag_module):
    embedder, collection = patch_rag_module

    text = "word " * 120
    collection.ids_to_return = ["existing-chunk"]

    stored_chunks = rag.add_document(text, doc_id="test doc")

    assert stored_chunks == 1
    assert collection.get_calls == [{"where": {"source": "test doc"}, "include": []}]
    assert collection.delete_calls == [["existing-chunk"]]

    assert len(collection.add_calls) == 1
    add_call = collection.add_calls[0]
    assert add_call["documents"] == [text.strip()]
    assert add_call["metadatas"] == [{"source": "test doc", "chunk": 0, "total_chunks": 1}]
    assert add_call["ids"][0].startswith("test_doc_0_")

    assert embedder.calls[0]["texts"] == [text.strip()]
    assert embedder.calls[0]["kwargs"]["batch_size"] == rag.EMBED_BATCH_SIZE
    assert embedder.calls[0]["kwargs"]["normalize_embeddings"] is True


def test_add_document_rejects_empty_chunks(patch_rag_module):
    _, collection = patch_rag_module

    with pytest.raises(ValueError, match="Document produced no valid chunks"):
        rag.add_document("short text", doc_id="tiny")

    assert collection.add_calls == []


def test_query_documents_returns_formatted_context_with_sources(patch_rag_module):
    embedder, collection = patch_rag_module
    collection.query_result = {
        "documents": [["alpha chunk", "beta chunk"]],
        "metadatas": [[
            {"source": "alpha.md", "chunk": 0},
            {"source": "beta.md", "chunk": 3},
        ]],
        "distances": [[0.1, 0.2]],
    }

    result = rag.query_documents("what is test?", k=2)

    assert result == (
        "[Source: alpha.md | Chunk: 0]\nalpha chunk\n\n"
        "[Source: beta.md | Chunk: 3]\nbeta chunk"
    )
    assert embedder.calls[0]["texts"] == ["what is test?"]
    assert collection.query_calls == [
        {
            "query_embeddings": [[0.1, 0.2, 0.3]],
            "n_results": 2,
            "include": ["documents", "metadatas", "distances"],
        }
    ]


def test_query_documents_can_omit_sources(patch_rag_module):
    _, collection = patch_rag_module
    collection.query_result = {
        "documents": [["alpha chunk", "beta chunk"]],
        "metadatas": [[{}, {}]],
        "distances": [[0.1, 0.2]],
    }

    result = rag.query_documents("what is test?", k=2, include_sources=False)

    assert result == "alpha chunk\n\nbeta chunk"
