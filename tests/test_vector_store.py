from types import SimpleNamespace

import pytest

from API import vector_store


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": self.data}


def test_embedding_request_uses_passage_mode_and_configured_model(monkeypatch):
    request = {}
    monkeypatch.setattr(
        vector_store,
        "settings",
        SimpleNamespace(nvidia_api_key="test-key", nvidia_embedding_model="test/model"),
    )

    def fake_post(url, headers, json, timeout):
        request.update(url=url, headers=headers, body=json, timeout=timeout)
        return FakeResponse([{"index": 0, "embedding": [0.1] * 2048}])

    monkeypatch.setattr(vector_store.requests, "post", fake_post)
    vectors = vector_store._embed(["Subject: Interview"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 2048
    assert request["body"]["input_type"] == "passage"
    assert request["body"]["model"] == "test/model"
    assert request["headers"]["Authorization"] == "Bearer test-key"


def test_embedding_rejects_unexpected_vector_dimension(monkeypatch):
    monkeypatch.setattr(
        vector_store,
        "settings",
        SimpleNamespace(nvidia_api_key="test-key", nvidia_embedding_model="test/model"),
    )
    monkeypatch.setattr(
        vector_store.requests,
        "post",
        lambda *args, **kwargs: FakeResponse([{"index": 0, "embedding": [0.1]}]),
    )
    with pytest.raises(RuntimeError, match="dimension"):
        vector_store._embed(["email text"])


def test_pinecone_index_validation_rejects_wrong_dimension(monkeypatch):
    monkeypatch.setattr(
        vector_store,
        "settings",
        SimpleNamespace(pinecone_index_name="emails", nvidia_embedding_model="test/model"),
    )

    class FakePinecone:
        def describe_index(self, name):
            return SimpleNamespace(dimension=1024)

    monkeypatch.setattr(vector_store, "get_pinecone_client", lambda: FakePinecone())
    with pytest.raises(RuntimeError, match="must be at least 2048"):
        vector_store.validate_pinecone_index()


def test_pinecone_index_validation_explains_missing_index(monkeypatch):
    monkeypatch.setattr(
        vector_store,
        "settings",
        SimpleNamespace(pinecone_index_name="emails", nvidia_embedding_model="test/model"),
    )

    class NotFoundException(Exception):
        status = 404

    class FakePinecone:
        def describe_index(self, name):
            raise NotFoundException()

    monkeypatch.setattr(vector_store, "get_pinecone_client", lambda: FakePinecone())
    with pytest.raises(RuntimeError, match="index 'emails' was not found"):
        vector_store.validate_pinecone_index()


def test_upsert_zero_pads_embeddings_for_larger_existing_index(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        vector_store,
        "settings",
        SimpleNamespace(
            pinecone_index_name="emails",
            nvidia_embedding_model="test/model",
            embedding_batch_size=2,
            pinecone_batch_size=10,
        ),
    )

    class FakeIndex:
        def upsert(self, vectors, namespace):
            captured["vectors"] = vectors
            captured["namespace"] = namespace

    monkeypatch.setattr(vector_store, "get_pinecone_index", lambda: FakeIndex())
    monkeypatch.setattr(vector_store, "pinecone_index_dimension", lambda: 3072)
    monkeypatch.setattr(vector_store, "_embed", lambda texts: [[0.25] * 2048 for _ in texts])

    count = vector_store.upsert_emails(
        "user-1", [{"email_id": "message-1", "body": "email body"}]
    )

    values = captured["vectors"][0]["values"]
    assert count == 1
    assert captured["namespace"] == "user-1"
    assert len(values) == 3072
    assert values[:2048] == [0.25] * 2048
    assert values[2048:] == [0.0] * 1024


def test_query_uses_user_namespace_and_pads_query_vector(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        vector_store,
        "_embed",
        lambda texts, input_type="passage": [
            ([0.5] * 2048 if input_type == "query" else [0.25] * 2048)
        ],
    )
    monkeypatch.setattr(vector_store, "pinecone_index_dimension", lambda: 3072)

    class FakeIndex:
        def query(self, vector, top_k, namespace, include_metadata):
            captured.update(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=include_metadata,
            )
            return SimpleNamespace(
                matches=[
                    SimpleNamespace(
                        id="user-1:email-1",
                        score=0.9,
                        metadata={"email_id": "email-1", "subject": "Interview"},
                    )
                ]
            )

    monkeypatch.setattr(vector_store, "get_pinecone_index", lambda: FakeIndex())
    results = vector_store.query_emails("user-1", "interview time", top_k=3)

    assert len(captured["vector"]) == 3072
    assert captured["vector"][:2048] == [0.5] * 2048
    assert captured["vector"][2048:] == [0.0] * 1024
    assert captured["namespace"] == "user-1"
    assert captured["top_k"] == 3
    assert captured["include_metadata"] is True
    assert results[0]["email_id"] == "email-1"
