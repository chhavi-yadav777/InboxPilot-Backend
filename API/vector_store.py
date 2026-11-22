"""NVIDIA text embedding and Pinecone upsert helpers."""

from __future__ import annotations

from functools import lru_cache

import requests
from pinecone import Pinecone

from API.config import settings


EMBEDDING_DIMENSION = 2048
NVIDIA_EMBEDDINGS_URL = "https://integrate.api.nvidia.com/v1/embeddings"


@lru_cache(maxsize=1)
def get_pinecone_client() -> Pinecone:
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE is not configured")
    return Pinecone(api_key=settings.pinecone_api_key)


@lru_cache(maxsize=1)
def get_pinecone_index():
    if not settings.pinecone_index_name:
        raise RuntimeError("PINECONE_INDEX_NAME is not configured")
    return get_pinecone_client().Index(settings.pinecone_index_name)


def validate_pinecone_index() -> None:
    if not settings.pinecone_index_name:
        raise RuntimeError("PINECONE_INDEX_NAME is not configured")
    try:
        description = get_pinecone_client().describe_index(settings.pinecone_index_name)
    except Exception as exc:
        if type(exc).__name__ == "NotFoundException" or getattr(exc, "status", None) == 404:
            raise RuntimeError(
                f"Pinecone index {settings.pinecone_index_name!r} was not found. "
                "Set PINECONE_INDEX_NAME to an existing index in the project for this API key, "
                "or create that index with dimension 2048."
            ) from exc
        raise RuntimeError(
            f"Could not inspect Pinecone index {settings.pinecone_index_name!r} "
            f"({type(exc).__name__}). Check the API key and network connection."
        ) from exc
    dimension = getattr(description, "dimension", None)
    if not isinstance(dimension, int) or dimension < EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Pinecone index {settings.pinecone_index_name!r} has dimension {dimension}; "
            f"it must be at least {EMBEDDING_DIMENSION} for {settings.nvidia_embedding_model}"
        )


def pinecone_index_dimension() -> int:
    description = get_pinecone_client().describe_index(settings.pinecone_index_name)
    dimension = getattr(description, "dimension", None)
    if not isinstance(dimension, int):
        raise RuntimeError(f"Could not read the dimension for Pinecone index {settings.pinecone_index_name!r}")
    return dimension


def _embed(texts: list[str], *, input_type: str = "passage") -> list[list[float]]:
    if not settings.nvidia_api_key:
        raise RuntimeError("NVAPI is not configured")
    response = requests.post(
        NVIDIA_EMBEDDINGS_URL,
        headers={"Authorization": f"Bearer {settings.nvidia_api_key}"},
        json={
            "model": settings.nvidia_embedding_model,
            "input": texts,
            "input_type": input_type,
            "encoding_format": "float",
            "truncate": "END",
        },
        timeout=(10, 90),
    )
    response.raise_for_status()
    data = response.json().get("data", [])
    ordered = sorted(data, key=lambda row: row.get("index", 0))
    vectors = [row["embedding"] for row in ordered]
    if len(vectors) != len(texts) or any(len(vector) != EMBEDDING_DIMENSION for vector in vectors):
        raise RuntimeError("NVIDIA returned an unexpected embedding count or dimension")
    return vectors


def _to_index_dimension(vector: list[float]) -> list[float]:
    target_dimension = pinecone_index_dimension()
    if target_dimension < EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Pinecone index dimension {target_dimension} is smaller than the model output "
            f"dimension {EMBEDDING_DIMENSION}"
        )
    if target_dimension > EMBEDDING_DIMENSION:
        return vector + [0.0] * (target_dimension - EMBEDDING_DIMENSION)
    return vector


def query_emails(user_id: str, query: str, top_k: int = 5) -> list[dict]:
    """Return relevant email metadata from the authenticated user's namespace."""
    if not query.strip():
        raise ValueError("Query cannot be empty")
    vector = _to_index_dimension(_embed([query], input_type="query")[0])
    result = get_pinecone_index().query(
        vector=vector,
        top_k=top_k,
        namespace=user_id,
        include_metadata=True,
    )
    matches = getattr(result, "matches", None)
    if matches is None and isinstance(result, dict):
        matches = result.get("matches", [])
    sources = []
    for match in matches or []:
        if isinstance(match, dict):
            vector_id = match.get("id", "")
            score = match.get("score")
            metadata = match.get("metadata") or {}
        else:
            vector_id = getattr(match, "id", "")
            score = getattr(match, "score", None)
            metadata = getattr(match, "metadata", None) or {}
        sources.append(
            {
                "vector_id": vector_id,
                "score": score,
                "email_id": metadata.get("email_id", ""),
                "thread_id": metadata.get("thread_id", ""),
                "subject": metadata.get("subject", ""),
                "sender": metadata.get("sender", ""),
                "receiver": metadata.get("receiver", ""),
                "date": metadata.get("date", ""),
                "body": metadata.get("body", ""),
            }
        )
    return sources


def _embedding_text(email: dict) -> str:
    return "\n".join(
        value
        for value in (
            f"Subject: {email.get('subject', '')}",
            f"From: {email.get('sender', '')}",
            f"To: {email.get('receiver', '')}",
            f"Date: {email.get('date', '')}",
            f"Body: {email.get('body', '')}",
        )
        if value.strip()
    )


def _metadata_body(body: str) -> str:
    # Keep metadata below Pinecone's per-record size limit while preserving full text in MongoDB.
    return body.encode("utf-8")[:30000].decode("utf-8", errors="ignore")


def upsert_emails(user_id: str, emails: list[dict]) -> int:
    """Embed and upsert a user's MongoDB email records in bounded batches."""
    index = get_pinecone_index()
    target_dimension = pinecone_index_dimension()
    if target_dimension < EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Pinecone index dimension {target_dimension} is smaller than the model output "
            f"dimension {EMBEDDING_DIMENSION}"
        )
    count = 0
    for start in range(0, len(emails), settings.embedding_batch_size):
        batch = emails[start : start + settings.embedding_batch_size]
        texts = [_embedding_text(email) for email in batch]
        vectors = _embed(texts)
        records = []
        for email, vector in zip(batch, vectors):
            # Appending zeros preserves dense-vector similarity while fitting a larger index.
            vector = vector + [0.0] * (target_dimension - EMBEDDING_DIMENSION)
            records.append(
                {
                    "id": f"{user_id}:{email['email_id']}",
                    "values": vector,
                    "metadata": {
                        "user_id": user_id,
                        "email_id": email["email_id"],
                        "thread_id": email.get("thread_id", ""),
                        "subject": email.get("subject", ""),
                        "sender": email.get("sender", ""),
                        "receiver": email.get("receiver", ""),
                        "date": email.get("date", ""),
                        "body": _metadata_body(email.get("body", "")),
                    },
                }
            )
        for offset in range(0, len(records), settings.pinecone_batch_size):
            chunk = records[offset : offset + settings.pinecone_batch_size]
            index.upsert(vectors=chunk, namespace=user_id)
            count += len(chunk)
    return count
