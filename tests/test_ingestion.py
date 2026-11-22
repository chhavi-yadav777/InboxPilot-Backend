from datetime import datetime, timezone

import pytest

from API import ingestion


class FakeCollection:
    def __init__(self):
        self.updated = []
        self.document = {"checkpoint_timestamp": None}

    def update_one(self, query, update):
        self.updated.append((query, update))
        maximum = update["$max"]
        self.document["checkpoint_ms"] = maximum["checkpoint_ms"]
        self.document["checkpoint_timestamp"] = maximum["checkpoint_timestamp"]

    def find_one(self, query, projection=None):
        return self.document


def _message(email_id="m1", timestamp=1_800_000_000_000):
    return {
        "email_id": email_id,
        "thread_id": "t1",
        "subject": "Interview Schedule",
        "sender": "hr@company.com",
        "receiver": "user@gmail.com",
        "date": "2026-11-04",
        "body": "Your interview is scheduled for Monday...",
        "internal_date_ms": timestamp,
    }


def test_ingestion_advances_checkpoint_only_after_indexing(monkeypatch):
    user_collection = FakeCollection()
    message = _message()
    checkpoint_calls = []

    monkeypatch.setattr(ingestion, "credentials_for_user", lambda user: object())
    monkeypatch.setattr(ingestion, "fetch_gmail_messages", lambda credentials, checkpoint_ms: [message])
    monkeypatch.setattr(ingestion, "_store_emails", lambda user_id, messages: len(messages))
    monkeypatch.setattr(ingestion, "_emails_for_user", lambda user_id, messages=None: [message])
    monkeypatch.setattr(ingestion, "upsert_emails", lambda user_id, emails: len(emails))
    monkeypatch.setattr(ingestion, "get_users_collection", lambda: user_collection)
    original_advance = ingestion._advance_checkpoint

    def track_advance(user_id, timestamp_ms):
        checkpoint_calls.append(timestamp_ms)
        return original_advance(user_id, timestamp_ms)

    monkeypatch.setattr(ingestion, "_advance_checkpoint", track_advance)
    result = ingestion.ingest_mailbox({"google_sub": "google-user"}, incremental=True)

    assert result["fetched"] == result["stored"] == result["indexed"] == 1
    assert checkpoint_calls == [message["internal_date_ms"]]
    assert user_collection.document["checkpoint_timestamp"] == datetime.fromtimestamp(
        message["internal_date_ms"] / 1000, timezone.utc
    )


def test_ingestion_does_not_advance_checkpoint_when_indexing_fails(monkeypatch):
    message = _message()
    checkpoint_calls = []
    monkeypatch.setattr(ingestion, "credentials_for_user", lambda user: object())
    monkeypatch.setattr(ingestion, "fetch_gmail_messages", lambda credentials, checkpoint_ms: [message])
    monkeypatch.setattr(ingestion, "_store_emails", lambda user_id, messages: len(messages))
    monkeypatch.setattr(ingestion, "_emails_for_user", lambda user_id, messages=None: [message])

    def fail_index(user_id, emails):
        raise RuntimeError("simulated Pinecone outage")

    monkeypatch.setattr(ingestion, "upsert_emails", fail_index)
    monkeypatch.setattr(
        ingestion,
        "_advance_checkpoint",
        lambda user_id, timestamp_ms: checkpoint_calls.append(timestamp_ms),
    )

    with pytest.raises(RuntimeError, match="simulated Pinecone outage"):
        ingestion.ingest_mailbox({"google_sub": "google-user"}, incremental=True)
    assert checkpoint_calls == []


def test_email_response_returns_only_the_public_email_fields():
    result = ingestion.email_response({**_message(), "user_id": "private-user", "_id": "mongo-id"})
    assert set(result) == {
        "email_id",
        "thread_id",
        "subject",
        "sender",
        "receiver",
        "date",
        "body",
    }
    assert result["email_id"] == "m1"
