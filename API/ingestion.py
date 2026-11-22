"""Mailbox ingestion orchestration and email persistence."""

from __future__ import annotations

from datetime import datetime, timezone

from pymongo import ReplaceOne

from API.fetchEmails import fetch_gmail_messages
from API.security import credentials_for_user
from API.vector_store import upsert_emails
from DB.dbClient import get_emails_collection, get_users_collection


def _checkpoint_datetime(timestamp_ms: int | None) -> datetime | None:
    if not timestamp_ms:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc)


def _store_emails(user_id: str, messages: list[dict]) -> int:
    if not messages:
        return 0
    collection = get_emails_collection()
    unique_messages = {message["email_id"]: message for message in messages}
    operations = []
    for message in unique_messages.values():
        document = {**message, "user_id": user_id}
        operations.append(
            ReplaceOne(
                {"user_id": user_id, "email_id": message["email_id"]},
                document,
                upsert=True,
            )
        )
    collection.bulk_write(operations, ordered=False)
    return len(operations)


def _emails_for_user(user_id: str, messages: list[dict] | None = None) -> list[dict]:
    collection = get_emails_collection()
    if messages is None:
        return list(collection.find({"user_id": user_id}))
    ids = list({message["email_id"] for message in messages})
    if not ids:
        return []
    return list(collection.find({"user_id": user_id, "email_id": {"$in": ids}}))


def _advance_checkpoint(user_id: str, max_timestamp_ms: int | None) -> str | None:
    if not max_timestamp_ms:
        user = get_users_collection().find_one({"google_sub": user_id}, {"checkpoint_timestamp": 1})
        return user.get("checkpoint_timestamp") if user else None
    timestamp = _checkpoint_datetime(max_timestamp_ms)
    get_users_collection().update_one(
        {"google_sub": user_id},
        {
            "$max": {"checkpoint_ms": max_timestamp_ms, "checkpoint_timestamp": timestamp},
            "$set": {"updated_at": datetime.now(timezone.utc)},
        },
    )
    user = get_users_collection().find_one({"google_sub": user_id}, {"checkpoint_timestamp": 1})
    return user.get("checkpoint_timestamp") if user else timestamp


def ingest_mailbox(user: dict, *, incremental: bool) -> dict:
    user_id = user["google_sub"]
    checkpoint_ms = user.get("checkpoint_ms") if incremental else None
    credentials = credentials_for_user(user)
    messages = fetch_gmail_messages(credentials, checkpoint_ms=checkpoint_ms)
    stored = _store_emails(user_id, messages)

    # A full import also repairs vectors for records already in MongoDB from an earlier attempt.
    records_to_index = _emails_for_user(user_id) if not incremental else _emails_for_user(user_id, messages)
    indexed = upsert_emails(user_id, records_to_index)
    max_timestamp_ms = max(
        (message.get("internal_date_ms", 0) for message in messages), default=0
    )
    checkpoint = _advance_checkpoint(user_id, max_timestamp_ms)
    return {
        "fetched": len(messages),
        "stored": stored,
        "indexed": indexed,
        "checkpoint_timestamp": checkpoint,
    }


def reindex_mailbox(user: dict) -> dict:
    user_id = user["google_sub"]
    emails = _emails_for_user(user_id)
    indexed = upsert_emails(user_id, emails)
    return {"indexed": indexed}


def email_response(document: dict) -> dict:
    return {
        "email_id": document["email_id"],
        "thread_id": document.get("thread_id", ""),
        "subject": document.get("subject", ""),
        "sender": document.get("sender", ""),
        "receiver": document.get("receiver", ""),
        "date": document.get("date", ""),
        "body": document.get("body", ""),
    }
