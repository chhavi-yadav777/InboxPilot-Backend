"""MongoDB connection and collection access."""

from __future__ import annotations

from functools import lru_cache

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from API.config import settings


@lru_cache(maxsize=1)
def get_mongo_client() -> MongoClient:
    options = {"serverSelectionTimeoutMS": 8000, "appname": "InboxPilot", "tz_aware": True}
    # Credentials are supplied separately when DB_URI does not already include them.
    if settings.mongodb_username and "@" not in settings.mongodb_uri.split("://", 1)[-1].split("/", 1)[0]:
        options.update(
            username=settings.mongodb_username,
            password=settings.mongodb_password,
            authSource="admin",
        )
    return MongoClient(settings.mongodb_uri, **options)


def get_database() -> Database:
    return get_mongo_client()[settings.mongodb_database]


def get_users_collection() -> Collection:
    return get_database()[settings.users_collection]


def get_emails_collection() -> Collection:
    return get_database()[settings.emails_collection]


def get_chats_collection() -> Collection:
    return get_database()[settings.chats_collection]


def ensure_indexes() -> None:
    get_users_collection().create_index("google_sub", unique=True)
    get_emails_collection().create_index(
        [("user_id", 1), ("email_id", 1)], unique=True, name="user_email_unique"
    )
    get_emails_collection().create_index(
        [("user_id", 1), ("internal_date_ms", -1)], name="user_email_date"
    )
    get_chats_collection().create_index(
        [("user_id", 1), ("conversation_id", 1), ("created_at", 1)],
        name="user_conversation_messages",
    )
    get_chats_collection().create_index(
        [("user_id", 1), ("created_at", -1)], name="user_chat_history"
    )


def ping_database() -> None:
    get_database().command("ping")
