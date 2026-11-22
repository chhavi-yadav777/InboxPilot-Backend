"""Environment-backed configuration for the InboxPilot API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


@dataclass(frozen=True)
class Settings:
    google_client_id: str = _env("GOOGLE_CLIENT_ID", "CLIENT_ID")
    google_client_secret: str = _env("GOOGLE_CLIENT_SECRET")
    google_redirect_uri: str = _env(
        "GOOGLE_REDIRECT_URI", default="http://127.0.0.1:8000/auth/google/callback"
    )
    frontend_url: str = _env("FRONTEND_URL", default="http://127.0.0.1:5173/")
    mongodb_uri: str = _env("MONGODB_URI", "DB_URI")
    mongodb_username: str = _env("MONGODB_USERNAME", "DB_USERNAME")
    mongodb_password: str = _env("MONGODB_PASSWORD", "DB_PASSWORD")
    mongodb_database: str = _env(
        "MONGODB_DATABASE", "DATABASE_NAME", "InboxPilot", default="InboxPilot"
    )
    users_collection: str = _env(
        "MONGODB_USERS_COLLECTION", "USERS_COLLECTION", "IP_users", default="IP_users"
    )
    emails_collection: str = _env(
        "MONGODB_EMAILS_COLLECTION", "EMAILS_COLLECTION", default="IP_emails"
    )
    chats_collection: str = _env(
        "MONGODB_CHATS_COLLECTION", "CHATS_COLLECTION", default="IP_chats"
    )

    pinecone_api_key: str = _env("PINECONE_API_KEY", "PINECONE")
    pinecone_index_name: str = _env("PINECONE_INDEX_NAME")
    nvidia_api_key: str = _env("NVIDIA_API_KEY", "NVAPI")
    nvidia_embedding_model: str = _env(
        "NVIDIA_EMBEDDING_MODEL", default="nvidia/nemotron-3-embed-1b"
    )
    gemini_api_key: str = _env("GEMINI_API_KEY", "GEMINI")
    gemini_model: str = _env("GEMINI_MODEL", default="gemini-3.1-flash-lite")

    secret_key: str = _env("SECRET_KEY")
    jwt_algorithm: str = _env("ALGORITHM", default="HS256")
    access_token_expire_minutes: int = int(
        _env("ACCESS_TOKEN_EXPIRE_MINUTES", default="60")
    )
    email_fetch_workers: int = int(_env("EMAIL_FETCH_WORKERS", default="8"))
    embedding_batch_size: int = int(_env("EMBEDDING_BATCH_SIZE", default="32"))
    pinecone_batch_size: int = int(_env("PINECONE_BATCH_SIZE", default="100"))
    sync_overlap_seconds: int = int(_env("SYNC_OVERLAP_SECONDS", default="2"))

    @property
    def oauth_client_config(self) -> dict:
        return {
            "web": {
                "client_id": self.google_client_id,
                "client_secret": self.google_client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [self.google_redirect_uri],
            }
        }

    def validate_api(self) -> None:
        missing = []
        for label, value in (
            ("CLIENT_ID", self.google_client_id),
            ("GOOGLE_CLIENT_SECRET", self.google_client_secret),
            ("DB_URI", self.mongodb_uri),
            ("PINECONE", self.pinecone_api_key),
            ("PINECONE_INDEX_NAME", self.pinecone_index_name),
            ("NVAPI", self.nvidia_api_key),
            ("GEMINI", self.gemini_api_key),
            ("SECRET_KEY", self.secret_key),
        ):
            if not value:
                missing.append(label)
        if missing:
            raise RuntimeError("Missing required configuration: " + ", ".join(missing))
        if self.email_fetch_workers < 1:
            raise RuntimeError("EMAIL_FETCH_WORKERS must be at least 1")
        if self.embedding_batch_size < 1 or self.pinecone_batch_size < 1:
            raise RuntimeError("Embedding and Pinecone batch sizes must be at least 1")


settings = Settings()
