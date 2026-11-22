"""OAuth token encryption, API JWTs, and authenticated-user dependencies."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone

import jwt
from cryptography.fernet import Fernet, InvalidToken
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials

from API.config import settings
from DB.dbClient import get_users_collection


bearer = HTTPBearer(auto_error=False)
AUTH_COOKIE_NAME = "inboxpilot_access_token"


def _fernet() -> Fernet:
    key = base64.urlsafe_b64encode(hashlib.sha256(settings.secret_key.encode("utf-8")).digest())
    return Fernet(key)


def encrypt_oauth_credentials(credentials: Credentials) -> str:
    return _fernet().encrypt(credentials.to_json().encode("utf-8")).decode("ascii")


def decrypt_oauth_credentials(encrypted: str) -> dict:
    try:
        info = json.loads(_fernet().decrypt(encrypted.encode("ascii")).decode("utf-8"))
        # google-auth's authorized-user loader requires this key even when Google
        # did not issue a refresh token (for example, during a re-consent flow).
        info.setdefault("refresh_token", None)
        return info
    except (InvalidToken, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored Google credentials could not be decrypted",
        ) from exc


def credentials_for_user(user: dict) -> Credentials:
    info = decrypt_oauth_credentials(user["oauth_credentials"])
    credentials = Credentials.from_authorized_user_info(info)
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(GoogleAuthRequest())
        get_users_collection().update_one(
            {"google_sub": user["google_sub"]},
            {"$set": {"oauth_credentials": encrypt_oauth_credentials(credentials)}},
        )
    if not credentials.valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google authorization expired; sign in again",
        )
    return credentials


def create_access_token(google_sub: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": google_sub,
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
        "type": "access",
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict:
    token = credentials.credentials if credentials else request.cookies.get(AUTH_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Bearer token required")
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "exp"]},
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired access token") from exc
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid access token")
    user = get_users_collection().find_one({"google_sub": payload["sub"]})
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return user


def public_user(user: dict) -> dict:
    return {
        "id": str(user.get("_id", "")),
        "google_sub": user["google_sub"],
        "email": user.get("email"),
        "name": user.get("name"),
        "picture": user.get("picture"),
        "checkpoint_timestamp": user.get("checkpoint_timestamp"),
        "created_at": user.get("created_at"),
        "updated_at": user.get("updated_at"),
    }
