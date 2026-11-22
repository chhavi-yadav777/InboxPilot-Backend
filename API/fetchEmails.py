"""Parallel Gmail message retrieval and Gmail payload normalization."""

from __future__ import annotations

import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from API.config import settings


_worker_state = threading.local()


class _TextFromHtml(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _init_worker(credentials_info: dict) -> None:
    credentials_info.setdefault("refresh_token", None)
    credentials = Credentials.from_authorized_user_info(credentials_info)
    _worker_state.gmail = build("gmail", "v1", credentials=credentials, cache_discovery=False)


def _header(headers: list[dict], name: str) -> str:
    name = name.casefold()
    for item in headers:
        if item.get("name", "").casefold() == name:
            return item.get("value", "")
    return ""


def _decode_body(data: str) -> str:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8", errors="replace")


def _body_from_part(part: dict) -> tuple[str, bool]:
    mime_type = part.get("mimeType", "")
    data = part.get("body", {}).get("data")
    if data:
        decoded = _decode_body(data)
        if mime_type == "text/plain":
            return decoded, True
        if mime_type == "text/html":
            parser = _TextFromHtml()
            parser.feed(decoded)
            return " ".join(parser.parts), False

    plain_fallback = ""
    html_fallback = ""
    for child in part.get("parts", []):
        text, is_plain = _body_from_part(child)
        if text and is_plain:
            plain_fallback += ("\n" if plain_fallback else "") + text
        elif text:
            html_fallback += ("\n" if html_fallback else "") + text
    if plain_fallback:
        return plain_fallback, True
    return html_fallback, False


def _message_date(internal_date_ms: int, headers: list[dict]) -> str:
    if internal_date_ms:
        return datetime.fromtimestamp(internal_date_ms / 1000, timezone.utc).date().isoformat()
    raw = _header(headers, "Date")
    if raw:
        try:
            parsed = parsedate_to_datetime(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.date().isoformat()
        except (TypeError, ValueError, OverflowError):
            pass
    return datetime.now(timezone.utc).date().isoformat()


def _fetch_one(message_ref: dict) -> dict:
    raw = (
        _worker_state.gmail.users()
        .messages()
        .get(userId="me", id=message_ref["id"], format="full")
        .execute()
    )
    payload = raw.get("payload", {})
    headers = payload.get("headers", [])
    body, _ = _body_from_part(payload)
    internal_date_ms = int(raw.get("internalDate") or 0)
    return {
        "email_id": raw["id"],
        "thread_id": raw.get("threadId", message_ref.get("threadId", "")),
        "subject": _header(headers, "Subject"),
        "sender": _header(headers, "From"),
        "receiver": _header(headers, "To"),
        "date": _message_date(internal_date_ms, headers),
        "body": body,
        "internal_date_ms": internal_date_ms,
    }


def fetch_gmail_messages(credentials: Credentials, checkpoint_ms: int | None = None) -> list[dict]:
    """List mailbox message IDs, then fetch message payloads concurrently."""
    service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
    query = None
    if checkpoint_ms:
        after_seconds = max(0, checkpoint_ms // 1000 - settings.sync_overlap_seconds)
        query = f"after:{after_seconds}"

    message_refs: list[dict] = []
    page_token = None
    while True:
        response = (
            service.users()
            .messages()
            .list(
                userId="me",
                maxResults=500,
                pageToken=page_token,
                q=query,
                includeSpamTrash=False,
            )
            .execute()
        )
        message_refs.extend(response.get("messages", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    if not message_refs:
        return []

    credentials_info = credentials.to_json()
    fetched: list[dict] = []
    with ThreadPoolExecutor(
        max_workers=min(settings.email_fetch_workers, len(message_refs)),
        initializer=_init_worker,
        initargs=(json.loads(credentials_info),),
        thread_name_prefix="gmail-fetch",
    ) as executor:
        futures = [executor.submit(_fetch_one, ref) for ref in message_refs]
        for future in as_completed(futures):
            fetched.append(future.result())
    return fetched
