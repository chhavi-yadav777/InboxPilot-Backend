"""FastAPI application for Google sign-in and Gmail ingestion."""

from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import AuthorizedSession
from pydantic import BaseModel, Field

from API.config import settings
from API.chat_service import answer_email_question
from API.ingestion import email_response, ingest_mailbox, reindex_mailbox
from API.security import (
    AUTH_COOKIE_NAME,
    create_access_token,
    encrypt_oauth_credentials,
    get_current_user,
    public_user,
)
from API.vector_store import validate_pinecone_index
from DB.dbClient import (
    ensure_indexes,
    get_emails_collection,
    get_chats_collection,
    get_users_collection,
    ping_database,
)


logger = logging.getLogger("inboxpilot.api")
OAUTH_STATE_COOKIE = "inboxpilot_oauth_state"
OAUTH_VERIFIER_COOKIE = "inboxpilot_oauth_code_verifier"
OAUTH_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.readonly",
]


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.validate_api()
    ping_database()
    ensure_indexes()
    validate_pinecone_index()
    yield


app = FastAPI(
    title="InboxPilot Gmail API",
    version="1.0.0",
    description="Google OAuth, Gmail ingestion, MongoDB storage, and Pinecone indexing.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OAuthStartResponse(BaseModel):
    authorization_url: str


class IngestionResponse(BaseModel):
    fetched: int
    stored: int
    indexed: int
    checkpoint_timestamp: datetime | None = None


class ReindexResponse(BaseModel):
    indexed: int


class EmailListResponse(BaseModel):
    items: list[dict]
    limit: int
    offset: int
    total: int


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=100)
    top_k: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[dict]


def _oauth_flow(state: str | None = None, code_verifier: str | None = None) -> Flow:
    flow = Flow.from_client_config(
        settings.oauth_client_config,
        scopes=OAUTH_SCOPES,
        state=state,
        code_verifier=code_verifier,
    )
    flow.redirect_uri = settings.google_redirect_uri
    return flow


def _oauth_error_code(exc: Exception) -> str | None:
    """Extract only Google's non-sensitive OAuth error identifier."""
    code = getattr(exc, "error", None)
    if isinstance(code, str):
        return code
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            body = response.json()
            code = body.get("error") if isinstance(body, dict) else None
            if isinstance(code, str):
                return code
        except (ValueError, AttributeError):
            pass
    return None


@app.get("/auth/google", response_model=OAuthStartResponse, tags=["auth"])
def google_login(response: Response) -> OAuthStartResponse:
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(status_code=503, detail="Google OAuth is not configured")
    flow = _oauth_flow()
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    response.set_cookie(
        OAUTH_STATE_COOKIE,
        state,
        httponly=True,
        secure=settings.google_redirect_uri.startswith("https://"),
        samesite="lax",
        max_age=600,
        path="/",
    )
    if not flow.code_verifier:
        raise HTTPException(status_code=500, detail="OAuth PKCE verifier could not be created")
    response.set_cookie(
        OAUTH_VERIFIER_COOKIE,
        flow.code_verifier,
        httponly=True,
        secure=settings.google_redirect_uri.startswith("https://"),
        samesite="lax",
        max_age=600,
        path="/",
    )
    return OAuthStartResponse(authorization_url=authorization_url)


@app.get("/auth/google/callback", tags=["auth"])
def google_callback(
    request: Request,
    code: str,
    state: str,
    error: str | None = None,
) -> RedirectResponse:
    if error:
        raise HTTPException(status_code=400, detail="Google authorization was not completed")
    cookie_state = request.cookies.get(OAUTH_STATE_COOKIE, "")
    if not cookie_state:
        raise HTTPException(
            status_code=400,
            detail=(
                "OAuth state cookie is missing. Restart sign-in from GET /auth/google; "
                "frontend fetch calls must use credentials: 'include'."
            ),
        )
    if not hmac.compare_digest(cookie_state, state):
        raise HTTPException(
            status_code=400,
            detail="OAuth state does not match this browser session. Restart sign-in and use the newest authorization URL.",
        )
    code_verifier = request.cookies.get(OAUTH_VERIFIER_COOKIE, "")
    if not code_verifier:
        raise HTTPException(
            status_code=400,
            detail="OAuth PKCE verifier cookie is missing. Restart sign-in from GET /auth/google.",
        )

    flow = _oauth_flow(state=state, code_verifier=code_verifier)
    try:
        flow.fetch_token(code=code)
    except Exception as exc:
        # Never log exception text, as HTTP libraries may include request details.
        error_code = _oauth_error_code(exc)
        logger.warning(
            "Google OAuth token exchange failed (type=%s, error=%s)",
            type(exc).__name__,
            error_code or "unknown",
        )
        details = {
            "invalid_client": (
                "Google rejected the OAuth client credentials. Ensure GOOGLE_CLIENT_SECRET "
                "is from the same OAuth client as CLIENT_ID, then restart the API."
            ),
            "redirect_uri_mismatch": (
                "Google rejected GOOGLE_REDIRECT_URI. It must exactly match an authorized "
                "redirect URI on this OAuth client."
            ),
            "invalid_grant": (
                "Google rejected this authorization code. It may be expired or already used, or "
                "the code may have been issued for a different client or redirect URI. Restart the API, "
                "start a new sign-in, and let the callback run once without reloading it. Verify that "
                "the authorization URL's client_id and redirect_uri match the current OAuth client settings."
            ),
            "unauthorized_client": (
                "This Google OAuth client is not authorized for the requested flow. "
                "Check that it is a Web application client."
            ),
        }
        detail = details.get(
            error_code,
            "Google rejected the authorization code. Start a fresh sign-in and verify the OAuth client settings.",
        )
        raise HTTPException(
            status_code=502,
            detail=detail,
        ) from exc

    credentials = flow.credentials
    try:
        profile_response = AuthorizedSession(credentials).get(
            "https://openidconnect.googleapis.com/v1/userinfo", timeout=15
        )
        profile_response.raise_for_status()
        profile = profile_response.json()
    except Exception as exc:
        logger.warning("Google user profile request failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail="Google authorized the sign-in, but the user profile request failed. Retry sign-in.",
        ) from exc

    google_sub = profile.get("sub")
    if not google_sub or not profile.get("email"):
        raise HTTPException(status_code=502, detail="Google did not return a usable user profile")

    users = get_users_collection()
    existing = users.find_one({"google_sub": google_sub})
    # Google may omit a refresh token when this account previously granted access.
    if not credentials.refresh_token and existing:
        from API.security import decrypt_oauth_credentials

        previous = decrypt_oauth_credentials(existing["oauth_credentials"])
        credentials.refresh_token = previous.get("refresh_token")
    now = datetime.now(timezone.utc)
    user_document = {
        "google_sub": google_sub,
        "email": profile.get("email"),
        "name": profile.get("name"),
        "picture": profile.get("picture"),
        "email_verified": profile.get("email_verified", False),
        "oauth_credentials": encrypt_oauth_credentials(credentials),
        "created_at": existing.get("created_at", now) if existing else now,
        "updated_at": now,
        "checkpoint_ms": existing.get("checkpoint_ms") if existing else None,
        "checkpoint_timestamp": existing.get("checkpoint_timestamp") if existing else None,
    }
    users.update_one({"google_sub": google_sub}, {"$set": user_document}, upsert=True)
    redirect = RedirectResponse(settings.frontend_url, status_code=303)
    redirect.set_cookie(
        AUTH_COOKIE_NAME,
        create_access_token(google_sub),
        httponly=True,
        secure=settings.google_redirect_uri.startswith("https://"),
        samesite="lax",
        max_age=settings.access_token_expire_minutes * 60,
        path="/",
    )
    redirect.delete_cookie(OAUTH_STATE_COOKIE, httponly=True, samesite="lax", path="/")
    redirect.delete_cookie(OAUTH_VERIFIER_COOKIE, httponly=True, samesite="lax", path="/")
    return redirect


@app.post("/auth/logout", status_code=204, tags=["auth"])
def logout(response: Response) -> Response:
    response.delete_cookie(
        AUTH_COOKIE_NAME,
        httponly=True,
        secure=settings.google_redirect_uri.startswith("https://"),
        samesite="lax",
        path="/",
    )
    response.status_code = 204
    return response


@app.get("/users/me", tags=["users"])
def get_me(user: dict = Depends(get_current_user)) -> dict:
    return public_user(user)


def _run_ingestion(user: dict, incremental: bool) -> IngestionResponse:
    try:
        return IngestionResponse(**ingest_mailbox(user, incremental=incremental))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Mailbox %s failed (%s)", "sync" if incremental else "fetch", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail="Mailbox ingestion failed; the checkpoint was not advanced. Retry the request.",
        ) from exc


@app.post("/emails/fetch", response_model=IngestionResponse, tags=["emails"])
def fetch_all_emails(user: dict = Depends(get_current_user)) -> IngestionResponse:
    """Fetch all available Gmail messages, persist them, then index MongoDB records."""
    return _run_ingestion(user, incremental=False)


@app.post("/emails/sync", response_model=IngestionResponse, tags=["emails"])
def sync_emails(user: dict = Depends(get_current_user)) -> IngestionResponse:
    """Fetch messages since the user's saved checkpoint (or do an initial full fetch)."""
    return _run_ingestion(user, incremental=True)


@app.post("/emails/index", response_model=ReindexResponse, tags=["emails"])
def index_stored_emails(user: dict = Depends(get_current_user)) -> ReindexResponse:
    """Embed and upsert all of this user's messages currently stored in MongoDB."""
    try:
        return ReindexResponse(**reindex_mailbox(user))
    except Exception as exc:
        logger.exception("Email reindex failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Email indexing failed; retry the request") from exc


@app.get("/emails", response_model=EmailListResponse, tags=["emails"])
def list_emails(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(get_current_user),
) -> EmailListResponse:
    collection = get_emails_collection()
    query = {"user_id": user["google_sub"]}
    cursor = (
        collection.find(query)
        .sort([("internal_date_ms", -1), ("email_id", 1)])
        .skip(offset)
        .limit(limit)
    )
    return EmailListResponse(
        items=[email_response(document) for document in cursor],
        limit=limit,
        offset=offset,
        total=collection.count_documents(query),
    )


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat_with_emails(
    payload: ChatRequest,
    user: dict = Depends(get_current_user),
) -> ChatResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be blank")
    try:
        result = answer_email_question(
            user_id=user["google_sub"],
            question=question,
            top_k=payload.top_k,
            conversation_id=payload.conversation_id,
        )
        return ChatResponse(**result)
    except Exception as exc:
        logger.exception("Email chat failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Could not answer from email search; retry the request") from exc


@app.get("/chats", tags=["chat"])
def list_chats(
    limit: int = Query(default=20, ge=1, le=100),
    user: dict = Depends(get_current_user),
) -> dict:
    conversations = list(
        get_chats_collection().aggregate(
            [
                {"$match": {"user_id": user["google_sub"]}},
                {"$sort": {"created_at": -1}},
                {
                    "$group": {
                        "_id": "$conversation_id",
                        "last_turn": {"$first": "$$ROOT"},
                        "turn_count": {"$sum": 1},
                    }
                },
                {"$sort": {"last_turn.created_at": -1}},
                {"$limit": limit},
                {
                    "$project": {
                        "_id": 0,
                        "conversation_id": "$_id",
                        "last_activity": "$last_turn.created_at",
                        "last_question": "$last_turn.question",
                        "turn_count": 1,
                    }
                },
            ]
        )
    )
    return {"items": conversations, "limit": limit}


@app.get("/chats/{conversation_id}", tags=["chat"])
def get_chat_history(
    conversation_id: str,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(get_current_user),
) -> dict:
    query = {"user_id": user["google_sub"], "conversation_id": conversation_id}
    turns = list(
        get_chats_collection()
        .find(query, {"_id": 0, "user_id": 0})
        .sort("created_at", 1)
        .skip(offset)
        .limit(limit)
    )
    return {"conversation_id": conversation_id, "items": turns, "limit": limit, "offset": offset}


@app.get("/health", tags=["health"])
def health() -> JSONResponse:
    try:
        ping_database()
        validate_pinecone_index()
    except Exception as exc:
        logger.warning("Health check failed (%s)", type(exc).__name__)
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    return JSONResponse(content={"status": "ok"})


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"name": "InboxPilot Gmail API", "docs": "/docs", "health": "/health"}
