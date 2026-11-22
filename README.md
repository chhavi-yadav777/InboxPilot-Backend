# InboxPilot Gmail API

FastAPI backend for Google sign-in, Gmail ingestion, MongoDB storage, Pinecone indexing, and email Q&A.

## Setup

1. Install dependencies with `python -m pip install -r requirements.txt`.
2. Copy `.env.example` to `.env` and fill in the Google, MongoDB, Pinecone, NVIDIA, Gemini, and JWT settings. Existing `.env` values can use the legacy names shown in the example (`CLIENT_ID`, `DB_URI`, `PINECONE`, `NVAPI`, and `GEMINI`).
3. Set `PINECONE_INDEX_NAME` to an existing index with dimension 2048 or larger. If it is larger, the API pads the 2048-dimensional NVIDIA embeddings with zeros before upsert.
4. Add `http://127.0.0.1:8000/auth/google/callback` to the authorized redirect URIs for the Google OAuth web client. The backend redirects successful sign-ins to `FRONTEND_URL` (local default: `http://127.0.0.1:5173/`). Enable the Gmail API for that Google project.
5. Start the API from the repository root:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn API.index:app --host 0.0.0.0 --port 8000
   ```

The API checks MongoDB connectivity and the Pinecone index at startup. Set a long random `SECRET_KEY`; it signs API JWTs and derives the key used to encrypt Google OAuth credentials in MongoDB. Changing it invalidates existing JWTs and prevents decrypting previously stored Google credentials.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/auth/google` | Return a Google authorization URL. Open it in a browser to sign in. |
| `GET` | `/auth/google/callback` | Complete OAuth, set an HTTP-only session cookie, and redirect to `FRONTEND_URL`. |
| `POST` | `/auth/logout` | Clear the browser's HTTP-only session cookie. |
| `POST` | `/auth/logout` | Clear the browser's HTTP-only session cookie. |
| `GET` | `/users/me` | Return the authenticated user's profile and sync checkpoint. |
| `POST` | `/emails/fetch` | Fetch all available messages, store them in MongoDB, then index the user's stored messages. |
| `POST` | `/emails/sync` | Fetch after the saved checkpoint; if no checkpoint exists, perform an initial full fetch. |
| `POST` | `/emails/index` | Re-embed and upsert all of the user's MongoDB email records. |
| `GET` | `/emails?limit=50&offset=0` | List the authenticated user's stored emails. |
| `POST` | `/chat` | Search the user's Pinecone namespace, ask Gemini using the top matching emails, and save the turn in MongoDB. |
| `GET` | `/chats?limit=20` | List the user's recent conversations. |
| `GET` | `/chats/{conversation_id}` | Read saved turns and cited sources for one conversation. |
| `GET` | `/health` | Check MongoDB and Pinecone availability. |

Use the HTTP-only session cookie set by Google sign-in for browser requests, or `Authorization: Bearer <access_token>` for API clients. Browser requests must use `credentials: "include"`. Fetch and sync requests run synchronously and use `EMAIL_FETCH_WORKERS` threads for message detail requests. MongoDB records include internal user and Gmail timestamp fields; email list responses return the requested email fields. The checkpoint advances only after persistence and Pinecone indexing complete successfully.

Send chat requests as JSON, for example `{"question":"When is my interview?","top_k":5}`. Reuse the returned `conversation_id` on subsequent turns; omit it to start a new conversation. The API returns the Gemini answer with source metadata and persists each question/answer turn in `IP_chats`. Email evidence is embedded with NVIDIA in query mode, fetched only from the authenticated user's Pinecone namespace, and sent to the configured Gemini model (`GEMINI_MODEL`, default `gemini-3.8-flash`).

Interactive API docs are available at `/docs`.

For browser application setup, OAuth callback handling, API examples, and chat integration, see [readme-frontend.md](readme-frontend.md).
