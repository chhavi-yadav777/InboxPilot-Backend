import base64
import hashlib
from urllib.parse import parse_qs, urlparse

from google_auth_oauthlib.flow import Flow


CLIENT_CONFIG = {
    "web": {
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost:8000/auth/google/callback"],
    }
}


def test_callback_flow_reuses_pkce_verifier_from_authorization_flow():
    scopes = ["openid", "https://www.googleapis.com/auth/gmail.readonly"]
    start_flow = Flow.from_client_config(CLIENT_CONFIG, scopes=scopes)
    start_flow.redirect_uri = CLIENT_CONFIG["web"]["redirect_uris"][0]
    authorization_url, state = start_flow.authorization_url(access_type="offline")

    verifier = start_flow.code_verifier
    callback_flow = Flow.from_client_config(
        CLIENT_CONFIG,
        scopes=scopes,
        state=state,
        code_verifier=verifier,
    )
    callback_flow.redirect_uri = CLIENT_CONFIG["web"]["redirect_uris"][0]

    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    query = parse_qs(urlparse(authorization_url).query)
    assert verifier
    assert callback_flow.code_verifier == verifier
    assert query["code_challenge"] == [challenge]
    assert query["code_challenge_method"] == ["S256"]
