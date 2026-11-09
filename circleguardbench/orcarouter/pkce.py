"""OrcaRouter "Connect with OrcaRouter" -- OAuth 2.0 + PKCE.

Flow A (loopback redirect) is used: CircleGuardBench runs as a local CLI on a
machine that has a browser, can bind 127.0.0.1, and its install address is not
stable (the redirect goes to an ephemeral loopback port, so nothing has to be
pre-registered).  Flow B (out-of-band code) is offered as a fallback for
headless boxes and Flow C (device grant) for machines without any browser;
neither replaces PKCE for Flow B -- S256 is mandatory there.

PKCE invariants enforced here:

* a fresh verifier and state per attempt, from :mod:`secrets`;
* ``code_challenge = base64url(sha256(verifier))`` without padding, S256 only;
* the verifier never leaves the process except in the exchange body -- it is
  never logged, printed or placed in a URL;
* for Flow A ``state`` is compared in constant time before the code is used;
* every failure path (deny, state mismatch, timeout, exchange 4xx/5xx, network)
  raises :class:`OrcaRouterAuthError` with an actionable message.

Client secret: none.  Redirect URI pre-registration: none.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import logging
import secrets
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, List, Optional

from circleguardbench.orcarouter.credentials import (
    DEFAULT_AUTH_BASE,
    DEFAULT_API_BASE,
    MASK,
    Credential,
    CredentialScopeError,
    FileCredentialStore,
    OrcaRouterAuthError,
    check_granted_scope,
    validate_key_shape,
)

logger = logging.getLogger("guardbench.orcarouter.pkce")

APP_NAME = "CircleGuardBench"
AUTHORIZE_PATH = "/auth"
EXCHANGE_PATH = "/api/v1/auth/keys"
DEVICE_CODE_PATH = "/api/v1/auth/device/code"
DEVICE_TOKEN_PATH = "/api/v1/auth/device/token"
DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def new_verifier() -> str:
    # 32 bytes -> 43 chars of base64url, the RFC 7636 minimum length.
    return b64url(secrets.token_bytes(32))


def challenge_for(verifier: str) -> str:
    return b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def new_state() -> str:
    return b64url(secrets.token_bytes(16))


def states_match(expected: str, received: Optional[str]) -> bool:
    if not expected or not received:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), received.encode("utf-8"))


def _normalise_origin(url: str, name: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"{name} must be an absolute URL, got {url!r}")
    if parsed.scheme == "https":
        pass
    elif parsed.scheme == "http":
        host = (parsed.hostname or "").lower()
        if host not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError(f"{name} must use HTTPS (HTTP is allowed only for loopback): {url!r}")
    else:
        raise ValueError(f"{name} must use http(s): {url!r}")
    return url.rstrip("/")


@dataclass(frozen=True)
class OrcaEndpoints:
    """Public origins.

    Authentication lives on ``auth_base`` (``/auth`` and ``/api/v1/auth/...``);
    inference and the model catalog live on ``api_base`` (``/v1``).  They are
    separate origins and are never derived from one another.
    """

    auth_base: str = DEFAULT_AUTH_BASE
    api_base: str = DEFAULT_API_BASE

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "OrcaEndpoints":
        import os

        env = os.environ if env is None else env
        shared = env.get("ORCA_BASE_URL")
        auth = env.get("ORCA_AUTH_BASE_URL") or shared or DEFAULT_AUTH_BASE
        api = env.get("ORCA_API_BASE_URL") or shared or DEFAULT_API_BASE
        return cls(auth_base=_normalise_origin(auth, "ORCA_AUTH_BASE_URL"),
                   api_base=_normalise_origin(api, "ORCA_API_BASE_URL"))

    @property
    def authorize_url(self) -> str:
        return f"{self.auth_base}{AUTHORIZE_PATH}"

    @property
    def exchange_url(self) -> str:
        # NOTE: auth origin, /api/v1/auth -- NOT api_base + /auth/keys.
        return f"{self.auth_base}{EXCHANGE_PATH}"

    @property
    def device_code_url(self) -> str:
        return f"{self.auth_base}{DEVICE_CODE_PATH}"

    @property
    def device_token_url(self) -> str:
        return f"{self.auth_base}{DEVICE_TOKEN_PATH}"

    @property
    def models_url(self) -> str:
        return f"{self.api_base}/models"


@dataclass
class PkceRequest:
    verifier: str
    state: str
    challenge: str
    callback_url: str
    authorize_url: str
    created_at: float = field(default_factory=time.monotonic)

    def __repr__(self) -> str:  # never leak the verifier
        return f"PkceRequest(callback_url={self.callback_url!r}, state=<redacted>, verifier=<redacted>)"

    __str__ = __repr__


def build_authorization_url(
    endpoints: OrcaEndpoints,
    challenge: str,
    state: str,
    callback_url: str,
    app_name: str = APP_NAME,
    scope: str = "api",
) -> str:
    params = {
        "callback_url": callback_url,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "app_name": app_name,
        "scope": scope,
    }
    return f"{endpoints.authorize_url}?{urllib.parse.urlencode(params)}"


def start_authorization(
    endpoints: OrcaEndpoints,
    callback_url: Optional[str],
    app_name: str = APP_NAME,
) -> PkceRequest:
    """Create a pending PKCE attempt. When ``callback_url`` is None the
    out-of-band flow (``callback_url=oob``) is used."""
    verifier = new_verifier()
    state = new_state()
    challenge = challenge_for(verifier)
    cb = callback_url or "oob"
    url = build_authorization_url(endpoints, challenge, state, cb, app_name=app_name)
    return PkceRequest(verifier=verifier, state=state, challenge=challenge,
                       callback_url=cb, authorize_url=url)


# --------------------------------------------------------------------------
# HTTP helpers (stdlib; tests point endpoints at a local fake auth server)
# --------------------------------------------------------------------------


def _post_json(url: str, body: Dict, timeout: float = 30.0) -> Dict:
    data = urllib.parse.urlencode(body).encode("ascii")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as exc:
        payload = exc.read() or b""
        raise _exchange_error(exc.code, payload) from None
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise OrcaRouterAuthError(
            f"Could not reach OrcaRouter at {url}: {exc}. Check your network and retry."
        ) from None
    import json

    try:
        return json.loads(payload.decode("utf-8"))
    except ValueError:
        raise OrcaRouterAuthError("OrcaRouter returned a non-JSON response.") from None


def _exchange_error(status: int, payload: bytes) -> OrcaRouterAuthError:
    detail = ""
    try:
        import json

        obj = json.loads(payload.decode("utf-8"))
        detail = str(obj.get("error_description") or obj.get("error") or "")[:300]
    except ValueError:
        pass
    if status == 400:
        return OrcaRouterAuthError(
            "OrcaRouter rejected the PKCE exchange (400): code challenge method "
            f"mismatch or invalid request. {detail}".strip()
        )
    if status == 403:
        return OrcaRouterAuthError(
            "OrcaRouter rejected the code (403): it was denied, expired (>10 min), "
            f"already used, or the verifier did not match. {detail}".strip()
        )
    if status == 429:
        return OrcaRouterAuthError(
            "OrcaRouter rate limited the authorization (429). Wait a minute and retry."
        )
    return OrcaRouterAuthError(f"OrcaRouter exchange failed with HTTP {status}. {detail}".strip())


def exchange_code(
    endpoints: OrcaEndpoints,
    code: str,
    verifier: str,
    store: FileCredentialStore,
    scope: str = "api",
    timeout: float = 30.0,
) -> Credential:
    """Redeem an auth code. The verifier is only ever sent in this body."""
    code = (code or "").strip()
    if not code:
        raise OrcaRouterAuthError("Empty authorization code.")
    data = _post_json(
        endpoints.exchange_url,
        {"code": code, "code_verifier": verifier, "code_challenge_method": "S256"},
        timeout=timeout,
    )
    key = data.get("key")
    if not key:
        raise OrcaRouterAuthError("OrcaRouter exchange succeeded but returned no key.")
    validate_key_shape(key)
    granted = data.get("scope")
    # Fail closed on a downgraded grant.  We request `api`; anything that does
    # not include it is reported instead of assumed.
    # A grant narrower than `api` does not cover inference.
    if scope == "api":
        check_granted_scope(granted)
    elif "api" not in (granted or "").split():
        raise CredentialScopeError(
            f"OrcaRouter granted scope {granted!r}, which does not cover inference."
        )
    return persist_key(store, key, granted, data.get("user_id"), "pkce", endpoints.auth_base)


def persist_key(
    store: FileCredentialStore,
    key: str,
    granted_scope: Optional[str],
    user_id: Optional[str],
    grant: str,
    auth_base: str,
) -> Credential:
    """Store a freshly obtained key and return the Credential for it."""
    generation = uuid.uuid4().hex
    from circleguardbench.orcarouter.credentials import _now

    store.save({
        "version": 1,
        "generation": generation,
        "key": key,
        # How this key reaches OrcaRouter: "pkce" keys are backed by an OAuth
        # grant that can be revoked from the console; "api" keys are an
        # OrcaRouter account key with the same lifecycle and inference scope.
        "grant": grant,
        "scope": granted_scope,
        "user_id": user_id,
        "status": "active",
        "created_at": _now(),
        "updated_at": _now(),
        "auth_base": auth_base,
    })
    logger.info("Stored OrcaRouter credential from %s login (key %s).", grant, MASK)
    return Credential(api_key=key, source="pkce", generation=generation)


# --------------------------------------------------------------------------
# Flow A -- loopback redirect
# --------------------------------------------------------------------------

_DONE_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>CircleGuardBench</title></head>
<body style="font-family:sans-serif;padding:3em"><h2>{title}</h2><p>{message}</p></body></html>"""


class LoopbackListener:
    """Binds 127.0.0.1:<ephemeral>/cb before the browser opens."""

    def __init__(self, expected_state: str, host: str = "127.0.0.1"):
        self.expected_state = expected_state
        self.host = host
        self.code: Optional[str] = None
        self.error: Optional[str] = None
        self._done = threading.Event()
        listener = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):  # keep server logs quiet
                pass

            def do_GET(self):  # noqa: N802
                url = urllib.parse.urlparse(self.path)
                if url.path != "/cb":
                    self.send_response(404)
                    self.end_headers()
                    return
                params = urllib.parse.parse_qs(url.query)
                state = (params.get("state") or [None])[0]
                if not states_match(listener.expected_state, state):
                    listener.error = "state mismatch"
                    body = _DONE_PAGE.format(title="Connection failed",
                                             message="State mismatch. Close this tab and retry.")
                    self.send_response(400)
                else:
                    err = (params.get("error") or [None])[0]
                    if err:
                        listener.error = err
                        body = _DONE_PAGE.format(title="OrcaRouter authorization denied",
                                                 message=html.escape(err) + ". You can close this tab.")
                    else:
                        listener.code = (params.get("code") or [None])[0]
                        body = _DONE_PAGE.format(title="Connected to OrcaRouter",
                                                 message="You can close this tab and return to the terminal.")
                    self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))
                listener._done.set()

        self._server = ThreadingHTTPServer((host, 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def callback_url(self) -> str:
        host = "[::1]" if self.host == "::1" else self.host
        return f"http://{host}:{self.port}/cb"

    def __enter__(self) -> "LoopbackListener":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._done.set()

    def wait(self, timeout: float) -> str:
        """Block until the browser hits /cb; returns the code or raises."""
        deadline = time.monotonic() + timeout
        while not self._done.wait(0.25):
            if time.monotonic() >= deadline:
                raise OrcaRouterAuthError(
                    "Timed out waiting for the OrcaRouter browser authorization. "
                    "Run `guardbench orcarouter login` again."
                )
        if self.error == "state mismatch":
            raise OrcaRouterAuthError(
                "Authorization response failed the state check (possible CSRF); "
                "nothing was stored. Retry `guardbench orcarouter login`."
            )
        if self.error:
            raise OrcaRouterAuthError(f"OrcaRouter authorization denied: {self.error}.")
        if not self.code:
            raise OrcaRouterAuthError("The browser returned no authorization code.")
        return self.code


def _open_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:  # pragma: no cover - browserless boxes
        pass


def login_with_loopback(
    endpoints: OrcaEndpoints,
    store: FileCredentialStore,
    open_browser: Callable[[str], None] = _open_browser,
    timeout: float = 180.0,
    app_name: str = APP_NAME,
    out: Callable[[str], None] = print,
    host: str = "127.0.0.1",
) -> Credential:
    """Flow A end-to-end: listener -> browser -> callback -> exchange -> persist."""
    state = new_state()
    with LoopbackListener(state, host=host) as listener:
        req = start_authorization(endpoints, listener.callback_url, app_name=app_name)
        # The request's own verifier must be bound to the listener state.
        listener.expected_state = req.state
        out("Opening your browser to authorize CircleGuardBench with OrcaRouter…")
        out(f"If it does not open, visit:\n  {req.authorize_url}")
        try:
            open_browser(req.authorize_url)
            code = listener.wait(timeout)
        finally:
            listener.close()
    return exchange_code(endpoints, code, req.verifier, store, scope="api")


def login_with_oob(
    endpoints: OrcaEndpoints,
    store: FileCredentialStore,
    open_browser: Callable[[str], None] = _open_browser,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
    app_name: str = APP_NAME,
) -> Credential:
    """Flow B: user pastes the displayed code. S256 is mandatory."""
    req = start_authorization(endpoints, None, app_name=app_name)
    out("Open this URL in a browser and approve access to CircleGuardBench:")
    out(f"  {req.authorize_url}")
    open_browser(req.authorize_url)
    code = input_fn("Paste the authorization code: ")
    return exchange_code(endpoints, code, req.verifier, store, scope="api")


# --------------------------------------------------------------------------
# Flow C -- device grant (extra convenience for browserless machines; it is
# NOT a substitute for PKCE and is not used by `connect` by default).
# --------------------------------------------------------------------------


def login_with_device_grant(
    endpoints: OrcaEndpoints,
    store: FileCredentialStore,
    out: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    app_name: str = APP_NAME,
) -> Credential:
    start = _post_json(endpoints.device_code_url, {"app_name": app_name, "scope": "api"})
    complete = start.get("verification_uri_complete")
    if complete:
        out(f"Open this on any device:\n  {complete}")
    else:
        out(f"Open {start.get('verification_uri')} and enter code: {start.get('user_code')}")
    interval = float(start.get("interval") or 5)
    deadline = time.monotonic() + float(start.get("expires_in") or 600)
    while time.monotonic() < deadline:
        sleep(interval)
        try:
            tok = _post_json(endpoints.device_token_url, {
                "device_code": start.get("device_code"),
                "grant_type": DEVICE_GRANT_TYPE,
            })
        except OrcaRouterAuthError as exc:
            # "authorization_pending" / "slow_down" arrive as HTTP errors in some
            # deployments; anything else is terminal.
            msg = str(exc)
            if "authorization_pending" in msg:
                continue
            if "slow_down" in msg:
                interval += 5
                continue
            raise
        err = tok.get("error")
        if err in (None, ""):
            key = validate_key_shape(tok.get("key") or "")
            check_granted_scope(tok.get("scope"))
            return persist_key(store, key, tok.get("scope"), tok.get("user_id"), "device",
                               endpoints.auth_base)
        if err == "authorization_pending":
            continue
        if err == "slow_down":
            interval += 5
            continue
        raise OrcaRouterAuthError(f"Device authorization failed: {err}.")
    raise OrcaRouterAuthError("Device authorization expired. Start a new login.")
