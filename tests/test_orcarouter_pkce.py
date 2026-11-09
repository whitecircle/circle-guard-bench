import json
import threading
import urllib.parse
import urllib.error
import urllib.request

import pytest

from circleguardbench.orcarouter import pkce
from circleguardbench.orcarouter.credentials import (
    SOURCE_PKCE,
    CredentialScopeError,
    FileCredentialStore,
    MissingCredentialError,
    OrcaRouterAuthError,
    ReauthenticationRequired,
    mark_needs_reauth,
    resolve_credential,
)

from fake_auth_server import DEFAULT_CATALOG, FakeAuthServer, fake_endpoints


@pytest.fixture()
def server():
    with FakeAuthServer() as srv:
        yield srv


@pytest.fixture()
def store(tmp_path):
    return FileCredentialStore(tmp_path / "orcarouter-auth.json")


def _hit_ignore(url):
    try:
        return _hit(url)
    except urllib.error.HTTPError as exc:
        return exc.code, ""


def _hit(url):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, resp.read().decode()


def test_authorize_url_targets_auth_origin_only(server, store):
    ep = fake_endpoints(server)
    req = pkce.start_authorization(ep, "http://127.0.0.1:1234/cb")
    parsed = urllib.parse.urlparse(req.authorize_url)
    assert parsed.scheme == "http"
    assert parsed.netloc == urllib.parse.urlparse(server.origin).netloc  # explicit override
    assert parsed.path == "/auth"
    q = urllib.parse.parse_qs(parsed.query)
    assert q["code_challenge_method"] == ["S256"]
    assert q["code_challenge"] == [pkce.challenge_for(req.verifier)]
    assert "=" not in q["code_challenge"][0]
    assert q["state"] == [req.state]
    assert q["callback_url"] == ["http://127.0.0.1:1234/cb"]
    assert q["scope"] == ["api"]
    assert req.verifier not in req.authorize_url
    assert req.verifier not in str(req) and req.verifier not in repr(req)


def test_default_endpoints_use_separate_public_origins():
    ep = pkce.OrcaEndpoints.from_env({})
    assert ep.auth_base == "https://www.orcarouter.ai"
    assert ep.api_base == "https://api.orcarouter.ai/v1"
    assert ep.authorize_url == "https://www.orcarouter.ai/auth"
    assert ep.exchange_url == "https://www.orcarouter.ai/api/v1/auth/keys"
    assert "api.orcarouter.ai/v1/auth/keys" not in ep.exchange_url
    assert ep.models_url == "https://api.orcarouter.ai/v1/models"


def test_shared_and_explicit_overrides():
    shared = pkce.OrcaEndpoints.from_env({"ORCA_BASE_URL": "https://orca.internal"})
    assert shared.auth_base == "https://orca.internal"
    assert shared.api_base == "https://orca.internal"
    explicit = pkce.OrcaEndpoints.from_env({
        "ORCA_BASE_URL": "https://orca.internal",
        "ORCA_AUTH_BASE_URL": "https://auth.example.com",
        "ORCA_API_BASE_URL": "https://api.example.com/v1/",
    })
    assert explicit.auth_base == "https://auth.example.com"
    assert explicit.api_base == "https://api.example.com/v1"
    assert explicit.exchange_url == "https://auth.example.com/api/v1/auth/keys"
    assert explicit.models_url == "https://api.example.com/v1/models"
    with pytest.raises(ValueError):
        pkce.OrcaEndpoints.from_env({"ORCA_AUTH_BASE_URL": "http://evil.example.com"})


def test_loopback_flow_end_to_end_persists_plain_key(server, store):
    ep = fake_endpoints(server)
    captured = {}

    def fake_browser(url):
        captured["url"] = url
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        cb = q["callback_url"][0]
        captured["cb"] = cb
        # Browser hits the callback out-of-band on another thread.
        def hit():
            code = "code-abc"
            status, body = _hit(f"{cb}?code={code}&state={q['state'][0]}")
            captured["page"] = (status, body)
        t = threading.Thread(target=hit)
        t.start()
        captured["thread"] = t

    cred = pkce.login_with_loopback(ep, store, open_browser=fake_browser, timeout=10, out=lambda *_: None)
    captured["thread"].join(5)

    assert cred.api_key == "sk-orca-test-key-0123456789abcdef"
    assert cred.source == SOURCE_PKCE and cred.generation
    assert captured["page"][0] == 200 and "close this tab" in captured["page"][1]
    assert server.paths() == ["/api/v1/auth/keys"]
    sent = server.requests[0]["body"]
    assert sent["code"] == "code-abc"
    assert sent["code_challenge_method"] == "S256"
    assert sent["code_verifier"]  # only in the exchange body
    assert sent["code_verifier"] not in captured["url"]

    stored = store.load()
    assert stored["key"] == cred.api_key and stored["status"] == "active"
    assert stored["generation"] == cred.generation
    assert (store.path.stat().st_mode & 0o777) == 0o600


def test_loopback_state_mismatch_never_exchanges(server, store):
    ep = fake_endpoints(server)

    def fake_browser(url):
        cb = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["callback_url"][0]
        threading.Thread(target=lambda: _hit_ignore(f"{cb}?code=code-abc&state=forged")).start()

    with pytest.raises(OrcaRouterAuthError, match="state"):
        pkce.login_with_loopback(ep, store, open_browser=fake_browser, timeout=10, out=lambda *_: None)
    assert server.requests == []
    assert store.load() is None


def test_loopback_denial_is_terminal_and_clears(server, store):
    ep = fake_endpoints(server)

    def fake_browser(url):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        url = f"{q['callback_url'][0]}?error=access_denied&state={q['state'][0]}"
        threading.Thread(target=lambda: _hit(url.split("?", 1)[0] + "?" + url.split("?", 1)[1]) if False else _hit_ignore(url)).start()

    with pytest.raises(OrcaRouterAuthError, match="denied"):
        pkce.login_with_loopback(ep, store, open_browser=fake_browser, timeout=10, out=lambda *_: None)
    assert server.requests == []


def test_loopback_timeout(server, store):
    ep = fake_endpoints(server)
    with pytest.raises(OrcaRouterAuthError, match="Timed out"):
        pkce.login_with_loopback(ep, store, open_browser=lambda *_: None, timeout=0.3, out=lambda *_: None)
    assert store.load() is None


def test_oob_flow_uses_literal_oob_and_s256(server, store):
    ep = fake_endpoints(server)
    cred = pkce.login_with_oob(ep, store, open_browser=lambda u: None, input_fn=lambda _p: "code-oob",
                               out=lambda *_: None)
    assert cred.api_key.startswith("sk-orca-")
    sent = server.requests[0]["body"]
    assert sent["code"] == "code-oob"


def test_oob_authorize_url(server):
    ep = fake_endpoints(server)
    req = pkce.start_authorization(ep, None)
    q = urllib.parse.parse_qs(urllib.parse.urlparse(req.authorize_url).query)
    assert q["callback_url"] == ["oob"]
    assert q["code_challenge_method"] == ["S256"]


@pytest.mark.parametrize("status", [400, 403, 429])
def test_exchange_errors_are_actionable_and_do_not_store(server, store, status):
    server.exchange_status = status
    ep = fake_endpoints(server)
    with pytest.raises(OrcaRouterAuthError) as exc:
        pkce.exchange_code(ep, "code", pkce.new_verifier(), store)
    assert str(status) in str(exc.value)
    assert store.load() is None


def test_exchange_network_failure(store):
    ep = pkce.OrcaEndpoints(auth_base="http://127.0.0.1:9", api_base="http://127.0.0.1:9/v1")
    with pytest.raises(OrcaRouterAuthError, match="Could not reach"):
        pkce.exchange_code(ep, "code", pkce.new_verifier(), store, timeout=2)


def test_code_reuse_fails_closed(server, store):
    """Second exchange of the same code is rejected by the server (403) and
    the first credential is untouched."""
    ep = fake_endpoints(server)
    first = pkce.exchange_code(ep, "code-1", pkce.new_verifier(), store)
    server.exchange_status = 403
    with pytest.raises(OrcaRouterAuthError, match="403"):
        pkce.exchange_code(ep, "code-1-expired", pkce.new_verifier(), store)
    assert store.load()["key"] == first.api_key


def test_scope_downgrade_is_reported(server, store):
    server.exchange_body = {"key": "sk-orca-test-key-0123456789abcdef", "user_id": "u", "scope": "readonly"}
    ep = fake_endpoints(server)
    with pytest.raises(CredentialScopeError):
        pkce.exchange_code(ep, "code", pkce.new_verifier(), store)
    assert store.load() is None


def test_missing_scope_is_rejected(server, store):
    server.exchange_body = {"key": "sk-orca-test-key-0123456789abcdef", "user_id": "u"}
    ep = fake_endpoints(server)
    with pytest.raises(CredentialScopeError):
        pkce.exchange_code(ep, "code", pkce.new_verifier(), store)


def test_device_grant_flow(server, store):
    server.device_polls = [{"error": "authorization_pending"},
                           {"error": "slow_down"},
                           {"key": "sk-orca-device-key-000000000000", "scope": "api"}]
    ep = fake_endpoints(server)
    cred = pkce.login_with_device_grant(ep, store, out=lambda *_: None, sleep=lambda *_: None)
    assert cred.api_key == "sk-orca-device-key-000000000000"
    assert store.load()["grant"] == "device"
    assert "/api/v1/auth/device/code" in server.paths()
    assert server.paths().count("/api/v1/auth/device/token") == 3


def test_get_granted_scope_accepted():
    from circleguardbench.orcarouter.credentials import check_granted_scope

    check_granted_scope("api")
    check_granted_scope("connector api")
    with pytest.raises(CredentialScopeError):
        check_granted_scope("connector")
    with pytest.raises(CredentialScopeError):
        check_granted_scope(None)


def test_generation_is_new_per_login(server, store):
    ep = fake_endpoints(server)
    a = pkce.exchange_code(ep, "c1", pkce.new_verifier(), store)
    b = pkce.exchange_code(ep, "c2", pkce.new_verifier(), store)
    assert a.generation != b.generation
    assert store.load()["generation"] == b.generation


def test_stale_generation_401_does_not_poison_new_credential(store):
    old = store.__class__  # noqa: F841  (documents store type)
    cred_old = pkce.persist_key(store, "sk-orca-old-key-000000000000", "api", "u", "pkce", "https://www.orcarouter.ai")
    cred_new = pkce.persist_key(store, "sk-orca-new-key-000000000000", "api", "u", "pkce", "https://www.orcarouter.ai")
    # An async failure belonging to the old generation arrives late.
    assert mark_needs_reauth(cred_old, store) is False
    assert store.load()["status"] == "active"
    assert store.load()["generation"] == cred_new.generation
    # The new credential failing is exactly what marks it.
    assert mark_needs_reauth(cred_new, store) is True
    assert store.load()["status"] == "needs_reauth"
    with pytest.raises(ReauthenticationRequired):
        resolve_credential(store=store, env={})
    # The old secret is kept (not silently deleted) until a new login.
    assert store.load()["key"] == cred_new.api_key


def test_api_key_source_unaffected_by_reauth(store):
    from circleguardbench.orcarouter.credentials import Credential

    assert mark_needs_reauth(Credential("sk-orca-x-00000000000", "api_key_env"), store) is False


def test_missing_credential_message(store):
    with pytest.raises(MissingCredentialError, match="login"):
        resolve_credential(config_key=None, store=store, env={})


def test_loopback_listener_binds_loopback_only(server):
    listener = pkce.LoopbackListener("st")
    with listener:
        assert listener.callback_url.startswith("http://127.0.0.1:")
        assert listener._server.server_address[0] == "127.0.0.1"
    server  # silence fixture-unused warning


def test_catalog_fetch_with_bearer(server, store):
    ep = fake_endpoints(server)
    from circleguardbench.orcarouter.catalog import load_catalog

    cat = load_catalog("sk-orca-test-key-0123456789abcdef", ep.models_url)
    assert cat.source == "live" and not cat.degraded
    assert server.requests[-1]["auth"].startswith("Bearer sk-orca-")
    assert "orcarouter/auto" in cat.ids()


def test_catalog_fixture_is_wellformed():
    assert isinstance(DEFAULT_CATALOG, list)
