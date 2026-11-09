import logging
import urllib.parse

import pytest

from circleguardbench.orcarouter import pkce
from circleguardbench.orcarouter.catalog import (
    CAPABILITY_EMBEDDING,
    CAPABILITY_IMAGE,
    CAPABILITY_INFERENCE,
    CAPABILITY_MODERATION,
    CAPABILITY_RERANK,
    CAPABILITY_VIDEO,
    DEFAULT_MODEL,
    GENERAL_FALLBACK_MODEL,
    RECOMMENDED_CYBER_MODELS,
    VERIFIED_SEED_CATALOG,
    CatalogError,
    load_catalog,
    parse_catalog,
)
from circleguardbench.orcarouter.credentials import (
    SOURCE_ENV,
    SOURCE_PKCE,
    ApiKeyCredentialSource,
    Credential,
    FileCredentialStore,
    ReauthenticationRequired,
    resolve_credential,
)

from fake_auth_server import DEFAULT_CATALOG, FakeAuthServer, fake_endpoints

FAKE_KEY = "sk-orca-unit-test-key-0123456789"


@pytest.fixture()
def server():
    with FakeAuthServer() as srv:
        yield srv


@pytest.fixture()
def store(tmp_path):
    return FileCredentialStore(tmp_path / "orcarouter-auth.json")


# --------------------------------------------------------------------------
# API Key adapter
# --------------------------------------------------------------------------


def test_api_key_adapter_from_env(store):
    cred = resolve_credential(config_key=None, store=store, env={"ORCAROUTER_API_KEY": FAKE_KEY})
    assert isinstance(cred, Credential)
    assert cred.api_key == FAKE_KEY and cred.source == SOURCE_ENV


def test_api_key_save_read_clear_and_mask(store):
    saved = pkce.persist_key(store, FAKE_KEY, "api", "u1", "api", "https://www.orcarouter.ai")
    loaded = resolve_credential(config_key=None, store=store, env={})
    assert loaded.api_key == FAKE_KEY
    assert loaded.generation == saved.generation
    assert FAKE_KEY not in loaded.masked()
    assert FAKE_KEY not in str(loaded) and FAKE_KEY not in repr(loaded)
    assert store.clear() is True
    assert store.load() is None
    assert store.clear() is False


def test_api_key_config_wins_over_env(store):
    cred = resolve_credential(config_key=FAKE_KEY + "x", store=store, env={"ORCAROUTER_API_KEY": FAKE_KEY})
    assert cred.api_key == FAKE_KEY + "x"


def test_corrupt_key_is_terminal(store):
    from circleguardbench.orcarouter.credentials import InvalidCredentialError

    with pytest.raises(InvalidCredentialError):
        ApiKeyCredentialSource("not-a-key").obtain()
    pkce_store = store.path
    pkce_store.write_text('{"key": "garbage", "generation": "g"}')
    with pytest.raises(InvalidCredentialError):
        resolve_credential(store=store, env={})


# --------------------------------------------------------------------------
# Both adapters produce the same downstream credential type
# --------------------------------------------------------------------------


def test_both_adapters_yield_same_credential_kind_and_disable_free_provider(server, store):
    ep = fake_endpoints(server)
    pkce_cred = pkce.exchange_code(ep, "code", pkce.new_verifier(), store)
    api_cred = resolve_credential(config_key=FAKE_KEY, store=store, env={})
    assert type(pkce_cred) is type(api_cred) is Credential
    assert pkce_cred.source == SOURCE_PKCE and api_cred.source != SOURCE_PKCE

    # The provider path (credential seam -> catalog -> client) does not depend
    # on provenance: swap credentials and the discovered options are identical.
    cat_pkce = load_catalog(pkce_cred.api_key, ep.models_url)
    cat_api = load_catalog(api_cred.api_key, ep.models_url)
    assert cat_pkce.options(CAPABILITY_INFERENCE) == cat_api.options(CAPABILITY_INFERENCE)
    model_calls = [r["auth"] for r in server.requests if r["path"].startswith("/v1/models")]
    assert model_calls == [f"Bearer {pkce_cred.api_key}", f"Bearer {api_cred.api_key}"]


def test_provider_uses_resolved_credential_into_openai_client(server, store, monkeypatch):
    """The guard provider's key comes from the seam; requests go to api_base/v1."""
    from circleguardbench.models_config import ModelConfig
    from circleguardbench.providers import MODEL_PROVIDERS_REGISTRY

    served = {}

    class FakeClient:
        def __init__(self, **kw):
            served["client"] = kw
            self.chat = self
            self.completions = self

        async def create(self, **kw):
            served["request"] = kw
            msg = type("M", (), {"content": "After the assessment, my verdict on the user's last message is: SAFE."})
            return type("C", (), {"choices": [type("Ch", (), {"message": msg})()]})()

    monkeypatch.setenv("ORCA_API_BASE_URL", server.origin + "/v1")
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ORCA_AUTH_BASE_URL", raising=False)
    monkeypatch.delenv("ORCA_BASE_URL", raising=False)
    monkeypatch.setattr(
        "circleguardbench.providers.utils.get_openai_client",
        lambda endpoint, api_key: FakeClient(base_url=endpoint, api_key=api_key),
    )
    pkce.exchange_code(fake_endpoints(server), "code", pkce.new_verifier(), store)
    monkeypatch.setattr("circleguardbench.orcarouter.credentials._default_credential_path",
                        lambda: store.path)

    import jinja2

    templates = {
        name: jinja2.Template("{{ dialog }}")
        for name in (
            "cot_prompt_eval_regexp", "cot_answer_eval_regexp", "prompt_eval_regexp",
            "answer_eval_regexp", "prompt_eval_so", "answer_eval_so", "default_prompt_eval",
            "default_answer_eval", "cot_prompt_eval_so", "cot_answer_eval_so",
        )
    }
    cfg = ModelConfig(
        name="orca-test", type="llm_regexp", inference_engine="orcarouter",
        params={"api_model_name": DEFAULT_MODEL},
    )
    provider = MODEL_PROVIDERS_REGISTRY.create_provider(cfg, templates)
    assert provider.endpoint == server.origin + "/v1/"
    assert provider.api_model_name == DEFAULT_MODEL
    assert DEFAULT_MODEL in provider.compatible_models
    assert server.requests[-1]["path"].startswith("/v1/models")

    import asyncio

    out = asyncio.run(provider._get_prompt_verdict([]))
    assert out.verdict.value == "safe"
    assert served["request"]["model"] == DEFAULT_MODEL
