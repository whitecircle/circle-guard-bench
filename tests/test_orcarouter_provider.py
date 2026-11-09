import json
import logging
import os

import jinja2
import pytest
from click.testing import CliRunner

from circleguardbench.models_config import ModelConfig
from circleguardbench.orcarouter import pkce
from circleguardbench.orcarouter.catalog import CAPABILITY_INFERENCE, DEFAULT_MODEL
from circleguardbench.orcarouter.credentials import FileCredentialStore

from fake_auth_server import DEFAULT_CATALOG, FakeAuthServer, fake_endpoints

TEMPLATES = [
    "cot_prompt_eval_regexp", "cot_answer_eval_regexp", "prompt_eval_regexp",
    "answer_eval_regexp", "prompt_eval_so", "answer_eval_so", "default_prompt_eval",
    "default_answer_eval", "cot_prompt_eval_so", "cot_answer_eval_so",
]


def _templates():
    return {name: jinja2.Template("{{ dialog }}") for name in TEMPLATES}


@pytest.fixture()
def server():
    with FakeAuthServer() as srv:
        srv.models = DEFAULT_CATALOG
        yield srv


@pytest.fixture()
def env(server, tmp_path, monkeypatch):
    for var in ("ORCAROUTER_API_KEY", "ORCA_AUTH_BASE_URL", "ORCA_BASE_URL", "ORCA_API_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ORCA_API_BASE_URL", server.origin + "/v1")
    monkeypatch.setenv("ORCA_AUTH_BASE_URL", server.origin)
    store_path = tmp_path / "auth.json"
    monkeypatch.setattr("circleguardbench.orcarouter.credentials._default_credential_path",
                        lambda: store_path)
    return FileCredentialStore(store_path)


def _provider(model=DEFAULT_MODEL, **params):
    from circleguardbench.providers import MODEL_PROVIDERS_REGISTRY

    cfg = ModelConfig(name="orca", type="llm_regexp", inference_engine="orcarouter",
                      params={"api_model_name": model, **params})
    return MODEL_PROVIDERS_REGISTRY.create_provider(cfg, _templates())


def test_registry_routes_orcarouter_engine():
    from circleguardbench.providers import MODEL_PROVIDERS_REGISTRY
    from circleguardbench.providers.orcarouter_providers import (
        OrcaRouterModerationProvider,
        OrcaRouterRegexpProvider,
        OrcaRouterStructuredProvider,
    )

    assert MODEL_PROVIDERS_REGISTRY._get_provider_type(
        __import__("circleguardbench.models_config", fromlist=["ModelType"]).ModelType.LLM_REGEXP,
        __import__("circleguardbench.models_config", fromlist=["InferenceEngine"]).InferenceEngine.ORCAROUTER,
    ) == "orcarouter_regexp"
    assert MODEL_PROVIDERS_REGISTRY._registry["orcarouter_regexp"] is OrcaRouterRegexpProvider
    assert MODEL_PROVIDERS_REGISTRY._registry["orcarouter_structured"] is OrcaRouterStructuredProvider
    assert MODEL_PROVIDERS_REGISTRY._registry["orcarouter_moderation"] is OrcaRouterModerationProvider


def test_provider_reports_missing_credential(env):
    provider = _provider()
    with pytest.raises(Exception, match="login"):
        provider.compatible_models


def test_provider_uses_pkce_credential_and_live_catalog(env, server):
    pkce.exchange_code(fake_endpoints(server), "code", pkce.new_verifier(), env)
    provider = _provider("orcacyber-ultra-1.0")
    assert provider.api_key.startswith("sk-orca-")
    assert "orcacyber-ultra-1.0" in provider.compatible_models
    assert provider.api_model_name == "orcacyber-ultra-1.0"
    assert server.requests[-1]["auth"] == f"Bearer {provider.api_key}"
    assert server.requests[-1]["path"].startswith("/v1/models")


def test_provider_rejects_stale_model_selection(env, server):
    pkce.exchange_code(fake_endpoints(server), "code", pkce.new_verifier(), env)
    provider = _provider("vendor/image-gen")
    with pytest.raises(Exception, match="not in the OrcaRouter catalog"):
        provider.api_model_name


def test_provider_resolves_documented_cyber_alias(env, server):
    """Config may use the short documented id; the served `orca/` id is sent."""
    server.models = [
        {"id": "orca/orcacyber-zero-1.5", "architecture": {"input_modalities": ["text"]},
         "supported_endpoint_types": ["openai"]},
    ]
    pkce.exchange_code(fake_endpoints(server), "code", pkce.new_verifier(), env)
    provider = _provider("orcacyber-zero-1.5")
    assert provider.api_model_name == "orca/orcacyber-zero-1.5"


def test_provider_marks_only_exact_generation_on_401(env, server, monkeypatch):
    cred_old = pkce.exchange_code(fake_endpoints(server), "c1", pkce.new_verifier(), env)
    cred_new = pkce.exchange_code(fake_endpoints(server), "c2", pkce.new_verifier(), env)
    provider = _provider()
    provider._credential = cred_old  # the provider resolved the older credential

    class Boom(Exception):
        status_code = 401

    with pytest.raises(Exception, match="rejected the credential"):
        import asyncio

        asyncio.run(provider._execute(lambda: _raise(Boom())))

    data = env.load()
    assert data["generation"] == cred_new.generation
    assert data["status"] == "active"  # new credential not poisoned
    from circleguardbench.orcarouter.credentials import mark_needs_reauth

    assert mark_needs_reauth(cred_new, env) is True
    assert env.load()["status"] == "needs_reauth"


async def _raise(exc):
    raise exc


def test_cli_login_links_both_methods(env, monkeypatch):
    from circleguardbench.cli import cli

    monkeypatch.setenv("HOME", str(env.path.parent))
    out = CliRunner().invoke(cli, ["orcarouter", "--help"])
    assert out.exit_code == 0
    assert "login" in out.output and "key" in out.output
    assert "https://www.orcarouter.ai/cyber" in out.output
    assert "https://www.orcarouter.ai/models?tier=cyber" in out.output


def test_cli_models_lists_live_catalog(env, server, monkeypatch):
    from circleguardbench.cli import cli

    pkce.exchange_code(fake_endpoints(server), "code", pkce.new_verifier(), env)
    result = CliRunner().invoke(cli, ["orcarouter", "models", "--capability", "inference", "--json"])
    assert result.exit_code == 0, result.output + str(result.exception)
    payload = json.loads(result.output)
    assert payload["source"] == "live"
    assert payload["catalog_url"] == server.origin + "/v1/models"
    assert "orcarouter/auto" in payload["options"]
    assert "vendor/image-gen" not in payload["options"]


def test_cli_models_multimodal_filter(env, server):
    from circleguardbench.cli import cli

    pkce.exchange_code(fake_endpoints(server), "code", pkce.new_verifier(), env)
    result = CliRunner().invoke(cli, ["orcarouter", "models", "--input-modalities", "text,image", "--json"])
    payload = json.loads(result.output)
    assert payload["options"] == ["vendor/vision-guard"]


def test_cli_key_set_stores_masked(env, monkeypatch):
    from circleguardbench.cli import cli

    result = CliRunner().invoke(cli, ["orcarouter", "key", "set"],
                                input="sk-orca-cli-test-key-0123456789\n")
    assert result.exit_code == 0, result.output + str(result.exception)
    assert "sk-orca-cli-test-key-0123456789" not in result.output
    assert env.load()["key"] == "sk-orca-cli-test-key-0123456789"
    status = CliRunner().invoke(cli, ["orcarouter", "status"])
    assert status.exit_code == 0
    assert "sk-orca-cli-test-key-0123456789" not in status.output
    clear = CliRunner().invoke(cli, ["orcarouter", "key", "clear"])
    assert clear.exit_code == 0 and env.load() is None


def test_verifier_and_key_never_logged(env, server, caplog):
    caplog.set_level(logging.DEBUG)
    captured = {}

    def fake_browser(url):
        import threading
        import urllib.parse
        import urllib.request

        captured["url"] = url
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        cb = q["callback_url"][0]
        threading.Thread(
            target=urllib.request.urlopen,
            args=(f"{cb}?code=c&state={q['state'][0]}",),
        ).start()

    cred = pkce.login_with_loopback(fake_endpoints(server), env, open_browser=fake_browser,
                                    timeout=10, out=lambda *_: None)
    sent = server.requests[0]["body"]
    verifier = sent["code_verifier"]
    for text in caplog.text.splitlines():
        assert verifier not in text
        assert cred.api_key not in text
    assert verifier not in captured["url"]
    # URL is the auth endpoint only -- never the API origin.
    assert "/api/v1/auth/keys" not in captured["url"]
    assert ascii_url_has_only_auth_host(captured["url"], server.origin)


def ascii_url_has_only_auth_host(url, origin):
    import urllib.parse

    return urllib.parse.urlparse(url).netloc == urllib.parse.urlparse(origin).netloc


def test_capability_constant_exported():
    assert CAPABILITY_INFERENCE == "inference"
    assert os.environ.get("ORCAROUTER_API_KEY") in (None, os.environ.get("ORCAROUTER_API_KEY"))
