"""Opt-in live checks against the real OrcaRouter endpoint.

Skipped unless ORCAROUTER_API_KEY is set.  Requests go through the
integration's own credential seam, catalog loader and guard provider -- no
separate curl call stands in for the wired path.

Run with:  ORCAROUTER_API_KEY=... python3 -m pytest tests/test_orcarouter_live.py
"""

import asyncio
import os

import jinja2
import pytest

from circleguardbench.models_config import ModelConfig
from circleguardbench.orcarouter.catalog import (
    CAPABILITY_INFERENCE,
    DEFAULT_MODEL,
    load_catalog,
)
from circleguardbench.orcarouter.credentials import SOURCE_ENV, resolve_credential
from circleguardbench.orcarouter.pkce import OrcaEndpoints
from circleguardbench.providers import MODEL_PROVIDERS_REGISTRY
from pathlib import Path

needs_key = pytest.mark.skipif(
    not os.environ.get("ORCAROUTER_API_KEY"),
    reason="ORCAROUTER_API_KEY not set",
)

# Public models our verification key can call; the cyber tier may be locked
# (403) for it.  This is a *verification* model, not a user-facing default.
LIVE_TEXT_MODEL = os.environ.get("ORCAROUTER_LIVE_MODEL", "deepseek/deepseek-v4.1-flash")

# The repository's real prompt templates (same loader the CLI uses), so the
# live request exercises the wired prompt path rather than a stub.
_PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
_env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(_PROMPTS)))
TEMPLATES = {p.stem: _env.get_template(p.name) for p in _PROMPTS.glob("*.jinja")}


@needs_key
def test_live_credential_resolves_from_env():
    cred = resolve_credential(config_key=None, store=None)
    assert cred.source == SOURCE_ENV
    assert cred.api_key.startswith("sk-orca-")


@needs_key
def test_live_catalog_is_live_and_filters_are_consistent():
    ep = OrcaEndpoints.from_env()
    assert ep.api_base == "https://api.orcarouter.ai/v1"
    cat = load_catalog(resolve_credential().api_key, ep.models_url)
    assert cat.source == "live", cat.error
    text_opts = cat.options(CAPABILITY_INFERENCE)
    assert text_opts, "text capability filter returned nothing"
    assert DEFAULT_MODEL in text_opts
    for excluded in ("embeddings", "image-generation", "openai-video", "jina-rerank"):
        for m in cat.compatible(CAPABILITY_INFERENCE):
            assert excluded not in m.endpoint_types
    multimodal = cat.options(CAPABILITY_INFERENCE, ["text", "image"])
    assert set(multimodal) <= set(text_opts)
    declared = {m.id: m.input_modalities for m in cat.models}
    assert all("image" in declared[mid] for mid in multimodal)
    print(f"live text models: {len(text_opts)}; multimodal text models: {len(multimodal)}")


def _verdict_once(model_id: str, model_type: str):
    cfg = ModelConfig(name="orca-live", type=model_type, inference_engine="orcarouter",
                      params={"api_model_name": model_id})
    provider = MODEL_PROVIDERS_REGISTRY.create_provider(cfg, TEMPLATES)
    assert model_id in provider.compatible_models
    out = asyncio.run(
        provider._get_prompt_verdict([{"role": "User", "message": "Hello there"}])
    )
    assert out.run_time_ms is None or out.run_time_ms >= 0
    return out


@needs_key
def test_live_guard_provider_request():
    """One real verdict through the OrcaRouter guard provider."""
    out = _verdict_once(LIVE_TEXT_MODEL, "llm_regexp")
    assert out.verdict.value in ("safe", "unsafe"), out.error_message
    print("live verdict:", out.verdict.value, (out.raw_prediction or "")[:120])


@needs_key
@pytest.mark.skipif(
    not os.environ.get("ORCAROUTER_LIVE_CYBER_MODEL"),
    reason="ORCAROUTER_LIVE_CYBER_MODEL not set (cyber tier may be 403 for this key)",
)
def test_live_cyber_guard_provider_request():
    """Recommended cyber model through the same provider path, when entitled."""
    out = _verdict_once(os.environ["ORCAROUTER_LIVE_CYBER_MODEL"], "llm_regexp")
    assert out.verdict.value in ("safe", "unsafe"), out.error_message
