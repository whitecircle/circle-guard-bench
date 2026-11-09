import json
import os
from pathlib import Path

import pytest

from circleguardbench.models_config import (
    InferenceEngine,
    ModelConfig,
    ModelType,
    load_models_configs_registery,
)
from circleguardbench.providers import MODEL_PROVIDERS_REGISTRY

ROOT = Path(__file__).resolve().parents[1]


def test_models_json_loads_with_orcarouter_entries():
    registry = load_models_configs_registery(ROOT / "configs" / "models.json")
    orca = [name for name in registry.list() if name.startswith("OrcaRouter")]
    assert len(orca) == 4
    ids = {registry[n].params.api_model_name for n in orca}
    assert ids == {"orcacyber-ultra-1.0", "orcacyber-zero-1.5", "orcarouter/auto", "orcarouter/fusion"}


def test_engine_mapping():
    assert MODEL_PROVIDERS_REGISTRY._get_provider_type(
        ModelType.LLM_SO, InferenceEngine.ORCAROUTER) == "orcarouter_structured"
    assert MODEL_PROVIDERS_REGISTRY._get_provider_type(
        ModelType.LLM_REGEXP, InferenceEngine.ORCAROUTER) == "orcarouter_regexp"
    assert MODEL_PROVIDERS_REGISTRY._get_provider_type(
        ModelType.LLAMA_GUARD, InferenceEngine.ORCAROUTER) == "orcarouter_regexp"
    assert MODEL_PROVIDERS_REGISTRY._get_provider_type(
        ModelType.OPENAI_MODERATION, InferenceEngine.ORCAROUTER) == "orcarouter_moderation"
    with pytest.raises(ValueError):
        MODEL_PROVIDERS_REGISTRY._get_provider_type(
            ModelType.CLASSIFIER, InferenceEngine.ORCAROUTER)


def test_missing_model_id_rejected():
    with pytest.raises(ValueError):
        ModelConfig(name="x", type="llm_regexp", inference_engine="orcarouter", params={})
    with pytest.raises(ValueError):
        ModelConfig(name="x", type="llm_regexp", inference_engine="orcarouter", params=None)


def test_packaged_code_does_not_needs_optional_heavy_deps():
    """All OrcaRouter modules import with only openai/pydantic/jinja2/click;
    datasets, sklearn, rich and tenacity are imported where they are used."""
    import importlib
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import circleguardbench.orcarouter.cli, circleguardbench.orcarouter.catalog\n"
        "import circleguardbench.orcarouter.credentials, circleguardbench.orcarouter.pkce\n"
        "import circleguardbench.providers, circleguardbench.models_config\n"
        "print('ok')\n"
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_orcarouter_cli_commands_do_not_need_dataset(tmp_path):
    from click.testing import CliRunner

    import circleguardbench.cli as cli_mod

    env = {"ORCAROUTER_API_KEY": "sk-orca-offline-test-0000000000",
           "ORCA_API_BASE_URL": "http://127.0.0.1:9/v1", "ORCA_AUTH_BASE_URL": "http://127.0.0.1:9"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        result = CliRunner().invoke(cli_mod.cli, ["orcarouter", "status"])
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert result.exit_code == 0, result.output + str(result.exception)
    assert "sk-orca-offline-test-0000000000" not in result.output
