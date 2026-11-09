import pytest

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
    Catalog,
    CatalogError,
    ModelEntry,
    load_catalog,
    parse_catalog,
)

from fake_auth_server import DEFAULT_CATALOG, FakeAuthServer

KEY = "sk-orca-catalog-test-0123456789"


@pytest.fixture()
def server():
    with FakeAuthServer() as srv:
        srv.models = DEFAULT_CATALOG
        yield srv


def test_live_catalog_is_parsed_from_api(server):
    cat = load_catalog(KEY, server.origin + "/v1/models")
    assert cat.source == "live" and not cat.degraded
    assert cat.catalog_url == server.origin + "/v1/models"
    # The malformed entries are dropped, the real ones kept.
    assert "" not in cat.ids()
    assert "orcarouter/auto" in cat.ids()
    auto = [m for m in cat.models if m.id == "orcarouter/auto"][0]
    assert auto.context_length == 1000000
    assert set(auto.input_modalities) == {"text"}


def test_inference_filter_excludes_non_text_endpoints(server):
    cat = load_catalog(KEY, server.origin + "/v1/models")
    opts = cat.options(CAPABILITY_INFERENCE)
    assert opts == [
        "orcarouter/auto",
        "orcacyber-ultra-1.0",
        "orcacyber-zero-1.5",
        "orcarouter/fusion",
        "vendor/text-only-guard",
        "vendor/text-only-guard-lookalike",
        "vendor/vision-guard",
    ]
    for excluded in ("vendor/image-gen", "vendor/embed-guard", "vendor/video-gen",
                     "vendor/rerank-guard", "vendor/text-endpoint-but-image-gen"):
        assert excluded not in opts
    assert cat.options(CAPABILITY_MODERATION) == opts


def test_multimodal_filter_is_fail_closed(server):
    cat = load_catalog(KEY, server.origin + "/v1/models")
    opts = cat.options(CAPABILITY_INFERENCE, ["text", "image"])
    assert opts == ["vendor/vision-guard"]
    # Undeclared capability never leaks into a multimodal selector.
    assert "vendor/text-only-guard-lookalike" not in opts
    assert "orcarouter/auto" not in opts  # seed/API entry declares text only


@pytest.mark.parametrize("capability,expected", [
    (CAPABILITY_EMBEDDING, ["vendor/embed-guard"]),
    # image generation matches the declared endpoint type strictly; a model
    # that also offers chat endpoints is still image-capable.
    (CAPABILITY_IMAGE, ["vendor/image-gen", "vendor/text-endpoint-but-image-gen"]),
    (CAPABILITY_VIDEO, ["vendor/video-gen"]),
    (CAPABILITY_RERANK, ["vendor/rerank-guard"]),
])
def test_specialised_capability_filters(server, capability, expected):
    cat = load_catalog(KEY, server.origin + "/v1/models")
    assert cat.options(capability) == expected


def test_selection_validation_clears_stale_model(server):
    cat = load_catalog(KEY, server.origin + "/v1/models")
    assert cat.validate_selection("orcacyber-ultra-1.0", CAPABILITY_INFERENCE) == "orcacyber-ultra-1.0"
    # Not offered (e.g. gone from the catalog / wrong capability) -> cleared.
    assert cat.validate_selection("vendor/image-gen", CAPABILITY_INFERENCE) is None
    assert cat.validate_selection("removed/model", CAPABILITY_INFERENCE) is None
    # Switching to an image attachment clears a text-only selection.
    assert cat.validate_selection("orcarouter/auto", CAPABILITY_INFERENCE, ["text", "image"]) is None
    assert cat.validate_selection("vendor/vision-guard", CAPABILITY_INFERENCE, ["image"]) == "vendor/vision-guard"


def test_cyber_alias_resolves_to_the_served_id():
    cat = Catalog(
        models=[ModelEntry(id="orca/orcacyber-zero-1.5", name="x", input_modalities=("text",),
                           endpoint_types=("openai",)),
                ModelEntry(id="orcarouter/auto", name="y", input_modalities=("text",),
                           endpoint_types=("openai",))],
        source="live",
    )
    assert cat.resolve_id("orcacyber-zero-1.5", CAPABILITY_INFERENCE) == "orca/orcacyber-zero-1.5"
    assert cat.resolve_id("orca/orcacyber-zero-1.5", CAPABILITY_INFERENCE) == "orca/orcacyber-zero-1.5"
    # No guessing beyond the documented alias.
    assert cat.resolve_id("cyber-zero-1.5", CAPABILITY_INFERENCE) is None
    assert cat.resolve_id("vendor/missing", CAPABILITY_INFERENCE) is None
    assert cat.resolve_id("orca/orcacyber-zero-1.5", CAPABILITY_INFERENCE, ["text", "image"]) is None


def test_provider_switch_recomputes_options(server):
    cat = load_catalog(KEY, server.origin + "/v1/models")
    first = cat.options(CAPABILITY_INFERENCE)
    server.models = [{"id": "vendor/only-chat", "architecture": {"input_modalities": ["text"]},
                      "supported_endpoint_types": ["openai"]}]
    cat2 = load_catalog(KEY, server.origin + "/v1/models")
    assert cat2.options(CAPABILITY_INFERENCE) == ["vendor/only-chat"]
    assert first != cat2.options(CAPABILITY_INFERENCE)


def test_auth_error_degrades_to_seed_not_free_text(server):
    server.models_status = 401
    cat = load_catalog(KEY, server.origin + "/v1/models")
    assert cat.source == "seed" and cat.degraded
    assert "401" in cat.error
    assert cat.ids() == [DEFAULT_MODEL]
    for m in cat.models:
        assert m.seed
    # Seed is never mixed into a live response:
    server.models_status = 200
    live = load_catalog(KEY, server.origin + "/v1/models")
    assert all(not m.seed for m in live.models)


def test_network_error_degrades_and_can_raise():
    cat = load_catalog(KEY, "http://127.0.0.1:9/v1/models", timeout=1)
    assert cat.degraded and cat.source == "seed"
    with pytest.raises(CatalogError):
        load_catalog(KEY, "http://127.0.0.1:9/v1/models", timeout=1, use_seed_on_error=False)


def test_empty_catalog_is_an_error(server):
    server.models = []
    cat = load_catalog(KEY, server.origin + "/v1/models")
    assert cat.degraded and cat.ids() == [DEFAULT_MODEL]


def test_model_names_are_never_used_to_guess_capabilities(server):
    server.models = [
        {"id": "vendor/vision-guard-vl", "architecture": {"input_modalities": ["text"]},
         "supported_endpoint_types": ["openai"]},
        {"id": "vendor/embed-text-model", "architecture": {"input_modalities": ["text"]},
         "supported_endpoint_types": ["openai"]},
    ]
    cat = load_catalog(KEY, server.origin + "/v1/models")
    assert "vendor/vision-guard-vl" not in cat.options(CAPABILITY_INFERENCE, ["image"])
    assert cat.options(CAPABILITY_EMBEDDING) == []


def test_seed_contains_only_branded_gateway_model():
    from circleguardbench.orcarouter.catalog import VERIFIED_SEED_CATALOG

    ids = [raw["id"] for raw in VERIFIED_SEED_CATALOG]
    assert ids == [DEFAULT_MODEL]
    assert DEFAULT_MODEL == "orcarouter/auto"
    assert GENERAL_FALLBACK_MODEL == "orcarouter/fusion"
    assert RECOMMENDED_CYBER_MODELS == ("orcacyber-ultra-1.0", "orcacyber-zero-1.5")


def test_parse_catalog_rejects_malformed_payload():
    with pytest.raises(CatalogError):
        parse_catalog({"nope": 1})
