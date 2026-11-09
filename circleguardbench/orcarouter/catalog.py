"""OrcaRouter Cyber Inference -- model catalog and capability filters.

The single source of truth for guard-model choices is the live catalog at
``GET {api_base}/models`` of the configured OrcaRouter origin, fetched with the
user's own credential (their workspace may unlock extra models).  The
``orcarouter/auto`` router model is always offered because it lets the gateway
pick a compatible model.

If discovery fails, a tiny, verified cold-start seed is used and reported as
``degraded`` so the CLI can say so.  An expired selection (a model that is no
longer in the compatible list) is cleared instead of silently kept.
"""

from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger("guardbench.orcarouter.catalog")

DEFAULT_MODEL = "orcarouter/auto"
RECOMMENDED_CYBER_MODELS = ("orcacyber-ultra-1.0", "orcacyber-zero-1.5")
GENERAL_FALLBACK_MODEL = "orcarouter/fusion"

# The public catalog lists the dedicated cyber models under an `orca/` prefix
# (``orca/orcacyber-ultra-1.0``), while the short documented names omit it.
# Both spellings are accepted for configuration and resolved to whatever id the
# live catalog actually serves; nothing is guessed beyond this one prefix.
CYBER_MODEL_ALIAS_PREFIX = "orca/"

PROVIDER_DOCS_URL = "https://www.orcarouter.ai/cyber"
CYBER_TIER_URL = "https://www.orcarouter.ai/models?tier=cyber"

# Outage fallback only.  Each entry was verified against the OrcaRouter
# catalog (`GET /v1/models` + the capability flags below).  It is never mixed
# into a successful live response.
VERIFIED_SEED_CATALOG: List[Dict] = [
    {
        "id": DEFAULT_MODEL,
        "name": DEFAULT_MODEL,
        "context_length": None,
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "supported_endpoint_types": ["openai", "anthropic", "openai-response"],
        "supported_parameters": [],
        "_seed": True,
        "_note": "Gateway auto-router; routes to a compatible model.",
    },
]

TEXT_ENDPOINTS = {"openai", "anthropic", "gemini", "openai-response"}
NON_TEXT_ENDPOINTS = {"image-generation", "openai-video", "jina-rerank", "embeddings"}

# Guard-model roles the benchmark can evaluate with `inference_engine=orcarouter`.
CAPABILITY_INFERENCE = "inference"  # text chat: verdict extraction (llm_regexp / llm_so)
CAPABILITY_MODERATION = "moderation"  # text chat: category moderation verdict
CAPABILITY_EMBEDDING = "embedding"
CAPABILITY_IMAGE = "image"
CAPABILITY_VIDEO = "video"
CAPABILITY_RERANK = "rerank"
ALL_CAPABILITIES = (
    CAPABILITY_INFERENCE,
    CAPABILITY_MODERATION,
    CAPABILITY_EMBEDDING,
    CAPABILITY_IMAGE,
    CAPABILITY_VIDEO,
    CAPABILITY_RERANK,
)


class CatalogError(Exception):
    """Discovery failed (auth, network, malformed payload)."""


@dataclass(frozen=True)
class ModelEntry:
    id: str
    name: str
    context_length: Optional[int] = None
    input_modalities: Sequence[str] = field(default_factory=tuple)
    endpoint_types: Sequence[str] = field(default_factory=tuple)
    seed: bool = False

    @classmethod
    def from_api(cls, raw: Dict) -> Optional["ModelEntry"]:
        mid = raw.get("id")
        if not isinstance(mid, str) or not mid:
            return None
        arch = raw.get("architecture") or {}
        mods = arch.get("input_modalities") if isinstance(arch, dict) else None
        eps = raw.get("supported_endpoint_types") or raw.get("endpoint_types") or []
        ctx = raw.get("context_length") or raw.get("context_window")
        return cls(
            id=mid,
            name=str(raw.get("name") or mid),
            context_length=int(ctx) if isinstance(ctx, (int, float)) else None,
            input_modalities=tuple(m for m in (mods or []) if isinstance(m, str)),
            endpoint_types=tuple(e for e in eps if isinstance(e, str)),
            seed=bool(raw.get("_seed")),
        )

    def is_text_chat(self) -> bool:
        eps = set(self.endpoint_types)
        if not (eps & TEXT_ENDPOINTS):
            return False
        return not (eps & NON_TEXT_ENDPOINTS)

    def supports_modality(self, modality: str) -> bool:
        # Fail closed: a model must *declare* the modality.
        return modality in self.input_modalities

    def supports(self, capability: str) -> bool:
        eps = set(self.endpoint_types)
        if capability in (CAPABILITY_INFERENCE, CAPABILITY_MODERATION):
            if not self.is_text_chat():
                return False
            # Text chat only.  Multimodal guard dialogs additionally require a
            # declared input modality (see `compatible`).
            return True
        if capability == CAPABILITY_EMBEDDING:
            return "embeddings" in eps
        if capability == CAPABILITY_IMAGE:
            return "image-generation" in eps
        if capability == CAPABILITY_VIDEO:
            return "openai-video" in eps
        if capability == CAPABILITY_RERANK:
            return "jina-rerank" in eps
        raise ValueError(f"Unknown capability {capability!r}")


@dataclass
class Catalog:
    models: List[ModelEntry]
    source: str  # "live" | "seed"
    degraded: bool = False
    error: Optional[str] = None
    catalog_url: Optional[str] = None

    def ids(self) -> List[str]:
        return [m.id for m in self.models]

    def compatible(self, capability: str, modalities: Iterable[str] = ()) -> List[ModelEntry]:
        wanted = [m for m in modalities if m and m != "text"]
        result = []
        for m in self.models:
            if not m.supports(capability):
                continue
            if capability in (CAPABILITY_INFERENCE, CAPABILITY_MODERATION):
                if any(not m.supports_modality(mod) for mod in wanted):
                    continue
            result.append(m)
        return result

    def options(self, capability: str, modalities: Iterable[str] = ()) -> List[str]:
        return [m.id for m in self.compatible(capability, modalities)]

    def validate_selection(
        self, selected: Optional[str], capability: str, modalities: Iterable[str] = ()
    ) -> Optional[str]:
        """Return the selection if still compatible, else None (clear it)."""
        opts = self.options(capability, modalities)
        return selected if selected in opts else None

    def resolve_id(
        self, selected: Optional[str], capability: str, modalities: Iterable[str] = ()
    ) -> Optional[str]:
        """Map a configured id to the id the catalog serves, or None.

        Only the documented ``orca/`` alias of the dedicated cyber models is
        accepted; every other id must match the catalog exactly.  The result
        is always a real catalog entry, never a guess.
        """
        opts = self.options(capability, modalities)
        if selected in opts:
            return selected
        if selected and not selected.startswith(CYBER_MODEL_ALIAS_PREFIX):
            alias = CYBER_MODEL_ALIAS_PREFIX + selected
            if alias in opts:
                return alias
        return None


def _fetch_json(url: str, api_key: str, timeout: float) -> Dict:
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise CatalogError("OrcaRouter rejected the credential (HTTP 401).") from None
        raise CatalogError(f"OrcaRouter model catalog returned HTTP {exc.code}.") from None
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise CatalogError(f"Could not reach the OrcaRouter model catalog: {exc}") from None
    except ValueError:
        raise CatalogError("OrcaRouter model catalog returned malformed JSON.") from None


def parse_catalog(payload: Dict) -> List[ModelEntry]:
    items = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise CatalogError("OrcaRouter model catalog has no 'data' list.")
    models = [ModelEntry.from_api(item) for item in items if isinstance(item, dict)]
    return [m for m in models if m is not None]


def load_catalog(
    api_key: str,
    models_url: str,
    timeout: float = 30.0,
    use_seed_on_error: bool = True,
) -> Catalog:
    try:
        payload = _fetch_json(models_url, api_key, timeout)
        models = parse_catalog(payload)
        if not models:
            raise CatalogError("OrcaRouter model catalog is empty.")
        return Catalog(models=models, source="live", catalog_url=models_url)
    except CatalogError as exc:
        if not use_seed_on_error:
            raise
        logger.warning(
            "OrcaRouter model discovery failed (%s); using the verified cold-start seed. "
            "Run `guardbench orcarouter models --refresh` to retry.",
            exc,
        )
        seed = [ModelEntry.from_api(dict(raw)) for raw in VERIFIED_SEED_CATALOG]
        return Catalog(
            models=[m for m in seed if m],
            source="seed",
            degraded=True,
            error=str(exc),
            catalog_url=models_url,
        )
