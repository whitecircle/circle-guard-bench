"""OrcaRouter Cyber Inference guard providers for CircleGuardBench.

CircleGuardBench already has OpenAI-compatible guard providers
(``OpenAIRegexpModelProvider`` / ``OpenAIStructuredOutputsModelProvider``) and
a moderation provider.  The three classes below reuse their verdict handling
verbatim and differ only in how the request is wired:

* credential -> :func:`resolve_credential` (API key adapter or PKCE adapter,
  same downstream key) -- never read from a model selector;
* endpoint -> :attr:`OrcaEndpoints.api_base` (``https://api.orcarouter.ai/v1``
  by default, ``ORCA_API_BASE_URL`` override);
* model -> the id chosen from the live OrcaRouter catalog
  (:mod:`circleguardbench.orcarouter.catalog`) for the guard's capability.

Only the resolved key is passed to the OpenAI client; neither the key nor the
PKCE verifier is ever logged.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import jinja2

from circleguardbench.models_config import ModelConfig, ModelType, OrcaRouterParams
from circleguardbench.orcarouter.catalog import (
    CAPABILITY_INFERENCE,
    CAPABILITY_MODERATION,
    Catalog,
    load_catalog,
)
from circleguardbench.orcarouter.credentials import (
    Credential,
    FileCredentialStore,
    OrcaRouterAuthError,
    mark_needs_reauth,
    resolve_credential,
)
from circleguardbench.orcarouter.pkce import OrcaEndpoints
from circleguardbench.providers.model_providers import (
    BaseModelProvider,
    OpenAIModerationModelProvider,
    OpenAIRegexpModelProvider,
    OpenAIStructuredOutputsModelProvider,
)
from circleguardbench.providers.utils import get_openai_client
from circleguardbench.storage.objects import ModelOutput, Verdict

logger = logging.getLogger("guardbench.orcarouter.provider")


class OrcaRouterMixin(BaseModelProvider):
    """Shared credential / endpoint / catalog wiring for OrcaRouter guards."""

    capability: str = CAPABILITY_INFERENCE

    def __init__(
        self, model_config: ModelConfig, prompts_templates: Dict[str, jinja2.Template]
    ):
        # Bypass the mixin of the OpenAI-compat parent (its endpoint/key parsing
        # is replaced by the credential seam below).
        BaseModelProvider.__init__(self, model_config, prompts_templates)
        params = model_config.params
        if not isinstance(params, OrcaRouterParams):  # defensive; validator normalises
            params = OrcaRouterParams(**dict(params or {}))
        self.params = params
        self.endpoints = OrcaEndpoints.from_env(
            {**os.environ, **({"ORCA_API_BASE_URL": params.endpoint} if params.endpoint else {})}
        )
        self.store = FileCredentialStore()
        self.catalog: Optional[Catalog] = None
        self._credential: Optional[Credential] = None

    # -- credential ------------------------------------------------------
    def _resolve(self) -> Credential:
        self._credential = resolve_credential(config_key=self.params.api_key, store=self.store)
        return self._credential

    @property
    def api_key(self) -> str:
        return self._resolve().api_key

    @api_key.setter
    def api_key(self, value: str) -> None:  # keeps the OpenAI-compat API surface
        self.params = self.params.model_copy(update={"api_key": value})

    @property
    def endpoint(self) -> str:
        # The OpenAI client expects a base URL; /v1 is part of api_base.
        return self.endpoints.api_base + "/"

    # -- model -----------------------------------------------------------
    def load_catalog(self, refresh: bool = False) -> Catalog:
        if self.catalog is None or refresh:
            self.catalog = load_catalog(self.api_key, self.endpoints.models_url)
        return self.catalog

    @property
    def compatible_models(self) -> List[str]:
        return self.load_catalog().options(self.capability, self.params.input_modalities)

    @property
    def api_model_name(self) -> str:
        """Return the configured model, validated against the catalog.

        An id that is no longer offered for this capability is reported
        instead of being sent blindly; ``orcarouter/auto`` is always offered.
        """
        wanted = self.params.api_model_name
        catalog = self.load_catalog()
        resolved = catalog.resolve_id(wanted, self.capability, self.params.input_modalities)
        if resolved is None:
            options = catalog.options(self.capability, self.params.input_modalities)
            raise OrcaRouterAuthError(
                f"Model '{wanted}' is not in the OrcaRouter catalog for capability "
                f"'{self.capability}' (available: {', '.join(options[:10]) or 'none'}). "
                "Run `guardbench orcarouter models` and pick another id."
            )
        return resolved

    # -- error classification -------------------------------------------
    def _classify(self, exc: BaseException) -> BaseException:
        status = getattr(exc, "status_code", None)
        if status == 401 and self._credential is not None:
            mark_needs_reauth(self._credential, self.store)
            return OrcaRouterAuthError(
                "OrcaRouter rejected the credential (HTTP 401). If you signed in with "
                "`guardbench orcarouter login`, run it again; otherwise check ORCAROUTER_API_KEY."
            )
        return exc

    async def _execute(self, coro_fn):
        try:
            return await coro_fn()
        except BaseException as exc:  # re-raised after classification
            raise self._classify(exc)

    async def _moderate_text(self, text: str) -> ModelOutput:  # pragma: no cover - overridden
        raise NotImplementedError

    def __del__(self):
        # these providers hold no engines to shut down (unlike vLLM/SGLang)
        pass


class OrcaRouterRegexpProvider(OrcaRouterMixin, OpenAIRegexpModelProvider):
    """OrcaRouter-backed ``llm_regexp`` / ``atla_selene`` / ``llama_guard`` guard."""

    capability = CAPABILITY_INFERENCE

    async def _get_prompt_verdict(self, dialog) -> ModelOutput:
        return await self._execute(lambda: super(OrcaRouterRegexpProvider, self)._get_prompt_verdict(dialog))

    async def _get_answer_verdict(self, dialog) -> ModelOutput:
        return await self._execute(lambda: super(OrcaRouterRegexpProvider, self)._get_answer_verdict(dialog))


class OrcaRouterStructuredProvider(OrcaRouterMixin, OpenAIStructuredOutputsModelProvider):
    """OrcaRouter-backed ``llm_so`` guard (JSON-schema verdict)."""

    capability = CAPABILITY_INFERENCE

    async def _get_prompt_verdict(self, dialog) -> ModelOutput:
        return await self._execute(lambda: super(OrcaRouterStructuredProvider, self)._get_prompt_verdict(dialog))

    async def _get_answer_verdict(self, dialog) -> ModelOutput:
        return await self._execute(lambda: super(OrcaRouterStructuredProvider, self)._get_answer_verdict(dialog))


class OrcaRouterModerationProvider(OrcaRouterMixin, OpenAIModerationModelProvider):
    """OrcaRouter-backed moderation verdict (openai_moderation model type)."""

    capability = CAPABILITY_MODERATION

    @property
    def endpoint(self) -> str:  # type: ignore[override]
        return self.endpoints.api_base + "/moderations"

    @property
    def api_model_name(self) -> str:  # type: ignore[override]
        wanted = self.params.api_model_name
        catalog = self.load_catalog()
        resolved = catalog.resolve_id(wanted, self.capability, self.params.input_modalities)
        if resolved is None:
            options = catalog.options(self.capability, self.params.input_modalities)
            raise OrcaRouterAuthError(
                f"Model '{wanted}' is not offered for moderation in the OrcaRouter "
                f"catalog (available: {', '.join(options[:10]) or 'none'})."
            )
        return resolved

    async def _moderate_text(self, text: str) -> ModelOutput:
        return await self._execute(lambda: super(OrcaRouterModerationProvider, self)._moderate_text(text))


def register_orcarouter_providers(registry) -> None:
    registry.register("orcarouter_regexp")(OrcaRouterRegexpProvider)
    registry.register("orcarouter_structured")(OrcaRouterStructuredProvider)
    registry.register("orcarouter_moderation")(OrcaRouterModerationProvider)
