"""`guardbench orcarouter ...` -- authentication and model selection for
OrcaRouter Cyber Inference.

Authentication has two first-class choices, both producing the same ordinary
OrcaRouter API key for the ``orcarouter`` inference engine:

    guardbench orcarouter login            # Connect with OrcaRouter (OAuth 2.0 + PKCE, loopback)
    guardbench orcarouter login --oob      # same, for machines that cannot listen
    guardbench orcarouter login --device   # device grant, for machines without a browser
    guardbench orcarouter key set          # paste an existing API key (stored 0600)
    ORCAROUTER_API_KEY=...                 # or just use the environment variable

The PKCE login is a *plain* click and is wired straight to the provider's
credential seam; the pasted key goes through the very same seam.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import click

from circleguardbench.orcarouter.catalog import (
    ALL_CAPABILITIES,
    CYBER_TIER_URL,
    DEFAULT_MODEL,
    GENERAL_FALLBACK_MODEL,
    PROVIDER_DOCS_URL,
    RECOMMENDED_CYBER_MODELS,
    CatalogError,
    load_catalog,
)
from circleguardbench.orcarouter.credentials import (
    API_KEY_ENV,
    FileCredentialStore,
    OrcaRouterAuthError,
    describe_sources,
    mask_secret,
    resolve_credential,
    validate_key_shape,
)
from circleguardbench.orcarouter import pkce


def _resolve(explicit_key: Optional[str] = None):
    """Credential for the CLI: explicit key/config -> env -> env file -> PKCE store."""
    if explicit_key:
        return resolve_credential(config_key=explicit_key, env={})
    # Order: ORCAROUTER_API_KEY -> stored credential (PKCE login or a key
    # pasted with `orcarouter key set`).
    return resolve_credential(config_key=None, env={})


def _api_key() -> str:
    return _resolve().api_key


def _store(ctx: click.Context) -> FileCredentialStore:
    path = ctx.obj.get("credential_file") if isinstance(ctx.obj, dict) else None
    return FileCredentialStore(path) if path else FileCredentialStore()


@click.group()
@click.option(
    "--credential-file",
    default=None,
    help="Override the credential store location (default ~/.config/guardbench/orcarouter-auth.json)",
)
@click.pass_context
def orcarouter(ctx: click.Context, credential_file: Optional[str]):
    """OrcaRouter Cyber Inference: models and defenses for security teams.

    \b
    Provider page:  https://www.orcarouter.ai/cyber
    Cyber models:   https://www.orcarouter.ai/models?tier=cyber
    Release-screened, authorization-scoped security inference for red and blue
    teams; guard models are evaluated through the same OpenAI-compatible
    endpoint that OrcaRouter's own gateway controls sit in front of.
    """
    ctx.ensure_object(dict)
    ctx.obj["credential_file"] = credential_file


@orcarouter.command("login")
@click.option("--oob", is_flag=True, help="Out-of-band flow: paste the code shown by the browser.")
@click.option("--device", is_flag=True, help="Device grant (no browser on this machine).")
@click.option("--timeout", default=180.0, show_default=True, help="Loopback wait (seconds).")
@click.pass_context
def login(ctx: click.Context, oob: bool, device: bool, timeout: float):
    """Connect with OrcaRouter using OAuth 2.0 + PKCE (no client secret)."""
    store = _store(ctx)
    endpoints = pkce.OrcaEndpoints.from_env()
    try:
        if device:
            cred = pkce.login_with_device_grant(endpoints, store)
        elif oob:
            cred = pkce.login_with_oob(endpoints, store, input_fn=lambda prompt: click.prompt(prompt))
        else:
            cred = pkce.login_with_loopback(endpoints, store, timeout=timeout,
                                            out=click.echo)
    except OrcaRouterAuthError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    click.echo(f"Connected to OrcaRouter. Stored key {mask_secret(cred.api_key)} "
               f"in {store.path} (generation {cred.generation}).")
    click.echo("Try: guardbench orcarouter models --capability inference")


@orcarouter.command("logout")
@click.pass_context
def logout(ctx: click.Context):
    """Remove the stored OrcaRouter login (environment/config keys untouched)."""
    removed = _store(ctx).clear()
    click.echo("Removed stored OrcaRouter login." if removed else "No stored OrcaRouter login.")


@orcarouter.group("key")
def key_group():
    """Manage a pasted OrcaRouter API key (the other authentication method)."""


@key_group.command("set")
@click.pass_context
def key_set(ctx: click.Context):
    """Paste an existing OrcaRouter API key (stored 0600; never echoed)."""
    value = click.prompt("OrcaRouter API key", hide_input=True)
    try:
        value = validate_key_shape(value)
    except OrcaRouterAuthError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    store = FileCredentialStore()
    import uuid
    from circleguardbench.orcarouter.credentials import _now

    store.save({"version": 1, "generation": uuid.uuid4().hex, "key": value,
                "grant": "api", "scope": "api", "status": "active",
                "created_at": _now(), "updated_at": _now()})
    click.echo(f"Stored OrcaRouter API key {mask_secret(value)} in {store.path}.")


@key_group.command("clear")
@click.pass_context
def key_clear(ctx: click.Context):
    """Clear the stored OrcaRouter credential."""
    removed = _store(ctx).clear()
    click.echo("Cleared." if removed else "Nothing stored.")


@orcarouter.command("status")
@click.pass_context
def status(ctx: click.Context):
    """Show which OrcaRouter credentials are available (keys are masked)."""
    rows = describe_sources(store=_store(ctx))
    endpoints = pkce.OrcaEndpoints.from_env()
    click.echo(f"Auth origin: {endpoints.auth_base}")
    click.echo(f"API origin:  {endpoints.api_base}")
    if not rows:
        click.echo("No OrcaRouter credential. Use `guardbench orcarouter login` or "
                   f"set {API_KEY_ENV}.")
        return
    for row in rows:
        click.echo(json.dumps(row, sort_keys=True))


def _model_options(capability: str, modalities: str) -> None:
    try:
        cred = resolve_credential()
    except OrcaRouterAuthError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    endpoints = pkce.OrcaEndpoints.from_env()
    catalog = load_catalog(cred.api_key, endpoints.models_url)
    mods = [m.strip() for m in modalities.split(",") if m.strip()]
    entries = catalog.compatible(capability, mods)
    if catalog.degraded:
        click.echo(f"[degraded] live catalog unavailable ({catalog.error}); "
                   "showing verified cold-start seed only. Retry with --refresh.", err=True)
    click.echo(f"OrcaRouter catalog: {catalog.catalog_url} ({catalog.source}, "
               f"{len(catalog.models)} models)")
    click.echo(f"Compatible for capability={capability} input_modalities={mods or ['text']}:")
    for e in entries:
        rec = "  (default)" if e.id == DEFAULT_MODEL else ""
        if e.id in RECOMMENDED_CYBER_MODELS:
            rec = "  (recommended for security teams)"
        elif e.id == GENERAL_FALLBACK_MODEL:
            rec = "  (general fallback)"
        ctx_len = f" ctx={e.context_length}" if e.context_length else ""
        click.echo(f"  - {e.id}{ctx_len}{rec}")
    click.echo("")
    click.echo(f"Security-model docs: {PROVIDER_DOCS_URL}")
    click.echo(f"Cyber model tier:    {CYBER_TIER_URL}")


@orcarouter.command("models")
@click.option("--capability", default="inference",
              type=click.Choice(ALL_CAPABILITIES), help="Catalog capability filter.")
@click.option("--input-modalities", default="text",
              help="Comma-separated modalities this guard sends (e.g. text,image).")
@click.option("--refresh", is_flag=True,
              help="Re-fetch instead of reusing the last good catalog for this run.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
@click.pass_context
def models(ctx: click.Context, capability: str, input_modalities: str, refresh: bool, as_json: bool):
    """List OrcaRouter models from the live catalog (GET /v1/models)."""
    if as_json:
        try:
            cred = resolve_credential()
        except OrcaRouterAuthError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        endpoints = pkce.OrcaEndpoints.from_env()
        try:
            catalog = load_catalog(cred.api_key, endpoints.models_url)
        except CatalogError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        mods = [m.strip() for m in input_modalities.split(",") if m.strip()]
        click.echo(json.dumps({
            "source": catalog.source,
            "degraded": catalog.degraded,
            "catalog_url": catalog.catalog_url,
            "options": catalog.options(capability, mods),
        }, indent=2))
        return
    _model_options(capability, input_modalities)
