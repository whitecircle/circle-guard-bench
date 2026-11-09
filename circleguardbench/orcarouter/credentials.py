"""OrcaRouter credential seam.

Both ways of obtaining an OrcaRouter credential -- pasting an existing
``sk-orca-...`` API key, or "Connect with OrcaRouter" (OAuth 2.0 + PKCE) --
implement :class:`CredentialSource` and produce the very same
:class:`Credential`.  Providers and model discovery only ever see the
resulting key; they never know (or care) how it was obtained.

Credential resolution order for the ``orcarouter`` inference engine:

1. ``params.api_key`` in ``configs/models.json`` (explicit, wins);
2. ``ORCAROUTER_API_KEY`` environment variable;
3. the stored PKCE credential from the local credential store.
"""

from __future__ import annotations

import abc
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("guardbench.orcarouter.credentials")

DEFAULT_API_BASE = "https://api.orcarouter.ai/v1"
DEFAULT_AUTH_BASE = "https://www.orcarouter.ai"
API_KEY_ENV = "ORCAROUTER_API_KEY"
# Purpose-neutral placeholder written to logs / display instead of a secret.
MASK = "********"

_KEY_RE = re.compile(r"^sk-orca-[A-Za-z0-9._\-]{8,}$")

# Credential sources
SOURCE_CONFIG = "api_key_config"
SOURCE_ENV = "api_key_env"
SOURCE_PKCE = "pkce"


class OrcaRouterAuthError(Exception):
    """Base class for credential problems that the user must act on."""

    terminal = False


class MissingCredentialError(OrcaRouterAuthError):
    """No usable credential could be resolved."""

    terminal = True

    def __init__(self, message: str, source: Optional[str] = None):
        super().__init__(message)
        self.source = source


class InvalidCredentialError(OrcaRouterAuthError):
    """The credential has the wrong shape (corrupt key)."""

    terminal = True


class ReauthenticationRequired(OrcaRouterAuthError):
    """The credential was rejected by OrcaRouter (HTTP 401).

    The generation is the one that issued the rejected request.  A failure for
    an old generation must never mark a freshly stored credential as broken.
    """

    terminal = True

    def __init__(self, message: str, source: str, generation: Optional[str]):
        super().__init__(message)
        self.source = source
        self.generation = generation


class CredentialScopeError(OrcaRouterAuthError):
    """The granted OAuth scope does not cover inference requests."""

    terminal = True


@dataclass(frozen=True)
class Credential:
    """A plain OrcaRouter API key, plus the provenance needed for lifecycle."""

    api_key: str
    source: str
    generation: Optional[str] = None

    def __str__(self) -> str:
        return f"Credential(source={self.source}, generation={self.generation}, api_key={MASK})"

    def __repr__(self) -> str:
        return str(self)

    def masked(self) -> str:
        return mask_secret(self.api_key)


def mask_secret(value: Optional[str]) -> str:
    if not value:
        return ""
    if len(value) <= 12:
        return MASK
    return f"{value[:8]}…{value[-4:]}"


def validate_key_shape(key: str) -> str:
    key = (key or "").strip()
    if not _KEY_RE.match(key):
        raise InvalidCredentialError(
            f"Credentials are malformed (expected the OrcaRouter key prefix, got {mask_secret(key)!r})"
        )
    return key


def is_key_shape(key: Optional[str]) -> bool:
    return bool(key) and bool(_KEY_RE.match(key.strip()))


def _default_credential_path() -> Path:
    explicit = os.environ.get("ORCAROUTER_CREDENTIAL_FILE")
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".config" / "guardbench" / "orcarouter-auth.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _restrict_permissions(path: Path) -> None:
    """Best effort 0600 -- the file holds a secret."""
    try:
        os.chmod(path, 0o600)
    except OSError:  # e.g. non-POSIX filesystems
        pass


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".orcarouter-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    _restrict_permissions(path)


class FileCredentialStore:
    """Small JSON store for the PKCE-obtained key.

    ``{"version": 1, "generation": "<uuid>", "key": "sk-orca-…", ...}``
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else _default_credential_path()

    def load(self) -> Optional[Dict]:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            logger.warning("Stored OrcaRouter credential is unreadable: %s", exc)
            return None
        if not isinstance(data, dict) or not isinstance(data.get("key"), str):
            return None
        return data

    def save(self, data: Dict) -> None:
        _atomic_write(self.path, json.dumps(data, indent=2, sort_keys=True))

    def clear(self) -> bool:
        try:
            self.path.unlink()
            return True
        except FileNotFoundError:
            return False

    def update_status(self, generation: Optional[str], status: str) -> bool:
        """Mark a stored credential's status (e.g. ``needs_reauth``) if it is
        still the current generation. Returns True if the record changed."""
        data = self.load()
        if not data or data.get("generation") != generation:
            return False
        data["status"] = status
        data["updated_at"] = _now()
        self.save(data)
        return True


class CredentialSource(abc.ABC):
    """Adapter interface: obtain a credential from one source."""

    source_name: str = "unknown"

    @abc.abstractmethod
    def obtain(self) -> Credential:  # pragma: no cover - interface
        raise NotImplementedError


class ApiKeyCredentialSource(CredentialSource):
    """Adapter for an existing key (config file or environment variable)."""

    source_name = SOURCE_CONFIG

    def __init__(self, key: Optional[str], source_name: str = SOURCE_CONFIG):
        self._key = key
        self.source_name = source_name

    def obtain(self) -> Credential:
        if not self._key or not self._key.strip():
            raise MissingCredentialError(
                f"No OrcaRouter API key found. Set {API_KEY_ENV} or pass "
                "'params.api_key', or run `guardbench orcarouter connect`.",
                source=self.source_name,
            )
        return Credential(api_key=validate_key_shape(self._key), source=self.source_name)


REQUIRED_SCOPES = ("api",)


def check_granted_scope(granted: Optional[str]) -> None:
    """Fail closed unless the *granted* scope covers inference."""
    scopes = set((granted or "").split())
    if not scopes or not scopes.intersection(REQUIRED_SCOPES):
        raise CredentialScopeError(
            f"OrcaRouter granted scope {granted!r}; inference requires 'api'."
        )


def resolve_credential(
    config_key: Optional[str] = None,
    store: Optional[FileCredentialStore] = None,
    env: Optional[Dict[str, str]] = None,
) -> Credential:
    """Resolve one credential through the seam. Providers only need this."""
    env = os.environ if env is None else env
    store = store or FileCredentialStore()

    if config_key:
        return ApiKeyCredentialSource(config_key, SOURCE_CONFIG).obtain()

    env_key = env.get(API_KEY_ENV)
    if env_key:
        return ApiKeyCredentialSource(env_key, SOURCE_ENV).obtain()

    stored = store.load()
    if stored and stored.get("status") == "needs_reauth":
        raise ReauthenticationRequired(
            "Your stored OrcaRouter login was revoked or expired. Run "
            "`guardbench orcarouter login` again, or set "
            f"{API_KEY_ENV}.",
            source=SOURCE_PKCE,
            generation=stored.get("generation"),
        )
    if stored and stored.get("key"):
        cred = Credential(
            api_key=validate_key_shape(stored["key"]),
            source=SOURCE_PKCE,
            generation=stored.get("generation"),
        )
        return cred

    raise MissingCredentialError(
        f"No OrcaRouter credential found. Set {API_KEY_ENV}, pass 'params.api_key', "
        "or run `guardbench orcarouter login` (OAuth 2.0 + PKCE).",
        source=None,
    )


def mark_needs_reauth(credential: Credential, store: Optional[FileCredentialStore] = None) -> bool:
    """Terminal reauthentication: only mark the *exact* credential generation
    that issued the rejected request. API-key sources are left untouched."""
    if credential.source != SOURCE_PKCE or not credential.generation:
        return False
    store = store or FileCredentialStore()
    changed = store.update_status(credential.generation, "needs_reauth")
    if changed:
        logger.warning(
            "OrcaRouter credential generation %s was rejected (HTTP 401); "
            "marked needs_reauth. Run `guardbench orcarouter login`.",
            credential.generation,
        )
    return changed


def describe_sources(
    config_key: Optional[str] = None,
    store: Optional[FileCredentialStore] = None,
    env: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    """Credential status for `guardbench orcarouter status` (never the key)."""
    env = os.environ if env is None else env
    store = store or FileCredentialStore()
    stored = store.load()
    out = []
    if config_key:
        out.append({"source": SOURCE_CONFIG, "status": "configured", "key": mask_secret(config_key)})
    if env.get(API_KEY_ENV):
        out.append({"source": SOURCE_ENV, "status": "configured", "key": mask_secret(env[API_KEY_ENV])})
    if stored:
        status = stored.get("status", "active")
        item = {
            "source": SOURCE_PKCE,
            "status": status,
            "key": mask_secret(stored.get("key")),
            "generation": str(stored.get("generation")),
            "scope": str(stored.get("scope")),
            "created_at": str(stored.get("created_at")),
        }
        out.append(item)
    return out


def with_generation(credential: Credential, generation: str) -> Credential:
    return replace(credential, generation=generation)
