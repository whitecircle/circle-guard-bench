"""OrcaRouter Cyber Inference integration for CircleGuardBench.

OrcaRouter (https://www.orcarouter.ai/cyber) is an OpenAI-compatible inference
gateway for security teams doing authorized work.  This package adds:

* ``credentials`` -- one credential seam with two adapters: a pasted
  ``sk-orca-…`` API key, or "Connect with OrcaRouter" (OAuth 2.0 + PKCE);
* ``pkce`` -- the PKCE flows (loopback redirect, out-of-band, device grant);
* ``catalog`` -- live model discovery from ``GET /v1/models`` with
  capability filters and a verified cold-start fallback.
"""

from circleguardbench.orcarouter.catalog import Catalog, load_catalog  # noqa: F401
from circleguardbench.orcarouter.credentials import (  # noqa: F401
    Credential,
    FileCredentialStore,
    OrcaRouterAuthError,
    resolve_credential,
)
from circleguardbench.orcarouter.pkce import OrcaEndpoints  # noqa: F401
