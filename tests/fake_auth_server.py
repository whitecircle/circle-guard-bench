import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional


class FakeAuthServer:
    """Tiny local stand-in for www.orcarouter.ai's auth API.

    Records every request so tests can assert *where* requests went and what
    they carried (the verifier must only ever appear in the exchange body).
    """

    def __init__(self):
        self.requests: List[Dict] = []
        self.exchange_status = 200
        self.exchange_body: Dict = {"key": "sk-orca-test-key-0123456789abcdef", "user_id": "u_test", "scope": "api"}
        self.device_polls: List[Dict] = []
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def _read(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length).decode("utf-8") if length else ""
                ctype = self.headers.get("Content-Type", "")
                if "json" in ctype:
                    return json.loads(raw) if raw else {}
                return {k: v[0] for k, v in urllib.parse.parse_qs(raw).items()}

            def _send(self, status, obj):
                body = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = self._read()
                recorder.requests.append({"path": self.path, "body": body,
                                          "auth": self.headers.get("Authorization")})
                if self.path == "/api/v1/auth/keys":
                    if recorder.exchange_status != 200:
                        self._send(recorder.exchange_status,
                                   {"error": "invalid_grant", "error_description": "code expired"})
                    else:
                        self._send(200, recorder.exchange_body)
                elif self.path == "/api/v1/auth/device/code":
                    self._send(200, {"device_code": "dev-secret", "user_code": "ABCD-EFGH",
                                     "verification_uri": "https://www.orcarouter.ai/device",
                                     "verification_uri_complete": "https://www.orcarouter.ai/device?code=ABCD-EFGH",
                                     "expires_in": 600, "interval": 1})
                elif self.path == "/api/v1/auth/device/token":
                    poll = recorder.device_polls.pop(0) if recorder.device_polls else {"error": "expired_token"}
                    self._send(200 if not poll.get("error") else 400, poll)
                else:
                    self._send(404, {"error": "not_found"})

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                recorder.requests.append({"path": self.path, "body": dict(urllib.parse.parse_qs(parsed.query)),
                                          "auth": self.headers.get("Authorization")})
                if parsed.path == "/v1/models":
                    if recorder.models_status != 200:
                        self._send(recorder.models_status, {"error": "unauthorized"})
                    else:
                        self._send(200, {"data": recorder.models})
                else:
                    self._send(404, {"error": "not_found"})

        self.models_status = 200
        self.models = DEFAULT_CATALOG
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._server.shutdown()
        self._server.server_close()

    def paths(self) -> List[str]:
        return [r["path"].split("?")[0] for r in self.requests]


def _m(mid, eps, mods=("text",), ctx=None):
    raw = {"id": mid, "name": mid,
           "architecture": {"input_modalities": list(mods), "output_modalities": ["text"]},
           "supported_endpoint_types": list(eps)}
    if ctx:
        raw["context_length"] = ctx
    return raw


# Fixture covering every capability the benchmark can drive, including the
# three OrcaRouter-branded models plus a name that *looks* multimodal but
# declares nothing (must fail closed).
DEFAULT_CATALOG = [
    _m("orcarouter/auto", ["openai", "anthropic", "openai-response"], ctx=1000000),
    _m("orcacyber-ultra-1.0", ["openai", "anthropic"], ctx=200000),
    _m("orcacyber-zero-1.5", ["openai"], ctx=200000),
    _m("orcarouter/fusion", ["openai"]),
    _m("vendor/text-only-guard", ["openai"], ctx=32000),
    _m("vendor/text-only-guard-lookalike", ["openai"], mods=("text",)),
    _m("vendor/vision-guard", ["openai"], mods=("text", "image")),
    _m("vendor/image-gen", ["image-generation"], mods=("text",)),
    _m("vendor/embed-guard", ["embeddings"], mods=("text",)),
    _m("vendor/video-gen", ["openai-video"], mods=("text",)),
    _m("vendor/rerank-guard", ["jina-rerank"], mods=("text",)),
    _m("vendor/text-endpoint-but-image-gen", ["openai", "image-generation"]),
    {"id": "", "architecture": {}},
    "not-a-dict",
]


def fake_endpoints(server: FakeAuthServer):
    from circleguardbench.orcarouter.pkce import OrcaEndpoints

    return OrcaEndpoints(auth_base=server.origin, api_base=server.origin + "/v1")
