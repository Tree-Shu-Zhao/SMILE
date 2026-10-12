"""Client for an OpenAI-compatible VLM server (vLLM), and the image encoding of the prompts."""

import base64
import io
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI
from PIL import Image


def image_content(path: str, max_side: int) -> Dict[str, Any]:
    """Prompt part for the image at ``path``: a base64 JPEG data URL, with the longer side
    resized to at most ``max_side``."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


class RequestRejected(RuntimeError):
    """The server rejected the request itself (HTTP 4xx, e.g. more images than
    --limit-mm-per-prompt or a prompt longer than --max-model-len): resending it cannot help."""
    retryable = False


def _retryable(exc: Exception) -> bool:
    """Connection errors, timeouts, 429 and 5xx are transient, as is vLLM's tokenizer race
    (HTTP 400 "Already borrowed"), which succeeds when re-sent. Other 4xx responses are not."""
    status = getattr(exc, "status_code", None)
    if status is None or status == 429 or status >= 500:
        return True
    return "Already borrowed" in str(exc)


class VLMClient:
    """Chat completions against ``api_base``. Transient errors are retried ``max_retries`` times
    with backoff, then raised; any other error raises ``RequestRejected`` at once. ``timeout``
    (seconds) is per attempt."""

    def __init__(self, model: str, api_base: str = "http://localhost:8000/v1",
                 max_retries: int = 3, timeout: float = 300.0) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.max_retries = max_retries
        self._client = OpenAI(base_url=self.api_base, api_key="EMPTY", max_retries=0, timeout=timeout)

    def served_models(self) -> List[str]:
        return [m.id for m in self._client.models.list().data]

    def vllm_version(self) -> Optional[str]:
        try:
            r = requests.get(self.api_base.removesuffix("/v1") + "/version", timeout=10)
            return r.json().get("version") if r.status_code == 200 else None
        except Exception:
            return None

    def chat(self, content: List[Dict[str, Any]], temperature: float = 0.0, max_tokens: int = 512,
             n: int = 1) -> List[str]:
        """Send one user message; return the ``n`` answers (the text after a ``</think>`` tag, if any)."""
        messages = [{"role": "user", "content": content}]
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
                break
            except Exception as e:
                if not _retryable(e):
                    raise RequestRejected(f"VLM server rejected the request: {e}") from e
                if attempt == self.max_retries:
                    raise RuntimeError(f"VLM API error after {attempt + 1} attempts: {e}") from e
                print(f"Warning: API error (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                time.sleep(2 ** attempt)
        return [(c.message.content or "").rsplit("</think>", 1)[-1].strip() for c in resp.choices]
