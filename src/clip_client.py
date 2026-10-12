"""HTTP client for the CLIP retrieval server."""

import time
from typing import List

import requests
from requests.adapters import HTTPAdapter


class ServerError(RuntimeError):
    """HTTP error response from the server. ``retryable`` is False for 4xx other than 429, since
    resending such a request cannot succeed."""

    def __init__(self, url: str, status: int, detail: str):
        super().__init__(f"HTTP {status} from {url}: {detail}")
        self.status = status
        self.detail = detail
        self.retryable = status >= 500 or status == 429


class CLIPClient:
    TIMEOUT = (10, 300)  # (connect, read) seconds
    RETRIES = 3

    def __init__(self, api_base: str = "http://localhost:8001") -> None:
        self.api_base = api_base.rstrip("/")
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def _post(self, path: str, body: dict) -> dict:
        url = f"{self.api_base}{path}"
        for attempt in range(self.RETRIES + 1):
            try:
                resp = self._session.post(url, json=body, timeout=self.TIMEOUT)
                if resp.status_code >= 400:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except ValueError:
                        detail = resp.text
                    raise ServerError(url, resp.status_code, str(detail)[:2000])
                return resp.json()
            except (requests.ConnectionError, requests.Timeout, ServerError) as e:
                if (isinstance(e, ServerError) and not e.retryable) or attempt == self.RETRIES:
                    raise
                print(f"Warning: CLIP server error (attempt {attempt + 1}/{self.RETRIES + 1}): {e}")
                time.sleep(2 ** attempt)

    def retrieve(self, dataset: str, query_id: str, k: int) -> List[str]:
        """IDs of the ``k`` training images most similar to test image ``query_id`` (RICES)."""
        return self._post("/v1/retrieve", {"dataset": dataset, "query_id": query_id, "k": k})["ids"]

    def encode_text(self, texts: List[str]) -> List[List[float]]:
        """L2-normalized CLIP text embeddings."""
        return self._post("/v1/encode_text", {"texts": texts})["embeddings"]
