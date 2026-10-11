"""Tiny JSON-over-HTTP helper with retries. Standard library only."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Dict, Optional


class ModelAPIError(RuntimeError):
    pass


def post_json(url: str, payload: dict, headers: Optional[Dict[str, str]] = None, timeout: float = 600, retries: int = 4) -> dict:
    body = json.dumps(payload).encode()
    hdrs = {"content-type": "application/json"}
    hdrs.update(headers or {})
    delay = 2.0
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            if exc.code in (408, 409, 425, 429, 500, 502, 503, 504, 529) and attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise ModelAPIError(f"{exc.code} from {url}: {detail}") from None
        except urllib.error.URLError as exc:
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise ModelAPIError(f"cannot reach {url}: {exc.reason}") from None
    raise ModelAPIError(f"gave up on {url}")
