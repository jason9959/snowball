"""Small urllib adapter matching the subset of requests used by providers."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HttpResponse:
    content: bytes
    url: str

    @property
    def text(self) -> str:
        return self.content.decode("utf-8-sig")

    def json(self) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        return None


def get(url: str, *, params=None, headers=None, timeout=30) -> HttpResponse:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    seconds = max(timeout) if isinstance(timeout, tuple) else timeout
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=seconds) as response:
        return HttpResponse(content=response.read(), url=response.geturl())
