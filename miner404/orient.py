"""The renderer interface the pipeline uses."""

from __future__ import annotations

from typing import Protocol

from witness_client import RenderOptions


class Renderer(Protocol):
    def render(self, source: str, options: RenderOptions) -> bytes: ...
