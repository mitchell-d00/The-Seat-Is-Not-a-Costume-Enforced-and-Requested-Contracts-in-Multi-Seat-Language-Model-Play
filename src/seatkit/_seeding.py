"""Process-independent seed derivation.

Python's built-in `hash()` on strings and tuples is salted per process
(PYTHONHASHSEED), so a seed built from `hash(seat_id)` changes on every run.
That silently broke two promises the library makes: the mock was not actually
reproducible across processes, and `CachedBackend` (keyed partly on seed)
missed on every rerun, so "reruns are free" was false and a cached live run
re-billed every call. Every seed in the library now goes through `stable_int`.
"""

from __future__ import annotations

import zlib


def stable_int(*parts: object, mod: int = 10**6) -> int:
    """Deterministic non-negative int from arbitrary parts, stable across runs."""
    blob = "\x1f".join(repr(p) for p in parts).encode()
    return zlib.crc32(blob) % mod
