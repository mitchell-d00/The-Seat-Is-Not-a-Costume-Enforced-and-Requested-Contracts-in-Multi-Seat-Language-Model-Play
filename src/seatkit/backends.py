"""Model backends.

The protocol is deliberately narrow: a prompt goes in, a `Completion` comes out.
Everything E1-E4 needs sits behind that one method, which is why adding a
provider does not touch experimental logic.

    mock                      deterministic, offline, no key
    openai:<model-id>         OPENAI_API_KEY
    anthropic:<model-id>      ANTHROPIC_API_KEY
    grok:<model-id>           XAI_API_KEY        (OpenAI-compatible endpoint)
    gemini:<model-id>         GEMINI_API_KEY or GOOGLE_API_KEY

Wrap any of them in `CachedBackend` to make reruns free.

A NOTE ON WHAT THE MOCK IS FOR. `MockBackend` uses a forbidden fact only when
that fact's value appears in the prompt it was handed, and never invents one.
Prior bleed is therefore exactly zero, so any nonzero L3/L4 leak in a mock run
is a routing bug in `ladder.py` and nothing else. That makes it a good
regression harness and a bad experiment: its L0 > L1 > L2 ordering is stipulated
by three constructor constants, not discovered. No number it produces belongs in
a results table (paper 9.5).

A NOTE ON CROSS-FAMILY COMPARISON. `prompt_tokens` is whatever the provider
reports, and tokenizers differ between families. It is fine for cost accounting
and wrong for comparing leak rate against context length across providers, which
is exactly what E1 does. Use `prompt_words`, which every backend computes the
same way from the same string. See docs/backends.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class Completion:
    """One model response.

    Attributes:
        text: the completion.
        backend: provider:model string, recorded in results so a table can never
            be silently assembled from mixed sources.
        prompt_tokens: provider-reported input tokens. NOT comparable across
            families; use for cost, not for analysis.
        prompt_words: whitespace-delimited word count of the prompt, computed
            identically everywhere. This is the cross-family context-length
            measure E1 should regress on.
        raw_usage: the provider's usage object, verbatim, for auditing.
    """

    text: str
    backend: str
    prompt_tokens: int
    prompt_words: int = 0
    raw_usage: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    name: str

    def generate(self, prompt: str, *, temperature: float = 1.0,
                 max_tokens: int = 300, seed: int | None = None) -> Completion:
        ...


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

HEDGES = (
    "it may be", "arguably", "to some extent", "broadly speaking",
    "that said", "it's worth noting", "in fairness", "roughly",
)

GEO_MOVES = (
    "The banding is consistent with seasonal deposition.",
    "I can't date it more tightly than the varve count allows.",
    "The grain sorting points to a low-energy depositional setting.",
    "Nothing in the section requires an unconformity.",
)

BIO_MOVES = (
    "The preservation is too good for long transport.",
    "I'd want a second core before committing to the assemblage.",
    "The taphonomy suggests rapid burial.",
    "There's no trace fossil evidence for bioturbation at that level.",
)


class MockBackend:
    """Deterministic pseudo-model for offline runs and CI.

    Within the shared-window levels it models compliance decay: the chance of
    using a visible forbidden fact rises with prompt length, falls when a
    prohibition is stated (L1), and falls further when one is re-injected at the
    tail (L2).

    HONESTY NOTE. That L0 > L1 > L2 ordering is *stipulated by these constants*,
    not discovered. Mock runs demonstrate that the harness is wired correctly and
    that the L3/L4 architectural claim holds by construction. They are not
    evidence for the behavioural claim, and no number produced by this class
    belongs in a results table. Paper section 9.5 requires a live backend on at
    least two model families.
    """

    name = "mock"

    def __init__(self, leak_base: float = 0.05, decay_scale: float = 2200.0,
                 stated_discount: float = 0.55,
                 reminder_discount: float = 0.5) -> None:
        self.leak_base = leak_base
        self.decay_scale = decay_scale
        self.stated_discount = stated_discount
        self.reminder_discount = reminder_discount

    def _rng(self, prompt: str, seed: int | None) -> random.Random:
        h = hashlib.sha256(f"{seed}:{prompt}".encode()).hexdigest()
        return random.Random(int(h[:16], 16))

    def _leak_probability(self, prompt: str) -> float:
        """Stipulated compliance model. See the honesty note in the class docstring.

        Three factors, matching the drivers named in paper section 3:
          - context length (conditioning drift),
          - presence of a prohibition anywhere (a stated request),
          - presence of a prohibition near the tail (the L2 manipulation).
        """
        n = len(prompt.split())
        p = self.leak_base + (1 - self.leak_base) * (
            1 - pow(2.718281828, -n / self.decay_scale)
        )
        if "must not use it" in prompt:
            p *= self.stated_discount           # L1: the paragraph, stated once
        if "REMINDER" in prompt[-1200:]:
            p *= self.reminder_discount         # L2: re-injected at the tail
        return min(p, 0.97)

    # Values look like "18.4 ppm", "142.7 SI units", "37.1 per kilogram".
    _VALUE_RE = re.compile(r"\bis (\d+\.\d+ [A-Za-z][A-Za-z ]*?)\.")

    def _visible_forbidden(self, prompt: str) -> list[str]:
        """Values the prompt actually contains.

        The mock never invents a value it was not shown. That sets prior bleed
        to exactly zero, so any nonzero L3/L4 leak in a mock run is a routing
        bug in ladder.py and nothing else.
        """
        return [m.group(1).strip() for m in self._VALUE_RE.finditer(prompt)]

    def generate(self, prompt: str, *, temperature: float = 1.0,
                 max_tokens: int = 300, seed: int | None = None) -> Completion:
        rng = self._rng(prompt, seed)
        is_geo = "geolog" in prompt.lower()[:600]
        moves = GEO_MOVES if is_geo else BIO_MOVES
        out = [rng.choice(moves)]

        # Dialect capture: hedge rate rises with how many hedges are already in
        # the window. This is the section 3 "third driver" made mechanical.
        window_hedges = sum(prompt.lower().count(h) for h in HEDGES)
        if rng.random() < min(0.15 + 0.04 * window_hedges, 0.8):
            out.insert(0, rng.choice(HEDGES).capitalize() + ",")

        # Agreement pressure: convergence toward the other seat's last claim.
        if rng.random() < min(0.05 + 0.03 * (len(prompt.split()) / 400), 0.6):
            out.append("I don't think we actually disagree there.")

        visible = self._visible_forbidden(prompt)
        if visible and temperature > 0:
            p = self._leak_probability(prompt) * min(temperature, 1.0)
            if rng.random() < p:
                out.append(f"Which lines up with {rng.choice(visible)}, if that holds.")

        words = len(prompt.split())
        return Completion(text=" ".join(out), backend=self.name,
                          prompt_tokens=words, prompt_words=words)


# ---------------------------------------------------------------------------
# Shared live-backend plumbing
# ---------------------------------------------------------------------------

class BackendError(RuntimeError):
    pass


def _require_key(env_names: tuple[str, ...], explicit: str | None,
                 provider: str) -> str:
    if explicit:
        return explicit
    for n in env_names:
        if os.environ.get(n):
            return os.environ[n]
    raise BackendError(
        f"No API key for {provider}. Set one of {', '.join(env_names)}, "
        f"or pass api_key=. Use 'mock' for offline runs."
    )


TRANSIENT = ("rate", "429", "timeout", "timed out", "overloaded", "500", "502",
             "503", "504", "unavailable", "connection")


def _retry(fn, *, attempts: int = 5, base: float = 1.5, provider: str = "",
           sleep=time.sleep):
    """Exponential backoff on transient failures.

    Retries rate limits, timeouts, and 5xx. Does not retry auth or bad-request
    errors, because retrying those just burns the budget more slowly. A run of
    several thousand calls will hit rate limits; a run that silently dropped
    those failures would have an unrecorded exclusion rule, which is exactly
    what the preregistration forbids.
    """
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - providers raise unrelated types
            last = exc
            if not any(t in str(exc).lower() for t in TRANSIENT):
                raise
            if i == attempts - 1:
                break
            sleep(base ** i)
    raise BackendError(f"{provider}: giving up after {attempts} attempts: {last}")


class _OpenAICompatible:
    """Base for any provider exposing the OpenAI chat-completions shape.

    OpenAI and xAI both do. The only differences are the base URL, the env var,
    and the label that ends up in `Completion.backend` - which must stay
    distinct, so a results table can never be assembled from two providers
    without that being visible in the data.
    """

    name = "openai"
    base_url: str | None = None
    env_keys: tuple[str, ...] = ("OPENAI_API_KEY",)

    def __init__(self, model: str, api_key: str | None = None,
                 system: str = "", base_url: str | None = None,
                 timeout: float = 120.0) -> None:
        self.model = model
        self.system = system
        self.timeout = timeout
        self._key = _require_key(self.env_keys, api_key, self.name)
        self._base_url = base_url or self.base_url
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise BackendError("pip install openai") from exc
        kwargs: dict[str, Any] = {"api_key": self._key, "timeout": timeout}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(**kwargs)

    def _messages(self, prompt: str) -> list[dict]:
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def generate(self, prompt: str, *, temperature: float = 1.0,
                 max_tokens: int = 300, seed: int | None = None) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            # Best-effort determinism. Provider-supported but not guaranteed;
            # do not describe cross-run results as reproducible on this basis.
            kwargs["seed"] = seed

        try:
            resp = _retry(lambda: self._client.chat.completions.create(**kwargs),
                          provider=self.name)
        except Exception as exc:
            # Some models reject `max_tokens`, `temperature`, or `seed` outright.
            # Drop the contested parameters once and retry, rather than failing a
            # multi-thousand-call run over a parameter name.
            msg = str(exc).lower()
            retried = dict(kwargs)
            changed = False
            if "max_tokens" in msg or "max_completion_tokens" in msg:
                retried.pop("max_tokens", None)
                retried["max_completion_tokens"] = max_tokens
                changed = True
            if "temperature" in msg:
                retried.pop("temperature", None)
                changed = True
            if "seed" in msg:
                retried.pop("seed", None)
                changed = True
            if not changed:
                raise
            resp = _retry(lambda: self._client.chat.completions.create(**retried),
                          provider=self.name)

        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        if hasattr(usage, "model_dump"):
            raw = usage.model_dump()
        elif usage:
            raw = {"prompt_tokens": pt}
        else:
            raw = {}
        return Completion(text=text, backend=f"{self.name}:{self.model}",
                          prompt_tokens=pt, prompt_words=len(prompt.split()),
                          raw_usage=raw)


class OpenAIBackend(_OpenAICompatible):
    """OpenAI chat completions. Requires OPENAI_API_KEY and `pip install openai`."""

    name = "openai"
    base_url = None
    env_keys = ("OPENAI_API_KEY",)


class GrokBackend(_OpenAICompatible):
    """xAI Grok via its OpenAI-compatible endpoint.

    Requires XAI_API_KEY and `pip install openai` - no separate SDK. The label
    stays "grok" so results never silently merge with OpenAI's.
    """

    name = "grok"
    base_url = "https://api.x.ai/v1"
    env_keys = ("XAI_API_KEY", "GROK_API_KEY")


class AnthropicBackend:
    """Anthropic messages API. Requires ANTHROPIC_API_KEY and `pip install anthropic`."""

    name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None,
                 system: str = "", timeout: float = 120.0) -> None:
        self.model = model
        self.system = system
        self._key = _require_key(("ANTHROPIC_API_KEY",), api_key, self.name)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise BackendError("pip install anthropic") from exc
        self._client = anthropic.Anthropic(api_key=self._key, timeout=timeout)

    def generate(self, prompt: str, *, temperature: float = 1.0,
                 max_tokens: int = 300, seed: int | None = None) -> Completion:
        # No seed parameter on this API. Accepted and ignored so the protocol
        # stays uniform; runs are not reproducible on this backend.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.system:
            kwargs["system"] = self.system

        resp = _retry(lambda: self._client.messages.create(**kwargs),
                      provider=self.name)
        text = "".join(b.text for b in resp.content
                       if getattr(b, "type", "") == "text").strip()
        return Completion(
            text=text, backend=f"{self.name}:{self.model}",
            prompt_tokens=resp.usage.input_tokens,
            prompt_words=len(prompt.split()),
            raw_usage={"input_tokens": resp.usage.input_tokens,
                       "output_tokens": resp.usage.output_tokens},
        )


class GeminiBackend:
    """Google Gemini.

    Prefers the `google-genai` SDK and falls back to the older
    `google-generativeai` package, because which one is current has changed more
    than once. If both import paths fail, install one of them.
    """

    name = "gemini"

    def __init__(self, model: str, api_key: str | None = None,
                 system: str = "", timeout: float = 120.0) -> None:
        self.model = model
        self.system = system
        self.timeout = timeout
        self._key = _require_key(("GEMINI_API_KEY", "GOOGLE_API_KEY"),
                                 api_key, self.name)
        try:
            from google import genai
            self._client = genai.Client(api_key=self._key)
            self._mode = "genai"
        except ImportError:
            try:
                import google.generativeai as legacy
                legacy.configure(api_key=self._key)
                self._client = legacy
                self._mode = "legacy"
            except ImportError as exc:  # pragma: no cover
                raise BackendError(
                    "pip install google-genai  (or google-generativeai)"
                ) from exc

    def generate(self, prompt: str, *, temperature: float = 1.0,
                 max_tokens: int = 300, seed: int | None = None) -> Completion:
        words = len(prompt.split())

        if self._mode == "genai":
            from google.genai import types
            cfg: dict[str, Any] = {"temperature": temperature,
                                   "max_output_tokens": max_tokens}
            if self.system:
                cfg["system_instruction"] = self.system
            if seed is not None:
                cfg["seed"] = seed

            def call(c=cfg):
                return self._client.models.generate_content(
                    model=self.model, contents=prompt,
                    config=types.GenerateContentConfig(**c))

            try:
                resp = _retry(call, provider=self.name)
            except Exception as exc:
                if seed is None or "seed" not in str(exc).lower():
                    raise
                cfg.pop("seed")
                resp = _retry(call, provider=self.name)
        else:
            gen_cfg = {"temperature": temperature, "max_output_tokens": max_tokens}
            model = self._client.GenerativeModel(
                self.model, system_instruction=self.system or None,
                generation_config=gen_cfg)
            resp = _retry(lambda: model.generate_content(prompt),
                          provider=self.name)

        usage = getattr(resp, "usage_metadata", None)
        pt = getattr(usage, "prompt_token_count", 0) or 0
        return Completion(
            text=(getattr(resp, "text", "") or "").strip(),
            backend=f"{self.name}:{self.model}",
            prompt_tokens=pt, prompt_words=words,
            raw_usage={"prompt_token_count": pt},
        )


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class CachedBackend:
    """Disk cache wrapping any backend. Satisfies the same protocol.

    A full grid is scenes x turns x seats x levels calls, which reaches five
    figures quickly. The cache makes reruns free and, more usefully, makes
    re-analysis free: E2 re-judges transcripts E1 already generated, and there is
    no reason to pay for those completions twice.

    Keyed on (provider, model, prompt, temperature, max_tokens, seed).

    CAVEAT. For backends without a working seed parameter this caches one sample
    from a stochastic process. That is right for re-analysis and wrong if you
    wanted fresh draws - a cached run is one run replayed, not a replication.
    Clear the directory or pass write=False to force new samples.
    """

    def __init__(self, inner: Backend, cache_dir: str | Path = ".cache/seatkit",
                 write: bool = True) -> None:
        self.inner = inner
        self.name = f"cached({getattr(inner, 'name', 'backend')})"
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.write = write
        self.hits = 0
        self.misses = 0

    def _key(self, prompt: str, temperature: float, max_tokens: int,
             seed: int | None) -> Path:
        blob = json.dumps([getattr(self.inner, "name", "backend"),
                           getattr(self.inner, "model", ""),
                           prompt, temperature, max_tokens, seed],
                          sort_keys=True)
        return self.dir / (hashlib.sha256(blob.encode()).hexdigest() + ".json")

    def generate(self, prompt: str, *, temperature: float = 1.0,
                 max_tokens: int = 300, seed: int | None = None) -> Completion:
        path = self._key(prompt, temperature, max_tokens, seed)
        if path.exists():
            try:
                self.hits += 1
                return Completion(**json.loads(path.read_text()))
            except Exception:
                self.hits -= 1
                path.unlink(missing_ok=True)   # corrupt entry; refetch
        self.misses += 1
        comp = self.inner.generate(prompt, temperature=temperature,
                                   max_tokens=max_tokens, seed=seed)
        if self.write:
            path.write_text(json.dumps({
                "text": comp.text, "backend": comp.backend,
                "prompt_tokens": comp.prompt_tokens,
                "prompt_words": comp.prompt_words,
                "raw_usage": comp.raw_usage,
            }))
        return comp

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0}


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, type] = {
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "grok": GrokBackend,
    "xai": GrokBackend,          # alias
    "gemini": GeminiBackend,
    "google": GeminiBackend,     # alias
}

FAMILY_OF = {
    "openai": "openai", "anthropic": "anthropic", "grok": "xai", "xai": "xai",
    "gemini": "google", "google": "google", "mock": "mock",
}

ENV_KEY_FOR = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "grok": ("XAI_API_KEY", "GROK_API_KEY"),
    "xai": ("XAI_API_KEY", "GROK_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "mock": (),
}


def parse_spec(spec: str) -> tuple[str, str]:
    """Split 'provider:model-id' into its parts. 'mock' takes no model."""
    spec = spec.strip()
    if spec == "mock":
        return "mock", ""
    if ":" not in spec:
        raise ValueError(
            f"backend spec {spec!r} needs a model id, e.g. 'openai:<model-id>'. "
            f"Known providers: {', '.join(sorted(PROVIDERS))}, mock."
        )
    provider, model = spec.split(":", 1)
    provider = provider.strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(
            f"unknown provider {provider!r}. "
            f"Known: {', '.join(sorted(PROVIDERS))}, mock."
        )
    if not model.strip():
        raise ValueError(f"backend spec {spec!r} has an empty model id")
    return provider, model.strip()


def get_backend(spec: str, *, cache: bool | str | Path = False,
                system: str = "", **kwargs) -> Backend:
    """Resolve a backend from a config string.

        get_backend("mock")
        get_backend("openai:<model-id>")
        get_backend("anthropic:<model-id>", cache=True)
        get_backend("grok:<model-id>")
        get_backend("gemini:<model-id>", cache=".cache/run7")

    Every experiment takes its backend through here, so adding a provider needs
    no change to E1-E4.
    """
    provider, model = parse_spec(spec)
    inner: Backend = (MockBackend() if provider == "mock"
                      else PROVIDERS[provider](model=model, system=system, **kwargs))
    if cache:
        cache_dir = cache if isinstance(cache, (str, Path)) else ".cache/seatkit"
        return CachedBackend(inner, cache_dir=cache_dir)
    return inner


def family_of(spec: str) -> str:
    """Model family for a spec. Paper 9.5 requires at least two distinct ones.

    Grok and OpenAI share a wire protocol and are still different families; the
    shape of the endpoint has nothing to do with the weights.
    """
    return FAMILY_OF[parse_spec(spec)[0]]


def available_backends() -> dict[str, bool]:
    """Which providers have a key present. For preflight, never for logic."""
    out = {"mock": True}
    for p in sorted(set(PROVIDERS)):
        out[p] = any(os.environ.get(k) for k in ENV_KEY_FOR[p])
    return out
