"""Model backends.

Two implementations ship:

  MockBackend       deterministic, offline, no API key. Its leak behaviour is
                    driven entirely by what is in the assembled view, which
                    means running the experiments against it demonstrates the
                    *architectural* claim (L3/L4 cannot leak because the tokens
                    are absent) without pretending to demonstrate the
                    *behavioural* claim (compliance decays with context length).
                    Only a real model can show the second.

  AnthropicBackend  live calls. Requires ANTHROPIC_API_KEY.

The distinction matters and the repository should not blur it. Numbers produced
by the mock are a smoke test of the harness. Numbers reported in a paper must
come from a live backend on at least two model families (paper 9.5).
"""

from __future__ import annotations

import hashlib
import os
import random
import re
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Completion:
    text: str
    backend: str
    prompt_tokens: int


class Backend(Protocol):
    name: str

    def generate(self, prompt: str, *, temperature: float = 1.0,
                 max_tokens: int = 300, seed: int | None = None) -> Completion:
        ...


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

    Leak model: the mock will use a forbidden fact only if the fact's value
    string is present in the prompt it was handed. It never invents one. This
    is a deliberate simplification with a purpose - it isolates routing bleed
    from compliance bleed and sets prior bleed to exactly zero, so any nonzero
    L3/L4 leak in a mock run is unambiguously a bug in ladder.py.

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

        return Completion(text=" ".join(out), backend=self.name,
                          prompt_tokens=len(prompt.split()))


class AnthropicBackend:
    """Live backend. Model id is passed in, not hardcoded to a default."""

    name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None,
                 system: str = "") -> None:
        self.model = model
        self.system = system
        self._key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._key:
            raise RuntimeError(
                "No API key. Set ANTHROPIC_API_KEY or pass api_key=. "
                "Use MockBackend for offline runs."
            )
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install anthropic") from exc

    def generate(self, prompt: str, *, temperature: float = 1.0,
                 max_tokens: int = 300, seed: int | None = None) -> Completion:
        import anthropic

        client = anthropic.Anthropic(api_key=self._key)
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.system:
            kwargs["system"] = self.system
        resp = client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return Completion(text=text, backend=f"{self.name}:{self.model}",
                          prompt_tokens=resp.usage.input_tokens)


def get_backend(spec: str) -> Backend:
    """Resolve a backend from a config string: 'mock' or 'anthropic:<model-id>'."""
    if spec == "mock":
        return MockBackend()
    if spec.startswith("anthropic:"):
        return AnthropicBackend(model=spec.split(":", 1)[1])
    raise ValueError(f"unknown backend spec: {spec!r}")
