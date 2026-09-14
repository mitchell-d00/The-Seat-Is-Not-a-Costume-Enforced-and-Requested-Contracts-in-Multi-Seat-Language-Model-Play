"""Backend resolution, the shared protocol, retries, and caching.

All offline. Provider classes are exercised by injecting fake clients, so the
adapter code paths are genuinely covered without a key or a network call.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seatkit import (CachedBackend, Completion, MockBackend, available_backends,
                     family_of, get_backend, parse_spec)
from seatkit.backends import (AnthropicBackend, BackendError, GeminiBackend,
                              GrokBackend, OpenAIBackend, _retry)


# -- spec parsing ---------------------------------------------------------

@pytest.mark.parametrize("spec,provider,model", [
    ("mock", "mock", ""),
    ("openai:gpt-x", "openai", "gpt-x"),
    ("anthropic:claude-x", "anthropic", "claude-x"),
    ("grok:grok-x", "grok", "grok-x"),
    ("xai:grok-x", "xai", "grok-x"),
    ("gemini:gemini-x", "gemini", "gemini-x"),
    ("google:gemini-x", "google", "gemini-x"),
    ("  openai:gpt-x  ", "openai", "gpt-x"),
])
def test_parse_spec(spec, provider, model):
    assert parse_spec(spec) == (provider, model)


def test_model_id_may_contain_colons():
    """Vendor ids sometimes carry versions after a colon; split only on the first."""
    assert parse_spec("openai:ft:org:model-v2") == ("openai", "ft:org:model-v2")


@pytest.mark.parametrize("bad", ["openai", "", "openai:", "nope:model", "  "])
def test_parse_spec_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_spec(bad)


def test_unknown_provider_lists_the_known_ones():
    with pytest.raises(ValueError, match="anthropic"):
        parse_spec("acme:model")


def test_family_separates_grok_from_openai():
    """Sharing a wire protocol is not sharing weights.

    Paper 9.5 requires two distinct model families; if these collapsed to one
    label, a run could satisfy that requirement without satisfying it.
    """
    assert family_of("grok:x") != family_of("openai:x")
    assert family_of("xai:x") == family_of("grok:x")
    assert family_of("gemini:x") == family_of("google:x")


def test_available_backends_reports_mock_always():
    assert available_backends()["mock"] is True


# -- the protocol ---------------------------------------------------------

def test_mock_satisfies_protocol_and_is_deterministic():
    b = MockBackend()
    c1 = b.generate("a geologist speaks", seed=7)
    c2 = b.generate("a geologist speaks", seed=7)
    assert c1.text == c2.text
    assert c1.backend == "mock"
    assert c1.prompt_words == len("a geologist speaks".split())


def test_mock_never_invents_a_value_it_was_not_shown():
    """Prior bleed is exactly zero by construction; that is what makes the mock
    a usable regression harness for routing bugs."""
    b = MockBackend()
    for seed in range(60):
        assert "ppm" not in b.generate("no facts here at all", seed=seed).text


def test_prompt_words_is_the_cross_family_measure():
    """prompt_tokens is provider-reported and tokenizer-dependent. prompt_words
    is computed identically everywhere, which is what E1 must regress on."""
    prompt = "one two three four five"
    assert MockBackend().generate(prompt).prompt_words == 5


# -- provider adapters, with injected clients -----------------------------

def _fake_openai_response(text="hi", prompt_tokens=11):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, model_dump=lambda: {
            "prompt_tokens": prompt_tokens}),
    )


def _openai_like(cls, monkeypatch, capture, response=None, fail_with=None):
    b = cls.__new__(cls)
    b.model, b.system, b.timeout = "m", "", 30.0

    def create(**kwargs):
        capture.append(kwargs)
        if fail_with and len(capture) == 1:
            raise RuntimeError(fail_with)
        return response or _fake_openai_response()

    b._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return b


def test_openai_backend_shapes_a_completion(monkeypatch):
    cap = []
    b = _openai_like(OpenAIBackend, monkeypatch, cap)
    c = b.generate("hello there", temperature=0.3, max_tokens=50, seed=5)
    assert c.text == "hi"
    assert c.backend == "openai:m"
    assert c.prompt_tokens == 11 and c.prompt_words == 2
    assert cap[0]["seed"] == 5 and cap[0]["max_tokens"] == 50


def test_grok_is_labelled_separately_from_openai(monkeypatch):
    b = _openai_like(GrokBackend, monkeypatch, [])
    assert b.generate("x").backend.startswith("grok:")
    assert GrokBackend.base_url and "x.ai" in GrokBackend.base_url


def test_openai_backend_retries_without_rejected_parameter(monkeypatch):
    """A model that rejects `temperature` must not kill a several-thousand-call run."""
    cap = []
    b = _openai_like(OpenAIBackend, monkeypatch, cap,
                     fail_with="Unsupported value: 'temperature' is not supported")
    c = b.generate("hello", temperature=0.2)
    assert c.text == "hi"
    assert "temperature" in cap[0] and "temperature" not in cap[1]


def test_openai_backend_does_not_swallow_real_errors(monkeypatch):
    b = _openai_like(OpenAIBackend, monkeypatch, [], fail_with="invalid_api_key")
    with pytest.raises(RuntimeError, match="invalid_api_key"):
        b.generate("hello")


def test_anthropic_backend_shapes_a_completion():
    b = AnthropicBackend.__new__(AnthropicBackend)
    b.model, b.system = "m", ""
    b._client = SimpleNamespace(messages=SimpleNamespace(create=lambda **k: SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(input_tokens=9, output_tokens=3))))
    c = b.generate("a b c")
    assert c.text == "ok" and c.backend == "anthropic:m"
    assert c.prompt_tokens == 9 and c.prompt_words == 3


def test_gemini_backend_shapes_a_completion():
    b = GeminiBackend.__new__(GeminiBackend)
    b.model, b.system, b.timeout, b._mode = "m", "", 30.0, "legacy"

    class FakeModel:
        def __init__(self, *a, **k):
            pass

        def generate_content(self, prompt):
            return SimpleNamespace(
                text="fine",
                usage_metadata=SimpleNamespace(prompt_token_count=13))

    b._client = SimpleNamespace(GenerativeModel=FakeModel)
    c = b.generate("a b")
    assert c.text == "fine" and c.backend == "gemini:m"
    assert c.prompt_tokens == 13 and c.prompt_words == 2


def test_missing_key_raises_backend_error(monkeypatch):
    for var in ("OPENAI_API_KEY", "XAI_API_KEY", "GROK_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(BackendError, match="OPENAI_API_KEY"):
        OpenAIBackend(model="m")


# -- retry ----------------------------------------------------------------

def test_retry_backs_off_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 rate limit exceeded")
        return "ok"

    assert _retry(flaky, provider="test", sleep=lambda s: None) == "ok"
    assert calls["n"] == 3


def test_retry_does_not_retry_permanent_errors():
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise RuntimeError("401 invalid api key")

    with pytest.raises(RuntimeError):
        _retry(bad, provider="test", sleep=lambda s: None)
    assert calls["n"] == 1, "permanent errors must not be retried"


def test_retry_gives_up_and_says_so():
    with pytest.raises(BackendError, match="giving up"):
        _retry(lambda: (_ for _ in ()).throw(RuntimeError("503 unavailable")),
               attempts=2, provider="test", sleep=lambda s: None)


# -- caching --------------------------------------------------------------

class CountingBackend:
    name = "counting"
    model = "c"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, *, temperature=1.0, max_tokens=300, seed=None):
        self.calls += 1
        return Completion(f"resp{self.calls}", "counting:c", 1, len(prompt.split()))


def test_cache_hits_avoid_a_second_call(tmp_path):
    inner = CountingBackend()
    b = CachedBackend(inner, cache_dir=tmp_path)
    a1 = b.generate("same prompt", seed=1)
    a2 = b.generate("same prompt", seed=1)
    assert a1.text == a2.text == "resp1"
    assert inner.calls == 1
    assert b.stats() == {"hits": 1, "misses": 1, "hit_rate": 0.5}


def test_cache_key_separates_parameters(tmp_path):
    inner = CountingBackend()
    b = CachedBackend(inner, cache_dir=tmp_path)
    b.generate("p", seed=1)
    b.generate("p", seed=2)
    b.generate("p", temperature=0.5, seed=1)
    b.generate("other", seed=1)
    assert inner.calls == 4


def test_cache_survives_a_corrupt_entry(tmp_path):
    inner = CountingBackend()
    b = CachedBackend(inner, cache_dir=tmp_path)
    b.generate("p", seed=1)
    next(tmp_path.glob("*.json")).write_text("{not json")
    assert b.generate("p", seed=1).text == "resp2"


def test_cache_roundtrips_every_completion_field(tmp_path):
    class Rich:
        name, model = "rich", "r"

        def generate(self, prompt, *, temperature=1.0, max_tokens=300, seed=None):
            return Completion("t", "rich:r", 5, 2, {"input_tokens": 5})

    b = CachedBackend(Rich(), cache_dir=tmp_path)
    b.generate("a b")
    c = b.generate("a b")
    assert (c.prompt_tokens, c.prompt_words, c.raw_usage) == (5, 2, {"input_tokens": 5})


def test_get_backend_wraps_in_cache_when_asked(tmp_path):
    b = get_backend("mock", cache=tmp_path)
    assert isinstance(b, CachedBackend)
    assert b.generate("x", seed=1).text == b.generate("x", seed=1).text
    assert b.stats()["hits"] == 1
