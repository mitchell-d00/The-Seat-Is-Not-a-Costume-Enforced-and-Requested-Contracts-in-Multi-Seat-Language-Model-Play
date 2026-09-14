# Backends

The protocol is one method. A prompt goes in, a `Completion` comes out:

```python
class Backend(Protocol):
    name: str
    def generate(self, prompt: str, *, temperature: float = 1.0,
                 max_tokens: int = 300, seed: int | None = None) -> Completion: ...
```

Everything E1–E4 needs sits behind that, which is why adding a provider touches no experimental logic.

## Specs

| Spec | Key | SDK |
|---|---|---|
| `mock` | — | — |
| `openai:<model-id>` | `OPENAI_API_KEY` | `pip install openai` |
| `anthropic:<model-id>` | `ANTHROPIC_API_KEY` | `pip install anthropic` |
| `grok:<model-id>` (alias `xai:`) | `XAI_API_KEY` or `GROK_API_KEY` | `pip install openai` |
| `gemini:<model-id>` (alias `google:`) | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `pip install google-genai` |

```bash
pip install -e ".[live]"        # all four SDKs
python experiments/preflight.py --backends openai:<model> gemini:<model>
python experiments/e1_leak_probe.py --backend grok:<model> --cache .cache/grok
```

Grok uses xAI's OpenAI-compatible endpoint, so no separate SDK. It keeps its own `name` and family label anyway: **sharing a wire protocol is not sharing weights**, and if the two collapsed to one label a run could satisfy the two-family requirement without satisfying it. `test_family_separates_grok_from_openai` pins this.

Gemini prefers `google-genai` and falls back to `google-generativeai`, because which one is current has changed more than once.

## `prompt_tokens` is not comparable across families

This is the one thing that will quietly corrupt a cross-family table.

E1 regresses leak rate on context length. Every provider reports input tokens under its own tokenizer, so the same prompt is a different number of tokens on each. Comparing leak-vs-tokens across families confounds the wall effect with tokenizer granularity.

`Completion` therefore carries two counts:

- **`prompt_tokens`** — provider-reported. Correct for cost accounting. Wrong for analysis.
- **`prompt_words`** — `len(prompt.split())`, computed identically everywhere. **This is what cross-family analysis uses.**

Within a single family either works. Across families only `prompt_words` does.

## Determinism

| Provider | `seed` | Reproducible? |
|---|---|---|
| mock | yes | fully |
| openai / grok | accepted, best-effort | no guarantee |
| anthropic | no such parameter; accepted and ignored | no |
| gemini | passed when supported, dropped on rejection | no |

Only the mock is reproducible. Do not describe live cross-run results as reproducible on the basis of a seed argument being present.

## Parameter rejection and retries

Models reject `temperature`, `max_tokens`, or `seed` often enough that a grid dying on a parameter name is a real failure mode. The OpenAI-compatible path drops the contested parameter once and retries, and records nothing else — a run where half the cells silently ran at a different temperature would be worse than a crash.

`_retry` backs off on rate limits, timeouts, and 5xx. It does **not** retry auth or bad-request errors; retrying those just burns budget more slowly. After the last attempt it raises `BackendError` rather than returning empty text, because a run that silently dropped failures would have an unrecorded exclusion rule, which the preregistration forbids.

## Caching

```python
backend = get_backend("openai:<model-id>", cache=".cache/run7")
```

Keyed on provider, model, prompt, temperature, max_tokens, and seed. A full grid is scenes × turns × seats × levels calls — five figures quickly — and E2 re-judges transcripts E1 already generated, so re-analysis should not be paid for twice.

**A cached run is one run replayed, not a replication.** For providers without working seeds the cache freezes a single sample from a stochastic process. That is right for re-analysis and wrong if you wanted fresh draws. Clear the directory or pass `write=False`.

## Adding a provider

1. Subclass `_OpenAICompatible` (set `name`, `base_url`, `env_keys`) if the endpoint is OpenAI-shaped, otherwise write a class with a `generate` method.
2. Populate `prompt_words` from `len(prompt.split())`. Not from the provider's count.
3. Label `Completion.backend` as `provider:model`.
4. Register in `PROVIDERS` and `FAMILY_OF`. Give it its own family unless it is genuinely the same weights.
5. Add a fake-client test in `tests/test_backends.py`. No test should need a key.
