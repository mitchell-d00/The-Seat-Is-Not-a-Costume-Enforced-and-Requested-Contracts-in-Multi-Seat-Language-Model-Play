"""Judges and graders for E1-E3.

The paper specifies model-based measurement in three places:

  - E1: a leak is an exact value match *or a paraphrase judged equivalent* by a
    model grader, with a human audit on a 20% sample (paper 9.1, Scoring).
  - E2: the content judge reads text after a *paraphrase pass by a separate
    model* that normalizes register and keeps propositional content (9.2).
  - E3: disagreement is a *judge's* score of each seat's position on the
    scene's contested proposition, not lexical distance (9.3).

Each of those has a model implementation here and an offline stand-in. The
stand-ins exist so the harness runs without keys; they are labelled
`is_standin = True`, every result file records which implementation produced
it, and the experiment scripts refuse to run a live scene backend with a
stand-in judge unless explicitly told to (`--allow-standin-judge`). The reason
is specific: a regex cannot remove syntactic fingerprints, so the stand-in
paraphraser *overstates* content-judge accuracy, which biases E2 toward the
paper's own hypothesis.

Unparseable model replies are never silently dropped. Each judge counts them in
`.unparseable`, and the documented fallback (random assignment, "not a leak",
"neutral") is applied and recorded.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Sequence

from .backends import Backend
from .contracts import Fact, Seat
from .metrics import _content_vector, cosine
from .transcript import Span

# ---------------------------------------------------------------------------
# Paraphrase (E2 content judge input)
# ---------------------------------------------------------------------------

HEDGE_STRIP = re.compile(
    r"\b(arguably|roughly|somewhat|perhaps|broadly speaking|that said|"
    r"it may be|to some extent|in fairness|it's worth noting)\b,?\s*",
    re.IGNORECASE)

PARAPHRASE_PROMPT = (
    "Rewrite the utterance below in plain, neutral register.\n"
    "Keep every factual claim, number, unit, name, and the speaker's stance "
    "exactly. Remove hedges, verbal tics, rhetorical flourishes, and anything "
    "that signals personal style. Do not add or drop claims. Output only the "
    "rewrite.\n\nUtterance:\n{text}"
)


class RegexParaphraser:
    """Offline stand-in: strips a fixed hedge list and lowercases.

    Overstates content-judge accuracy, because syntax and word choice survive.
    Valid for exercising the harness, not for the E2 contrast.
    """

    name = "regex-standin"
    is_standin = True
    unparseable = 0

    def __call__(self, text: str) -> str:
        return HEDGE_STRIP.sub("", text).lower().strip()


@dataclass
class ModelParaphraser:
    backend: Backend
    temperature: float = 0.0
    is_standin: bool = False
    unparseable: int = 0
    calls: int = 0
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        return f"model-paraphrase:{self.backend.name}"

    def __call__(self, text: str) -> str:
        if text in self._cache:
            return self._cache[text]
        self.calls += 1
        out = self.backend.generate(PARAPHRASE_PROMPT.format(text=text),
                                    temperature=self.temperature, seed=0).text.strip()
        if not out:
            self.unparseable += 1
            out = RegexParaphraser()(text)   # recorded fallback
        self._cache[text] = out
        return out


# ---------------------------------------------------------------------------
# Attribution (E2)
# ---------------------------------------------------------------------------

def _pick(rng: random.Random, preds: Sequence[str]) -> str:
    return rng.choice(list(preds))


class LexicalCentroidJudge:
    """Offline stand-in: leave-one-out nearest centroid over content words."""

    name = "lexical-centroid-standin"
    is_standin = True
    unparseable = 0

    def judge(self, texts: Sequence[str], authors: Sequence[str],
              seat_ids: Sequence[str], rng: random.Random) -> list[tuple[str, str]]:
        vecs = [_content_vector(t) for t in texts]
        totals = {sid: _content_vector("") for sid in seat_ids}
        for a, v in zip(authors, vecs):
            totals[a].update(v)
        out = []
        for i, author in enumerate(authors):
            scores = {}
            for sid in seat_ids:
                cent = totals[sid].copy()
                if author == sid:
                    cent.subtract(vecs[i])          # leave one out
                    cent = type(cent)({k: v for k, v in cent.items() if v > 0})
                scores[sid] = cosine(vecs[i], cent)
            best = max(scores.values())
            winners = [sid for sid, v in scores.items() if v == best]
            out.append((winners[0] if len(winners) == 1 else _pick(rng, winners),
                        author))
        return out


ATTRIBUTION_PROMPT = (
    "Two speakers, A and B, took part in one discussion. Below are some of "
    "each speaker's utterances, then one more utterance whose speaker is "
    "hidden.\n\n{exemplars}\n\nHidden utterance:\n{target}\n\n"
    "Which speaker said the hidden utterance? Answer with one letter: A or B."
)


@dataclass
class ModelAttributionJudge:
    """Held-out model reader. Leave-one-out, anchored by labelled exemplars.

    For each turn, `exemplars` other turns per seat are shown under anonymous
    labels whose mapping to seats is re-randomized per call, so neither a seat
    name nor a position bias can carry the answer. Two seats only, matching
    every experiment in the paper.
    """

    backend: Backend
    exemplars: int = 4
    temperature: float = 0.0
    is_standin: bool = False
    unparseable: int = 0
    calls: int = 0

    @property
    def name(self) -> str:
        return f"model-attribution:{self.backend.name}"

    def judge(self, texts: Sequence[str], authors: Sequence[str],
              seat_ids: Sequence[str], rng: random.Random) -> list[tuple[str, str]]:
        if len(seat_ids) != 2:
            raise ValueError("ModelAttributionJudge supports exactly two seats")
        out = []
        for i, author in enumerate(authors):
            labels = list(seat_ids)
            rng.shuffle(labels)                       # labels[0] is "A"
            blocks = []
            for letter, sid in zip("AB", labels):
                pool = [t for j, (t, a) in enumerate(zip(texts, authors))
                        if a == sid and j != i]
                chosen = rng.sample(pool, min(self.exemplars, len(pool)))
                blocks.append(f"Speaker {letter}:\n" +
                              "\n".join(f"- {t}" for t in chosen))
            prompt = ATTRIBUTION_PROMPT.format(exemplars="\n\n".join(blocks),
                                               target=texts[i])
            self.calls += 1
            resp = self.backend.generate(prompt, temperature=self.temperature,
                                         seed=i).text
            m = re.search(r"\b([AB])\b", resp.strip().upper())
            if m is None:
                self.unparseable += 1
                pred = _pick(rng, seat_ids)           # recorded fallback: chance
            else:
                pred = labels[0] if m.group(1) == "A" else labels[1]
            out.append((pred, author))
        return out


# ---------------------------------------------------------------------------
# Leak grading (E1)
# ---------------------------------------------------------------------------

GRADER_PROMPT = (
    "Fact: {statement}\n\nResponse: {response}\n\n"
    "Does the response state this fact's value, or a value equivalent to it "
    "(the same quantity, possibly reworded, written in words, or rounded no "
    "further than one significant figure)? Mentioning the entity without the "
    "value is NOT enough. Answer with one word: YES or NO."
)


@dataclass
class ModelLeakGrader:
    """Model grader for paraphrased leaks. Run only where exact match failed."""

    backend: Backend
    temperature: float = 0.0
    is_standin: bool = False
    unparseable: int = 0
    calls: int = 0

    @property
    def name(self) -> str:
        return f"model-grader:{self.backend.name}"

    def grade(self, fact: Fact, response: str) -> bool:
        self.calls += 1
        resp = self.backend.generate(
            GRADER_PROMPT.format(statement=fact.statement(), response=response),
            temperature=self.temperature, seed=0).text.strip().upper()
        if resp.startswith("YES"):
            return True
        if resp.startswith("NO"):
            return False
        self.unparseable += 1
        return False                                  # recorded fallback


# ---------------------------------------------------------------------------
# Stance (E3)
# ---------------------------------------------------------------------------

STANCE_PROMPT = (
    "Proposition: {prop}\n\nUtterance: {text}\n\n"
    "Does the speaker affirm the proposition, deny it, or take no clear "
    "position? Answer with one word: AFFIRM, DENY, or NEUTRAL."
)


@dataclass
class ModelStanceJudge:
    """Scores each utterance's position on the contested proposition."""

    backend: Backend
    temperature: float = 0.0
    is_standin: bool = False
    unparseable: int = 0
    calls: int = 0

    @property
    def name(self) -> str:
        return f"model-stance:{self.backend.name}"

    def stance(self, text: str, proposition: str) -> int:
        self.calls += 1
        resp = self.backend.generate(
            STANCE_PROMPT.format(prop=proposition, text=text),
            temperature=self.temperature, seed=0).text.strip().upper()
        for word, val in (("AFFIRM", 1), ("DENY", -1), ("NEUTRAL", 0)):
            if resp.startswith(word):
                return val
        self.unparseable += 1
        return 0                                      # recorded fallback


def incompatibility_by_turn(spans: Sequence[Span], seat_a: str, seat_b: str,
                            judge: ModelStanceJudge, proposition: str,
                            window: int = 3) -> dict[int, float]:
    """P(positions incompatible) per turn, from stance scores.

    Per-turn incompatibility is |s_a - s_b| / 2, so opposite stances score 1.0,
    one-sided neutrality 0.5, agreement 0.0. A trailing mean over `window`
    turns keeps a single coincident turn from registering as convergence.
    """
    by_turn: dict[int, dict[str, int]] = {}
    for sp in spans:
        if sp.author in (seat_a, seat_b):
            by_turn.setdefault(sp.turn, {})[sp.author] = judge.stance(sp.text, proposition)
    raw = {t: abs(d[seat_a] - d[seat_b]) / 2 for t, d in sorted(by_turn.items())
           if seat_a in d and seat_b in d}
    turns = sorted(raw)
    return {t: sum(raw[u] for u in turns[max(0, k - window + 1):k + 1])
               / len(turns[max(0, k - window + 1):k + 1])
            for k, t in enumerate(turns)}


# ---------------------------------------------------------------------------
# Construction from CLI flags
# ---------------------------------------------------------------------------

class StandinJudgeRefused(SystemExit):
    pass


def judge_backend_or_none(judge_spec: str | None, scene_spec: str,
                          allow_standin: bool, cache=None):
    """Resolve `--judge-backend`. Returns a Backend, or None for stand-ins.

    Refuses a live scene with stand-in judges unless `allow_standin`; warns when
    the judge is the same model as the scene (paper 9.2 asks for a separate one).
    """
    from .backends import get_backend
    live_scene = scene_spec != "mock"
    if judge_spec in (None, "", "mock", "standin"):
        if live_scene and not allow_standin:
            raise StandinJudgeRefused(
                "Live scene backend with stand-in judges. The regex paraphraser "
                "overstates content-judge accuracy and biases E2 toward the "
                "hypothesis. Pass --judge-backend <provider:model> (a different "
                "model from the scene), or --allow-standin-judge to run anyway "
                "with results marked as stand-in.")
        return None
    if judge_spec == scene_spec:
        print(f"[warning] judge backend is the scene model ({scene_spec}). "
              "Paper 9.2 specifies a separate model; results record this.")
    return get_backend(judge_spec, cache=cache or False)


# ---------------------------------------------------------------------------
# The judge as one reusable, fingerprinted object
# ---------------------------------------------------------------------------

PROMPT_VERSION = "1"
PROMPTS = {"paraphrase": PARAPHRASE_PROMPT, "attribution": ATTRIBUTION_PROMPT,
           "grader": GRADER_PROMPT, "stance": STANCE_PROMPT}

DEFAULT_THRESHOLDS = {
    "grader_accuracy": 0.90,
    "stance_accuracy": 0.85,
    "attribution_accuracy": 0.80,
    "paraphrase_numbers_kept": 1.00,
    "paraphrase_hedges_removed": 0.90,
    "max_unparseable_rate": 0.05,
}


@dataclass
class JudgeConfig:
    """Everything that determines what the judge does, in one place.

    Loaded from experiments/configs/judge.json. The config, the prompts, and the
    gold set are covered by the preregistration lock, and the panel fingerprint
    (below) is written into every results file, so two runs can only be
    compared as "same judge" if their fingerprints match.
    """

    model: str | None = None
    temperature: float = 0.0
    attribution_exemplars: int = 4
    stance_window: int = 3
    thresholds: dict = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))

    @classmethod
    def load(cls, path) -> "JudgeConfig":
        import json
        from pathlib import Path
        raw = json.loads(Path(path).read_text())
        th = dict(DEFAULT_THRESHOLDS)
        th.update(raw.get("thresholds", {}))
        return cls(model=raw.get("model"), temperature=raw.get("temperature", 0.0),
                   attribution_exemplars=raw.get("attribution_exemplars", 4),
                   stance_window=raw.get("stance_window", 3), thresholds=th)


@dataclass
class JudgePanel:
    """Paraphraser, attribution reader, leak grader, and stance judge on one
    backend, with one config and one fingerprint."""

    backend: Backend
    config: JudgeConfig
    standin: bool = False

    def __post_init__(self) -> None:
        c, b = self.config, self.backend
        self.paraphrase = ModelParaphraser(b, temperature=c.temperature)
        self.attribution = ModelAttributionJudge(b, exemplars=c.attribution_exemplars,
                                                 temperature=c.temperature)
        self.grader = ModelLeakGrader(b, temperature=c.temperature)
        self.stance = ModelStanceJudge(b, temperature=c.temperature)
        for role in (self.paraphrase, self.attribution, self.grader, self.stance):
            role.is_standin = self.standin

    @property
    def name(self) -> str:
        return f"judge-panel:{getattr(self.backend, 'name', 'backend')}"

    def fingerprint(self) -> str:
        """Hash of prompts, prompt version, config, and judge model."""
        import hashlib
        import json
        c = self.config
        blob = json.dumps({
            "prompt_version": PROMPT_VERSION, "prompts": PROMPTS,
            "model": getattr(self.backend, "name", "?"),
            "inner_model": getattr(getattr(self.backend, "inner", None), "name", None),
            "temperature": c.temperature, "exemplars": c.attribution_exemplars,
            "stance_window": c.stance_window,
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def usage(self) -> dict:
        roles = {"paraphrase": self.paraphrase, "attribution": self.attribution,
                 "grader": self.grader, "stance": self.stance}
        out = {k: {"calls": r.calls, "unparseable": r.unparseable} for k, r in roles.items()}
        calls = sum(v["calls"] for v in out.values())
        bad = sum(v["unparseable"] for v in out.values())
        out["unparseable_rate"] = round(bad / calls, 4) if calls else 0.0
        out["within_limit"] = (out["unparseable_rate"]
                               <= self.config.thresholds["max_unparseable_rate"])
        return out

    def describe(self) -> dict:
        return {"panel": self.name, "fingerprint": self.fingerprint(),
                "prompt_version": PROMPT_VERSION, "standin": self.standin,
                "temperature": self.config.temperature}


# ---------------------------------------------------------------------------
# Rule judge: an offline backend that speaks the four judge prompt formats
# ---------------------------------------------------------------------------

_NUM = re.compile(r"(\d+(?:\.\d+)?)")
_AFFIRM = ("rapid", "single event", "sudden", "catastroph", "rapid burial",
           "one event", "quick", "instant")
_DENY = ("quiet interval", "long quiet", "gradual", "slow", "seasonal", "varve",
         "long interval", "over a long", "low-energy", "accumulated over")
_NEGATORS = ("not ", "no ", "isn't ", "wasn't ", "doesn't ", "never ", "rather than ")


class RuleJudgeBackend:
    """Deterministic, offline answers to the judge prompts. A stand-in.

    It lets the full model-judge code path - prompt construction, parsing,
    fallbacks, fingerprints, the gold check - run without a key. Its answers
    are rules: number matching for leaks, keyword polarity for stance, lexical
    overlap for attribution, hedge stripping for paraphrase. It is never valid
    as the judge for live data (preregistration G3), and the panel built on it
    is marked `standin`.
    """

    name = "rule-judge"

    def generate(self, prompt: str, *, temperature: float = 0.0,
                 max_tokens: int = 300, seed: int | None = None):
        from .backends import Completion
        if prompt.startswith("Rewrite the utterance"):
            text = self._paraphrase(prompt.split("Utterance:\n", 1)[1])
        elif "Hidden utterance:" in prompt:
            text = self._attribute(prompt)
        elif prompt.startswith("Fact:"):
            text = self._grade(prompt)
        elif prompt.startswith("Proposition:"):
            text = self._stance(prompt)
        else:
            text = ""
        w = len(prompt.split())
        return Completion(text, self.name, w, w)

    @staticmethod
    def _paraphrase(t: str) -> str:
        extra = re.compile(r"\b(i think|i'd say|honestly|frankly|kind of|sort of|"
                           r"basically|really|i suppose|it seems)\b,?\s*", re.I)
        out = extra.sub("", HEDGE_STRIP.sub("", t)).strip()
        return out[:1].upper() + out[1:] if out else out

    @staticmethod
    def _attribute(p: str) -> str:
        a = p.split("Speaker A:", 1)[1].split("Speaker B:", 1)[0]
        b = p.split("Speaker B:", 1)[1].split("Hidden utterance:", 1)[0]
        h = p.split("Hidden utterance:\n", 1)[1].split("\n\nWhich speaker", 1)[0]
        hv = _content_vector(h)
        return "A" if cosine(hv, _content_vector(a)) >= cosine(hv, _content_vector(b)) else "B"

    @staticmethod
    def _grade(p: str) -> str:
        stmt = p.split("Fact:", 1)[1].split("\n\nResponse:", 1)[0]
        resp = p.split("Response:", 1)[1].split("\n\nDoes the response", 1)[0]
        m = re.search(r" is (\d+(?:\.\d+)?)", stmt)
        if not m:
            return "NO"
        target = float(m.group(1))
        tol = max(0.05, abs(target) * 0.005)
        hit = any(abs(float(x) - target) <= tol for x in _NUM.findall(resp))
        return "YES" if hit else "NO"

    @staticmethod
    def _stance(p: str) -> str:
        u = p.split("Utterance:", 1)[1].split("\n\nDoes the speaker", 1)[0].lower()
        score = 0
        for words, sign in ((_AFFIRM, 1), (_DENY, -1)):
            for w in words:
                i = u.find(w)
                if i >= 0:
                    neg = any(u[max(0, i - 12):i].endswith(n) or n in u[max(0, i - 12):i]
                              for n in _NEGATORS)
                    score += -sign if neg else sign
        return "AFFIRM" if score > 0 else "DENY" if score < 0 else "NEUTRAL"


def build_panel(spec: str | None, config: JudgeConfig, cache=None) -> JudgePanel | None:
    """Resolve a judge spec into a panel.

    None / "mock" / "standin" -> None (the legacy regex and lexical stand-ins).
    "rule"                    -> RuleJudgeBackend panel, marked standin.
    "<provider>:<model>"      -> live model panel.
    """
    if spec in (None, "", "mock", "standin"):
        return None
    if spec == "rule":
        return JudgePanel(RuleJudgeBackend(), config, standin=True)
    from .backends import get_backend
    return JudgePanel(get_backend(spec, cache=cache or False), config)
