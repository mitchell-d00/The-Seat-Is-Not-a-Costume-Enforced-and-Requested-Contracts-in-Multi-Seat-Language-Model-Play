"""Model judges and graders (seatkit/judges.py), exercised with fake backends.

The point is to cover the code paths a live run takes, offline: prompt
construction, label anonymization, parsing, recorded fallbacks, and the refusal
to pair a live scene with stand-in judges.
"""
import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seatkit import (Completion, ProbeResult, Span, Origin, grade_probes,
                     leak_rates, make_facts)
from seatkit.judges import (LexicalCentroidJudge, ModelAttributionJudge,
                            ModelLeakGrader, ModelParaphraser, ModelStanceJudge,
                            RegexParaphraser, StandinJudgeRefused,
                            incompatibility_by_turn, judge_backend_or_none)


class Fake:
    name = "fake"

    def __init__(self, fn):
        self.fn, self.prompts = fn, []

    def generate(self, prompt, **kw):
        self.prompts.append(prompt)
        return Completion(self.fn(prompt), "fake", 0, 0)


# -- paraphrase -------------------------------------------------------------

def test_model_paraphraser_calls_model_once_per_text():
    be = Fake(lambda p: "NEUTRAL VERSION")
    para = ModelParaphraser(be)
    assert para("Arguably, the banding is seasonal.") == "NEUTRAL VERSION"
    para("Arguably, the banding is seasonal.")
    assert len(be.prompts) == 1 and not para.is_standin


def test_empty_paraphrase_falls_back_and_is_counted():
    para = ModelParaphraser(Fake(lambda p: "   "))
    assert para("Arguably, it is seasonal.") == "it is seasonal."
    assert para.unparseable == 1


def test_regex_paraphraser_is_marked_standin():
    assert RegexParaphraser.is_standin and LexicalCentroidJudge.is_standin


# -- attribution ------------------------------------------------------------

def oracle(prompt):
    """Answers correctly by matching the hidden utterance's keyword to a block."""
    hidden = prompt.split("Hidden utterance:\n")[1].split("\n")[0]
    key = hidden.split()[0]
    a_block = prompt.split("Speaker A:")[1].split("Speaker B:")[0]
    return "A" if key in a_block else "B"


def test_model_attribution_judge_maps_letters_back_to_seats():
    texts = [f"{'quartz' if i % 2 else 'shells'} remark {i}" for i in range(12)]
    authors = ["geo" if i % 2 else "bio" for i in range(12)]
    judge = ModelAttributionJudge(Fake(oracle), exemplars=3)
    out = judge.judge(texts, authors, ("geo", "bio"), random.Random(0))
    assert all(p == t for p, t in out)


def test_attribution_labels_are_reshuffled_per_call():
    seen = set()
    texts = ["quartz a", "shells b", "quartz c", "shells d"]
    authors = ["geo", "bio", "geo", "bio"]

    def spy(prompt):
        seen.add("quartz" in prompt.split("Speaker A:")[1].split("Speaker B:")[0])
        return "A"
    ModelAttributionJudge(Fake(spy), exemplars=1).judge(
        texts * 5, authors * 5, ("geo", "bio"), random.Random(1))
    assert seen == {True, False}


def test_hidden_turn_never_appears_among_its_own_exemplars():
    texts = [f"unique{i}" for i in range(8)]
    authors = ["geo", "bio"] * 4
    be = Fake(lambda p: "A")
    ModelAttributionJudge(be, exemplars=10).judge(texts, authors, ("geo", "bio"),
                                                  random.Random(0))
    for p, t in zip(be.prompts, texts):
        exemplars = p.split("Hidden utterance:")[0]
        assert f"- {t}\n" not in exemplars + "\n"


def test_unparseable_attribution_is_counted_not_dropped():
    judge = ModelAttributionJudge(Fake(lambda p: "I cannot tell."))
    out = judge.judge(["x one", "y two", "x three", "y four"],
                      ["geo", "bio", "geo", "bio"], ("geo", "bio"), random.Random(0))
    assert len(out) == 4 and judge.unparseable == 4


# -- leak grading -----------------------------------------------------------

def test_leak_grader_parses_and_counts():
    fact = make_facts(1, seed=3)[0]
    assert ModelLeakGrader(Fake(lambda p: "YES")).grade(fact, "x") is True
    assert ModelLeakGrader(Fake(lambda p: "no.")).grade(fact, "x") is False
    g = ModelLeakGrader(Fake(lambda p: "maybe"))
    assert g.grade(fact, "x") is False and g.unparseable == 1


def test_grade_probes_only_grades_what_exact_match_missed():
    fact = make_facts(1, seed=3)[0]
    probes = [ProbeResult(4, "geo", fact.fact_id, True, "elicited", "x", False),
              ProbeResult(4, "geo", fact.fact_id, False, "elicited", "x", False),
              ProbeResult(4, "geo", fact.fact_id, False, "elicited", "x", False,
                          disclosed=True)]
    grader = ModelLeakGrader(Fake(lambda p: "YES"))
    out = grade_probes(probes, {fact.fact_id: fact}, grader)
    assert grader.calls == 1
    assert [p.graded for p in out] == [None, True, None]
    assert probes[1].graded is None, "inputs must not be mutated"
    scoring = [p for p in out if p.counts]
    assert leak_rates(scoring).elicited == 0.5
    assert leak_rates(scoring, graded=True).elicited == 1.0


# -- stance -----------------------------------------------------------------

def test_stance_incompatibility_and_smoothing():
    stances = {"pro": "AFFIRM", "con": "DENY", "meh": "NEUTRAL"}
    judge = ModelStanceJudge(Fake(lambda p: stances[p.split("Utterance: ")[1].split()[0]]))
    spans = []
    for t, (g, b) in enumerate([("pro", "con"), ("pro", "con"), ("pro", "pro"),
                                ("pro", "pro"), ("pro", "pro")], start=1):
        spans += [Span(g, Origin.SEAT, "geo", frozenset(), t),
                  Span(b, Origin.SEAT, "bio", frozenset(), t)]
    inc = incompatibility_by_turn(spans, "geo", "bio", judge, "p", window=3)
    assert inc[1] == 1.0 and inc[2] == 1.0
    assert inc[3] == pytest.approx(2 / 3)       # one agreeing turn is not convergence
    assert inc[5] == 0.0


def test_unparseable_stance_is_neutral_and_counted():
    j = ModelStanceJudge(Fake(lambda p: "hmm"))
    assert j.stance("x", "p") == 0 and j.unparseable == 1


# -- refusal ----------------------------------------------------------------

def test_live_scene_with_standin_judges_is_refused():
    with pytest.raises(StandinJudgeRefused):
        judge_backend_or_none(None, "openai:some-model", allow_standin=False)


def test_standin_allowed_for_mock_or_when_explicit():
    assert judge_backend_or_none(None, "mock", allow_standin=False) is None
    assert judge_backend_or_none(None, "openai:m", allow_standin=True) is None
