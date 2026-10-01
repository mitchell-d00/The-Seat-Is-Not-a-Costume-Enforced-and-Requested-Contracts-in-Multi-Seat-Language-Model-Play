"""seatkit - a reference implementation for "The Seat Is Not a Costume".

The library exists to make one distinction executable: a prohibition written in
a shared window (requested) and a prohibition implemented in view assembly
(enforced) are different objects with different failure modes.

  >>> from seatkit import Level, make_facts, facing_pair, Bench, SceneConfig
  >>> from seatkit import MockBackend
  >>> a_facts, b_facts = make_facts(2, seed=1, prefix="a"), make_facts(2, seed=2, prefix="b")
  >>> geo, bio = facing_pair("geo", "bio", a_facts, b_facts)
  >>> bench = Bench([geo, bio], MockBackend())
  >>> res = bench.run(SceneConfig(turns=6, level=Level.PARTITIONED, probe_turns=(6,)))
  >>> res.routing_bleed_count
  0
"""

from .artifacts import LedgerEntry, SeatSheet, reseed, tail_carries_forbidden
from .backends import (AnthropicBackend, Backend, BackendError, CachedBackend,
                       Completion, GeminiBackend, GrokBackend, MockBackend,
                       OpenAIBackend, available_backends, family_of,
                       get_backend, parse_spec)
from .bench import Bench, SceneConfig, SceneResult
from .ceiling import CEILING, SeparatedBench
from .contracts import Contract, Fact, InstrumentRights, Seat, facing_pair
from .instruments import Instrument, InstrumentOutput, build_replay_log
from .ladder import Level, View, assemble_view, visibility_for
from .metrics import (
    LeakRates, Position, SurvivalScore, attribution_score, benjamini_hochberg,
    cohens_h, cohens_kappa, disagreement_half_life, paired_sign_test, hedge_index, leak_by_turn, leak_rates,
    lexical_convergence, normalize, survival,
)
from .probes import (
    Fact as ProbeFact, ProbeResult, elicited_probe, grade_probes, make_facts,
    score_volunteered, verify_prior,
)
from .transcript import Origin, Span, Transcript

__version__ = "0.2.1"

__all__ = [
    "Level", "View", "assemble_view", "visibility_for",
    "Contract", "Fact", "InstrumentRights", "Seat", "facing_pair",
    "Transcript", "Span", "Origin",
    "Bench", "SceneConfig", "SceneResult", "SeparatedBench", "CEILING",
    "Instrument", "InstrumentOutput", "build_replay_log",
    "Backend", "Completion", "BackendError", "get_backend", "parse_spec",
    "family_of", "available_backends", "CachedBackend",
    "MockBackend", "OpenAIBackend", "AnthropicBackend", "GrokBackend",
    "GeminiBackend",
    "make_facts", "verify_prior", "elicited_probe", "score_volunteered",
    "ProbeResult", "grade_probes",
    "leak_rates", "leak_by_turn", "LeakRates", "attribution_score",
    "lexical_convergence", "hedge_index", "disagreement_half_life",
    "survival", "SurvivalScore", "normalize", "Position",
    "benjamini_hochberg", "cohens_h", "cohens_kappa", "paired_sign_test",
    "SeatSheet", "LedgerEntry", "reseed", "tail_carries_forbidden",
    "__version__",
]
