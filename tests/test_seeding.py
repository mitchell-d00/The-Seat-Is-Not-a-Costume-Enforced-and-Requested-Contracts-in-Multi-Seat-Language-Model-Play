"""Reproducibility across processes.

Seeds used to be derived from Python's salted `hash()`, so the mock changed
with PYTHONHASHSEED and the response cache missed on every rerun. These tests
run the same scene in fresh interpreters with different hash seeds.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCENE = r"""
import json, sys
sys.path.insert(0, {src!r})
from seatkit import *
a, b = make_facts(3, seed=1, prefix="a"), make_facts(3, seed=2, prefix="b")
geo, bio = facing_pair("geo", "bio", a, b,
                       a_extra={{"persona": "a geologist"}},
                       b_extra={{"persona": "a paleobiologist"}})
inner = MockBackend()
calls = [0]
orig = inner.generate
def counted(*a, **k):
    calls[0] += 1
    return orig(*a, **k)
inner.generate = counted
be = CachedBackend(inner, cache_dir={cache!r}) if {cache!r} else inner
res = Bench([geo, bio], be).run(SceneConfig(turns=6, level=Level.REQUESTED, seed=3))
print(json.dumps({{"utt": [s.text for s in res.transcript if s.origin is Origin.SEAT],
                   "calls": calls[0]}}))
"""


def run(hashseed, cache=""):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    code = SCENE.format(src=str(ROOT / "src"), cache=cache)
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_mock_scene_is_identical_across_hash_seeds():
    assert run(1)["utt"] == run(2)["utt"]


def test_cache_hits_on_rerun_in_a_new_process(tmp_path):
    first = run(1, str(tmp_path))
    second = run(2, str(tmp_path))
    assert first["calls"] > 0
    assert second["calls"] == 0, "rerun should be served entirely from cache"
    assert first["utt"] == second["utt"]


def test_stable_int_is_not_builtin_hash():
    sys.path.insert(0, str(ROOT / "src"))
    from seatkit._seeding import stable_int
    assert stable_int("geo") == stable_int("geo")
    assert 0 <= stable_int("geo", mod=10) < 10
