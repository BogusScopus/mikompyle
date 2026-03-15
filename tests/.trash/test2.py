import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

# ── paths ────────────────────────────────────────────────────────────────────
_HERE   = Path(__file__).parent
_ASSETS = _HERE.parent / "assets" / "toy"

# ── optional package imports ─────────────────────────────────────────────────
try:
    sys.path.insert(0, str(_HERE))
    import my_ext
    HAS_MY_EXT = True
except ImportError:
    HAS_MY_EXT = False

try:
    import klay
    from klay.klay_ext import Circuit as KlayCircuit
    HAS_KLAY = True
except ImportError:
    HAS_KLAY = False

requires_my_ext = pytest.mark.skipif(
    not HAS_MY_EXT, reason="my_ext not importable – build the .so first"
)
requires_klay = pytest.mark.skipif(
    not HAS_KLAY, reason="klay not importable"
)
requires_both = pytest.mark.skipif(
    not (HAS_MY_EXT and HAS_KLAY),
    reason="both my_ext and klay are required"
)


# ─────────────────────────────────────────────────────────────────────────────
# Random DIMACS generation  (matches klay.utils.generate_random_dimacs exactly)
# ─────────────────────────────────────────────────────────────────────────────

def generate_random_dimacs(
    file_name: str,
    var_count: int,
    clause_count: int,
    seed: int = 1,
    clause_length: int = 3,
) -> None:
    """Generate a random k-CNF formula and save it in DIMACS format.

    Identical to ``klay.utils.generate_random_dimacs`` so results are
    reproducible across both test suites.
    """
    random.seed(seed)
    with open(file_name, "w") as f:
        f.write(f"p cnf {var_count} {clause_count}\n")
        for _ in range(clause_count):
            clause = [
                random.randint(1, var_count) * random.choice([1, -1])
                for _ in range(clause_length)
            ]
            f.write(" ".join(map(str, clause)) + " 0\n")


def make_temp_cnf(var_count: int, clause_count: int, seed: int, clause_length: int = 3) -> str:
    """Write a random CNF to a temp file and return its path."""
    path = tempfile.mktemp(suffix=".cnf")
    generate_random_dimacs(path, var_count, clause_count, seed=seed, clause_length=clause_length)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Circuit evaluation utilities  (log-semiring, numpy)
# ─────────────────────────────────────────────────────────────────────────────

def _unroll(csr: np.ndarray) -> np.ndarray:
    """CSR offset array → flat scatter index (matches klay's unroll_ixs)."""
    return np.repeat(np.arange(len(csr) - 1), np.diff(csr))


def encode_log(nb_vars: int, weights: List[float], neg_weights: List[float]) -> np.ndarray:
    """Initial activation vector in log semiring.

    Layout: [log 0, log 1, log w1, log(1-w1), log w2, log(1-w2), ...]
    (index 0 = False, index 1 = True, then pos/neg pairs per variable)
    """
    x: List[float] = [float("-inf"), 0.0]
    for i in range(nb_vars):
        x.append(weights[i])
        x.append(neg_weights[i])
    return np.array(x, dtype=np.float64)


def eval_circuit_log(
    ixs_in:  List[np.ndarray],
    ixs_out: List[np.ndarray],
    x:       np.ndarray,
) -> float:
    """Evaluate a klay/my_ext circuit in the log semiring.

    Layers alternate: product (even) → sum (odd).
    Returns log-WMC at the root.
    """
    for layer_idx, (ix_in, ix_out_csr) in enumerate(zip(ixs_in, ixs_out)):
        ix_out   = _unroll(np.asarray(ix_out_csr))
        out_size = len(ix_out_csr) - 1
        src      = x[np.asarray(ix_in)]

        if layer_idx % 2 == 0:           # product  →  sum in log space
            out = np.zeros(out_size, dtype=np.float64)
            np.add.at(out, ix_out, src)
        else:                             # sum  →  logsumexp
            out = np.full(out_size, float("-inf"), dtype=np.float64)
            for s, o in zip(src, ix_out):
                out[o] = np.logaddexp(out[o], s)

        x = out

    return float(x[0])


def circuit_wmc(
    circuit,
    nb_vars: int,
    weights: List[float],
    neg_weights: List[float],
) -> float:
    """Compute log-WMC for a circuit from my_ext or klay."""
    get_ix = getattr(circuit, "_get_indices", None) or getattr(circuit, "get_indices")
    ixs_in, ixs_out = get_ix()
    return eval_circuit_log(ixs_in, ixs_out, encode_log(nb_vars, weights, neg_weights))


# ─────────────────────────────────────────────────────────────────────────────
# Weight helpers  (match klay.utils.python_weights)
# ─────────────────────────────────────────────────────────────────────────────

def random_weights(nb_vars: int, seed: int = 42) -> Tuple[List[float], List[float]]:
    """Random log-weights in [log 0.1, log 0.9]."""
    rng = random.Random(seed)
    probs = [rng.uniform(0.1, 0.9) for _ in range(nb_vars)]
    return [math.log(p) for p in probs], [math.log(1.0 - p) for p in probs]


def uniform_weights(nb_vars: int) -> Tuple[List[float], List[float]]:
    """All variables have P(xi = T) = 0.5."""
    w = [math.log(0.5)] * nb_vars
    return w, w[:]


# ─────────────────────────────────────────────────────────────────────────────
# Brute-force reference WMC
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dimacs(cnf_path: str) -> Tuple[int, List[List[int]]]:
    nb_vars, clauses = 0, []
    with open(cnf_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            parts = line.split()
            if parts[0] == "p":
                nb_vars = int(parts[2])
            else:
                lits = [int(t) for t in parts if t != "0"]
                if lits:
                    clauses.append(lits)
    return nb_vars, clauses


def brute_force_wmc(cnf_path: str, weights: List[float], neg_weights: List[float]) -> float:
    """Enumerate all 2^n assignments; accumulate WMC in log space."""
    nb_vars, clauses = _parse_dimacs(cnf_path)
    assert nb_vars <= 20, "brute_force_wmc only feasible for ≤20 vars"

    log_wmc = float("-inf")
    for bits in range(1 << nb_vars):
        assignment = {v: bool((bits >> (v - 1)) & 1) for v in range(1, nb_vars + 1)}
        sat = all(
            any((lit > 0) == assignment[abs(lit)] for lit in clause)
            for clause in clauses
        )
        if sat:
            log_w = sum(
                weights[v - 1] if assignment[v] else neg_weights[v - 1]
                for v in range(1, nb_vars + 1)
            )
            log_wmc = np.logaddexp(log_wmc, log_w)

    return log_wmc


# ─────────────────────────────────────────────────────────────────────────────
# d4 NNF helpers for klay  (no external binary – files written inline)
# ─────────────────────────────────────────────────────────────────────────────

def klay_circuit_from_nnf(nnf_content: str) -> "KlayCircuit":
    """Load an inline d4 NNF string into a klay Circuit."""
    tmp = tempfile.mktemp(suffix=".nnf")
    with open(tmp, "w") as f:
        f.write(nnf_content)
    try:
        c = KlayCircuit()
        c.add_d4_from_file(tmp)
        return c
    finally:
        os.unlink(tmp)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark helpers
# ─────────────────────────────────────────────────────────────────────────────

def _time_fn(fn, repeats: int = 1) -> Tuple[object, float]:
    """Return (last_result, mean_elapsed_seconds)."""
    t0 = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = fn()
    return result, (time.perf_counter() - t0) / repeats


# ─────────────────────────────────────────────────────────────────────────────
# ── TEST CLASSES ──────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────


# ── 1. Sanity-check the brute-force reference ─────────────────────────────────

class TestBruteForceReference:

    # CNFs with known model counts (hand-verified)
    # toy0.cnf: p cnf 5 4  →  12 solutions
    TOY0 = str(_ASSETS / "toy0.cnf")
    UNSAT = str(_ASSETS / "unsat.cnf")

    def test_unsat_is_neg_inf(self):
        assert brute_force_wmc(self.UNSAT, *uniform_weights(1)) == float("-inf")

    def test_toy0_model_count(self):
        log_wmc = brute_force_wmc(self.TOY0, *uniform_weights(5))
        mc = math.exp(log_wmc) * (2 ** 5)
        assert abs(mc - 12) < 1e-9

    @pytest.mark.parametrize("var_count,clause_count,seed", [
        (5, 15, 1),
        (5, 15, 4),
        (8, 24, 2),
        (10, 30, 1),
    ])
    def test_random_cnf_wmc_is_finite_or_unsat(self, var_count, clause_count, seed):
        """Random CNF brute-force returns either -inf (UNSAT) or a finite log-WMC."""
        path = make_temp_cnf(var_count, clause_count, seed)
        try:
            result = brute_force_wmc(path, *uniform_weights(var_count))
            assert result == float("-inf") or math.isfinite(result)
        finally:
            os.unlink(path)


# ── 2. Ganak correctness vs brute force on random CNFs ───────────────────────

@requires_my_ext
class TestGanakVsBruteForce:
    """For each random CNF, ganak WMC must match brute-force WMC exactly."""

    TOL = 1e-6   # relative tolerance

    def _check(self, var_count: int, clause_count: int, seed: int, weight_seed: int = 0):
        path = make_temp_cnf(var_count, clause_count, seed)
        try:
            w, nw = random_weights(var_count, seed=weight_seed)
            circuit   = my_ext.compile_to_ganak(path)
            ganak_log = circuit_wmc(circuit, var_count, w, nw)
            ref_log   = brute_force_wmc(path, w, nw)

            if ref_log == float("-inf"):
                assert ganak_log == float("-inf"), \
                    f"ganak returned {ganak_log} for UNSAT formula"
                return

            g_wmc, r_wmc = math.exp(ganak_log), math.exp(ref_log)
            rel_err = abs(g_wmc - r_wmc) / (abs(r_wmc) + 1e-30)
            assert rel_err < self.TOL, (
                f"var={var_count} cls={clause_count} seed={seed} weight_seed={weight_seed}\n"
                f"  ganak={g_wmc:.8f}  brute_force={r_wmc:.8f}  rel_err={rel_err:.2e}"
            )
        finally:
            os.unlink(path)

    # Small CNFs – brute-force is exact, tests are fast
    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_5vars_15clauses(self, seed):
        """5 variables, ~3× clauses-per-var.  Multiple weight seeds per CNF."""
        for wseed in range(3):
            self._check(5, 15, seed=seed, weight_seed=wseed)

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_8vars_24clauses(self, seed):
        for wseed in range(3):
            self._check(8, 24, seed=seed, weight_seed=wseed)

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_10vars_30clauses(self, seed):
        for wseed in range(3):
            self._check(10, 30, seed=seed, weight_seed=wseed)

    @pytest.mark.parametrize("seed", [1, 2])
    def test_12vars_50clauses(self, seed):
        for wseed in range(3):
            self._check(12, 50, seed=seed, weight_seed=wseed)

    # Uniform weights sanity pass
    @pytest.mark.parametrize("var_count,clause_count,seed", [
        (5,  10, 1),
        (5,  15, 42),
        (8,  16, 7),
        (10, 20, 3),
    ])
    def test_uniform_weights(self, var_count, clause_count, seed):
        path = make_temp_cnf(var_count, clause_count, seed)
        try:
            w, nw = uniform_weights(var_count)
            circuit   = my_ext.compile_to_ganak(path)
            ganak_log = circuit_wmc(circuit, var_count, w, nw)
            ref_log   = brute_force_wmc(path, w, nw)

            if ref_log == float("-inf"):
                assert ganak_log == float("-inf")
                return

            rel_err = abs(math.exp(ganak_log) - math.exp(ref_log)) / (abs(math.exp(ref_log)) + 1e-30)
            assert rel_err < self.TOL
        finally:
            os.unlink(path)


# ── 3. Ganak vs d4 on random CNFs  ───────────────────────────────────────────
#
# The core comparison: the same random CNF compiled by two different solvers
# (ganak and d4 via klay.Circuit.add_d4_from_file) must give the same WMC.
# This is exactly the pattern of klay's fuzzer_torch.py.
#
# We drive d4 by generating the NNF at the command-line level – if a d4 binary
# is on PATH; otherwise we fall back to comparing ganak vs klay/SDD or simply
# skip the d4 leg.
# ─────────────────────────────────────────────────────────────────────────────

def _d4_binary() -> str | None:
    """Return path to d4 binary if available, else None."""
    import shutil
    return shutil.which("d4")


def compile_d4_to_nnf(cnf_path: str) -> str | None:
    """Run d4 on ``cnf_path``, return path to the .nnf output or None."""
    d4 = _d4_binary()
    if d4 is None:
        return None
    nnf_path = cnf_path.replace(".cnf", ".nnf")
    import subprocess
    result = subprocess.run(
        [d4, "-dDNNF", cnf_path, f"-out={nnf_path}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0 or not os.path.exists(nnf_path):
        return None
    return nnf_path


@requires_both
class TestGanakVsD4RandomCNF:
    """Ganak WMC == d4 WMC on the same random CNF.

    If the d4 binary is not on PATH these tests are automatically skipped.
    When d4 is available the full fuzzer loop runs:
        random CNF → ganak circuit → klay circuit(d4) → compare WMC.
    """

    TOL = 1e-9

    @pytest.fixture(autouse=True)
    def require_d4(self):
        if _d4_binary() is None:
            pytest.skip("d4 binary not found on PATH")

    def _compare(self, var_count: int, clause_count: int, seed: int, weight_seed: int = 0):
        path = make_temp_cnf(var_count, clause_count, seed)
        try:
            nnf_path = compile_d4_to_nnf(path)
            if nnf_path is None:
                pytest.skip(f"d4 failed to compile {path}")

            w, nw = random_weights(var_count, seed=weight_seed)

            ganak_c = my_ext.compile_to_ganak(path)
            d4_c    = KlayCircuit()
            d4_c.add_d4_from_file(nnf_path)

            g_log = circuit_wmc(ganak_c, var_count, w, nw)
            d_log = circuit_wmc(d4_c,    var_count, w, nw)

            if d_log == float("-inf"):
                assert g_log == float("-inf")
                return

            rel_err = abs(math.exp(g_log) - math.exp(d_log)) / (abs(math.exp(d_log)) + 1e-30)
            assert rel_err < self.TOL, (
                f"var={var_count} cls={clause_count} seed={seed}\n"
                f"  ganak={math.exp(g_log):.8f}  d4={math.exp(d_log):.8f}  "
                f"rel_err={rel_err:.2e}"
            )
        finally:
            os.unlink(path)
            if nnf_path and os.path.exists(nnf_path):
                os.unlink(nnf_path)

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_5vars_15clauses(self, seed):
        for wseed in range(3):
            self._compare(5, 15, seed=seed, weight_seed=wseed)

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_10vars_30clauses(self, seed):
        for wseed in range(3):
            self._compare(10, 30, seed=seed, weight_seed=wseed)

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_20vars_84clauses(self, seed):
        for wseed in range(3):
            self._compare(20, 84, seed=seed, weight_seed=wseed)

    @pytest.mark.parametrize("seed", [1, 2])
    def test_50vars_210clauses(self, seed):
        for wseed in range(3):
            self._compare(50, 210, seed=seed, weight_seed=wseed)


# ── 4. Performance benchmarks ─────────────────────────────────────────────────
#
# Mirrors klay's benchmark_klay_torch / benchmark_pysdd pattern:
#   - Time compilation for each CNF size
#   - Time circuit evaluation (NB_EVAL_REPEATS runs, 2 warm-up discarded)
#   - Print a comparison table
# ─────────────────────────────────────────────────────────────────────────────

NB_EVAL_REPEATS = 50   # matches klay's nb_repeats default

# (var_count, clause_count, label) – ratio ≈ 4.2 (phase-transition region)
BENCH_CONFIGS = [
    (10,  42,  "10 vars  /  42 clauses"),
    (20,  84,  "20 vars  /  84 clauses"),
    (30,  126, "30 vars  / 126 clauses"),
    (50,  210, "50 vars  / 210 clauses"),
    (100, 420, "100 vars / 420 clauses"),
]


@requires_my_ext
class TestBenchmarks:
    """Wall-clock benchmarks.  Run with ``pytest -v -s -k bench``."""

    BENCH_SEED = 1        # CNF generation seed
    WEIGHT_SEED = 0       # weight generation seed

    def _run(self, var_count: int, clause_count: int, label: str):
        path = make_temp_cnf(var_count, clause_count, seed=self.BENCH_SEED)
        try:
            w, nw = random_weights(var_count, seed=self.WEIGHT_SEED)

            # ── compile ────────────────────────────────────────────────────
            _, compile_s = _time_fn(
                lambda: my_ext.compile_to_ganak(path)
            )

            # ── evaluate (warm up + timed runs) ───────────────────────────
            circuit = my_ext.compile_to_ganak(path)
            timings = []
            for i in range(NB_EVAL_REPEATS + 2):
                t0 = time.perf_counter()
                circuit_wmc(circuit, var_count, w, nw)
                timings.append(time.perf_counter() - t0)
            eval_times = timings[2:]   # drop 2 warm-up runs

            mean_eval  = sum(eval_times) / len(eval_times)
            min_eval   = min(eval_times)

            # ── optional d4 comparison ────────────────────────────────────
            d4_mean_s = None
            if HAS_KLAY and _d4_binary():
                nnf_path = compile_d4_to_nnf(path)
                if nnf_path:
                    try:
                        d4_c = KlayCircuit()
                        d4_c.add_d4_from_file(nnf_path)
                        d4_timings = []
                        for i in range(NB_EVAL_REPEATS + 2):
                            t0 = time.perf_counter()
                            circuit_wmc(d4_c, var_count, w, nw)
                            d4_timings.append(time.perf_counter() - t0)
                        d4_mean_s = sum(d4_timings[2:]) / len(d4_timings[2:])
                    finally:
                        os.unlink(nnf_path)

            # ── print ─────────────────────────────────────────────────────
            d4_str   = f"{d4_mean_s*1e6:7.1f} µs" if d4_mean_s else "      N/A"
            ratio_s  = (
                f"  ({d4_mean_s/mean_eval:.2f}× slower)" if d4_mean_s else ""
            )
            print(
                f"\n  [{label}]\n"
                f"    nodes          : {circuit.nb_nodes()}\n"
                f"    compile time   : {compile_s*1e3:.2f} ms\n"
                f"    eval ganak     : {mean_eval*1e6:7.1f} µs  "
                f"(min {min_eval*1e6:.1f} µs,  n={NB_EVAL_REPEATS})\n"
                f"    eval d4        : {d4_str}{ratio_s}"
            )

        finally:
            os.unlink(path)

    @pytest.mark.parametrize("var_count,clause_count,label", BENCH_CONFIGS)
    def test_bench(self, var_count, clause_count, label):
        self._run(var_count, clause_count, label)

    @requires_both
    def test_bench_ganak_vs_d4_wmc_agreement(self):
        """On each benchmark CNF, ganak and d4 WMCs must agree (when d4 is available)."""
        if _d4_binary() is None:
            pytest.skip("d4 binary not on PATH")

        for var_count, clause_count, label in BENCH_CONFIGS:
            path = make_temp_cnf(var_count, clause_count, seed=self.BENCH_SEED)
            try:
                nnf_path = compile_d4_to_nnf(path)
                if nnf_path is None:
                    continue

                w, nw = random_weights(var_count, seed=self.WEIGHT_SEED)
                ganak_c = my_ext.compile_to_ganak(path)
                d4_c    = KlayCircuit()
                d4_c.add_d4_from_file(nnf_path)

                g_log = circuit_wmc(ganak_c, var_count, w, nw)
                d_log = circuit_wmc(d4_c,    var_count, w, nw)

                if d_log != float("-inf"):
                    rel_err = abs(math.exp(g_log) - math.exp(d_log)) / (abs(math.exp(d_log)) + 1e-30)
                    assert rel_err < 1e-6, (
                        f"{label}: ganak={math.exp(g_log):.8f} "
                        f"d4={math.exp(d_log):.8f} rel_err={rel_err:.2e}"
                    )

            finally:
                os.unlink(path)
                if nnf_path and os.path.exists(nnf_path):
                    os.unlink(nnf_path)
