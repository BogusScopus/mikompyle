import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest

_HERE   = Path(__file__).parent
_ASSETS = _HERE.parent / "assets" / "toy"

TOY0_CNF  = str(_ASSETS / "toy0.cnf")
TOY_CNF   = str(_ASSETS / "toy.cnf")
UNSAT_CNF = str(_ASSETS / "unsat.cnf")

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
    not HAS_MY_EXT, reason="my_ext not importable (build the .so first)"
)
requires_klay = pytest.mark.skipif(
    not HAS_KLAY, reason="klay not importable"
)
requires_both = pytest.mark.skipif(
    not (HAS_MY_EXT and HAS_KLAY),
    reason="both my_ext and klay are required"
)


# -----------------------------------------------------------------------------
# Helpers 
# -----------------------------------------------------------------------------

def _unroll(csr: np.ndarray) -> np.ndarray:
    """Convert CSR-style ix_out offsets into a flat scatter index array.

    klay stores ix_out as a length-(out_size+1) offset array; the number of
    edges pointing to output node k is csr[k+1]-csr[k].  This is equivalent
    to torch.repeat_interleave(arange(out_size), deltas).
    """
    deltas = np.diff(csr)
    return np.repeat(np.arange(len(deltas)), deltas)


def encode_log(nb_vars: int, weights: List[float], neg_weights: List[float]) -> np.ndarray:
    """Build the initial activation vector in the log semiring.

    Layout (matching klay's encode_input):
      index 0 = False-node  (log 0  = -inf)
      index 1 = True-node   (log 1  = 0)
      then for each variable i:  [log P(xi=T), log P(xi=F)]
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
    """Evaluate a klay / my_ext circuit in the log semiring (numpy).

    Layers alternate: product (even index) → sum (odd index).
    Returns log-WMC of the single root.
    """
    for layer_idx, (ix_in, ix_out_csr) in enumerate(zip(ixs_in, ixs_out)):
        ix_out   = _unroll(np.asarray(ix_out_csr))
        out_size = len(ix_out_csr) - 1
        src      = x[np.asarray(ix_in)]

        if layer_idx % 2 == 0:           # product layer  (sum in log space)
            out = np.zeros(out_size, dtype=np.float64)
            np.add.at(out, ix_out, src)
        else:                             # sum layer      (logsumexp)
            out = np.full(out_size, float("-inf"), dtype=np.float64)
            for s, o in zip(src, ix_out):
                out[o] = np.logaddexp(out[o], s)

        x = out

    return float(x[0])


def circuit_wmc(circuit, nb_vars: int, weights: List[float], neg_weights: List[float]) -> float:
    """Compute log-WMC for a circuit returned by my_ext or klay.

    Works with both my_ext.Circuit (.get_indices()) and
    klay.Circuit (._get_indices()).
    """
    get_ix = getattr(circuit, "_get_indices", None) or getattr(circuit, "get_indices")
    ixs_in, ixs_out = get_ix()
    x0 = encode_log(nb_vars, weights, neg_weights)
    return eval_circuit_log(ixs_in, ixs_out, x0)


# ─────────────────────────────────────────────────────────────────────────────
# Brute-force reference WMC
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dimacs(cnf_path: str):
    """Return (nb_vars, list_of_clauses) from a DIMACS file."""
    nb_vars = 0
    clauses = []
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
    """Enumerate all 2^n assignments and accumulate WMC in log space.
    Only practical for CNFs with ≤ 20 variables.
    """
    nb_vars, clauses = _parse_dimacs(cnf_path)
    assert nb_vars <= 25, "brute_force_wmc is only feasible for small CNFs"

    log_wmc = float("-inf")
    for bits in range(1 << nb_vars):
        assignment = {v: bool((bits >> (v - 1)) & 1) for v in range(1, nb_vars + 1)}

        satisfied = True
        for clause in clauses:
            if not any((lit > 0) == assignment[abs(lit)] for lit in clause):
                satisfied = False
                break

        if satisfied:
            log_w = sum(
                weights[v - 1] if assignment[v] else neg_weights[v - 1]
                for v in range(1, nb_vars + 1)
            )
            log_wmc = np.logaddexp(log_wmc, log_w)

    return log_wmc


def make_random_weights(nb_vars: int, seed: int = 42):
    """Return (log_weights, log_neg_weights) with P(xi=T) ~ Uniform(0.1, 0.9)."""
    rng = random.Random(seed)
    probs       = [rng.uniform(0.1, 0.9) for _ in range(nb_vars)]
    weights     = [math.log(p)       for p in probs]
    neg_weights = [math.log(1.0 - p) for p in probs]
    return weights, neg_weights


def uniform_weights(nb_vars: int):
    """All variables have P(xi=T) = 0.5."""
    w = [math.log(0.5)] * nb_vars
    return w, w[:]


# ─────────────────────────────────────────────────────────────────────────────
# Inline d4 NNF helpers (no external d4 binary needed)
# ─────────────────────────────────────────────────────────────────────────────
# Format understood by klay.Circuit.add_d4_from_file:
#   Lines starting with 'o'/'a'/'f'/'t' declare OR/AND/FALSE/TRUE nodes
#   (numbered 1, 2, 3, … in declaration order – no explicit ID on the line).
#   Other lines are edges:  "<parent_id>  <child_id>  <lit1> ... 0"
# ─────────────────────────────────────────────────────────────────────────────

# x1 ∧ x2  →  1 solution, WMC = w1·w2
NNF_AND_X1_X2 = "a\nt\n1 2 1 2 0\n"

# x1 ∨ x2  →  3 solutions, WMC = w1 + w2 - w1·w2
NNF_OR_X1_X2 = "o\nt\nt\nt\n1 2 1 2 0\n1 3 1 -2 0\n1 4 -1 2 0\n"

# x1 ↔ x2  →  2 solutions, WMC = w1·w2 + (1-w1)·(1-w2)
NNF_XNOR_X1_X2 = "o\nt\nt\n1 2 1 2 0\n1 3 -1 -2 0\n"

# UNSAT (False at root)
NNF_UNSAT = "f\n"

# x1 ∨ ¬x1  →  tautology for 1 variable, WMC = 1
NNF_TAUT_X1 = "o\nt\nt\n1 2 1 0\n1 3 -1 0\n"


def _klay_circuit_from_nnf(nnf_content: str) -> "KlayCircuit":
    """Write NNF to a temp file, load it into a klay Circuit, return circuit."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".nnf", delete=False)
    tmp.write(nnf_content)
    tmp.close()
    try:
        c = KlayCircuit()
        c.add_d4_from_file(tmp.name)
        return c
    finally:
        os.unlink(tmp.name)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBruteForceReference:
    """Sanity-check the brute-force evaluator itself."""

    def test_unsat_returns_neg_inf(self):
        assert brute_force_wmc(UNSAT_CNF, *uniform_weights(1)) == float("-inf")

    def test_toy0_model_count(self):
        """toy0.cnf has exactly 12 satisfying assignments out of 2^5 = 32."""
        nb_vars = 5
        log_wmc = brute_force_wmc(TOY0_CNF, *uniform_weights(nb_vars))
        model_count = math.exp(log_wmc) * (2 ** nb_vars)
        assert abs(model_count - 12) < 1e-9, f"expected 12, got {model_count}"


@requires_klay
class TestNumpyEvaluator:
    """Verify the numpy circuit evaluator on manually-constructed circuits."""

    def test_and_x1_x2(self):
        c = KlayCircuit()
        l1 = c.literal_node(1); l2 = c.literal_node(2)
        c.set_root(c.and_node([l1, l2]))
        wmc = circuit_wmc(c, 2, *uniform_weights(2))
        assert abs(math.exp(wmc) - 0.25) < 1e-9

    def test_or_x1_x2(self):
        c = KlayCircuit()
        l1 = c.literal_node(1); l2 = c.literal_node(2)
        nl1 = c.literal_node(-1); nl2 = c.literal_node(-2)
        c.set_root(c.or_node([c.and_node([l1, l2]), c.and_node([l1, nl2]), c.and_node([nl1, l2])]))
        wmc = circuit_wmc(c, 2, *uniform_weights(2))
        assert abs(math.exp(wmc) - 0.75) < 1e-9


@requires_klay
class TestD4CircuitWMC:
    """WMC of klay circuits built from inline d4 NNF files."""

    def _check(self, nnf_content, nb_vars, weights, neg_weights, expected, tol=1e-9):
        c   = _klay_circuit_from_nnf(nnf_content)
        wmc = circuit_wmc(c, nb_vars, weights, neg_weights)
        got = math.exp(wmc)
        assert abs(got - expected) < tol, f"got {got:.8f}, expected {expected:.8f}"

    def test_and_uniform(self):
        self._check(NNF_AND_X1_X2, 2, *uniform_weights(2), 0.25)

    def test_or_uniform(self):
        self._check(NNF_OR_X1_X2, 2, *uniform_weights(2), 0.75)

    def test_xnor_uniform(self):
        self._check(NNF_XNOR_X1_X2, 2, *uniform_weights(2), 0.50)

    def test_taut_uniform(self):
        self._check(NNF_TAUT_X1, 1, *uniform_weights(1), 1.0)

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_and_random_weights(self, seed):
        """WMC(x1∧x2) = P(x1=T) · P(x2=T)."""
        w, nw = make_random_weights(2, seed)
        p1, p2 = math.exp(w[0]), math.exp(w[1])
        self._check(NNF_AND_X1_X2, 2, w, nw, p1 * p2)

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_or_random_weights(self, seed):
        """WMC(x1∨x2) = P(x1=T) + P(x2=T) - P(x1=T)·P(x2=T)."""
        w, nw = make_random_weights(2, seed)
        p1, p2 = math.exp(w[0]), math.exp(w[1])
        self._check(NNF_OR_X1_X2, 2, w, nw, p1 + p2 - p1 * p2)


@requires_my_ext
class TestGanakCorrectnessVsBruteForce:
    """Core correctness: ganak circuit WMC must match brute-force WMC."""

    TOL = 1e-6   # relative tolerance

    def _assert_matches(self, cnf_path, nb_vars, weights, neg_weights):
        circuit   = my_ext.compile_to_ganak(cnf_path)
        ganak_log = circuit_wmc(circuit, nb_vars, weights, neg_weights)
        ref_log   = brute_force_wmc(cnf_path, weights, neg_weights)

        if ref_log == float("-inf"):
            assert ganak_log == float("-inf"), \
                f"Expected UNSAT (-inf) from ganak, got {ganak_log}"
            return

        ganak_wmc = math.exp(ganak_log)
        ref_wmc   = math.exp(ref_log)
        rel_err   = abs(ganak_wmc - ref_wmc) / (abs(ref_wmc) + 1e-30)
        assert rel_err < self.TOL, (
            f"WMC mismatch: ganak={ganak_wmc:.8f}, "
            f"brute_force={ref_wmc:.8f}, rel_err={rel_err:.2e}"
        )

    def test_unsat(self):
        self._assert_matches(UNSAT_CNF, 1, *uniform_weights(1))

    def test_toy0_uniform(self):
        """WMC with uniform weights == 12/32 == 0.375."""
        self._assert_matches(TOY0_CNF, 5, *uniform_weights(5))

    @pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
    def test_toy0_random_weights(self, seed):
        """Fuzz toy0.cnf with several random weight vectors."""
        self._assert_matches(TOY0_CNF, 5, *make_random_weights(5, seed))

    def test_toy_cnf(self):
        nb_vars, _ = _parse_dimacs(TOY_CNF)
        if nb_vars > 20:
            pytest.skip("toy.cnf has too many variables for brute-force")
        self._assert_matches(TOY_CNF, nb_vars, *uniform_weights(nb_vars))


@requires_both
class TestGanakVsD4WMC:
    """Compare ganak circuits against equivalent klay/d4 NNF circuits.

    For each formula:
      1. Write as DIMACS CNF  → compile with ganak.
      2. Write as NNF         → load with klay.Circuit.add_d4_from_file.
      3. Evaluate both with the same weights and assert equality.
    """

    TOL = 1e-9

    def _make_cnf(self, header, clauses):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".cnf", delete=False)
        tmp.write(header + "\n")
        for c in clauses:
            tmp.write(c + "\n")
        tmp.close()
        return tmp.name

    def _compare(self, cnf_path, nnf_content, nb_vars, weights, neg_weights):
        ganak_c = my_ext.compile_to_ganak(cnf_path)
        d4_c    = _klay_circuit_from_nnf(nnf_content)

        g_log = circuit_wmc(ganak_c, nb_vars, weights, neg_weights)
        d_log = circuit_wmc(d4_c,    nb_vars, weights, neg_weights)

        if d_log == float("-inf"):
            assert g_log == float("-inf")
            return

        rel_err = abs(math.exp(g_log) - math.exp(d_log)) / (abs(math.exp(d_log)) + 1e-30)
        assert rel_err < self.TOL, (
            f"ganak WMC={math.exp(g_log):.8f}  d4 WMC={math.exp(d_log):.8f}  "
            f"rel_err={rel_err:.2e}"
        )

    # ── x1 ∧ x2 ──────────────────────────────────────────────────────────────

    def test_and_uniform(self):
        path = self._make_cnf("p cnf 2 2", ["1 0", "2 0"])
        try:
            self._compare(path, NNF_AND_X1_X2, 2, *uniform_weights(2))
        finally:
            os.unlink(path)

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_and_random_weights(self, seed):
        path = self._make_cnf("p cnf 2 2", ["1 0", "2 0"])
        try:
            self._compare(path, NNF_AND_X1_X2, 2, *make_random_weights(2, seed))
        finally:
            os.unlink(path)

    # ── x1 ∨ x2 ──────────────────────────────────────────────────────────────

    def test_or_uniform(self):
        path = self._make_cnf("p cnf 2 1", ["1 2 0"])
        try:
            self._compare(path, NNF_OR_X1_X2, 2, *uniform_weights(2))
        finally:
            os.unlink(path)

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_or_random_weights(self, seed):
        path = self._make_cnf("p cnf 2 1", ["1 2 0"])
        try:
            self._compare(path, NNF_OR_X1_X2, 2, *make_random_weights(2, seed))
        finally:
            os.unlink(path)

    # ── toy0 integration ─────────────────────────────────────────────────────

    def test_toy0_model_count(self):
        """ganak circuit WMC on toy0.cnf must equal 12/32 with uniform weights."""
        if not os.path.exists(TOY0_CNF):
            pytest.skip("toy0.cnf not found")
        circuit = my_ext.compile_to_ganak(TOY0_CNF)
        wmc = math.exp(circuit_wmc(circuit, 5, *uniform_weights(5)))
        assert abs(wmc - 12 / 32) < 1e-6, f"got {wmc:.6f}, expected {12/32:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# Performance benchmarks
# ─────────────────────────────────────────────────────────────────────────────

NB_EVAL_REPEATS = 50


def _time_fn(fn, repeats=1):
    t0 = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = fn()
    return result, (time.perf_counter() - t0) / repeats


@requires_my_ext
class TestBenchmarks:
    """Wall-clock benchmarks.  Run with ``pytest -v -s -k bench``."""

    def _bench_cnf(self, label, cnf_path, nb_vars=None):
        if not os.path.exists(cnf_path):
            pytest.skip(f"{cnf_path} not found")
        if nb_vars is None:
            nb_vars, _ = _parse_dimacs(cnf_path)

        # Compilation
        _, compile_s = _time_fn(lambda: my_ext.compile_to_ganak(cnf_path))

        # Evaluation (reuse circuit)
        circuit = my_ext.compile_to_ganak(cnf_path)
        w, nw   = make_random_weights(nb_vars, seed=0)
        _, eval_s = _time_fn(
            lambda: circuit_wmc(circuit, nb_vars, w, nw),
            repeats=NB_EVAL_REPEATS,
        )

        # Optional klay/d4 comparison (same circuit, same indices)
        d4_eval_s = None
        if HAS_KLAY:
            get_ix = getattr(circuit, "get_indices", None)
            if get_ix:
                ixs = get_ix()
                x0  = encode_log(nb_vars, w, nw)
                _, d4_eval_s = _time_fn(
                    lambda: eval_circuit_log(ixs[0], ixs[1], x0.copy()),
                    repeats=NB_EVAL_REPEATS,
                )

        d4_str = f"{d4_eval_s*1e6:.1f} µs" if d4_eval_s is not None else "N/A"
        print(
            f"\n  [{label}]\n"
            f"    nodes          : {circuit.nb_nodes()}\n"
            f"    compile time   : {compile_s*1e3:.2f} ms\n"
            f"    eval (ganak)   : {eval_s*1e6:.1f} µs  (numpy, {NB_EVAL_REPEATS} runs)\n"
            f"    eval (d4 ref)  : {d4_str}\n"
        )

    def test_bench_toy0(self):
        """Benchmark toy0.cnf (5 vars – smallest, fastest)."""
        self._bench_cnf("toy0.cnf  5 vars", TOY0_CNF, nb_vars=5)

    def test_bench_toy(self):
        """Benchmark toy.cnf."""
        self._bench_cnf("toy.cnf", TOY_CNF)

    @requires_both
    def test_bench_ganak_vs_d4_simple(self):
        """Head-to-head: ganak vs d4 circuit for AND(x1,x2)."""
        nb_vars = 2
        w, nw   = make_random_weights(nb_vars, seed=0)

        cnf_path = tempfile.mktemp(suffix=".cnf")
        with open(cnf_path, "w") as f:
            f.write("p cnf 2 2\n1 0\n2 0\n")
        try:
            ganak_c = my_ext.compile_to_ganak(cnf_path)
            d4_c    = _klay_circuit_from_nnf(NNF_AND_X1_X2)

            _, ganak_t = _time_fn(
                lambda: circuit_wmc(ganak_c, nb_vars, w, nw),
                repeats=NB_EVAL_REPEATS,
            )
            _, d4_t = _time_fn(
                lambda: circuit_wmc(d4_c, nb_vars, w, nw),
                repeats=NB_EVAL_REPEATS,
            )

            g_wmc = math.exp(circuit_wmc(ganak_c, nb_vars, w, nw))
            d_wmc = math.exp(circuit_wmc(d4_c,    nb_vars, w, nw))

            print(
                f"\n  [ganak vs d4 – AND(x1, x2)]\n"
                f"    WMC ganak  : {g_wmc:.8f}\n"
                f"    WMC d4     : {d_wmc:.8f}\n"
                f"    eval ganak : {ganak_t*1e6:.1f} µs\n"
                f"    eval d4    : {d4_t*1e6:.1f} µs\n"
                f"    speedup    : {d4_t/max(ganak_t,1e-12):.2f}×"
            )
            assert abs(g_wmc - d_wmc) < 1e-9
        finally:
            os.unlink(cnf_path)
