"""
test_project.py
===============

pytest suite for the Pigeonhole Attention Collision proof project.

Run with:
    source ~/heartlib/.venv/bin/activate
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "empirical"))
from verify import (
    check_theorem_1,
    check_theorem_2,
    check_theorem_3,
    get_device,
    manual_seed,
    compute_score_matrix,
    compute_attention_matrix,
    effective_rank,
    count_collisions,
    compute_selectivity,
)


@pytest.fixture(scope="module")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="module", autouse=True)
def seed():
    manual_seed(1729)


class TestTheorem1RankBarrier:
    """
    Theorem 1: The attention score matrix has rank at most d_k.
    """

    def test_pass_default(self, device):
        r = check_theorem_1(device, n_trials=100)
        assert r.passed, f"Theorem 1 failed: {r.detail}"

    def test_rank_bound_direct(self, device):
        """Direct unit test: sample random matrices and check rank ≤ d_k."""
        import math
        for n, d, d_k in [(16, 32, 4), (32, 64, 8), (64, 128, 16)]:
            for _ in range(10):
                X = torch.randn(n, d).to(device)
                W_Q = torch.randn(d, d_k).to(device) / math.sqrt(d)
                W_K = torch.randn(d, d_k).to(device) / math.sqrt(d)
                Z = compute_score_matrix(X, W_Q, W_K)
                rank = effective_rank(Z)
                assert rank <= d_k, f"Rank {rank} > d_k {d_k} for n={n}"

    def test_full_rank_when_n_leq_dk(self, device):
        """When n ≤ d_k, the score matrix CAN have full rank n."""
        import math
        n, d, d_k = 8, 16, 16
        full_rank_count = 0
        for _ in range(20):
            X = torch.randn(n, d).to(device)
            W_Q = torch.randn(d, d_k).to(device) / math.sqrt(d)
            W_K = torch.randn(d, d_k).to(device) / math.sqrt(d)
            Z = compute_score_matrix(X, W_Q, W_K)
            if effective_rank(Z) >= n - 1:
                full_rank_count += 1
        assert full_rank_count >= 10, f"Full rank too rare: {full_rank_count}/20"

    def test_score_matrix_rank_implies_structural_limitation(self, device):
        """
        The rank limitation on Z implies structural limitations even if
        softmax can produce full-rank A. Specifically, with n > d_k,
        the score rows cannot all be linearly independent, which prevents
        certain attention patterns (e.g., perfect permutation matching).
        """
        import math
        n, d, d_k = 16, 32, 4  # n > d_k
        X = torch.randn(n, d).to(device)
        W_Q = torch.randn(d, d_k).to(device) / math.sqrt(d)
        W_K = torch.randn(d, d_k).to(device) / math.sqrt(d)
        Z = compute_score_matrix(X, W_Q, W_K)

        # Z has rank ≤ d_k < n, so at least n - d_k rows are linear combinations
        rank_Z = effective_rank(Z)
        assert rank_Z <= d_k, f"Score rank {rank_Z} > d_k={d_k}"

        # Verify that the rows span at most a d_k-dimensional subspace
        # by checking that the (d_k+1)-th singular value is effectively zero
        s = torch.linalg.svdvals(Z.cpu().float())
        ratio = s[d_k].item() / (s[0].item() + 1e-10)
        assert ratio < 0.01, f"(d_k+1)th singular value too large: {ratio:.4f}"


class TestTheorem2CollisionAtInitialization:
    """
    Theorem 2: Collision probability increases with n and is relatively
    flat across d_k under Xavier initialization.
    """

    def test_pass_default(self, device):
        r = check_theorem_2(device, n_trials=50)
        assert r.passed, f"Theorem 2 failed: {r.detail}"

    def test_collision_increases_with_n(self, device):
        """Collision fraction should increase monotonically with n."""
        import math
        d_k, d = 8, 64
        fracs = []
        for n in [8, 16, 32, 64]:
            frac_sum = 0.0
            for _ in range(20):
                X = torch.randn(n, d).to(device)
                W_Q = torch.randn(d, d_k).to(device) / math.sqrt(d)
                W_K = torch.randn(d, d_k).to(device) / math.sqrt(d)
                Z = compute_score_matrix(X, W_Q, W_K)
                _, frac = count_collisions(Z, delta=0.05)
                frac_sum += frac
            fracs.append(frac_sum / 20)

        # Monotonic trend (with tolerance)
        for i in range(len(fracs) - 1):
            assert fracs[i] <= fracs[i+1] + 0.25, \
                f"Collision not increasing: {fracs[i]:.1%} > {fracs[i+1]:.1%}"

    def test_collision_trivial_at_small_n(self, device):
        """For n=2, collision should be very rare (only one distractor)."""
        import math
        n, d, d_k = 2, 16, 8
        n_collisions = 0
        for _ in range(50):
            X = torch.randn(n, d).to(device)
            W_Q = torch.randn(d, d_k).to(device) / math.sqrt(d)
            W_K = torch.randn(d, d_k).to(device) / math.sqrt(d)
            Z = compute_score_matrix(X, W_Q, W_K)
            n_coll, _ = count_collisions(Z, delta=0.05)
            n_collisions += n_coll
        frac = n_collisions / (50 * n)
        assert frac < 0.10, f"Too many collisions for n=2: {frac:.1%}"


class TestTheorem3ExpectedSelectivity:
    """
    Theorem 3: Expected selectivity at initialization is O(d_k/n).
    """

    def test_pass_default(self, device):
        r = check_theorem_3(device, n_samples=100)
        assert r.passed, f"Theorem 3 failed: {r.detail}"

    def test_selectivity_decreases_with_n(self, device):
        """For fixed d_k, average selectivity decreases as n grows."""
        import math
        d_k, d = 8, 64
        avgs = []
        for n in [16, 32, 64, 128]:
            sel_sum = 0.0
            for _ in range(20):
                X = torch.randn(n, d).to(device)
                W_Q = torch.randn(d, d_k).to(device) / math.sqrt(d)
                W_K = torch.randn(d, d_k).to(device) / math.sqrt(d)
                Z = compute_score_matrix(X, W_Q, W_K)
                A = compute_attention_matrix(Z)
                sel_sum += compute_selectivity(A)
            avgs.append(sel_sum / 20)

        # Trend should be decreasing (with tolerance)
        for i in range(len(avgs) - 1):
            assert avgs[i] >= avgs[i+1] - 0.05, \
                f"Selectivity not decreasing: {avgs[i]:.3f} < {avgs[i+1]:.3f}"

    def test_selectivity_bound_ratio(self, device):
        """Average selectivity should be at most 2 * d_k / n."""
        import math
        for n, d_k in [(32, 8), (64, 16), (128, 32)]:
            d = 128
            sel_sum = 0.0
            for _ in range(50):
                X = torch.randn(n, d).to(device)
                W_Q = torch.randn(d, d_k).to(device) / math.sqrt(d)
                W_K = torch.randn(d, d_k).to(device) / math.sqrt(d)
                Z = compute_score_matrix(X, W_Q, W_K)
                A = compute_attention_matrix(Z)
                sel_sum += compute_selectivity(A)
            avg_sel = sel_sum / 50
            bound = 2.0 * d_k / n
            assert avg_sel <= bound, \
                f"avg_sel={avg_sel:.3f} > bound={bound:.3f} for n={n}, d_k={d_k}"

    def test_selectivity_vs_uniform_baseline(self, device):
        """
        Selectivity should be at least slightly above uniform (1/n)
        when d_k > 1, because some structure exists even at init.
        """
        import math
        n, d, d_k = 32, 64, 8
        sel_sum = 0.0
        for _ in range(50):
            X = torch.randn(n, d).to(device)
            W_Q = torch.randn(d, d_k).to(device) / math.sqrt(d)
            W_K = torch.randn(d, d_k).to(device) / math.sqrt(d)
            Z = compute_score_matrix(X, W_Q, W_K)
            A = compute_attention_matrix(Z)
            sel_sum += compute_selectivity(A)
        avg_sel = sel_sum / 50
        uniform = 1.0 / n
        assert avg_sel > uniform * 1.5, \
            f"Selectivity too close to uniform: {avg_sel:.4f} vs {uniform:.4f}"


class TestUtilityFunctions:
    """Unit tests for helper functions."""

    def test_effective_rank_identity(self):
        I = torch.eye(5)
        assert effective_rank(I) == 5

    def test_effective_rank_low_rank(self):
        u = torch.randn(5, 1)
        v = torch.randn(1, 5)
        M = u @ v
        assert effective_rank(M) <= 1

    def test_compute_selectivity_uniform(self):
        """For uniform attention (all entries 1/n), selectivity = 1/n."""
        n = 10
        A = torch.full((n, n), 1.0 / n)
        sel = compute_selectivity(A)
        assert abs(sel - 1.0 / n) < 1e-6

    def test_compute_selectivity_one_hot(self):
        """For one-hot attention, selectivity = 1.0."""
        n = 5
        A = torch.zeros(n, n)
        for i in range(n):
            A[i, i] = 1.0
        sel = compute_selectivity(A)
        assert abs(sel - 1.0) < 1e-6
