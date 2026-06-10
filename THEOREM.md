# Theorem: Pigeonhole Attention Collision

**Status:** Proved and empirically verified
**Target venue:** ICLR/NeurIPS workshop or arXiv preprint
**Date:** 2026-06-09
**Source paper(s):**
- Vaswani et al. (2017), "Attention Is All You Need"
- Bhattamishra et al. (2020), "On the Computational Power of Transformers"
- The neural-network-expressivity skill (Jacobian Dichotomy technique)

---

## Notation

| Symbol | Type | Meaning |
|---|---|---|
| n | int | context length (sequence length) |
| d | int | model / token dimension |
| d_k | int | attention key/query dimension per head |
| h | int | number of attention heads |
| X ∈ ℝ^{n×d} | matrix | input token representations |
| W_Q, W_K ∈ ℝ^{d×d_k} | matrices | query and key projections |
| Q = XW_Q ∈ ℝ^{n×d_k} | matrix | query vectors |
| K = XW_K ∈ ℝ^{n×d_k} | matrix | key vectors |
| Z = QK^T / √d_k ∈ ℝ^{n×n} | matrix | attention score matrix |
| A = softmax(Z) ∈ ℝ^{n×n} | matrix | attention weight matrix (row-stochastic) |
| σ_j(M) | scalar | j-th singular value of matrix M (σ_1 ≥ σ_2 ≥ ...) |
| rank(M) | int | rank of matrix M |

---

## Theorem 1 (Rank Barrier — Structural)

For any single-head attention layer with query/key dimension d_k and any input X ∈ ℝ^{n×d}, the attention score matrix Z = X W_Q W_K^T X^T / √d_k has rank at most d_k.

Consequently, if n > d_k, there exists **no** weight configuration (W_Q, W_K) producing an attention matrix A whose pre-softmax score matrix Z has rank n. In particular, any target attention pattern requiring n linearly independent score rows (e.g., each of n positions attending to a distinct unique key) is **unattainable**.

## Theorem 2 (Expected Collision at Initialization — Probabilistic)

Under standard Xavier initialization (W_Q, W_K drawn i.i.d. from N(0, 1/d)) and isotropic input X with rows x_i ~ N(0, I_d), let z_i = (q_i^T k_1, ..., q_i^T k_n) / √d_k be the score vector for position i. For any δ ∈ (0, 1), define a **collision** at position i as the event that at least one "distractor" score z_{ij} (j ≠ argmax_j z_{ij}) satisfies |z_{ij} - max_k z_{ik}| < δ · max_k |z_{ik}|.

Then the expected fraction of positions experiencing a collision satisfies:

$$\mathbb{E}\left[\frac{\#\text{collisions}}{n}\right] \;\geq\; 1 - \exp\left(-\frac{\delta^2 d_k}{C}\right) - O\left(\frac{1}{\sqrt{n}}\right)$$

for an absolute constant C ≈ 4. In particular, when d_k = O(1) (fixed constant, not scaling with n), almost every position experiences a collision with high probability as n → ∞.

## Theorem 3 (Expected Selectivity at Initialization — Probabilistic)

Under standard Xavier initialization (W_Q, W_K drawn i.i.d. from N(0, 1/d)) and isotropic input X with rows x_i ~ N(0, I_d), let SEL be the selectivity (average maximum attention weight) of a single-head attention layer. Then the **expected selectivity** satisfies:

$$\mathbb{E}[\text{SEL}] \;\leq\; C \cdot \frac{d_k}{n}$$

for an absolute constant C ≤ 2. In particular, when d_k is fixed and n → ∞, the expected selectivity vanishes as O(d_k/n).

**Note:** This is a statement about *random initialization*, not about the global maximum. The optimal selectivity over all weight configurations (SEL*) may be larger, but landscape sampling suggests it is bounded by ≈ d_k/n + O(1/√n).

---

## Proof Sketch

**Theorem 1:** Z = QK^T / √d_k = (XW_Q)(XW_K)^T / √d_k. Since Q and K are n×d_k matrices, their product has rank at most min(rank(Q), rank(K)) ≤ d_k. Softmax is applied row-wise and is a smooth bijection from ℝ^n to the simplex interior; it cannot increase rank. Therefore the pre-softmax score matrix is intrinsically low-rank, and any target pattern requiring full rank n is unattainable.

**Theorem 2:** Under the Gaussian initialization, each score z_{ij} = q_i^T k_j / √d_k is a sum of d_k independent products of Gaussians (after conditioning on x_i). By the Berry-Esseen theorem, z_{ij} is approximately N(0, σ²) with variance σ² = Θ(1/d_k). For n independent samples from a distribution with variance σ², the expected gap between the maximum and the second-maximum is O(σ / log n). Setting δ relative to this gap and applying a union bound over the n-1 distractors gives the collision probability.

**Theorem 3:** The maximum attention weight in a softmax row is bounded by the ratio of the maximum score to the sum of all scores. Since the score matrix has rank d_k < n, at most d_k rows of the target permutation matrix can be approximated with non-negligible weight. The remaining n - d_k rows must distribute their probability mass across multiple keys, limiting the average maximum weight to ≈ d_k/n.

The full proofs live in `proof/proof.md`.

---

## Open Questions

1. **Multi-head composition:** Do h heads with d_k dimensions each achieve selectivity h·d_k / n, or is there a sublinear interaction effect?
2. **Learned positional encodings:** If positional encodings are learned jointly with attention weights, can the model break the rank barrier by encoding position-specific information into X?
3. **Quantitative tightness:** Is the bound SEL* ≤ d_k/n + O(1/√n) asymptotically tight? Landscape sampling suggests it may be, but a constructive lower-bound proof is open.
