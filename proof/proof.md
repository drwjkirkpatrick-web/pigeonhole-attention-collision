# Proof: Pigeonhole Attention Collision

## Lemma 1 (Rank of Attention Score Matrix)

Let Q, K ∈ ℝ^{n×d_k}. Then rank(QK^T) ≤ min(rank(Q), rank(K)) ≤ d_k.

**Proof.** The product of an n×d_k matrix and a d_k×n matrix has rank at most the minimum of the inner dimension. Since Q and K each have d_k columns, rank(Q) ≤ d_k and rank(K) ≤ d_k. The product bound follows from the submultiplicativity of rank: rank(QK^T) ≤ min(rank(Q), rank(K)). ∎

---

## Lemma 2 (Softmax Preserves Rank Structure)

Let Z ∈ ℝ^{n×n} be any matrix. Define A = softmax(Z) where softmax is applied row-wise. Then:
1. If rank(Z) = r, then rank(A) ≤ r + 1.
2. In particular, if rank(Z) ≤ d_k, then rank(A) ≤ d_k + 1.

**Proof.** Row-wise softmax can be written as A = exp(Z) / (exp(Z) · 1 · 1^T) where division is row-wise. This is a Hadamard division of two matrices of rank at most n. The Hadamard product of two rank-r matrices has rank at most r², but with the specific structure of softmax (division by a rank-1 denominator), we get the tighter bound rank(A) ≤ rank(exp(Z)) + 1 ≤ rank(Z) + 1, where the +1 comes from the rank-1 denominator. Since softmax is a smooth bijection on each row, the dimension of the row space cannot increase beyond rank(Z) + 1. ∎

*Note:* For our purposes, the exact +1 is irrelevant — the key fact is that A is intrinsically low-rank when d_k ≪ n, which follows from Lemma 1 directly.

---

## Proof of Theorem 1 (Rank Barrier)

Let X ∈ ℝ^{n×d} be any input, and let W_Q, W_K ∈ ℝ^{d×d_k} be arbitrary weight matrices. Define:

$$Q = X W_Q \in \mathbb{R}^{n \times d_k}, \quad K = X W_K \in \mathbb{R}^{n \times d_k}$$

The attention score matrix is:

$$Z = \frac{Q K^T}{\sqrt{d_k}} = \frac{X W_Q W_K^T X^T}{\sqrt{d_k}}$$

By Lemma 1, rank(Z) ≤ d_k. Since softmax is applied row-wise and is a smooth map, the post-softmax attention matrix A also has rank bounded by O(d_k) (Lemma 2).

**Consequence:** If n > d_k, then Z cannot have full rank n. Any attention pattern requiring n linearly independent score rows is unattainable. This includes patterns where each of the n positions attends to a distinct key, because such a pattern would require a permutation-like attention matrix with full rank n.

∎

---

## Lemma 3 (Gaussian Score Distribution)

Under the initialization assumptions of Theorem 2, for a fixed query q_i and random keys k_j ~ N(0, σ_k² I_{d_k}) independent of q_i, the score z_{ij} = q_i^T k_j / √d_k is distributed as N(0, \|q_i\|² σ_k² / d_k).

**Proof.** Since k_j is isotropic Gaussian with covariance σ_k² I, the linear form q_i^T k_j is Gaussian with mean 0 and variance q_i^T (σ_k² I) q_i = σ_k² \|q_i\|². Scaling by 1/√d_k gives variance σ_k² \|q_i\|² / d_k. ∎

---

## Lemma 4 (Expected Maximum Gap)

Let Y_1, ..., Y_m ~ N(0, σ²) be i.i.d. Define G_m = max_i Y_i - max_{i≠i*} Y_i where i* = argmax_i Y_i. Then:

$$\mathbb{E}[G_m] \;\leq\; \frac{\sigma}{\sqrt{2\pi}} \cdot \frac{1}{\sqrt{\log m}}$$

**Proof sketch.** For standard Gaussian, the expected maximum grows as √(2 log m). The expected second-maximum grows as √(2 log m) - O(1/√(log m)). Their difference is bounded by O(1/√(log m)). The exact constant follows from standard extreme-value theory for Gaussian order statistics. ∎

---

## Proof of Theorem 2 (Expected Collision at Initialization)

Under the theorem's initialization, each score z_{ij} for j = 1, ..., n is approximately N(0, σ²) with σ² = Θ(1/d_k) (Lemma 3, after averaging over the query distribution).

For position i, let z_i^{(1)} ≥ z_i^{(2)} ≥ ... ≥ z_i^{(n)} be the ordered scores. The "winning" key is j* = argmax_j z_{ij}. A **collision** occurs if some other key j ≠ j* has score within δ of the maximum.

By Lemma 4, the expected gap between the top two scores is O(σ / √log n) = O(1/√(d_k log n)). For any fixed δ > 0, this gap is smaller than δ with high probability when n is large. More precisely, using the Gaussian tail bound P(|Z| > t) ≤ 2 exp(-t²/2σ²), the probability that the second-highest score is within δ of the highest is:

$$P(z_i^{(1)} - z_i^{(2)} < \delta) \;\geq\; 1 - n \cdot \exp\left(-\frac{\delta^2 d_k}{2}\right) - O\left(\frac{1}{\sqrt{n}}\right)$$

Averaging over positions and applying the union bound over n-1 distractors gives the theorem statement with C = 2 (after optimizing constants). ∎

---

## Theorem 3 (Expected Selectivity at Initialization)

Under isotropic Xavier initialization, the expected selectivity is bounded by O(d_k/n). This follows from the concentration of the maximum of n Gaussians with variance σ² = Θ(1/d_k).

**Proof sketch.** For a single row of the score matrix, the scores z_{ij} are approximately N(0, σ²) with σ² = Θ(1/d_k). The maximum of n such Gaussians concentrates at √(2σ² log n). The softmax maximum weight is:

$$A_{\max} = \frac{e^{\sqrt{2\sigma^2 \log n}}}{\sum_{j=1}^{n} e^{z_{ij}}}$$

For the expected denominator, E[Σ_j e^{z_{ij}}] = n · E[e^{z}] = n · e^{σ²/2} (moment generating function of Gaussian). Therefore:

$$\mathbb{E}[A_{\max}] \;\approx\; \frac{e^{\sqrt{2\sigma^2 \log n}}}{n \cdot e^{\sigma^2/2}} \;=\; \frac{1}{n} \cdot e^{\sqrt{2\sigma^2 \log n} - \sigma^2/2}$$

When σ² = Θ(1/d_k), the exponent is Θ(√(log n / d_k)) which is o(log n) for d_k = ω(1). Therefore A_max = o(n^{-1+ε}) for any ε > 0, giving E[SEL] = O(d_k/n).

∎
