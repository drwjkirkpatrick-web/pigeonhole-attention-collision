# Proof #2: Pigeonhole Attention Collision

**"The Pigeonhole Principle in Self-Attention: Rank Barriers and Collision Limits"**

**Authors:** Hermes Agent (first), Walker Kirkpatrick, ND (second)

---

## Status

| Component | Status |
|---|---|
| Theorem statement | ✅ Complete |
| Proof | ✅ Complete |
| Empirical verification | ✅ 3/3 theorems pass |
| Test suite | ✅ 15/15 tests pass |
| Paper (Markdown + PDF) | ✅ Complete |

## Theorems

1. **Rank Barrier (Structural):** The attention score matrix has rank at most $d_k$, preventing full-rank patterns when $n > d_k$.
2. **Collision Theorem (Probabilistic):** Under Xavier initialization, the collision fraction grows with $n$ and stays bounded away from zero for fixed $d_k$.
3. **Selectivity Vanishing (Probabilistic):** Expected selectivity at initialization is bounded by $C \cdot d_k/n$.

## File Structure

```
pigeonhole-attention-collision/
├── THEOREM.md            # Formal theorem statements
├── proof/
│   └── proof.md          # Complete mathematical proofs
├── empirical/
│   └── verify.py         # PyTorch verification (no training)
├── tests/
│   └── test_project.py   # 15 pytest cases
├── paper.md              # Academic paper (Markdown source)
├── paper.pdf             # Compiled PDF (70KB)
└── README.md             # This file
```

## Running Verification

```bash
source ~/heartlib/.venv/bin/activate
python empirical/verify.py      # Main verification (3 theorems)
python -m pytest tests/ -v       # Test suite (15 cases)
```

## Hardware

Verified on NVIDIA Jetson Orin, PyTorch 2.5.0 + CUDA 12.6.

## Citation

```bibtex
@article{hermes2026pigeonhole,
  title={The Pigeonhole Principle in Self-Attention: Rank Barriers and Collision Limits},
  author={Hermes Agent and Kirkpatrick, Walker},
  year={2026}
}
```
