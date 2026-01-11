# fast-neural-geometry

Fast, PyTorch-based analysis of neural manifolds. The core functions estimate manifold capacity, radius, and dimension from sampled points. The `original/` folder holds a reference
implementation used to validate the optimized code. The original implementation is from https://github.com/chung-neuroai-lab/cshl2022-deep-learning-manifolds-tutorial. I made minor modifications primarily to improve numerical stability.

## What you get
- Fast PyTorch-based solvers for manifold geometry and correlation analysis, supporting CPU and GPU with consistent NumPy outputs.
- Optimized implementations validated against reference solvers via accuracy and timing tests.
- For input sizes in `[1000, 4000]`, ~100× speedup on CPU and ~1000× on GPU compared to the original implementation.

## Performance and optimizations
- Replaced CVXOPT QP solves with a batched projected gradient descent (PGD) dual solver.
- Fully vectorized across all `n_t` optimization problems.
- Precomputed Gram matrices and Lipschitz constants per manifold.
- Applied Nesterov acceleration with adaptive restart.
- Moved heavy linear algebra to PyTorch to minimize Python overhead.

#### Notes
- Tests and benchmarks are in `test/`.
- The original implementation (`original/manifold_analysis_correlation.py`) fails one test.

## Requirements
- Python 3.9+
- Use-only (fast implementation): `requirements.txt` (torch, numpy)
- Use + test (includes original reference): `requirements-test.txt`

## Quick start

Each manifold is a 2D array with shape `(d, n)` where `d` is the ambient dimension and `n` is the number of sampled points. Pass a list of manifolds into the analysis functions.

### Manifold analysis (fast)
```python
import numpy as np
from manifold_analysis import manifold_analysis_fast

d = 64
n = 200
num_manifolds = 5

XtotT = [np.random.randn(d, n) for _ in range(num_manifolds)]
kappa = 0.1
n_t = 200

a_Mfull, R_M, D_M = manifold_analysis_fast(
    XtotT,
    kappa=kappa,
    n_t=n_t,
    device="cuda",  # or "cpu"
)
```

### Correlation analysis (fast)
```python
import numpy as np
from manifold_analysis_correlation import manifold_analysis_corr_fast

d = 64
n = 200
num_manifolds = 5

XtotT = [np.random.randn(d, n) for _ in range(num_manifolds)]
kappa = 0.1
n_t = 200

a_Mfull, R_M, D_M, res_coeff0, KK = manifold_analysis_corr_fast(
    XtotT,
    kappa=kappa,
    n_t=n_t,
    n_reps=10,
    device="cuda",
)
```

### Manifold SNR (pairwise)
`manifold_snr.py` implements the Sorscher et al. (2022) geometry-based SNR for binary classification. The code is from https://github.com/bsorsch/geometry-fewshot-learning/.
It extracts per-class manifold geometry via SVD and combines distance and overlap terms into an
SNR estimate.

```python
import numpy as np
from manifold_snr import pairwise_manifold_snr

n_samples = 200
n_features = 64

class_0 = np.random.randn(n_samples, n_features)
class_1 = np.random.randn(n_samples, n_features)

stats = pairwise_manifold_snr(class_0, class_1, m=1)
print(stats["snr"])
```

Returns a dict with `participation_ratio`, `dist_norm`, `signal_noise_overlap`, `signal`, `bias`,
`noise`, and `snr`. Inputs are `(n_samples, n_features)` arrays (one per class).

### Shape notes for `t_vecs`

- `t_vecs` is optional. Provide one matrix per manifold with shape `(D_reduced + 1, n_t)`.
- If `d > n`, the solver reduces the dimension to `D_reduced = n`, so `t_vecs[i]` must have `n + 1` rows.
- Outputs are NumPy arrays (or scalars) even when running on GPU.



## References

Chung, S., Lee, D. D., & Sompolinsky, H. (2018). Classification and geometry of general perceptual manifolds.
Physical Review X, 8(3), 031003.

Cohen, U., Chung, S., Lee, D. D., & Sompolinsky, H. (2020). Separability and geometry of object manifolds in deep
neural networks. Nature Communications, 11(1), 746.

Sorscher, B., Ganguli, S., & Sompolinsky, H. (2022). Neural representational geometry underlies few-shot concept
learning. Proceedings of the National Academy of Sciences, 119(43), e2200800119.
