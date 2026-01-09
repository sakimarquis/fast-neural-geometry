"""
Test script to verify the fast PyTorch correlation analysis matches the original implementation.
Also benchmarks performance differences.
"""

import time

import numpy as np
import torch

from manifold_analysis_correlation import (
    MGramSchmidt_fast,
    OptStiefelGBB_fast,
    fun_FA_fast,
    manifold_analysis_corr_fast as fast_manifold_corr,
    square_corrcoeff_full_cost_fast,
)
from original.manifold_analysis_correlation import (
    OptStiefelGBB as original_OptStiefelGBB,
    fun_FA as original_fun_FA,
    manifold_analysis_corr as original_manifold_corr,
    square_corrcoeff_full_cost as original_cost_fn,
)

TOL = 0.02  # 2% relative tolerance for stochastic optimization
ATOL = 1e-3


def _make_manifold_inputs(seed, num_manifolds, d, p, n_t, t_dim=None):
    """Build deterministic manifolds and t vectors for solver comparisons."""
    rng = np.random.default_rng(seed)
    if t_dim is None:
        t_dim = d + 1
    XtotT = []
    t_vecs = []
    for _ in range(num_manifolds):
        XtotT.append(rng.standard_normal((d, p)))
        t_vecs.append(rng.standard_normal((t_dim, n_t)))
    return XtotT, t_vecs


def _time_fn(fn, *args, **kwargs):
    """Time a function call."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def _print_section(title):
    """Print a consistent section header for test output."""
    line = "=" * 60
    print(line)
    print(title)
    print(line)


def test_gram_schmidt():
    """Test MGramSchmidt_fast produces orthonormal columns."""
    _print_section("GRAM-SCHMIDT TEST")

    np.random.seed(42)
    n, k = 20, 5
    V = np.random.randn(n, k)
    V_t = torch.as_tensor(V, dtype=torch.float32)

    V_orth = MGramSchmidt_fast(V_t)

    # Check orthonormality: V^T V should be identity
    VTV = V_orth.T @ V_orth
    I = torch.eye(k)
    error = torch.linalg.norm(VTV - I, ord='fro').item()

    print(f"Orthonormality error: {error:.2e}")
    print(f"Status: {'PASS' if error < 1e-6 else 'FAIL'}")

    return error < 1e-6


def test_cost_function_values():
    """Test that cost function values match between implementations."""
    print()
    _print_section("COST FUNCTION VALUE TEST")

    np.random.seed(42)

    test_cases = [
        (10, 5, 2),  # P=10, N=9, K=2
        (20, 15, 5),  # P=20, N=19, K=5
        (50, 40, 10),  # P=50, N=49, K=10
    ]

    all_passed = True

    for P, N, K in test_cases:
        print(f"\nTest: P={P}, N={N}, K={K}")

        # Create random orthonormal V
        V_rand = np.random.randn(N, K)
        V_np, _ = np.linalg.qr(V_rand)
        V_t = torch.as_tensor(V_np, dtype=torch.float32)

        # Create random X
        X_np = np.random.randn(P, N)
        X_t = torch.as_tensor(X_np, dtype=torch.float32)

        # Original cost
        cost_orig, grad_orig = original_cost_fn(V_np, X_np, grad=True)

        # Fast cost
        cost_fast, grad_fast = square_corrcoeff_full_cost_fast(V_t, X_t, grad=True)

        cost_diff = abs(cost_orig - cost_fast) / (abs(cost_orig) + 1e-10)
        grad_diff = np.linalg.norm(grad_orig - grad_fast.numpy()) / (np.linalg.norm(grad_orig) + 1e-10)

        passed = cost_diff < 1e-5 and grad_diff < 1e-4
        all_passed = all_passed and passed

        print(f"  Cost orig: {cost_orig:.6f}, fast: {cost_fast:.6f}, rel diff: {cost_diff:.2e}")
        print(f"  Grad rel diff: {grad_diff:.2e}")
        print(f"  Status: {'PASS' if passed else 'FAIL'}")

    return all_passed


def test_stiefel_optimizer():
    """Test Stiefel manifold optimizer produces similar results."""
    print()
    _print_section("STIEFEL OPTIMIZER TEST")

    np.random.seed(123)

    P, N, K = 15, 14, 3

    # Create X data
    X_np = np.random.randn(P, N)
    X_t = torch.as_tensor(X_np, dtype=torch.float32)

    # Create initial orthonormal V
    V_rand = np.random.randn(N, K)
    V0_np, _ = np.linalg.qr(V_rand)
    V0_t = torch.as_tensor(V0_np, dtype=torch.float32)

    opts = {'max_iter': 1000, 'gtol': 1e-6, 'xtol': 1e-6, 'ftol': 1e-8}

    # Run original optimizer
    from functools import partial
    V_orig, _ = original_OptStiefelGBB(
        V0_np, partial(original_cost_fn, grad=True), X_np, **opts
    )

    # Run fast optimizer
    V_fast = OptStiefelGBB_fast(
        V0_t.clone(), partial(square_corrcoeff_full_cost_fast, grad=True), X_t, **opts
    )

    # Compare final costs
    cost_orig, _ = original_cost_fn(V_orig, X_np, grad=False)
    cost_fast, _ = square_corrcoeff_full_cost_fast(V_fast, X_t, grad=False)

    cost_diff = abs(cost_orig - cost_fast) / (abs(cost_orig) + 1e-10)

    # Check orthonormality
    orth_error = torch.linalg.norm(V_fast.T @ V_fast - torch.eye(K), ord='fro').item()

    passed = cost_diff < 0.1 and orth_error < 1e-6  # Allow some difference due to floating point

    print(f"Original final cost: {cost_orig:.6f}")
    print(f"Fast final cost:     {cost_fast:.6f}")
    print(f"Cost rel diff:       {cost_diff:.2%}")
    print(f"Orthonormality err:  {orth_error:.2e}")
    print(f"Status:              {'PASS' if passed else 'FAIL'}")

    return passed


def test_fun_FA_comparison():
    """Test factor analysis function produces similar results."""
    print()
    _print_section("FACTOR ANALYSIS (fun_FA) TEST")

    np.random.seed(42)

    # Create test data - centers of manifolds
    N = 50  # Ambient dimension
    P = 20  # Number of manifolds
    centers = np.random.randn(N, P)

    maxK = 5
    max_iter = 5000
    n_reps = 3

    print(f"N={N}, P={P}, maxK={maxK}, n_reps={n_reps}")

    # Run original
    (norm_coeff_orig, norm_coeff_vec_orig, Proj_orig, V1_mat_orig,
     res_coeff_orig, res_coeff0_orig), t_orig = _time_fn(
        original_fun_FA, centers, maxK, max_iter, n_reps
    )

    # Run fast
    (norm_coeff_fast, norm_coeff_vec_fast, Proj_fast, V1_mat_fast,
     res_coeff_fast, res_coeff0_fast), t_fast = _time_fn(
        fun_FA_fast, centers, maxK, max_iter, n_reps, device="cpu"
    )

    # Compare results
    res_coeff0_diff = abs(res_coeff0_orig - res_coeff0_fast) / (abs(res_coeff0_orig) + 1e-10)

    # Compare final residual coefficients for each K
    res_passed = True
    print("\nResidual coefficients per K:")
    for k in range(min(len(res_coeff_orig), len(res_coeff_fast))):
        diff = abs(res_coeff_orig[k] - res_coeff_fast[k]) / (abs(res_coeff_orig[k]) + 1e-10)
        print(f"  K={k+1}: orig={res_coeff_orig[k]:.6f}, fast={res_coeff_fast[k]:.6f}, diff={diff:.2%}")
        if diff > 0.2:  # Allow 20% difference due to stochastic optimization
            res_passed = False

    print(f"\nInitial correlation (res_coeff0):")
    print(f"  orig={res_coeff0_orig:.6f}, fast={res_coeff0_fast:.6f}, diff={res_coeff0_diff:.2%}")

    print(f"\nTiming: orig={t_orig:.3f}s, fast={t_fast:.3f}s, speedup={t_orig/t_fast:.1f}x")

    passed = res_coeff0_diff < 1e-5 and res_passed
    print(f"Status: {'PASS' if passed else 'FAIL'}")

    return passed


def test_manifold_analysis_corr_comparison():
    """Compare full manifold_analysis_corr outputs."""
    print()
    _print_section("FULL MANIFOLD CORRELATION ANALYSIS TEST")

    np.random.seed(42)

    # Create test manifolds
    num_manifolds = 5
    N = 30  # Ambient dimension
    P = 50  # Points per manifold
    n_t = 100
    kappa = 0.1

    XtotT = [np.random.randn(N, P) for _ in range(num_manifolds)]

    print(f"num_manifolds={num_manifolds}, N={N}, P={P}, n_t={n_t}, kappa={kappa}")

    # Run original
    print("\nRunning original implementation...")
    (a_orig, R_orig, D_orig, res_coeff0_orig, KK_orig), t_orig = _time_fn(
        original_manifold_corr, XtotT, kappa, n_t, n_reps=3
    )

    # Run fast
    print("Running fast implementation...")
    (a_fast, R_fast, D_fast, res_coeff0_fast, KK_fast), t_fast = _time_fn(
        fast_manifold_corr, XtotT, kappa, n_t, n_reps=3, device="cpu"
    )

    # Compare results
    print("\nCapacity (a_Mfull):")
    a_diffs = []
    for i in range(num_manifolds):
        diff = abs(a_orig[i] - a_fast[i]) / (abs(a_orig[i]) + 1e-10)
        a_diffs.append(diff)
        print(f"  Manifold {i}: orig={a_orig[i]:.4f}, fast={a_fast[i]:.4f}, diff={diff:.2%}")

    print("\nRadius (R_M):")
    R_diffs = []
    for i in range(num_manifolds):
        diff = abs(R_orig[i] - R_fast[i]) / (abs(R_orig[i]) + 1e-10)
        R_diffs.append(diff)
        print(f"  Manifold {i}: orig={R_orig[i]:.4f}, fast={R_fast[i]:.4f}, diff={diff:.2%}")

    print("\nDimension (D_M):")
    D_diffs = []
    for i in range(num_manifolds):
        diff = abs(D_orig[i] - D_fast[i]) / (abs(D_orig[i]) + 1e-10)
        D_diffs.append(diff)
        print(f"  Manifold {i}: orig={D_orig[i]:.4f}, fast={D_fast[i]:.4f}, diff={diff:.2%}")

    print(f"\nres_coeff0: orig={res_coeff0_orig:.6f}, fast={res_coeff0_fast:.6f}")
    print(f"KK: orig={KK_orig}, fast={KK_fast}")

    print(f"\nTiming: orig={t_orig:.3f}s, fast={t_fast:.3f}s, speedup={t_orig/t_fast:.1f}x")

    # Check if results are close enough
    avg_a_diff = np.mean(a_diffs)
    avg_R_diff = np.mean(R_diffs)
    avg_D_diff = np.mean(D_diffs)

    passed = avg_a_diff < TOL and avg_R_diff < TOL and avg_D_diff < TOL

    print(f"\nAvg diffs: a={avg_a_diff:.2%}, R={avg_R_diff:.2%}, D={avg_D_diff:.2%}")
    print(f"Status: {'PASS' if passed else 'FAIL - check tolerances'}")

    return passed


def test_manifold_analysis_corr_with_t_vecs():
    """Test with fixed t_vecs for more deterministic comparison."""
    print()
    _print_section("MANIFOLD CORRELATION ANALYSIS WITH FIXED T_VECS")

    np.random.seed(123)

    num_manifolds = 3
    N = 20
    P = 30
    n_t = 50
    kappa = 0.1

    # Generate fixed inputs
    XtotT = [np.random.randn(N, P) for _ in range(num_manifolds)]

    # For t_vecs, we need the right dimension after QR reduction
    # After normalization and QR, dimension is min(D, m) + 1
    # D is the ambient dimension after preprocessing, m is points per manifold
    t_vecs = [np.random.randn(min(N, P) + 1, n_t) for _ in range(num_manifolds)]

    print(f"num_manifolds={num_manifolds}, N={N}, P={P}, n_t={n_t}")

    (a_orig, R_orig, D_orig, res_coeff0_orig, KK_orig), t_orig = _time_fn(
        original_manifold_corr, XtotT, kappa, n_t, t_vecs=t_vecs, n_reps=2
    )

    (a_fast, R_fast, D_fast, res_coeff0_fast, KK_fast), t_fast = _time_fn(
        fast_manifold_corr, XtotT, kappa, n_t, t_vecs=t_vecs, n_reps=2, device="cpu"
    )

    # Compare
    a_diff = np.mean(np.abs(a_orig - a_fast) / (np.abs(a_orig) + 1e-10))
    R_diff = np.mean(np.abs(R_orig - R_fast) / (np.abs(R_orig) + 1e-10))
    D_diff = np.mean(np.abs(D_orig - D_fast) / (np.abs(D_orig) + 1e-10))

    print(f"Avg capacity diff: {a_diff:.2%}")
    print(f"Avg radius diff:   {R_diff:.2%}")
    print(f"Avg dimension diff: {D_diff:.2%}")
    print(f"Timing: orig={t_orig:.3f}s, fast={t_fast:.3f}s, speedup={t_orig/t_fast:.1f}x")

    passed = a_diff < TOL and R_diff < TOL and D_diff < TOL
    print(f"Status: {'PASS' if passed else 'FAIL'}")

    return passed


def benchmark_performance():
    """Benchmark performance scaling."""
    print()
    _print_section("PERFORMANCE BENCHMARK")

    np.random.seed(42)

    print("\nScaling with number of manifolds (N=30, P=50, n_t=100):")
    print("-" * 50)

    N, P, n_t = 30, 50, 100
    kappa = 0.1

    for num_manifolds in [3, 5, 10]:
        XtotT = [np.random.randn(N, P) for _ in range(num_manifolds)]

        _, t_orig = _time_fn(original_manifold_corr, XtotT, kappa, n_t, n_reps=2)
        _, t_fast = _time_fn(fast_manifold_corr, XtotT, kappa, n_t, n_reps=2, device="cpu")

        speedup = t_orig / t_fast
        print(f"  num_manifolds={num_manifolds}: orig={t_orig:.3f}s, fast={t_fast:.3f}s, speedup={speedup:.1f}x")

    print("\nScaling with ambient dimension (num_manifolds=5, P=50, n_t=100):")
    print("-" * 50)

    num_manifolds, P, n_t = 5, 50, 100

    for N in [20, 30, 50]:
        XtotT = [np.random.randn(N, P) for _ in range(num_manifolds)]

        _, t_orig = _time_fn(original_manifold_corr, XtotT, kappa, n_t, n_reps=2)
        _, t_fast = _time_fn(fast_manifold_corr, XtotT, kappa, n_t, n_reps=2, device="cpu")

        speedup = t_orig / t_fast
        print(f"  N={N}: orig={t_orig:.3f}s, fast={t_fast:.3f}s, speedup={speedup:.1f}x")


def test_edge_cases():
    """Test edge cases and numerical stability."""
    print()
    _print_section("EDGE CASE TESTS")

    np.random.seed(42)

    def _run_edge_case(label, fn):
        """Run a single edge case and surface failures immediately."""
        print(f"\nTest: {label}")
        try:
            return fn()
        except Exception as exc:
            print(f"  Error: {exc} [FAIL]")
            raise

    # Keep cases small to isolate numerical issues quickly.
    def _case_small_manifolds():
        num_manifolds = 2
        N, P, n_t = 20, 30, 50
        kappa = 0.1
        XtotT = [np.random.randn(N, P) for _ in range(num_manifolds)]
        a, R, D, res_coeff0, KK = fast_manifold_corr(XtotT, kappa, n_t, n_reps=2, device="cpu")
        print(f"  Results: a={a.mean():.4f}, R={R.mean():.4f}, D={D.mean():.4f}, KK={KK} [PASS]")
        return a, R, D, res_coeff0, KK

    _run_edge_case("Small number of manifolds", _case_small_manifolds)

    def _case_high_dimensional():
        num_manifolds = 3
        N, P, n_t = 100, 30, 50
        kappa = 0.1
        XtotT = [np.random.randn(N, P) for _ in range(num_manifolds)]
        a, R, D, res_coeff0, KK = fast_manifold_corr(XtotT, kappa, n_t, n_reps=2, device="cpu")
        print(f"  Results: a={a.mean():.4f}, R={R.mean():.4f}, D={D.mean():.4f}, KK={KK} [PASS]")
        return a, R, D, res_coeff0, KK

    _run_edge_case("High dimensional (N=100)", _case_high_dimensional)

    def _case_small_kappa():
        num_manifolds = 3
        N, P, n_t = 30, 40, 50
        kappa = 1e-6
        XtotT = [np.random.randn(N, P) for _ in range(num_manifolds)]
        a, R, D, res_coeff0, KK = fast_manifold_corr(XtotT, kappa, n_t, n_reps=2, device="cpu")
        print(f"  Results: a={a.mean():.4f}, R={R.mean():.4f}, D={D.mean():.4f}, KK={KK} [PASS]")
        return a, R, D, res_coeff0, KK

    _run_edge_case("Very small kappa", _case_small_kappa)

    def _case_large_kappa():
        num_manifolds = 3
        N, P, n_t = 30, 40, 50
        kappa = 1.0
        XtotT = [np.random.randn(N, P) for _ in range(num_manifolds)]
        a, R, D, res_coeff0, KK = fast_manifold_corr(XtotT, kappa, n_t, n_reps=2, device="cpu")
        print(f"  Results: a={a.mean():.4f}, R={R.mean():.4f}, D={D.mean():.4f}, KK={KK} [PASS]")
        return a, R, D, res_coeff0, KK

    _run_edge_case("Large kappa", _case_large_kappa)


def test_gpu_if_available():
    """Test GPU acceleration if available."""
    print()
    _print_section("GPU ACCELERATION TEST")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return True

    np.random.seed(42)

    num_manifolds = 5
    N, P, n_t = 30, 50, 100
    kappa = 0.1

    XtotT = [np.random.randn(N, P) for _ in range(num_manifolds)]

    # Compare CPU and GPU timing on identical inputs.
    # CPU run
    _, t_cpu = _time_fn(fast_manifold_corr, XtotT, kappa, n_t, n_reps=2, device="cpu")

    # GPU run
    _, t_gpu = _time_fn(fast_manifold_corr, XtotT, kappa, n_t, n_reps=2, device="cuda")

    speedup = t_cpu / t_gpu
    print(f"CPU time: {t_cpu:.3f}s")
    print(f"GPU time: {t_gpu:.3f}s")
    print(f"GPU speedup: {speedup:.1f}x")

    return True


if __name__ == "__main__":
    results = []

    results.append(("Gram-Schmidt", test_gram_schmidt()))
    results.append(("Cost Function", test_cost_function_values()))
    results.append(("Stiefel Optimizer", test_stiefel_optimizer()))
    results.append(("Factor Analysis", test_fun_FA_comparison()))
    results.append(("Full Analysis", test_manifold_analysis_corr_comparison()))
    results.append(("Fixed t_vecs", test_manifold_analysis_corr_with_t_vecs()))

    test_edge_cases()
    benchmark_performance()
    test_gpu_if_available()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed
        print(f"  {name}: [{status}]")

    print("-" * 60)
    if all_passed:
        print("ALL CORE TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - check implementation or tolerances")
    print("=" * 60)
