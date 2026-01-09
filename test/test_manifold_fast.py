"""
Test script to verify the batch PGD solver matches the original cvxopt implementation.
Also benchmarks performance differences.
"""

import time

import numpy as np
import torch

from manifold_analysis import (
    _prepare_dual_solver,
    _solve_dual_nesterov,
    each_manifold_analysis_D1_fast as fast_solver,
    manifold_analysis_fast as fast_manifold_analysis,
)
from original.manifold_analysis import (
    each_manifold_analysis_D1 as original_solver,
    manifold_analysis as original_manifold_analysis,
    minimize_vt_sq,
)

TOL = 0.01   # 1% tolerance
ATOL = 1e-3


def _make_manifold_inputs(seed, num_manifolds, d, p, n_t, t_dim=None):
    """Build deterministic manifolds and t vectors for solver comparisons."""
    # Keep RNG and shapes controlled so both solvers see identical inputs.
    rng = np.random.default_rng(seed)
    if t_dim is None:
        t_dim = d + 1
    XtotT = []
    t_vecs = []
    for _ in range(num_manifolds):
        XtotT.append(rng.standard_normal((d, p)))
        t_vecs.append(rng.standard_normal((t_dim, n_t)))
    return XtotT, t_vecs


def _time_solver(solver, sD1, kappa, n_t, t_vec):
    start = time.perf_counter()
    result = solver(sD1, kappa, n_t, t_vec=t_vec.copy())
    elapsed = time.perf_counter() - start
    return result, elapsed


def _print_section(title):
    """Print a consistent section header for test output."""
    line = "=" * 60
    print(line)
    print(title)
    print(line)


def test_accuracy_comparison():
    """Compare outputs of original vs fast implementation."""
    _print_section("ACCURACY TEST: Comparing original cvxopt vs batch PGD")

    np.random.seed(42)

    test_cases = [
        (10, 50, 100, 0.1),
        (20, 100, 200, 0.05),
        (50, 200, 300, 0.2),
        (30, 500, 100, 0.1),
    ]

    all_passed = True

    for i, (D, m, n_t, kappa) in enumerate(test_cases, start=1):
        print(f"\nTest {i}: D={D}, m={m}, n_t={n_t}, kappa={kappa}")
        # Deterministic data for repeatable comparisons.
        sD1 = np.random.randn(D + 1, m)
        sD1[-1, :] = 1.0

        t_vec = np.random.randn(D + 1, n_t)

        (a_orig, R_orig, D_orig), t_orig = _time_solver(
            original_solver, sD1, kappa, n_t, t_vec
        )
        (a_fast, R_fast, D_fast), t_fast = _time_solver(
            fast_solver, sD1, kappa, n_t, t_vec
        )

        # Relative differences for pass/fail.
        a_diff = abs(a_orig - a_fast) / (abs(a_orig) + 1e-10)
        R_diff = abs(R_orig - R_fast) / (abs(R_orig) + 1e-10)
        D_diff = abs(D_orig - D_fast) / (abs(D_orig) + 1e-10)

        passed = a_diff < TOL and R_diff < TOL and D_diff < TOL
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        print(f"  Original: a={a_orig:.6f}, R={R_orig:.6f}, D={D_orig:.6f} ({t_orig:.3f}s)")
        print(f"  Fast:     a={a_fast:.6f}, R={R_fast:.6f}, D={D_fast:.6f} ({t_fast:.3f}s)")
        print(f"  Rel diff: a={a_diff:.2%}, R={R_diff:.2%}, D={D_diff:.2%}")
        print(f"  Speedup:  {t_orig/t_fast:.1f}x")
        print(f"  Status:   [{status}]")

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - check tolerances or implementation")
    print("=" * 60)

    return all_passed


def test_manifold_analysis_fast_matches_numpy_small():
    """Compare full manifold_analysis outputs without QR reduction."""
    # Use CPU and fixed inputs to keep the comparison deterministic.
    XtotT, t_vecs = _make_manifold_inputs(seed=11, num_manifolds=3, d=6, p=12, n_t=25)
    kappa = 0.05

    a_np, R_np, D_np = original_manifold_analysis(XtotT, kappa, 25, t_vecs=t_vecs)
    a_fast, R_fast, D_fast = fast_manifold_analysis(
        XtotT, kappa, 25, t_vecs=t_vecs, device="cpu"
    )

    np.testing.assert_allclose(a_fast, a_np, rtol=TOL, atol=ATOL)
    np.testing.assert_allclose(R_fast, R_np, rtol=TOL, atol=ATOL)
    np.testing.assert_allclose(D_fast, D_np, rtol=TOL, atol=ATOL)


def test_manifold_analysis_fast_matches_numpy_with_projection():
    """Compare outputs when manifolds trigger QR reduction (D > m)."""
    # For D > m, the reduced dimension is m, so t vectors must be m + 1.
    d, p, n_t = 20, 5, 15
    t_dim = p + 1
    XtotT, t_vecs = _make_manifold_inputs(
        seed=29, num_manifolds=2, d=d, p=p, n_t=n_t, t_dim=t_dim
    )
    kappa = 0.1

    a_np, R_np, D_np = original_manifold_analysis(XtotT, kappa, n_t, t_vecs=t_vecs)
    a_fast, R_fast, D_fast = fast_manifold_analysis(
        XtotT, kappa, n_t, t_vecs=t_vecs, device="cpu"
    )

    np.testing.assert_allclose(a_fast, a_np, rtol=TOL, atol=ATOL)
    np.testing.assert_allclose(R_fast, R_np, rtol=TOL, atol=ATOL)
    np.testing.assert_allclose(D_fast, D_np, rtol=TOL, atol=ATOL)


def test_minimize_vt_sq_comparison():
    """Test individual QP solutions match."""
    print()
    _print_section("QP SOLUTION TEST: Comparing individual optimizations")

    np.random.seed(123)

    D1 = 15
    m = 100
    n_test = 20
    kappa = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Keep input fixed across solvers.
    sD1 = np.random.randn(D1, m).astype(np.float64)
    sD1[-1, :] = 1.0
    sD1_t = torch.as_tensor(sD1, device=device, dtype=torch.float32)

    # Pre-compute Gram matrix and Lipschitz constant once.
    STS, L = _prepare_dual_solver(sD1_t)
    lr = 1.0 / L

    errors = []

    for _ in range(n_test):
        t = np.random.randn(D1, 1)

        v_orig, _, _, _, _ = minimize_vt_sq(t, sD1, kappa=kappa)

        t_batch_t = torch.as_tensor(t, device=device, dtype=torch.float32)
        ST_T = sD1_t.T @ t_batch_t
        b = ST_T + kappa

        alpha_init = torch.zeros((m, 1), device=device, dtype=torch.float32)

        alpha_fast = _solve_dual_nesterov(
            STS, alpha_init, b, lr, max_iter=500, tol=1e-8
        )
        v_fast = t_batch_t - sD1_t @ alpha_fast

        v_orig_t = torch.as_tensor(v_orig, device=device, dtype=torch.float32)
        v_error = torch.linalg.norm(v_orig_t.flatten() - v_fast.flatten()).item()
        errors.append(v_error)

    mean_error = np.mean(errors)
    max_error = np.max(errors)

    print(f"Mean V error: {mean_error:.2e}")
    print(f"Max V error:  {max_error:.2e}")
    print(f"Status: {'PASS' if max_error < 1e-4 else 'FAIL'}")


def benchmark_performance():
    """Benchmark performance scaling."""
    # Measure scaling across n_t and m for both solvers.
    print()
    _print_section("PERFORMANCE BENCHMARK")

    np.random.seed(42)

    print("\nScaling with n_t (D=30, m=200):")
    print("-" * 50)
    D, m = 30, 200
    sD1 = np.random.randn(D + 1, m)
    sD1[-1, :] = 1.0
    kappa = 0.1

    for n_t in [50, 100, 200, 500, 1000]:
        t_vec = np.random.randn(D + 1, n_t)

        _, t_orig = _time_solver(original_solver, sD1, kappa, n_t, t_vec)
        _, t_fast = _time_solver(fast_solver, sD1, kappa, n_t, t_vec)

        speedup = t_orig / t_fast
        print(f"  n_t={n_t:4d}: orig={t_orig:.3f}s, fast={t_fast:.3f}s, speedup={speedup:.1f}x")

    print("\nScaling with m (D=30, n_t=200):")
    print("-" * 50)
    D, n_t = 30, 200
    kappa = 0.1

    for m in [50, 100, 200, 500, 1000]:
        sD1 = np.random.randn(D + 1, m)
        sD1[-1, :] = 1.0
        t_vec = np.random.randn(D + 1, n_t)

        _, t_orig = _time_solver(original_solver, sD1, kappa, n_t, t_vec)
        _, t_fast = _time_solver(fast_solver, sD1, kappa, n_t, t_vec)

        speedup = t_orig / t_fast
        print(f"  m={m:4d}: orig={t_orig:.3f}s, fast={t_fast:.3f}s, speedup={speedup:.1f}x")


def test_full_run_average_timing(full_runs=10):
    """Compare average runtime over full solver runs."""
    print()
    _print_section(f"FULL RUN AVERAGE ({full_runs} runs)")

    np.random.seed(7)

    D, m, n_t = 30, 200, 200
    kappa = 0.1
    sD1 = np.random.randn(D + 1, m)
    sD1[-1, :] = 1.0
    t_vec = np.random.randn(D + 1, n_t)

    # Reuse the same data for consistent timing.
    orig_times = []
    fast_times = []

    for _ in range(full_runs):
        _, t_orig = _time_solver(original_solver, sD1, kappa, n_t, t_vec)
        _, t_fast = _time_solver(fast_solver, sD1, kappa, n_t, t_vec)
        orig_times.append(t_orig)
        fast_times.append(t_fast)

    avg_orig = float(np.mean(orig_times))
    avg_fast = float(np.mean(fast_times))
    speedup = avg_orig / avg_fast if avg_fast > 0 else float("inf")

    print(f"Avg orig={avg_orig:.3f}s, avg fast={avg_fast:.3f}s, speedup={speedup:.1f}x")


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

    # Small manifold, large t vectors to emphasize interior points.
    def _case_all_interior():
        D, m, n_t = 10, 50, 100
        sD1 = np.random.randn(D + 1, m) * 0.01
        sD1[-1, :] = 1.0
        t_vec = np.random.randn(D + 1, n_t) * 10
        kappa = 0.001
        a, R, D_M = fast_solver(sD1, kappa, n_t, t_vec=t_vec)
        print(f"  a={a:.4f}, R={R:.4f}, D={D_M:.4f} [PASS]")
        return a, R, D_M

    _run_edge_case("All interior points", _case_all_interior)

    # Small kappa to check stability near the margin.
    def _case_small_kappa():
        D, m, n_t = 10, 50, 100
        sD1 = np.random.randn(D + 1, m)
        sD1[-1, :] = 1.0
        t_vec = np.random.randn(D + 1, n_t)
        kappa = 1e-10
        a, R, D_M = fast_solver(sD1, kappa, n_t, t_vec=t_vec)
        print(f"  a={a:.4f}, R={R:.4f}, D={D_M:.4f} [PASS]")
        return a, R, D_M

    _run_edge_case("Very small kappa", _case_small_kappa)

    def _case_large_kappa():
        D, m, n_t = 10, 50, 100
        sD1 = np.random.randn(D + 1, m)
        sD1[-1, :] = 1.0
        t_vec = np.random.randn(D + 1, n_t)
        kappa = 1.0
        a, R, D_M = fast_solver(sD1, kappa, n_t, t_vec=t_vec)
        print(f"  a={a:.4f}, R={R:.4f}, D={D_M:.4f} [PASS]")
        return a, R, D_M

    _run_edge_case("Large kappa", _case_large_kappa)

    def _case_high_dimensional():
        D, m, n_t = 100, 50, 100
        sD1 = np.random.randn(D + 1, m)
        sD1[-1, :] = 1.0
        t_vec = np.random.randn(D + 1, n_t)
        kappa = 0.1
        a, R, D_M = fast_solver(sD1, kappa, n_t, t_vec=t_vec)
        print(f"  a={a:.4f}, R={R:.4f}, D={D_M:.4f} [PASS]")
        return a, R, D_M

    _run_edge_case("High dimensional (D=100)", _case_high_dimensional)


if __name__ == "__main__":
    test_accuracy_comparison()
    test_minimize_vt_sq_comparison()
    test_edge_cases()
    test_manifold_analysis_fast_matches_numpy_small()
    test_manifold_analysis_fast_matches_numpy_with_projection()
    benchmark_performance()
    test_full_run_average_timing(full_runs=20)
