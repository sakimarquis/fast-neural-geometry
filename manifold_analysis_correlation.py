# Fast manifold correlation analysis in PyTorch.
# Batch solver with vectorized tensor operations.

import math
import torch
from functools import partial

# Import fast solver components from the non-correlation version.
from manifold_analysis import (
    _resolve_device,
    _to_torch,
    each_manifold_analysis_D1_fast,
)


@torch.no_grad()
def manifold_analysis_corr_fast(XtotT, kappa, n_t, t_vecs=None, n_reps=10,
                                device=None, **pgd_kwargs):
    """Run correlation analysis on multiple manifolds with the fast solver.

    Args:
        XtotT: Sequence of 2D arrays of shape (N, P_i) where N is the dimensionality
                of the space, and P_i is the number of sampled points for the i_th manifold.
        kappa: Margin size to use in the analysis (scalar, kappa > 0)
        n_t: Number of gaussian vectors to sample per manifold
        t_vecs: Optional sequence of 2D arrays of shape (Dm_i, n_t) where Dm_i is the reduced
                dimensionality of the i_th manifold. Contains the gaussian vectors to be used in
                analysis.  If not supplied, they will be randomly sampled for each manifold.
        n_reps: Number of repetitions for finding stable solution in factor analysis (default 10)
        device: Optional torch device (string or torch.device). Defaults to CUDA if available.
        **pgd_kwargs: Additional arguments passed to each_manifold_analysis_D1_fast
                      (max_iter, tol, use_nesterov)

    Returns:
        a_Mfull_vec: 1D numpy array containing the capacity calculated from each manifold
        R_M_vec: 1D numpy array containing the calculated anchor radius of each manifold
        D_M_vec: 1D numpy array containing the calculated anchor dimension of each manifold.
        res_coeff0: Residual correlation (scalar)
        KK: Dimensionality of low rank structure (int)
    """
    device = _resolve_device(device)
    num_manifolds = len(XtotT)

    # Convert inputs to torch tensors.
    Xtot_tensors = [_to_torch(m, device) for m in XtotT]

    # Compute a global mean across all samples.
    Xori = torch.cat(Xtot_tensors, dim=1)
    X_origin = torch.mean(Xori, dim=1, keepdim=True)

    # Center each manifold around the global mean.
    Xtot0 = [Xtot_tensors[i] - X_origin for i in range(num_manifolds)]

    # Compute per-manifold centers.
    centers = torch.stack([torch.mean(Xtot_tensors[i], dim=1) for i in range(num_manifolds)], dim=1)
    center_mean = torch.mean(centers, dim=1, keepdim=True)

    # Center correlation analysis via SVD.
    U, S, Vh = torch.linalg.svd(centers - center_mean, full_matrices=False)

    # Choose a K threshold from cumulative variance.
    total = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
    maxK = int(torch.sum(total < 0.95).item()) + 11
    # Cap K to the available basis dimension.
    P = centers.shape[1]
    maxK = min(maxK, P - 1)
    if maxK < 1:
        raise ValueError("Need at least two manifold centers for correlation analysis.")

    # Fit the low-rank structure with factor analysis.
    norm_coeff, norm_coeff_vec, Proj, V1_mat, res_coeff, res_coeff0 = fun_FA_fast(
        centers, maxK, max_iter=20000, n_repeats=n_reps, device=device
    )

    # Select the best K by minimum residual.
    res_coeff_tensor = torch.tensor(res_coeff, device=device)
    KK = int(torch.argmin(res_coeff_tensor).item()) + 1

    # Project into the learned low-rank structure.
    V11 = Proj @ V1_mat[KK - 1]

    XtotInput = []
    for i in range(num_manifolds):
        Xr = Xtot0[i]
        # Project manifold data into the center-null space.
        Xr_ns = Xr - V11 @ (V11.T @ Xr)
        # Compute mean in the center-null space.
        Xr0_ns = torch.mean(Xr_ns, dim=1)
        # Normalize by the mean norm.
        Xr0_ns_norm = torch.linalg.norm(Xr0_ns)
        # Center and normalize the data.
        Xrr_ns = (Xr_ns - Xr0_ns.unsqueeze(1)) / Xr0_ns_norm
        XtotInput.append(Xrr_ns)

    a_Mfull_vec = torch.zeros(num_manifolds, device=device, dtype=torch.float32)
    R_M_vec = torch.zeros(num_manifolds, device=device, dtype=torch.float32)
    D_M_vec = torch.zeros(num_manifolds, device=device, dtype=torch.float32)

    # Analyze each manifold.
    for i in range(num_manifolds):
        S_r = XtotInput[i]
        D, m = S_r.shape

        # Project onto a smaller subspace if needed.
        if D > m:
            Q, _ = torch.linalg.qr(S_r, mode="reduced")
            S_r = Q.T @ S_r
            D, m = S_r.shape

        # Append the center dimension.
        sD1 = torch.cat([S_r, torch.ones((1, m), device=device, dtype=S_r.dtype)], dim=0)

        # Run the analysis on the i_th manifold.
        if t_vecs is not None:
            t_vec = _to_torch(t_vecs[i], device)
            a_Mfull, R_M, D_M = each_manifold_analysis_D1_fast(
                sD1, kappa, n_t, t_vec=t_vec, device=device, **pgd_kwargs
            )
        else:
            a_Mfull, R_M, D_M = each_manifold_analysis_D1_fast(
                sD1, kappa, n_t, device=device, **pgd_kwargs
            )

        a_Mfull_vec[i] = a_Mfull
        R_M_vec[i] = R_M
        D_M_vec[i] = D_M

    return (
        a_Mfull_vec.cpu().numpy(),
        R_M_vec.cpu().numpy(),
        D_M_vec.cpu().numpy(),
        res_coeff0,
        KK
    )


def fun_FA_fast(centers, maxK, max_iter, n_repeats, s_all=None, verbose=False, device=None):
    """Extract low-rank structure from manifold centers.

    Args:
        centers: 2D tensor of shape (N, P) where N is the ambient dimension and P is the number of centers
        maxK: Maximum rank to consider
        max_iter: Maximum number of iterations for the solver
        n_repeats: Number of repetitions to find the most stable solution at each iteration of K
        s_all: (Optional) iterable containing (P, 1) random normal vectors
        verbose: Whether to print progress
        device: Torch device

    Returns:
        norm_coeff: Ratio of center norms before and after optimization
        norm_coeff_vec: Mean ratio of center norms before and after optimization
        Proj: P-1 basis vectors
        V1_mat: Solution for each value of K
        res_coeff: Cost function after optimization for each K
        res_coeff0: Correlation before optimization
    """
    device = _resolve_device(device)
    centers = _to_torch(centers, device)
    N, P = centers.shape
    # Keep K within the P-1 dimensional basis.
    maxK = min(maxK, P - 1)
    if maxK < 1:
        raise ValueError("Need at least two centers to extract low-rank structure.")

    # Solver options.
    opts = {
        'max_iter': max_iter,
        'gtol': 1e-6,
        'xtol': 1e-6,
        'ftol': 1e-8
    }

    # Subtract the global mean.
    mean = torch.mean(centers.T, dim=0, keepdim=True)
    Xb = centers.T - mean
    xbnorm = torch.sqrt(torch.sum(Xb ** 2, dim=1, keepdim=True))

    # Build a P-1 basis via QR.
    q, r = torch.linalg.qr(Xb.T, mode='reduced')
    X = Xb @ q[:, :P - 1]

    # Store the data before extracting low-rank structure.
    X0 = X.clone()
    xnorm = torch.sqrt(torch.sum(X0 ** 2, dim=1, keepdim=True))

    # Compute the initial correlations.
    C0 = (X0 @ X0.T) / (xnorm @ xnorm.T)
    res_coeff0 = ((torch.sum(torch.abs(C0)) - P) / (P * (P - 1))).item()

    # Storage for results.
    V1_mat = []
    norm_coeff = []
    norm_coeff_vec = []
    res_coeff = []

    # Solve for ranks 1..maxK.
    V1 = None
    for i in range(1, maxK + 1):
        best_stability = -float("inf")
        best_V1 = None

        for j in range(1, n_repeats + 1):
            # Sample a random normal vector unless one is supplied.
            if s_all is not None and len(s_all) >= i * j:
                s = _to_torch(s_all[i * j - 1], device)
            else:
                s = torch.randn(P, 1, device=device, dtype=X.dtype)

            # Build the initial subspace.
            sX = s.T @ X
            if V1 is None:
                V0 = sX
            else:
                V0 = torch.cat([sX, V1.T], dim=0)
            V0, _ = torch.linalg.qr(V0.T, mode='reduced')

            # Optimize the subspace on the Stiefel manifold.
            V1tmp = OptStiefelGBB_fast(
                V0, partial(square_corrcoeff_full_cost_fast, grad=True), X, **opts
            )

            # Verify orthogonality.
            orth_error = torch.linalg.norm(V1tmp.T @ V1tmp - torch.eye(i, device=device), ord='fro')
            assert orth_error < 1e-6, f"Orthogonality violated: {orth_error}"

            # Remove the low-rank component.
            X0_tmp = X - (X @ V1tmp) @ V1tmp.T

            # Measure solution stability.
            # Guard the stability ratio against zero norms.
            denom = torch.clamp_min(torch.sqrt(torch.sum(X ** 2, dim=1)), 1e-12)
            stability = torch.min(torch.sqrt(torch.sum(X0_tmp ** 2, dim=1)) / denom).item()

            # Keep the most stable solution.
            if not math.isfinite(stability):
                if best_V1 is None and torch.isfinite(V1tmp).all().item():
                    best_V1 = V1tmp.clone()
                continue

            if stability > best_stability or best_V1 is None:
                best_stability = stability
                best_V1 = V1tmp.clone()

            if n_repeats > 1 and verbose:
                cost_after, _ = square_corrcoeff_full_cost_fast(V1tmp, X, grad=False)
                print(f"{j} cost={cost_after:.6f} stability={stability:.6f}")

        # Use the best solution.
        if best_V1 is None:
            raise ValueError(f"No valid Stiefel solution found for rank {i}.")
        V1 = best_V1

        # Remove the chosen low-rank component.
        XV1 = X @ V1
        X0 = X - XV1 @ V1.T

        # Compute the normalized cost.
        xnorm = torch.sqrt(torch.sum(X0 ** 2, dim=1, keepdim=True))
        C0 = (X0 @ X0.T) / (xnorm @ xnorm.T)
        current_cost = ((torch.sum(torch.abs(C0)) - P) / (P * (P - 1))).item()

        if verbose:
            print(f'K={i} mean={current_cost:.6f}')

        # Store the results.
        V1_mat.append(V1.clone())
        norm_coeff.append((xnorm / xbnorm).squeeze(1).cpu().numpy())
        norm_coeff_vec.append(torch.mean(xnorm / xbnorm).item())
        res_coeff.append(current_cost)

        # Stop after 3 consecutive cost increases.
        if (i > 4 and
                res_coeff[i - 1] > res_coeff[i - 2] and
                res_coeff[i - 2] > res_coeff[i - 3] and
                res_coeff[i - 3] > res_coeff[i - 4]):
            if verbose:
                print("Optimal K0 found")
            break

    return norm_coeff, norm_coeff_vec, q[:, :P - 1], V1_mat, res_coeff, res_coeff0


def OptStiefelGBB_fast(X, objective_function, A, **kwargs):
    """Find X that minimizes F(X) with X.T @ X = I_k.

    PyTorch port of the Wen-Yin method for optimization on the Stiefel manifold.

    Args:
        X: Initial 2D tensor of shape (n, k) such that X.T @ X = I_k
        objective_function: Objective function F(X, A) to minimize, returns (cost, gradient)
        A: Additional parameters for the objective function

    Keyword Args:
        max_iter: Maximum number of iterations
        xtol: Stop control for ||X_k - X_{k-1}||
        gtol: Stop control for the projected gradient
        ftol: Stop control for ||F_k - F_{k-1}||/(1 + |F_{k-1}|)

    Returns:
        X: Optimized tensor
    """
    # Default parameters.
    params = {
        'max_iter': 1000,
        'xtol': 1e-6,
        'gtol': 1e-6,
        'ftol': 1e-12,
        'rho': 1e-4,
        'eta': 0.2,
        'gamma': 0.85,
        'tau': 1e-3,
        'nt': 5,
        'projG': 1,
    }
    params.update(kwargs)

    device = X.device
    dtype = X.dtype
    n, k = X.shape

    eye_n = torch.eye(n, device=device, dtype=dtype)
    jitter = 1e-6 if dtype == torch.float32 else 1e-12

    if k < n / 2:
        invH = False
        eye2k = torch.eye(2 * k, device=device, dtype=dtype)
    else:
        invH = True
        eye2k = None

    # Compute the initial value and gradient.
    F, G = objective_function(X, A)

    GX = G.T @ X

    if invH:
        GXT = G @ X.T
        H = 0.5 * (GXT - GXT.T)
        RX = H @ X
    elif params['projG'] == 1:
        U = torch.cat([G, X], dim=1)
        V = torch.cat([X, -G], dim=1)
        VU = V.T @ U
        VX = V.T @ X
    else:  # projG == 2
        XpG = X.T @ G
        GB = G - 0.5 * X @ XpG
        U = torch.cat([GB, X], dim=1)
        V = torch.cat([X, -GB], dim=1)
        VU = V.T @ U
        VX = V.T @ X

    dtX = G - X @ GX
    nrmG = torch.linalg.norm(dtX, ord='fro')

    Q = 1.0
    Cval = F
    tau = params['tau']
    crit = torch.zeros((1, 3), device=device, dtype=dtype)

    # Main iteration loop.
    for itr in range(1, params['max_iter'] + 1):
        XP, FP, GP, dtXP = X.clone(), F, G.clone(), dtX.clone()
        nls = 1
        deriv = params['rho'] * nrmG ** 2

        while True:
            # Compute the next iterate. Solve the linear system with diagonal jitter for stability.
            if invH:
                system = eye_n + tau * H
                system = system + jitter * eye_n
                try:
                    X = torch.linalg.solve(system, XP - tau * RX)
                except RuntimeError: # Backtrack if the system is singular.
                    if nls >= 5:
                        X, F, G = XP, FP, GP
                        break
                    tau = params['eta'] * tau
                    nls += 1
                    continue
            else:
                system = eye2k + 0.5 * tau * VU
                system = system + jitter * eye2k
                try:
                    aa = torch.linalg.solve(system, VX)
                except RuntimeError:  # Backtrack if the system is singular.
                    if nls >= 5:
                        X, F, G = XP, FP, GP
                        break
                    tau = params['eta'] * tau
                    nls += 1
                    continue
                X = XP - tau * U @ aa

            # Evaluate objective and gradient.
            F, G = objective_function(X, A)
            # Backtrack if the objective becomes non-finite.
            if not math.isfinite(F) or not torch.isfinite(G).all():
                if nls >= 5:
                    X, F, G = XP, FP, GP
                    break
                tau = params['eta'] * tau
                nls += 1
                continue

            # Exit if the line search condition is met.
            if F <= (Cval - tau * deriv) or nls >= 5:
                break

            tau = params['eta'] * tau
            nls += 1

        # Compute post-step quantities.
        GX = G.T @ X
        if invH:
            GXT = G @ X.T
            H = 0.5 * (GXT - GXT.T)
            RX = H @ X
        elif params['projG'] == 1:
            U = torch.cat([G, X], dim=1)
            V = torch.cat([X, -G], dim=1)
            VU = V.T @ U
            VX = V.T @ X
        else:  # projG == 2
            XpG = X.T @ G
            GB = G - 0.5 * X @ XpG
            U = torch.cat([GB, X], dim=1)
            V = torch.cat([X, -GB], dim=1)
            VU = V.T @ U
            VX = V.T @ X

        dtX = G - X @ GX
        nrmG = torch.linalg.norm(dtX, ord='fro')

        S = X - XP
        XDiff = torch.linalg.norm(S, ord='fro') / (n ** 0.5)
        tau = params['tau']
        FDiff = abs(FP - F) / (abs(FP) + 1)

        # Update step size using the Barzilai-Borwein rule.
        Y = dtX - dtXP
        SY = torch.abs(torch.sum(S * Y))
        YY = torch.sum(Y * Y)
        eps = torch.finfo(dtype).eps

        # Guard the BB step against tiny denominators.
        SY_val = SY.item()
        YY_val = YY.item()
        if math.isfinite(SY_val) and math.isfinite(YY_val) and SY_val > eps and YY_val > eps:
            if itr % 2 == 0:
                tau = (torch.sum(S * S) / SY).item()
            else:
                tau = (SY / YY).item()
        else:
            tau = params['tau']

        if not math.isfinite(tau):
            tau = params['tau']

        tau = max(min(tau, 1e20), 1e-20)

        diff_arr = torch.tensor([[nrmG, XDiff, FDiff]], device=device, dtype=dtype)
        crit = torch.cat([crit, diff_arr], dim=0)
        mcrit = torch.mean(crit[max(1, itr - params['nt'] + 1):itr + 1, :], dim=0)

        # Check for convergence.
        cond1 = (XDiff < params['xtol']) and (FDiff < params['ftol'])
        cond2 = nrmG < params['gtol']
        cond3 = (mcrit[1] < 10 * params['xtol']) and (mcrit[2] < 10 * params['ftol'])

        if cond1 or cond2 or cond3:
            if itr <= 2:
                params['ftol'] *= 0.1
                params['xtol'] *= 0.1
                params['gtol'] *= 0.1
            else:
                break

        Qp = Q
        Q = params['gamma'] * Qp + 1
        Cval = (params['gamma'] * Qp * Cval + F) / Q

    # Enforce orthogonality via modified Gram-Schmidt if needed.
    feasi = torch.linalg.norm(X.T @ X - torch.eye(k, device=device, dtype=dtype), ord='fro')
    if feasi > 1e-13:
        X = MGramSchmidt_fast(X)

    return X


def square_corrcoeff_full_cost_fast(V, X, grad=True):
    """Cost function for correlation analysis using PyTorch.

    Measures the square difference in correlation coefficients after transforming
    to an orthonormal basis given by V.

    Args:
        V: 2D tensor of shape (N, K) with V.T @ V = I
        X: 2D tensor of shape (P, N) containing centers of P manifolds in an N=P-1 dimensional
            orthonormal basis
        grad: Whether to compute gradient

    Returns:
        cost: Scalar cost value
        gradient: Gradient tensor of shape (N, K), or None if grad=False
    """
    P, N = X.shape
    N_v, K = V.shape
    assert N_v == N

    # Calculate the cost.
    C = X @ X.T
    c = X @ V
    c0 = torch.diagonal(C).unsqueeze(1) - torch.sum(c ** 2, dim=1, keepdim=True)

    # Avoid division by zero.
    c0_safe = torch.clamp(c0, min=1e-12)
    c0c0T = c0_safe @ c0_safe.T

    Fmn = (C - c @ c.T) ** 2 / c0c0T
    cost = torch.sum(Fmn) / 2

    if not grad:
        return cost.item(), None

    # Calculate the gradient using torch operations.
    # Reshape for broadcasting.
    X1 = X.unsqueeze(0)  # (1, P, N)
    X2 = X.unsqueeze(1)  # (P, 1, N)
    C1 = c.unsqueeze(1)  # (P, 1, K)
    C2 = c.unsqueeze(0)  # (1, P, K)

    # Compute partial derivatives.
    ccT = c @ c.T
    PF1 = (C - ccT) / c0c0T  # (P, P)
    PF2 = ((C - ccT) ** 2) / (c0c0T ** 2)  # (P, P)

    # Gradient computation across pairs.
    # Term 1: -PF1 * C1 * X1.
    Gmni = -PF1.unsqueeze(-1).unsqueeze(-1) * C1.unsqueeze(-2) * X1.unsqueeze(-1)
    # Term 2: -PF1 * C2 * X2.
    Gmni = Gmni - PF1.unsqueeze(-1).unsqueeze(-1) * C2.unsqueeze(-2) * X2.unsqueeze(-1)
    # Term 3: PF2 * c0 * C2 * X1.
    c0_exp = c0_safe.unsqueeze(1)  # (P, 1, 1)
    Gmni = Gmni + PF2.unsqueeze(-1).unsqueeze(-1) * c0_exp.unsqueeze(-1) * C2.unsqueeze(-2) * X1.unsqueeze(-1)
    # Term 4: PF2 * c0.T * C1 * X2.
    c0T_exp = c0_safe.T.unsqueeze(-1)
    Gmni = Gmni + PF2.unsqueeze(-1).unsqueeze(-1) * c0T_exp.unsqueeze(-1) * C1.unsqueeze(-2) * X2.unsqueeze(-1)
    gradient = torch.sum(Gmni, dim=(0, 1))

    return cost.item(), gradient


def MGramSchmidt_fast(V):
    """Modified Gram-Schmidt for orthogonalization.

    Args:
        V: 2D tensor of shape (n, k) containing k vectors of dimension n

    Returns:
        V_out: 2D tensor of shape (n, k) containing k orthogonal unit vectors
    """
    n, k = V.shape
    V_out = V.clone()

    for i in range(k):
        for j in range(i):
            # Subtract the projection onto earlier vectors.
            proj_coeff = torch.dot(V_out[:, j], V_out[:, i]) / torch.dot(V_out[:, j], V_out[:, j])
            V_out[:, i] = V_out[:, i] - proj_coeff * V_out[:, j]
        # Normalize.
        V_out[:, i] = V_out[:, i] / torch.linalg.norm(V_out[:, i])

    return V_out
