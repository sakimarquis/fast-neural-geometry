# Fast manifold analysis using batch PGD in PyTorch.
# Vectorized dual optimization with on-device preprocessing.

import torch


def _resolve_device(device):
    """Resolve device, defaulting to CUDA when available."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but not available.")
    return resolved


def _to_torch(array, device):
    """Convert inputs to torch.float32 on the target device."""
    if torch.is_tensor(array):
        return array.to(device=device, dtype=torch.float32)
    return torch.as_tensor(array, device=device, dtype=torch.float32)


@torch.no_grad()
def _prepare_dual_solver(S_t, n_power_iter=20):
    """
    Compute S^T @ S and estimate its spectral norm for step sizing.

    Uses power iteration with a safety buffer to stabilize PGD steps.

    Args:
        S_t: Tensor of shape (D+1, m) already on target device
        n_power_iter: Number of power iterations for spectral norm estimate

    Returns:
        STS: Gram matrix S^T @ S of shape (m, m)
        L: Estimated spectral norm with 10% buffer for numerical stability
    """
    STS = S_t.T @ S_t
    m = STS.shape[0]

    # Power iteration to estimate the top eigenvalue.
    v = torch.randn(m, device=S_t.device, dtype=S_t.dtype)
    v = v / (torch.linalg.norm(v) + 1e-12)

    for _ in range(n_power_iter):
        u = STS @ v
        v = u / (torch.linalg.norm(u) + 1e-12)

    L = torch.linalg.norm(STS @ v).item() * 1.1
    return STS, L


@torch.no_grad()
def manifold_analysis_fast(XtotT, kappa, n_t, t_vecs=None, device=None, **pgd_kwargs):
    """Carry out the analysis on multiple manifolds using fast batch PGD solver.

    Args:
        XtotT: Sequence of 2D arrays of shape (N, P_i) where N is the dimensionality
                of the space, and P_i is the number of sampled points for the i_th manifold.
        kappa: Margin size to use in the analysis (scalar, kappa > 0)
        n_t: Number of gaussian vectors to sample per manifold
        t_vecs: Optional sequence of 2D arrays of shape (Dm_i, n_t) where Dm_i is the reduced
                dimensionality of the i_th manifold. Contains the gaussian vectors to be used in
                analysis.  If not supplied, they will be randomly sampled for each manifold.
        device: Optional torch device (string or torch.device). Defaults to CUDA if available.
        **pgd_kwargs: Additional arguments passed to each_manifold_analysis_D1_fast
                      (max_iter, tol, use_nesterov)

    Returns:
        a_Mfull_vec: 1D numpy array containing the capacity calculated from each manifold
        R_M_vec: 1D numpy array containing the calculated anchor radius of each manifold
        D_M_vec: 1D numpy array containing the calculated anchor dimension of each manifold.
    """
    num_manifolds = len(XtotT)
    device = _resolve_device(device)

    # Convert inputs and compute global mean.
    Xtot_tensors = [_to_torch(m, device) for m in XtotT]
    X_origin = torch.mean(torch.cat(Xtot_tensors, dim=1), dim=1, keepdim=True)

    a_Mfull_vec = torch.zeros(num_manifolds, device=device, dtype=torch.float32)
    R_M_vec = torch.zeros(num_manifolds, device=device, dtype=torch.float32)
    D_M_vec = torch.zeros(num_manifolds, device=device, dtype=torch.float32)

    for i, manifold_t in enumerate(Xtot_tensors):
        # Center and normalize each manifold on-device.
        centered = manifold_t - X_origin
        mean = torch.mean(centered, dim=1, keepdim=True)
        S_r = (centered - mean) / torch.linalg.norm(mean)
        D, m = S_r.shape

        # Project the data onto a smaller subspace.
        if D > m:
            Q, _ = torch.linalg.qr(S_r, mode="reduced")
            S_r = Q.T @ S_r
            D, m = S_r.shape

        # Add the center dimension.
        sD1 = torch.cat([S_r, torch.ones((1, m), device=device, dtype=S_r.dtype)], dim=0)

        # Carry out the analysis on the i_th manifold.
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

    return a_Mfull_vec.cpu().numpy(), R_M_vec.cpu().numpy(), D_M_vec.cpu().numpy()


@torch.no_grad()
def each_manifold_analysis_D1_fast(sD1, kappa, n_t, eps=1e-8, t_vec=None, max_iter=500, tol=1e-7,
                                   use_nesterov=True, device=None):
    """Batch dual PGD solver for manifold analysis.
    Solves all n_t optimization problems simultaneously.

    Primal for each t vector:
        min 0.5 ||V - T||^2
        s.t. S^T V <= -kappa  (for all manifold points S_j)

    Dual with alpha >= 0:
        V = T - S @ alpha
        min 0.5 * alpha^T (S^T S) alpha - alpha^T (S^T T + kappa)

    Args:
        sD1: 2D array of shape (D+1, m) where m is number of manifold points
        kappa: Margin size (scalar)
        n_t: Number of randomly sampled vectors to use
        eps: Minimal distance threshold (default 1e-8)
        t_vec: Optional 2D array of shape (D+1, n_t) containing sampled t vectors
        max_iter: Maximum PGD iterations (default 500)
        tol: Convergence tolerance (default 1e-7)
        use_nesterov: Whether to use Nesterov acceleration (default True)
        device: Optional torch device (string or torch.device). Defaults to CUDA if available.

    Returns:
        a_Mfull: Calculated capacity (scalar)
        R_M: Calculated radius (scalar)
        D_M: Calculated dimension (scalar)
    """
    device = _resolve_device(device)
    sD1_t = _to_torch(sD1, device)
    D1, m = sD1_t.shape
    D = D1 - 1

    if t_vec is None:
        t_vec_t = torch.randn(D1, n_t, device=device, dtype=sD1_t.dtype)
    else:
        t_vec_t = _to_torch(t_vec, device)

    # Step 1: Find max projections to split interior vs boundary points.
    ss_init, gg_init = maxproj(t_vec_t, sD1_t, device=device)

    # Interior: gg + kappa < 0, boundary: gg + kappa >= 0.
    needs_opt = gg_init + kappa >= 0
    n_opt = int(torch.sum(needs_opt).item())

    # Seed s_all with maxproj outputs for interior points.
    s_all = ss_init.clone()

    if n_opt > 0:
        # Extract the vectors that need optimization.
        t_opt = t_vec_t[:, needs_opt]

        # Dual objective for the active set.
        ST_T = sD1_t.T @ t_opt
        b = ST_T + kappa
        alpha = torch.zeros((m, n_opt), dtype=sD1_t.dtype, device=device)

        # Pre-compute Gram matrix and step size.
        STS, L_estimate = _prepare_dual_solver(sD1_t)
        lr = 1.0 / L_estimate

        if use_nesterov:
            alpha = _solve_dual_nesterov(STS, alpha, b, lr, max_iter, tol)
        else:
            alpha = _solve_dual_pgd(STS, alpha, b, lr, max_iter, tol)

        # Reconstruct the primal solution.
        S_alpha = sD1_t @ alpha
        V_opt = t_opt - S_alpha

        # Update optimized vectors; interior points stay unchanged.
        alpha_sum = torch.sum(alpha, dim=0)
        diff_norm = torch.linalg.norm(V_opt - t_opt, dim=0)
        is_boundary = diff_norm > eps

        # Guard division when reconstructing boundary anchors.
        safe_sum = torch.where(alpha_sum > 1e-12, alpha_sum, torch.ones_like(alpha_sum))
        s_boundary = S_alpha / safe_sum.unsqueeze(0)
        s_opt = torch.where(is_boundary.unsqueeze(0), s_boundary, ss_init[:, needs_opt])
        s_all[:, needs_opt] = s_opt

    # Step 2: Compute manifold capacity (Eq. 16-17).
    max_ts = torch.clamp_min(torch.sum(t_vec_t * s_all, dim=0) + kappa, 0.0)
    s_sum = torch.sum(s_all ** 2, dim=0)

    # Vectorized lambda with safe division.
    lamb = torch.zeros(n_t, device=device, dtype=sD1_t.dtype)
    valid_s = s_sum > 1e-12
    lamb[valid_s] = max_ts[valid_s] / s_sum[valid_s]

    slam = lamb ** 2 * s_sum
    mean_slam = torch.mean(slam)
    mean_slam_val = mean_slam.item()
    a_Mfull = 1.0 / mean_slam_val if mean_slam_val > 1e-12 else 0.0

    # Step 3: Compute manifold radius (Eq. 28).
    ds0 = s_all - torch.mean(s_all, dim=1, keepdim=True)
    denom = s_all[-1, :]
    valid_idx = torch.abs(denom) > 1e-12

    if torch.any(valid_idx):
        ds = ds0[:-1, valid_idx] / denom[valid_idx]
        R_M = torch.sqrt(torch.mean(torch.sum(ds ** 2, dim=0))).item()
    else:
        R_M = 0.0

    # Step 4: Compute manifold dimension (Eq. 29).
    t_norms = torch.linalg.norm(t_vec_t[:D, :], dim=0, keepdim=True)
    t_hat = t_vec_t[:D, :] / torch.clamp_min(t_norms, 1e-12)
    s_norms = torch.linalg.norm(s_all[:D, :], dim=0, keepdim=True)
    s_hat = s_all[:D, :] / torch.clamp_min(s_norms, 1e-12)
    ts_dot = torch.sum(t_hat * s_hat, dim=0)
    D_M = (D * torch.mean(ts_dot) ** 2).item()

    return a_Mfull, R_M, D_M


@torch.no_grad()
def _solve_dual_nesterov(STS, alpha, b, lr, max_iter, tol):
    """FISTA for the dual problem.
    Uses Nesterov acceleration with adaptive restart.

    Args:
        STS: Pre-computed Gram matrix S^T @ S of shape (m, m)
        alpha: Initial dual variables of shape (m, n_opt)
        b: RHS vector S^T @ T + kappa of shape (m, n_opt)
        lr: Step size (1 / L where L is spectral norm of STS)
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        alpha: Optimized dual variables
    """
    alpha_t = alpha.clone()
    y = alpha_t.clone()
    t_k = 1.0
    prev_obj = float("inf")
    b_scale = 1.0 + torch.max(torch.abs(b)).item()

    for k in range(max_iter):
        # Single matmul per iteration.
        grad = STS @ y - b

        # Gradient step with projection to the non-negative orthant.
        alpha_new = torch.clamp_min(y - lr * grad, 0.0)

        # Compute objective for restart check (every 10 iters).
        if k % 10 == 0:
            obj = 0.5 * torch.sum(alpha_new * (STS @ alpha_new)) - torch.sum(alpha_new * b)
            obj_val = obj.item()

            # Restart momentum if the objective increases.
            if obj_val > prev_obj:
                t_k = 1.0  # reset momentum
            prev_obj = obj_val

        # Nesterov momentum update.
        t_k_new = 0.5 * (1.0 + (1.0 + 4.0 * t_k * t_k) ** 0.5)
        beta = (t_k - 1.0) / t_k_new
        t_k = t_k_new

        y = alpha_new + beta * (alpha_new - alpha_t)
        alpha_t = alpha_new

        # Convergence check using KKT conditions every 25 iterations.
        if k % 25 == 24:
            active = alpha_t > 1e-12
            kkt_viol = torch.where(active, torch.abs(grad), torch.clamp_min(-grad, 0.0))
            if torch.max(kkt_viol).item() < tol * b_scale:
                break

    return alpha_t


@torch.no_grad()
def _solve_dual_pgd(STS, alpha, b, lr, max_iter, tol):
    """
    Standard Projected Gradient Descent for the dual problem.

    Args:
        STS: Pre-computed Gram matrix S^T @ S of shape (m, m)
        alpha: Initial dual variables of shape (m, n_opt)
        b: RHS vector S^T @ T + kappa of shape (m, n_opt)
        lr: Step size (1 / L where L is spectral norm of STS)
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        alpha: Optimized dual variables
    """
    alpha_t = alpha.clone()

    for k in range(max_iter):
        # Single matmul per iteration.
        grad = STS @ alpha_t - b

        # Gradient step with projection.
        alpha_new = torch.clamp_min(alpha_t - lr * grad, 0.0)

        # Convergence check.
        if k % 25 == 24:
            rel_change = torch.linalg.norm(alpha_new - alpha_t) / (torch.linalg.norm(alpha_t) + 1e-12)
            if rel_change.item() < tol:
                break

        alpha_t = alpha_new

    return alpha_t


@torch.no_grad()
def maxproj(t_vec, sD1, sc=1, device=None):
    """Find the point on the manifold with the largest projection onto each t vector.
    Fully vectorized implementation.

    Args:
        t_vec: 2D array of shape (D+1, n_t)
        sD1: 2D array of shape (D+1, m) where m is number of manifold points
        sc: Value for center dimension (scalar, default 1)
        device: Optional torch device (string or torch.device). Defaults to CUDA if available.

    Returns:
        s0: 2D torch tensor of shape (D+1, n_t) containing max projection points
        gt: 1D torch tensor of shape (n_t,) containing max projection values
    """
    device = _resolve_device(device)
    t_vec_t = _to_torch(t_vec, device)
    sD1_t = _to_torch(sD1, device)

    D1, n_t = t_vec_t.shape
    D = D1 - 1

    # Pairwise dot products for all t vectors.
    projections = t_vec_t[:D, :].T @ sD1_t[:D, :]

    # Select the best-matching manifold point per t vector.
    max_indices = torch.argmax(projections, dim=1)

    # Build s0 using the selected indices.
    s0 = torch.empty((D1, n_t), device=device, dtype=sD1_t.dtype)
    s0[:D, :] = sD1_t[:D, max_indices]
    s0[D, :] = sc

    # Compute projection values for the selected points.
    gt = torch.sum(t_vec_t * s0, dim=0)

    return s0, gt
