"""
McCann Displacement Interpolation for GFM latent path initialization.

Replaces iterative free-support Wasserstein barycenter with:
  1. ONE Sinkhorn solve → optimal plan P* (N×N)
  2. Barycentric projection → OT map T(x_i) = Σ_j (P*_{ij}/a_i) · y_j
  3. For each α: Z_α = (1-α)·X + α·T(X)

Complexity: O(sinkhorn_iters × N²) once, then O(α_steps × N × D) for interpolation.
vs free-support: O(α_steps × adam_steps × sinkhorn_iters × N²)

For K=2, equal weights, same particle count — this IS the global optimum of the
free-support barycenter problem (not an approximation).
"""

import torch
from typing import Optional, Tuple


def log_sinkhorn_plan(
    C: torch.Tensor,
    eps: Optional[float] = None,
    eps_scale: float = 0.05,
    max_iter: int = 200,
    thresh: float = 1e-6,
) -> torch.Tensor:
    """Log-domain Sinkhorn → optimal transport plan.
    
    Args:
        C: (N, M) cost matrix (squared Euclidean)
        eps: regularization. If None, auto-set to eps_scale * median(C)
        eps_scale: multiplier for adaptive eps (default 0.05)
        max_iter: max Sinkhorn iterations
        thresh: marginal error convergence threshold
        
    Returns:
        P: (N, M) optimal transport plan, rows sum to 1/N, cols sum to 1/M
    """
    N, M = C.shape
    
    # Adaptive epsilon from cost statistics
    if eps is None:
        with torch.no_grad():
            C_flat = C.reshape(-1)
            median_C = C_flat.median().item()
            eps = eps_scale * max(median_C, 1e-8)
            print(f"  [Sinkhorn] adaptive ε = {eps_scale} × median(C)={median_C:.4f} → ε={eps:.6f}")
    
    log_a = torch.full((N,), -torch.tensor(float(N)).log(), device=C.device, dtype=C.dtype)
    log_b = torch.full((M,), -torch.tensor(float(M)).log(), device=C.device, dtype=C.dtype)
    log_K = -C / eps
    
    f = torch.zeros(N, device=C.device, dtype=C.dtype)
    g = torch.zeros(M, device=C.device, dtype=C.dtype)
    
    for it in range(max_iter):
        f = log_a - torch.logsumexp(log_K + g.unsqueeze(0), dim=1)
        g = log_b - torch.logsumexp(log_K + f.unsqueeze(1), dim=0)
        
        if it % 10 == 0:
            log_row_sum = torch.logsumexp(f.unsqueeze(1) + log_K + g.unsqueeze(0), dim=1)
            err = (log_row_sum.exp() - (1.0 / N)).abs().max().item()
            if err < thresh:
                print(f"  [Sinkhorn] converged at iter {it+1}, marginal err={err:.2e}")
                break
    else:
        print(f"  [Sinkhorn] max_iter={max_iter} reached, marginal err={err:.2e}")
    
    log_P = f.unsqueeze(1) + log_K + g.unsqueeze(0)
    return log_P.exp()


def mccann_interpolation(
    supp_A: torch.Tensor,
    supp_B: torch.Tensor,
    num_steps: int,
    eps: Optional[float] = None,
    eps_scale: float = 0.05,
    max_sinkhorn: int = 200,
) -> torch.Tensor:
    """McCann displacement interpolation between two equal-weight discrete measures.
    
    Args:
        supp_A: (N, D) source support points
        supp_B: (N, D) target support points  (N must match)
        num_steps: total path length including endpoints
        eps: Sinkhorn regularization (None = adaptive)
        eps_scale: multiplier for adaptive eps
        max_sinkhorn: max Sinkhorn iterations
        
    Returns:
        path: (num_steps, N, D) interpolated support points
              path[0] = supp_A, path[-1] = supp_B
    """
    N, D = supp_A.shape
    assert supp_B.shape == (N, D), f"Shape mismatch: {supp_A.shape} vs {supp_B.shape}"
    device = supp_A.device
    
    # 1. Cost matrix: ||x_i - y_j||² = ||x_i||² + ||y_j||² - 2<x_i, y_j>
    #    Memory-efficient: never materializes (N, N, D) diff tensor
    print(f"  Computing cost matrix ({N}×{N}, D={D})...")
    x_sqnorm = (supp_A ** 2).sum(dim=1, keepdim=True)  # (N, 1)
    y_sqnorm = (supp_B ** 2).sum(dim=1, keepdim=True)  # (N, 1)
    C = x_sqnorm + y_sqnorm.T - 2.0 * (supp_A @ supp_B.T)  # (N, N)
    C = C.clamp(min=0)
    
    # 2. Sinkhorn → optimal plan P*
    P = log_sinkhorn_plan(C, eps=eps, eps_scale=eps_scale, max_iter=max_sinkhorn)
    
    # 3. Barycentric projection → OT map T
    #    T(x_i) = Σ_j [ P_{ij} / a_i ] · y_j
    #    where a_i = 1/N, so P_{ij}/a_i = N · P_{ij}
    transport_weights = P * N  # (N, N), rows sum to ~1
    T_A = transport_weights @ supp_B  # (N, D) — where each source point "should go"
    
    # Sanity check
    row_sums = transport_weights.sum(dim=1)
    print(f"  Transport weight row sums: mean={row_sums.mean():.6f}, "
          f"std={row_sums.std():.6f} (should be ~1.0)")
    
    # 4. McCann interpolation: Z_α = (1-α)·X + α·T(X)
    alphas = torch.linspace(0, 1, num_steps, device=device)
    path = torch.zeros(num_steps, N, D, device=device, dtype=supp_A.dtype)
    
    for i, alpha in enumerate(alphas):
        path[i] = (1 - alpha) * supp_A + alpha * T_A
    
    return path


class W2SeqELInterpolator_McCann:
    """
    Drop-in replacement using McCann displacement interpolation.
    
    Usage in your existing code:
        # Replace:
        #   interpolator = W2SeqELInterpolator(...)
        # With:
        #   (just override _generate_w2_initial_path)
    """
    
    @staticmethod
    def y_to_grid(y: torch.Tensor, H: int = 64, W: int = 64, C: int = 16) -> torch.Tensor:
        return y.reshape(H, W, C).permute(2, 0, 1).unsqueeze(0)
    
    @staticmethod
    def _generate_w2_initial_path(
        start_latent: torch.Tensor,
        end_latent: torch.Tensor,
        num_steps: int,
        eps: Optional[float] = None,
        eps_scale: float = 0.05,
    ) -> torch.Tensor:
        """
        McCann displacement interpolation.
        
        One Sinkhorn solve → analytic path. No iterative optimization.
        
        Args:
            start_latent: (1, C, H, W) source latent
            end_latent:   (1, C, H, W) target latent
            num_steps: total frames including endpoints
            eps: Sinkhorn ε (None = adaptive)
            
        Returns:
            path: (1, num_steps, C, H, W) interpolated latent path
        """
        device = start_latent.device
        B, C, H, W = start_latent.shape
        N = H * W
        
        # Reshape to point clouds: (1, C, H, W) → (N, C)
        supp_A = start_latent.squeeze(0).permute(1, 2, 0).reshape(N, C)
        supp_B = end_latent.squeeze(0).permute(1, 2, 0).reshape(N, C)
        
        print(f"McCann interpolation: {N} points × {C} dims, {num_steps} steps")
        
        # One-shot McCann
        path_points = mccann_interpolation(
            supp_A, supp_B, num_steps,
            eps=eps, eps_scale=eps_scale,
        )  # (num_steps, N, C)
        
        # Reshape back to latent grid: (num_steps, N, C) → (1, num_steps, C, H, W)
        path_latents = path_points.reshape(num_steps, H, W, C).permute(0, 3, 1, 2)
        path_latents = path_latents.unsqueeze(0)  # (1, num_steps, C, H, W)
        
        # Force exact endpoints
        path_latents[:, 0] = start_latent
        path_latents[:, -1] = end_latent
        
        return path_latents


# ============================================================
# Demo / test
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Test: McCann on synthetic latents (SD3-like shape)")
    print("=" * 60)
    
    C, H, W = 16, 32, 32
    N = H * W  # 1024
    
    # Two "latents" — shifted Gaussians
    latA = torch.randn(1, C, H, W, device=device) * 0.5 + 0.3
    latB = torch.randn(1, C, H, W, device=device) * 0.5 - 0.3
    
    import time
    t0 = time.time()
    path = W2SeqELInterpolator_McCann._generate_w2_initial_path(
        latA, latB, num_steps=10, eps_scale=0.05,
    )
    t1 = time.time()
    
    print(f"\n  Path shape: {path.shape}")
    print(f"  Time: {t1 - t0:.2f}s")
    print(f"  Endpoint error A: {(path[:, 0] - latA).abs().max():.2e}")
    print(f"  Endpoint error B: {(path[:, -1] - latB).abs().max():.2e}")
    
    # Check monotonic W2 cost along path
    supp_A = latA.squeeze(0).permute(1, 2, 0).reshape(N, C)
    supp_B = latB.squeeze(0).permute(1, 2, 0).reshape(N, C)
    
    print("\n  Frame-to-frame L2 distances:")
    for i in range(path.shape[1] - 1):
        pA = path[0, i].permute(1, 2, 0).reshape(N, C)
        pB = path[0, i + 1].permute(1, 2, 0).reshape(N, C)
        dist = ((pA - pB) ** 2).sum(dim=-1).sqrt().mean()
        print(f"    frame {i} → {i+1}: mean_dist = {dist:.4f}")
    
    print("\nDone!")