"""
Geodesic Interpolation for 3D Shape Latent Space

Flexible framework supporting multiple interpolation methods:
- Score-based geodesic optimization
- RBF kernel-based interpolation with anchors
- Custom metric-induced optimization

Follows extensible design - implement method-specific optimization strategies.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from gfm.models.models_class_cond import edm_sampler
from gfm.path.metrics import load_metric, h_diag_RBF

class BaseInterpolator(ABC):
    """
    Abstract base class for latent space interpolation methods.
    
    Provides common functionality and defines interface for method-specific optimization.
    """

    def __init__(self, diffusion_model, autoencoder, device="cuda"):
        """
        Args:
            diffusion_model: Trained EDM diffusion model
            autoencoder: Trained autoencoder for decoding validation
            device: Computing device
        """
        self.model = diffusion_model
        self.autoencoder = autoencoder
        self.device = device

        # Set models to eval mode
        self.model.eval()
        self.autoencoder.eval()

    def linear_interpolate(
        self, start_latent: torch.Tensor, end_latent: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        """
        Simple linear interpolation baseline.

        Args:
            start_latent: [batch, n_latents, channels] or [1, n_latents, channels]
            end_latent: [batch, n_latents, channels] or [1, n_latents, channels]
            num_steps: Number of interpolation steps

        Returns:
            interpolated: [batch, num_steps, n_latents, channels]
        """
        batch_size = start_latent.shape[0]
        t = torch.linspace(0, 1, num_steps, device=self.device).view(1, -1, 1, 1)
        interpolated = (
            start_latent.unsqueeze(1)
            + (end_latent.unsqueeze(1) - start_latent.unsqueeze(1)) * t
        )
        return interpolated

    def compute_force(
        self,
        latent: torch.Tensor,
        sigma: float,
        class_label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute force field from diffusion model.

        Force = -(x - x_pred) / sigma^2, where x_pred = model(x, sigma)

        Args:
            latent: [batch, n_latents, channels] latent codes
            sigma: Noise level for diffusion model
            class_label: Optional class conditioning

        Returns:
            force: [batch, n_latents, channels] force vectors
        """
        batch_size = latent.shape[0]

        # Prepare sigma tensor
        sigma_tensor = torch.full((batch_size,), sigma, device=self.device)

        # Prepare class labels
        if class_label is not None:
            if class_label.dim() == 0:  # scalar
                class_labels = class_label.repeat(batch_size)
            else:
                class_labels = class_label
        else:
            class_labels = None

        with torch.no_grad():
            # Get model prediction
            x_pred = self.model(latent, sigma_tensor, class_labels)

            # Compute force: gradient points toward clean data
            force = -(latent - x_pred) / (sigma**2)

        return force

    def add_noise(self, latent: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Add Gaussian noise to latent codes.

        Args:
            latent: [batch, n_latents, channels] clean latent codes
            sigma: Noise level

        Returns:
            noisy_latent: [batch, n_latents, channels] noisy latent codes
        """
        noise = torch.randn_like(latent) * sigma
        return latent + noise

    @abstractmethod
    def _optimize_path_impl(
        self,
        start_latent: torch.Tensor,
        end_latent: torch.Tensor,
        num_steps: int = 10,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Method-specific path optimization implementation.
        
        Args:
            start_latent: [batch, n_latents, channels] start points
            end_latent: [batch, n_latents, channels] end points  
            num_steps: Number of intermediate points
            **kwargs: Method-specific optimization parameters
            
        Returns:
            optimized_path: [batch, num_steps, n_latents, channels] optimized trajectories
            info: Dictionary with optimization info (losses, metrics, etc.)
        """
        pass
    
    def optimize_path(
        self,
        start_latent: torch.Tensor,
        end_latent: torch.Tensor,
        num_steps: int = 10,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        High-level interface for path optimization. Delegates to method-specific implementation.

        Args:
            start_latent: [batch, n_latents, channels] start points
            end_latent: [batch, n_latents, channels] end points
            num_steps: Number of intermediate points
            **kwargs: Method-specific optimization parameters

        Returns:
            optimized_path: [batch, num_steps, n_latents, channels] optimized trajectories
            info: Dictionary with optimization info and metrics
        """
        return self._optimize_path_impl(start_latent, end_latent, num_steps, **kwargs)

    def denoise_path(
        self,
        noisy_path: torch.Tensor,
        sigma_start: float,
        class_label: Optional[torch.Tensor] = None,
        num_denoise_steps: int = 18,
    ) -> torch.Tensor:
        """
        Denoise a path using the existing EDM sampler (DRY principle).

        Args:
            noisy_path: [batch, num_steps, n_latents, channels] noisy latent path
            sigma_start: Starting noise level
            class_label: Optional class conditioning [batch]
            num_denoise_steps: Number of EDM denoising steps

        Returns:
            clean_path: [batch, num_steps, n_latents, channels] denoised path
        """
        

        batch_size, num_steps = noisy_path.shape[:2]

        # Flatten for processing: [batch*num_steps, n_latents, channels]
        noisy_flat = noisy_path.view(-1, *noisy_path.shape[2:])

        # Expand class labels for all path points
        if class_label is not None:
            if class_label.numel() == 1:
                class_labels_expanded = class_label.repeat(noisy_flat.shape[0])
            else:
                # Repeat each class label for all steps in that batch
                class_labels_expanded = (
                    class_label.unsqueeze(1).repeat(1, num_steps).view(-1)
                )
        else:
            class_labels_expanded = None

        # Use EDM sampler to denoise from sigma_start to clean
        # noisy_flat is already at the correct noise level
        with torch.no_grad():
            clean_flat = edm_sampler(
                net=self.model,
                latents=noisy_flat,
                class_labels=class_labels_expanded,
                num_steps=num_denoise_steps,
                sigma_min=0.002,
                sigma_max=sigma_start,
                init_at_sigma=True,
            )

        # Reshape back to path format
        clean_path = clean_flat.view_as(noisy_path).float()
        return clean_path


class ELInterpolator(BaseInterpolator):
    """
    Euler-Lagrange Interpolator: Geodesic on sphere with score-based force.
    
    Minimizes: ||acc_geodesic + λ * force_tangent||²
    """
    
    def slerp_interpolate(
        self, start_latent: torch.Tensor, end_latent: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        """Standard SLERP for initialization."""
        original_shape = start_latent.shape
        batch_size = original_shape[0]
        
        start_flat = start_latent.view(batch_size, -1)
        end_flat = end_latent.view(batch_size, -1)
        
        start_norm = start_flat.norm(dim=-1, keepdim=True)
        end_norm = end_flat.norm(dim=-1, keepdim=True)
        dot = (start_flat * end_flat).sum(dim=-1, keepdim=True)
        
        cos_omega = (dot / (start_norm * end_norm + 1e-8)).clamp(-1 + 1e-7, 1 - 1e-7)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega)
        
        t = torch.linspace(0, 1, num_steps, device=self.device).view(1, num_steps, 1)
        
        s0 = torch.sin((1 - t) * omega.unsqueeze(1)) / (sin_omega.unsqueeze(1) + 1e-8)
        s1 = torch.sin(t * omega.unsqueeze(1)) / (sin_omega.unsqueeze(1) + 1e-8)
        
        small_angle = sin_omega.unsqueeze(1).abs() < 1e-6
        s0 = torch.where(small_angle, 1 - t, s0)
        s1 = torch.where(small_angle, t, s1)
        
        result_flat = s0 * start_flat.unsqueeze(1) + s1 * end_flat.unsqueeze(1)
        return result_flat.view(batch_size, num_steps, *original_shape[1:])

    @staticmethod
    def project_to_tangent_space(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project vector v onto tangent space of sphere at point x."""
        x_flat = x.flatten(start_dim=1)
        v_flat = v.flatten(start_dim=1)
        
        dot = (v_flat * x_flat).sum(dim=-1, keepdim=True)
        x_norm_sq = (x_flat * x_flat).sum(dim=-1, keepdim=True)
        
        normal_component = (dot / (x_norm_sq + 1e-8)) * x_flat
        tangent_v = v_flat - normal_component
        
        return tangent_v.view_as(v)

    def _optimize_path_impl(
        self,
        start_latent: torch.Tensor,
        end_latent: torch.Tensor,
        num_steps: int = 10,
        sigma: float = 0.1,
        lr: float = 0.05,
        max_iters: int = 200,
        class_label: Optional[torch.Tensor] = None,
        lam: float = 1.0,
        verbose: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Optimize geodesic on sphere with score-based force."""
        batch_size = start_latent.shape[0]
        
        path = self.slerp_interpolate(start_latent, end_latent, num_steps)
        
        if num_steps <= 2:
            return path, {"losses": [], "method": "euler_lagrange"}
        
        x0 = path[:, 0]
        x1 = path[:, -1]
        x_inner = path[:, 1:-1].clone().detach().requires_grad_(True)
        
        dt = 1.0 / (num_steps - 1)
        n_inner = num_steps - 2
        
        optimizer = torch.optim.Adam([x_inner], lr=lr, eps=1e-5)
        losses = []
        
        # Prepare expanded class labels once (for n_inner points per batch)
        if class_label is not None:
            if class_label.numel() == 1:
                cls_labels_expanded = class_label.expand(batch_size * n_inner)
            else:
                cls_labels_expanded = class_label.unsqueeze(1).expand(-1, n_inner).reshape(-1)
        else:
            cls_labels_expanded = None
        
        for iteration in range(max_iters):
            optimizer.zero_grad()
            
            full_path = torch.cat([x0.unsqueeze(1), x_inner, x1.unsqueeze(1)], dim=1)
            
            # Geodesic acceleration
            acc_euclidean = (full_path[:, 2:] - 2 * full_path[:, 1:-1] + full_path[:, :-2]) / (dt ** 2)
            x_mid = full_path[:, 1:-1].reshape(-1, *full_path.shape[2:])
            acc_flat = acc_euclidean.reshape(-1, *acc_euclidean.shape[2:])
            acc_geodesic = self.project_to_tangent_space(x_mid, acc_flat).view_as(acc_euclidean)
            
            # Reuse compute_force from BaseInterpolator
            x_inner_flat = x_inner.reshape(-1, *x_inner.shape[2:])
            force_euclidean = self.compute_force(x_inner_flat, sigma, cls_labels_expanded)
            
            # Project force to tangent space
            force_tangent = self.project_to_tangent_space(x_inner_flat, force_euclidean).view_as(x_inner)
            
            # Loss
            loss = ((acc_geodesic + lam * force_tangent) ** 2).mean()
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if verbose and iteration % 20 == 0:
                print(f"Iter {iteration}: loss = {loss.item():.6f}")
        
        final_path = torch.cat([x0.unsqueeze(1), x_inner.detach(), x1.unsqueeze(1)], dim=1)
        
        return final_path, {
            "losses": losses,
            "method": "euler_lagrange",
            "sigma": sigma,
            "lambda": lam,
            "iterations": max_iters
        }


class ScoreBasedInterpolator(BaseInterpolator):
    """
    Score-based geodesic interpolation using Jacobian of score function as metric.
    
    Metric: g_x(v, w) = v^T J^T J w, where J = ∇_x s(x, t)
    Energy: E[γ] = (1/2) Σ ||s(x_{i+1}) - s(x_i)||² / Δu
    """
    
    def slerp_interpolate(
        self, start_latent: torch.Tensor, end_latent: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        """Spherical linear interpolation (paper recommends for initialization)."""
        original_shape = start_latent.shape
        batch_size = original_shape[0]
        
        # Flatten to [batch, D]
        start_flat = start_latent.view(batch_size, -1)
        end_flat = end_latent.view(batch_size, -1)
        
        # Unit vectors and norms
        start_norm = start_flat.norm(dim=-1, keepdim=True)
        end_norm = end_flat.norm(dim=-1, keepdim=True)
        start_unit = start_flat / (start_norm + 1e-8)
        end_unit = end_flat / (end_norm + 1e-8)
        
        # Angle
        cos_theta = (start_unit * end_unit).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos_theta)
        sin_theta = torch.sin(theta)
        
        # Interpolation
        t = torch.linspace(0, 1, num_steps, device=self.device).view(1, num_steps, 1)
        w0 = torch.sin((1 - t) * theta.unsqueeze(1)) / (sin_theta.unsqueeze(1) + 1e-8)
        w1 = torch.sin(t * theta.unsqueeze(1)) / (sin_theta.unsqueeze(1) + 1e-8)
        
        # Fall back to linear for small angles
        small_angle = sin_theta.unsqueeze(1).abs() < 1e-6
        w0 = torch.where(small_angle, 1 - t, w0)
        w1 = torch.where(small_angle, t, w1)
        
        # Interpolate direction and norm
        norm_interp = start_norm.unsqueeze(1) * (1 - t) + end_norm.unsqueeze(1) * t
        interp_unit = w0 * start_unit.unsqueeze(1) + w1 * end_unit.unsqueeze(1)
        interp_unit = interp_unit / (interp_unit.norm(dim=-1, keepdim=True) + 1e-8)
        
        return (interp_unit * norm_interp).view(batch_size, num_steps, *original_shape[1:])

    def compute_score(
        self,
        latent: torch.Tensor,
        sigma: float,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score function s(x, σ) = (D(x, σ) - x) / σ² = -force
        
        Note: Needs gradient tracking, so we don't use no_grad here.
        """
        batch_size = latent.shape[0]
        sigma_tensor = torch.full((batch_size,), sigma, device=self.device, dtype=latent.dtype)
        
        x_pred = self.model(latent, sigma_tensor, class_labels)
        return (x_pred - latent) / (sigma ** 2)

    def _optimize_path_impl(
        self,
        start_latent: torch.Tensor,
        end_latent: torch.Tensor,
        num_steps: int = 10,
        sigma: float = 0.15,
        lr: float = 0.001,
        max_iters: int = 1000,
        class_label: Optional[torch.Tensor] = None,
        init_with_slerp: bool = True,
        lr_min_ratio: float = 0.1,
        verbose: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Minimize score-change energy (Eq. 11 from paper):
            E[γ] = (1/2) Σ ||s(x_{i+1}) - s(x_i)||² / Δu
        """
        batch_size = start_latent.shape[0]
        
        # Initialize with SLERP (paper recommendation) or LERP
        if init_with_slerp:
            path = self.slerp_interpolate(start_latent, end_latent, num_steps)
        else:
            path = self.linear_interpolate(start_latent, end_latent, num_steps)
        
        if num_steps <= 2:
            return path, {"losses": [], "method": "score_tangent"}
        
        # Optimize intermediate points only (endpoints fixed)
        intermediate = path[:, 1:-1].clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([intermediate], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_iters, eta_min=lr * lr_min_ratio
        )
        
        losses = []
        du = 1.0 / (num_steps - 1)
        
        for iteration in range(max_iters):
            optimizer.zero_grad()
            
            # Full path with fixed endpoints
            full_path = torch.cat([
                start_latent.unsqueeze(1),
                intermediate,
                end_latent.unsqueeze(1)
            ], dim=1)
            
            # Flatten for model: [batch * num_steps, ...]
            path_flat = full_path.view(-1, *full_path.shape[2:])
            
            # Expand class labels
            if class_label is not None:
                if class_label.numel() == 1:
                    class_labels_exp = class_label.expand(path_flat.shape[0])
                else:
                    class_labels_exp = class_label.unsqueeze(1).expand(-1, num_steps).reshape(-1)
            else:
                class_labels_exp = None
            
            # Compute scores
            scores = self.compute_score(path_flat, sigma, class_labels_exp).view_as(full_path)
            
            # Energy: E = (1/2) Σ ||s_{i+1} - s_i||² / Δu
            score_diff = scores[:, 1:] - scores[:, :-1]
            energy = 0.5 * (score_diff ** 2).sum() / du
            
            energy.backward()
            optimizer.step()
            scheduler.step()
            
            losses.append(energy.item())
            
            if verbose and iteration % 50 == 0:
                print(f"Iter {iteration}: Energy = {energy.item():.6f}, lr = {scheduler.get_last_lr()[0]:.6f}")
        
        final_path = torch.cat([
            start_latent.unsqueeze(1),
            intermediate.detach(),
            end_latent.unsqueeze(1)
        ], dim=1)
        
        return final_path, {
            "losses": losses,
            "method": "score_tangent", 
            "sigma": sigma,
            "iterations": max_iters
        }

class SteinScoreInterpolator(BaseInterpolator):
    """
    Stein Score Metric interpolation from Azeglio & Di Bernardo (2025).
    "What's Inside Your Diffusion Model? A Score-Based Riemannian Metric"
    
    Metric: g(x) = I + λ · s(x)s(x)^T
    Energy: E = (1/2) Σ [||v||² + λ(s^T v)²]
    
    The metric penalizes movement perpendicular to the data manifold
    while preserving movement along tangential directions.
    """
    
    def slerp_interpolate(
        self, start_latent: torch.Tensor, end_latent: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        """Standard SLERP for initialization."""
        original_shape = start_latent.shape
        batch_size = original_shape[0]
        
        start_flat = start_latent.view(batch_size, -1)
        end_flat = end_latent.view(batch_size, -1)
        
        start_norm = start_flat.norm(dim=-1, keepdim=True)
        end_norm = end_flat.norm(dim=-1, keepdim=True)
        dot = (start_flat * end_flat).sum(dim=-1, keepdim=True)
        
        cos_omega = (dot / (start_norm * end_norm + 1e-8)).clamp(-1 + 1e-7, 1 - 1e-7)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega)
        
        t = torch.linspace(0, 1, num_steps, device=self.device).view(1, num_steps, 1)
        
        s0 = torch.sin((1 - t) * omega.unsqueeze(1)) / (sin_omega.unsqueeze(1) + 1e-8)
        s1 = torch.sin(t * omega.unsqueeze(1)) / (sin_omega.unsqueeze(1) + 1e-8)
        
        small_angle = sin_omega.unsqueeze(1).abs() < 1e-6
        s0 = torch.where(small_angle, 1 - t, s0)
        s1 = torch.where(small_angle, t, s1)
        
        result_flat = s0 * start_flat.unsqueeze(1) + s1 * end_flat.unsqueeze(1)
        return result_flat.view(batch_size, num_steps, *original_shape[1:])

    def compute_score_with_grad(
        self,
        latent: torch.Tensor,
        sigma: float,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score s(x) = (D(x) - x) / σ² with gradient w.r.t. latent.
        
        Model output is detached, but latent remains in computation graph.
        This matches the paper's approach where score is treated as a 
        position-dependent field, not optimized through.
        """
        batch_size = latent.shape[0]
        sigma_tensor = torch.full((batch_size,), sigma, device=self.device, dtype=latent.dtype)
        
        with torch.no_grad():
            x_pred = self.model(latent, sigma_tensor, class_labels)
        
        return (x_pred - latent) / (sigma ** 2)

    def _optimize_path_impl(
        self,
        start_latent: torch.Tensor,
        end_latent: torch.Tensor,
        num_steps: int = 10,
        sigma: float = 0.5,
        lr: float = 0.001,
        max_iters: int = 500,
        lam: float = 1000.0,
        class_label: Optional[torch.Tensor] = None,
        init_with_slerp: bool = True,
        use_midpoint: bool = True,
        lr_min_ratio: float = 0.1,
        verbose: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Optimize geodesic using Stein Score Metric.
        
        Energy (Eq. 8 from paper):
            E[γ] = (1/2) Σ [||v_i||² + λ(s(γ_i)^T v_i)²]
        
        Args:
            lam: Penalty parameter λ controlling normal direction penalty.
                 Paper uses λ=1000 based on ablation study.
            use_midpoint: If True, evaluate score at segment midpoints (paper's approach).
        """
        batch_size = start_latent.shape[0]
        
        # Initialize
        if init_with_slerp:
            path = self.slerp_interpolate(start_latent, end_latent, num_steps)
        else:
            path = self.linear_interpolate(start_latent, end_latent, num_steps)
        
        if num_steps <= 2:
            return path, {"losses": [], "method": "stein_score"}
        
        x0 = path[:, 0]
        x1 = path[:, -1]
        intermediate = path[:, 1:-1].clone().detach().requires_grad_(True)
        
        n_inner = num_steps - 2
        n_segments = num_steps - 1
        
        optimizer = torch.optim.Adam([intermediate], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_iters, eta_min=lr * lr_min_ratio
        )
        
        # Prepare expanded class labels
        if class_label is not None:
            if class_label.numel() == 1:
                cls_labels_seg = class_label.expand(batch_size * n_segments)
            else:
                cls_labels_seg = class_label.unsqueeze(1).expand(-1, n_segments).reshape(-1)
        else:
            cls_labels_seg = None
        
        losses = []
        
        for iteration in range(max_iters):
            optimizer.zero_grad()
            
            # Full path: [batch, num_steps, ...]
            full_path = torch.cat([
                x0.unsqueeze(1),
                intermediate,
                x1.unsqueeze(1)
            ], dim=1)
            
            # Velocities: [batch, n_segments, ...]
            velocity = full_path[:, 1:] - full_path[:, :-1]
            
            # Points for score evaluation
            if use_midpoint:
                eval_points = 0.5 * (full_path[:, 1:] + full_path[:, :-1])
            else:
                eval_points = full_path[:, :-1]
              
            # Compute scores: [batch * n_segments, ...]
            eval_flat = eval_points.reshape(-1, *eval_points.shape[2:])
            scores_flat = self.compute_score_with_grad(eval_flat, sigma, cls_labels_seg)
            scores = scores_flat.view_as(velocity)
            
            # Flatten spatial dims for dot product
            vel_flat = velocity.flatten(start_dim=2)      # [batch, n_seg, D]
            score_flat = scores.flatten(start_dim=2)      # [batch, n_seg, D]
            
            # Stein energy: E = (1/2) Σ [||v||² + λ(s^T v)²]
            euclidean_term = (vel_flat ** 2).sum(dim=-1)  # [batch, n_seg]
            score_dot_vel = (score_flat * vel_flat).sum(dim=-1)  # [batch, n_seg]
            score_term = lam * (score_dot_vel ** 2)       # [batch, n_seg]
            
            energy = 0.5 * (euclidean_term + score_term).mean()
            
            energy.backward()
            optimizer.step()
            scheduler.step()
            
            losses.append(energy.item())
            
            if verbose and iteration % 50 == 0:
                euc_total = euclidean_term.sum().item()
                score_total = score_term.sum().item()
                print(f"Iter {iteration}: Energy = {energy.item():.4f} "
                      f"(euc={euc_total:.4f}, score={score_total:.4f}), "
                      f"lr = {scheduler.get_last_lr()[0]:.6f}")
        
        final_path = torch.cat([
            x0.unsqueeze(1),
            intermediate.detach(),
            x1.unsqueeze(1)
        ], dim=1)
        
        return final_path, {
            "losses": losses,
            "method": "stein_score",
            "sigma": sigma,
            "lambda": lam,
            "iterations": max_iters
        }


class RBFKernelInterpolator(BaseInterpolator):
    """
    RBF kernel-based geodesic interpolation using existing RiemannianMetric interface.
    """
    
    def __init__(self, diffusion_model, autoencoder, device="cuda"):
        super().__init__(diffusion_model, autoencoder, device)
        self.h = None
        self.metric = None  # RiemannianMetric instance
        self._fitted = False
    
    def fit(
        self, 
        latents: torch.Tensor, 
        n_centers: int = 1000, 
        kappa: float = 3.0,
        normalize_iters: int = 30000
    ) -> "RBFKernelInterpolator":
        """Fit RBF metric from latent data."""
        lt_size = latents.shape[1:]
        latents_flat = latents.view(-1, np.prod(lt_size))
        
        self.h = h_diag_RBF(
            n_centers=n_centers,
            latent_size=lt_size,
            ambiant_size=lt_size,
            data_to_fit_latent=latents_flat,
            data_to_fit_ambiant=latents_flat,
            kappa=kappa
        ).to(self.device)
        
        if normalize_iters > 0:
            self.h.normalize(latents_flat.to(self.device))
        
        # 直接用你的 load_metric
        self.metric = load_metric("conf", "rbf", self.h)
        self._fitted = True
        return self
    
    def load(self, h_path: str) -> "RBFKernelInterpolator":
        """Load pre-trained h_diag_RBF."""
        self.h = torch.load(h_path, map_location=self.device,weights_only=False)
        self.metric = load_metric("conf", "rbf", self.h)
        self._fitted = True
        return self
    
    def _optimize_path_impl(
        self,
        start_latent: torch.Tensor,
        end_latent: torch.Tensor,
        num_steps: int = 10,
        lr: float = 0.01,
        max_iters: int = 2000,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Optimize geodesic path minimizing Riemannian kinetic energy."""
        if not self._fitted:
            raise RuntimeError("Call fit() or load() first")
        
        path = self.linear_interpolate(start_latent, end_latent, num_steps)
        print(f'[DEBUG] RBFInterpolator: linear interpolate path shape: {path.shape}')
        if num_steps <= 2:
            return path, {"losses": [], "method": "rbf_conformal"}
        
        intermediate = path[:, 1:-1].clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([intermediate], lr=lr)
        
        losses = []
        dt = 1.0 / (num_steps - 1)
        
        for iteration in range(max_iters):
            optimizer.zero_grad()
            
            full_path = torch.cat([
                start_latent.unsqueeze(1),
                intermediate,
                end_latent.unsqueeze(1)
            ], dim=1)
            
            # 直接用你的 metric.kinetic()
            velocity = (full_path[:, 1:] - full_path[:, :-1]) / dt
            midpoints = 0.5 * (full_path[:, 1:] + full_path[:, :-1])
            energy = self.metric.kinetic(midpoints, velocity).mean()*dt
            
            loss = energy
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if iteration % 20 == 0:
                print(f"Iter {iteration}: loss={loss.item():.4f}")
                with torch.no_grad():
                    g_values = self.metric.g_fast(midpoints)  # = 1/h(x)
                    print(f"g: min={g_values.min():.2f}, max={g_values.max():.2f}, mean={g_values.mean():.2f}")
                    print(f"grad norm: {intermediate.grad.norm().item():.6f}")
        
        final_path = torch.cat([
            start_latent.unsqueeze(1),
            intermediate.detach(),
            end_latent.unsqueeze(1)
        ], dim=1)
        
        return final_path, {"losses": losses, "method": "rbf_conformal"}


class LandInterpolator(BaseInterpolator):
    """
    Landmark-based geodesic interpolation using h_diag_Land metric.
    """
    
    def __init__(self, diffusion_model, autoencoder, device="cuda"):
        super().__init__(diffusion_model, autoencoder, device)
        self.h = None
        self.metric = None
        self._fitted = False
    
    def fit(
        self, 
        latents: torch.Tensor, 
        n_reference: int = 1000, 
        gamma: float = None,  # None = data-driven
        normalize_iters: int = 3000
    ) -> "LandInterpolator":
        """Fit Land metric from latent data."""
        from gfm.path.metrics import h_diag_Land
        
        lt_size = latents.shape[1:]
        latents_flat = latents.view(-1, np.prod(lt_size)).to(self.device)
        
        n_total = latents_flat.shape[0]
        assert n_total > n_reference, f"Need more than {n_reference} samples"
        
        data_ref = latents_flat[:n_reference]
        data_to_fit = latents_flat[n_reference:]
        
        # Data-driven gamma
        if gamma is None:
            with torch.no_grad():
                n_sample = min(200, n_reference)
                sample_dists = torch.cdist(data_ref[:n_sample], data_ref[:n_sample])
                mask = ~torch.eye(n_sample, dtype=bool, device=self.device)
                gamma = sample_dists[mask].median().item() / 3
                print(f"[LandInterpolator] Data-driven gamma: {gamma:.2f}")
        
        self.h = h_diag_Land(data_ref, gamma=gamma).to(self.device)
        
        if normalize_iters > 0:
            self.h.normalize(data_to_fit)
        
        # Land 用 DiagonalMetric
        self.metric = load_metric("diag", "land", self.h)
        self._fitted = True
        return self
    
    def load(self, h_path: str) -> "LandInterpolator":
        """Load pre-trained h_diag_Land."""
        self.h = torch.load(h_path, map_location=self.device, weights_only=False)
        self.metric = load_metric("diag", "land", self.h)
        self._fitted = True
        return self
    
    def _optimize_path_impl(
        self,
        start_latent: torch.Tensor,
        end_latent: torch.Tensor,
        num_steps: int = 10,
        lr: float = 0.01,
        max_iters: int = 400,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Optimize geodesic path minimizing Riemannian kinetic energy."""
        if not self._fitted:
            raise RuntimeError("Call fit() or load() first")
        
        path = self.linear_interpolate(start_latent, end_latent, num_steps)
        print(f'[DEBUG] LandInterpolator: linear interpolate path shape: {path.shape}')
        if num_steps <= 2:
            return path, {"losses": [], "method": "land_diagonal"}

        intermediate = path[:, 1:-1].clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([intermediate], lr=lr)
        
        losses = []
        dt = 1.0 / (num_steps - 1)
        
        for iteration in range(max_iters):
            optimizer.zero_grad()
            
            full_path = torch.cat([
                start_latent.unsqueeze(1),
                intermediate,
                end_latent.unsqueeze(1)
            ], dim=1)
            
            velocity = (full_path[:, 1:] - full_path[:, :-1]) / dt
            midpoints = 0.5 * (full_path[:, 1:] + full_path[:, :-1])
            energy = self.metric.kinetic(midpoints, velocity).mean() * dt
            
            loss = energy
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if iteration % 20 == 0:
                print(f"Iter {iteration}: loss={loss.item():.4f}")
                with torch.no_grad():
                    g_values = self.metric.g_fast(midpoints)
                    
                    if iteration == 0:
                        init_intermediate = intermediate.clone().detach()
                        init_g = g_values.clone().clone().detach()

                    disp = (intermediate - init_intermediate).abs()
                    print(f"intermediate displacement: mean={disp.mean():.6f}, max={disp.max():.6f}")
                    
                    # g 值变化
                    g_diff = (g_values - init_g).abs()
                    print(f"g change: mean={g_diff.mean():.6f}, max={g_diff.max():.6f}")
        
        final_path = torch.cat([
            start_latent.unsqueeze(1),
            intermediate.detach(),
            end_latent.unsqueeze(1)
        ], dim=1)
        
        return final_path, {"losses": losses, "method": "land_diagonal"}

class SphericalInterpolator(BaseInterpolator):
    """Geodesic interpolation on a sphere."""
    def __init__(self, diffusion_model, autoencoder, device):
        super().__init__(diffusion_model, autoencoder, device)

    def slerp_interpolate(
        self, start_latent: torch.Tensor, end_latent: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        """Spherical linear interpolation (paper recommends for initialization)."""
        original_shape = start_latent.shape
        batch_size = original_shape[0]
        
        # Flatten to [batch, D]
        start_flat = start_latent.view(batch_size, -1)
        end_flat = end_latent.view(batch_size, -1)
        
        # Unit vectors and norms
        start_norm = start_flat.norm(dim=-1, keepdim=True)
        end_norm = end_flat.norm(dim=-1, keepdim=True)
        start_unit = start_flat / (start_norm + 1e-8)
        end_unit = end_flat / (end_norm + 1e-8)
        
        # Angle
        cos_theta = (start_unit * end_unit).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos_theta)
        sin_theta = torch.sin(theta)
        
        # Interpolation
        t = torch.linspace(0, 1, num_steps, device=self.device).view(1, num_steps, 1)
        w0 = torch.sin((1 - t) * theta.unsqueeze(1)) / (sin_theta.unsqueeze(1) + 1e-8)
        w1 = torch.sin(t * theta.unsqueeze(1)) / (sin_theta.unsqueeze(1) + 1e-8)
        
        # Fall back to linear for small angles
        small_angle = sin_theta.unsqueeze(1).abs() < 1e-6
        w0 = torch.where(small_angle, 1 - t, w0)
        w1 = torch.where(small_angle, t, w1)
        
        # Interpolate direction and norm
        norm_interp = start_norm.unsqueeze(1) * (1 - t) + end_norm.unsqueeze(1) * t
        interp_unit = w0 * start_unit.unsqueeze(1) + w1 * end_unit.unsqueeze(1)
        interp_unit = interp_unit / (interp_unit.norm(dim=-1, keepdim=True) + 1e-8)
        
        return (interp_unit * norm_interp).view(batch_size, num_steps, *original_shape[1:])

    def _optimize_path_impl(
        self,
        start_latent: torch.Tensor,
        end_latent: torch.Tensor,
        num_steps: int = 10,
        init_with_slerp: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if init_with_slerp:
            path = self.slerp_interpolate(start_latent, end_latent, num_steps)
        else:
            path = self.linear_interpolate(start_latent, end_latent, num_steps)
        return path
# Backward compatibility aliases
class EL2Interpolator(BaseInterpolator):
    """
    Euler-Lagrange Interpolator: Geodesic on sphere with score-based force.
    
    Minimizes: ||acc_geodesic + λ * force_tangent||²
    """
    
    def slerp_interpolate(
        self, start_latent: torch.Tensor, end_latent: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        """Standard SLERP for initialization."""
        original_shape = start_latent.shape
        batch_size = original_shape[0]
        
        start_flat = start_latent.view(batch_size, -1)
        end_flat = end_latent.view(batch_size, -1)
        
        start_norm = start_flat.norm(dim=-1, keepdim=True)
        end_norm = end_flat.norm(dim=-1, keepdim=True)
        dot = (start_flat * end_flat).sum(dim=-1, keepdim=True)
        
        cos_omega = (dot / (start_norm * end_norm + 1e-8)).clamp(-1 + 1e-7, 1 - 1e-7)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega)
        
        t = torch.linspace(0, 1, num_steps, device=self.device).view(1, num_steps, 1)
        
        s0 = torch.sin((1 - t) * omega.unsqueeze(1)) / (sin_omega.unsqueeze(1) + 1e-8)
        s1 = torch.sin(t * omega.unsqueeze(1)) / (sin_omega.unsqueeze(1) + 1e-8)
        
        small_angle = sin_omega.unsqueeze(1).abs() < 1e-6
        s0 = torch.where(small_angle, 1 - t, s0)
        s1 = torch.where(small_angle, t, s1)
        
        result_flat = s0 * start_flat.unsqueeze(1) + s1 * end_flat.unsqueeze(1)
        return result_flat.view(batch_size, num_steps, *original_shape[1:])

    @staticmethod
    def project_to_tangent_space(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project vector v onto tangent space of sphere at point x."""
        x_flat = x.flatten(start_dim=1)
        v_flat = v.flatten(start_dim=1)
        
        dot = (v_flat * x_flat).sum(dim=-1, keepdim=True)
        x_norm_sq = (x_flat * x_flat).sum(dim=-1, keepdim=True)
        
        normal_component = (dot / (x_norm_sq + 1e-8)) * x_flat
        tangent_v = v_flat - normal_component
        
        return tangent_v.view_as(v)

    def _optimize_path_impl(
        self,
        start_latent: torch.Tensor,
        end_latent: torch.Tensor,
        num_steps: int = 10,
        sigma: float = 0.1,
        lr: float = 0.05,
        max_iters: int = 200,
        class_label: Optional[torch.Tensor] = None,
        lam: float = 1.0,
        verbose: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Optimize geodesic on sphere with score-based force."""
        batch_size = start_latent.shape[0]
        
        path = self.slerp_interpolate(start_latent, end_latent, num_steps)
        
        if num_steps <= 2:
            return path, {"losses": [], "method": "euler_lagrange"}
        
        x0 = path[:, 0]
        x1 = path[:, -1]
        x_inner = path[:, 1:-1].clone().detach().requires_grad_(True)
        
        dt = 1.0 / (num_steps - 1)
        n_inner = num_steps - 2
        
        optimizer = torch.optim.Adam([x_inner], lr=lr, eps=1e-5)
        losses = []
        
        # Prepare expanded class labels once (for n_inner points per batch)
        if class_label is not None:
            if class_label.numel() == 1:
                cls_labels_expanded = class_label.expand(batch_size * n_inner)
            else:
                cls_labels_expanded = class_label.unsqueeze(1).expand(-1, n_inner).reshape(-1)
        else:
            cls_labels_expanded = None
        
        for iteration in range(max_iters):
            optimizer.zero_grad()
            
            full_path = torch.cat([x0.unsqueeze(1), x_inner, x1.unsqueeze(1)], dim=1)
            
            # Geodesic acceleration
            acc_euclidean = (full_path[:, 2:] - 2 * full_path[:, 1:-1] + full_path[:, :-2]) / (dt ** 2)
            x_mid = full_path[:, 1:-1].reshape(-1, *full_path.shape[2:])
            acc_flat = acc_euclidean.reshape(-1, *acc_euclidean.shape[2:])
            acc_geodesic = acc_euclidean
            
            # Reuse compute_force from BaseInterpolator
            x_inner_flat = x_inner.reshape(-1, *x_inner.shape[2:])
            force_euclidean = self.compute_force(x_inner_flat, sigma, cls_labels_expanded)
            
            # Project force to tangent space
            force_tangent = force_euclidean
            
            # Loss
            loss = ((acc_geodesic + lam * force_tangent) ** 2).mean()
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if verbose and iteration % 20 == 0:
                print(f"Iter {iteration}: loss = {loss.item():.6f}")
        
        final_path = torch.cat([x0.unsqueeze(1), x_inner.detach(), x1.unsqueeze(1)], dim=1)
        
        return final_path, {
            "losses": losses,
            "method": "euler_lagrange2",
            "sigma": sigma,
            "lambda": lam,
            "iterations": max_iters
        }
GeodesicInterpolator = ScoreBasedInterpolator
RBFInterpolator = RBFKernelInterpolator