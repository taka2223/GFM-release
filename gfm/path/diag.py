"""
Score Field Diagnostics for GFM Latent Space.

Usage:
    from score_diagnostics import ScoreDiagnostics
    diag = ScoreDiagnostics(sd3_pipe)
    diag.run_all(latA_noisy, latB_noisy, sigma_eff, output_dir="./output/diagnostics")
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path


class ScoreDiagnostics:
    def __init__(self, pipeline):
        """
        Args:
            pipeline: SD3GFMPipeline (or any object with .model.compute_force-like interface)
                      pipeline.model should support forward(latent, sigma) → denoised
        """
        self.pipe = pipeline
        self.model = pipeline.model
        self.device = pipeline.device
    
    def compute_force(self, z: torch.Tensor, sigma: float) -> torch.Tensor:
        """Compute score-based force: F = (x_pred - z) / σ² ∝ ∇log p_σ(z)."""
        sigma_t = torch.tensor([sigma], device=self.device)
        with torch.no_grad():
            x_pred = self.model(z, sigma_t)
        # v-prediction → denoised x₀, force = (x₀ - z_t) / σ
        # 这里 model.forward 在 nfsd 模式下直接返回 z + σ·grad_v
        # 需要根据你的 model mode 调整
        force = (x_pred - z) / (sigma + 1e-8)
        return force
    
    # ─────────────────────────────────────────
    # 1. Score magnitude along path × sigma
    # ─────────────────────────────────────────
    
    def score_magnitude_heatmap(
        self,
        latA: torch.Tensor,
        latB: torch.Tensor,
        sigma_current: float,
        n_points: int = 20,
        sigma_range: Optional[list] = None,
        save_path: Optional[str] = None,
    ) -> dict:
        """
        在 lerp(A,B) 路径上采样 n_points 个点，
        对每个点扫描不同 σ，计算 ||F(z, σ)||。
        
        输出: (n_sigmas × n_points) 热力图
        """
        if sigma_range is None:
            sigma_range = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        alphas = torch.linspace(0, 1, n_points, device=self.device)
        
        # 线性插值路径
        path_points = [(1 - a) * latA + a * latB for a in alphas]
        
        force_norms = np.zeros((len(sigma_range), n_points))
        
        print(f"  [Diag] Computing score magnitudes: {n_points} points × {len(sigma_range)} sigmas...")
        for si, sigma in enumerate(sigma_range):
            # 设置 model conditioning 对应的 sigma
            # 注意: 需要确保 model 的 conditioning 已设置好
            for pi, z in enumerate(path_points):
                force = self.compute_force(z, sigma)
                force_norms[si, pi] = force.flatten().norm().item()
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap
        ax = axes[0]
        im = ax.imshow(
            force_norms, aspect='auto', origin='lower',
            extent=[0, 1, sigma_range[0], sigma_range[-1]],
            cmap='viridis', interpolation='bilinear'
        )
        ax.set_xlabel('α  (0=imgA, 1=imgB)', fontsize=12)
        ax.set_ylabel('σ (noise level)', fontsize=12)
        ax.set_title('||Score Force||  along lerp path', fontsize=14)
        ax.axhline(y=sigma_current, color='r', linestyle='--', linewidth=2, label=f'σ_current={sigma_current}')
        ax.legend(fontsize=11)
        plt.colorbar(im, ax=ax, label='||F(z, σ)||')
        
        # Line plot at current sigma
        ax2 = axes[1]
        current_idx = np.argmin(np.abs(np.array(sigma_range) - sigma_current))
        for si in [0, current_idx, len(sigma_range) // 2, -1]:
            ax2.plot(
                alphas.cpu().numpy(), force_norms[si],
                label=f'σ={sigma_range[si]:.2f}', linewidth=2
            )
        ax2.set_xlabel('α  (0=imgA, 1=imgB)', fontsize=12)
        ax2.set_ylabel('||Force||', fontsize=12)
        ax2.set_title('Score magnitude profiles', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close()
        
        return {
            'force_norms': force_norms,
            'alphas': alphas.cpu().numpy(),
            'sigma_range': sigma_range,
        }
    
    # ─────────────────────────────────────────
    # 2. Tangential / Normal decomposition
    # ─────────────────────────────────────────
    
    def tangent_normal_decomposition(
        self,
        latA: torch.Tensor,
        latB: torch.Tensor,
        sigma: float,
        n_points: int = 20,
        save_path: Optional[str] = None,
    ) -> dict:
        """
        沿 lerp 路径，将 force 分解为:
          - tangential: proj_{path_dir} F  (加减速)
          - normal:     F - tangential      (弯曲路径, 这才是有用的)
        """
        alphas = torch.linspace(0, 1, n_points, device=self.device)
        
        # 路径方向 (归一化)
        path_dir = (latB - latA).flatten()
        path_dir = path_dir / (path_dir.norm() + 1e-8)
        
        tan_norms = []
        nor_norms = []
        total_norms = []
        
        print(f"  [Diag] Tangent/Normal decomposition at σ={sigma}...")
        for a in alphas:
            z = (1 - a) * latA + a * latB
            force = self.compute_force(z, sigma)
            f_flat = force.flatten()
            
            # Tangential component
            f_tan = (f_flat @ path_dir) * path_dir
            # Normal component
            f_nor = f_flat - f_tan
            
            tan_norms.append(f_tan.norm().item())
            nor_norms.append(f_nor.norm().item())
            total_norms.append(f_flat.norm().item())
        
        alphas_np = alphas.cpu().numpy()
        tan_norms = np.array(tan_norms)
        nor_norms = np.array(nor_norms)
        total_norms = np.array(total_norms)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(alphas_np, total_norms, 'k-', linewidth=2.5, label='||F|| total')
        ax.plot(alphas_np, tan_norms, 'b--', linewidth=2, label='||F_tangential|| (speed change)')
        ax.plot(alphas_np, nor_norms, 'r-', linewidth=2, label='||F_normal|| (path bending)')
        ax.fill_between(alphas_np, 0, nor_norms, alpha=0.15, color='red')
        ax.set_xlabel('α  (0=imgA, 1=imgB)', fontsize=12)
        ax.set_ylabel('Force magnitude', fontsize=12)
        ax.set_title(f'Force decomposition along lerp path (σ={sigma})', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close()
        
        return {
            'tangential': tan_norms,
            'normal': nor_norms,
            'total': total_norms,
            'alphas': alphas_np,
        }
    
    # ─────────────────────────────────────────
    # 3. PCA 2D score field visualization
    # ─────────────────────────────────────────
    
    def pca_score_field(
        self,
        latA: torch.Tensor,
        latB: torch.Tensor,
        sigma: float,
        grid_size: int = 15,
        extent: float = 1.5,
        save_path: Optional[str] = None,
    ) -> dict:
        """
        在 (latA, latB) 张成的 2D PCA 子空间里画 score vector field。
        
        basis:
          e1 = normalize(B - A)         — 连线方向
          e2 = normalize(rand - proj)   — 正交方向 (随机采样)
        """
        # 构建正交基
        A_flat = latA.flatten()
        B_flat = latB.flatten()
        
        e1 = B_flat - A_flat
        e1 = e1 / (e1.norm() + 1e-8)
        
        # 随机正交方向 (Gram-Schmidt)
        rand_dir = torch.randn_like(A_flat)
        rand_dir = rand_dir - (rand_dir @ e1) * e1
        e2 = rand_dir / (rand_dir.norm() + 1e-8)
        
        # 中心 = 路径中点
        center = 0.5 * (A_flat + B_flat)
        half_dist = 0.5 * (B_flat - A_flat).norm().item()
        
        # Grid in PCA coordinates
        s = torch.linspace(-extent, extent, grid_size, device=self.device)
        grid_x, grid_y = torch.meshgrid(s, s, indexing='ij')
        
        # Score field on grid
        Fx = np.zeros((grid_size, grid_size))
        Fy = np.zeros((grid_size, grid_size))
        Fmag = np.zeros((grid_size, grid_size))
        
        print(f"  [Diag] PCA score field: {grid_size}×{grid_size} grid at σ={sigma}...")
        for i in range(grid_size):
            for j in range(grid_size):
                # Map grid coords to full latent space
                z_flat = center + grid_x[i, j] * half_dist * e1 + grid_y[i, j] * half_dist * e2
                z = z_flat.view_as(latA)
                
                force = self.compute_force(z, sigma)
                f_flat = force.flatten()
                
                # Project force back to 2D
                Fx[i, j] = (f_flat @ e1).item()
                Fy[i, j] = (f_flat @ e2).item()
                Fmag[i, j] = f_flat.norm().item()
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        gx = grid_x.cpu().numpy()
        gy = grid_y.cpu().numpy()
        
        # Vector field
        ax = axes[0]
        # Normalize arrows for visibility
        Fnorm = np.sqrt(Fx**2 + Fy**2) + 1e-8
        ax.quiver(gx, gy, Fx/Fnorm, Fy/Fnorm, Fmag, cmap='hot', alpha=0.8)
        ax.plot([-1, 1], [0, 0], 'g-', linewidth=3, alpha=0.5, label='lerp A→B')
        ax.plot(-1, 0, 'go', markersize=12, label='A')
        ax.plot(1, 0, 'rs', markersize=12, label='B')
        ax.set_xlabel('e₁ (A→B direction)', fontsize=12)
        ax.set_ylabel('e₂ (orthogonal)', fontsize=12)
        ax.set_title(f'Score field in PCA subspace (σ={sigma})', fontsize=14)
        ax.legend(fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        # Magnitude heatmap
        ax2 = axes[1]
        im = ax2.pcolormesh(gx, gy, Fmag, cmap='viridis', shading='auto')
        ax2.plot([-1, 1], [0, 0], 'w-', linewidth=3, alpha=0.7)
        ax2.plot(-1, 0, 'wo', markersize=12)
        ax2.plot(1, 0, 'ws', markersize=12)
        ax2.set_xlabel('e₁ (A→B direction)', fontsize=12)
        ax2.set_ylabel('e₂ (orthogonal)', fontsize=12)
        ax2.set_title(f'||Score|| magnitude (σ={sigma})', fontsize=14)
        ax2.set_aspect('equal')
        plt.colorbar(im, ax=ax2, label='||F||')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close()
        
        return {'Fx': Fx, 'Fy': Fy, 'Fmag': Fmag, 'grid_x': gx, 'grid_y': gy}
    
    # ─────────────────────────────────────────
    # 4. Run all diagnostics
    # ─────────────────────────────────────────
    
    def run_all(
        self,
        latA: torch.Tensor,
        latB: torch.Tensor,
        sigma: float,
        output_dir: str = "./output/diagnostics",
    ):
        """Run all diagnostics and save plots."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("Score Field Diagnostics")
        print("=" * 60)
        
        # 1. Heatmap: score magnitude vs (alpha, sigma)
        print("\n[1/3] Score magnitude heatmap...")
        self.score_magnitude_heatmap(
            latA, latB, sigma,
            save_path=str(out / "score_heatmap.png")
        )
        
        # 2. Tangent/Normal decomposition
        print("\n[2/3] Tangent/Normal decomposition...")
        self.tangent_normal_decomposition(
            latA, latB, sigma,
            save_path=str(out / "tangent_normal.png")
        )
        # 也画几个不同sigma的对比
        for s in [0.2, 0.5, 0.8]:
            self.tangent_normal_decomposition(
                latA, latB, s,
                save_path=str(out / f"tangent_normal_sigma{s:.1f}.png")
            )
        
        # 3. PCA score field
        print("\n[3/3] PCA score field...")
        self.pca_score_field(
            latA, latB, sigma,
            save_path=str(out / f"pca_field_sigma{sigma:.2f}.png")
        )
        for s in [0.5, 0.8]:
            self.pca_score_field(
                latA, latB, s,
                save_path=str(out / f"pca_field_sigma{s:.1f}.png")
            )
        
        print(f"\nAll diagnostics saved to {output_dir}")