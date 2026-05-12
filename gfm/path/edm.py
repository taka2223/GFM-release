"""
EDM2 Wrapper for GFM experiments.
Class-conditional latent diffusion on ImageNet-512.

Same EDM convention as shape experiment:
    x_σ = x₀ + σ·ε,  model predicts D(x,σ) → x₀_pred
    score = -(x - x₀_pred) / σ²
    force = -(x - x₀_pred) / σ²  (same as BaseInterpolator.compute_force)

BaseInterpolator works WITHOUT ANY MODIFICATION.
Cross-class interpolation uses soft one-hot labels, handled transparently
inside the model wrapper.

Usage:
    python edm2_wrapper.py --preset edm2-img512-s-guid-dino --classA 207 --classB 281
"""

import os
import sys
sys.path.insert(0, 'edm2')

import pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any
from tqdm import tqdm
from diffusers.models import AutoencoderKL

# EDM2 repo must be on path for pickle loading
import dnnlib
dnnlib.util.set_cache_dir('/cns/USERS/zzhixuan/weights/edm2')

model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions'

config_presets = {
    'edm2-img512-xs-fid':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.135.pkl'),      # fid = 3.53
    'edm2-img512-xs-dino':             dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.200.pkl'),      # fd_dinov2 = 103.39
    'edm2-img512-s-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.130.pkl'),       # fid = 2.56
    'edm2-img512-s-dino':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.190.pkl'),       # fd_dinov2 = 68.64
    'edm2-img512-m-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.100.pkl'),       # fid = 2.25
    'edm2-img512-m-dino':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.155.pkl'),       # fd_dinov2 = 58.44
    'edm2-img512-l-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.085.pkl'),       # fid = 2.06
    'edm2-img512-l-dino':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.155.pkl'),       # fd_dinov2 = 52.25
    'edm2-img512-xl-fid':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.085.pkl'),      # fid = 1.96
    'edm2-img512-xl-dino':             dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.155.pkl'),      # fd_dinov2 = 45.96
    'edm2-img512-xxl-fid':             dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.070.pkl'),     # fid = 1.91
    'edm2-img512-xxl-dino':            dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.150.pkl'),     # fd_dinov2 = 42.84
    'edm2-img64-s-fid':                dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.075.pkl'),        # fid = 1.58
    'edm2-img64-m-fid':                dnnlib.EasyDict(net=f'{model_root}/edm2-img64-m-2147483-0.060.pkl'),        # fid = 1.43
    'edm2-img64-l-fid':                dnnlib.EasyDict(net=f'{model_root}/edm2-img64-l-1073741-0.040.pkl'),        # fid = 1.33
    'edm2-img64-xl-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img64-xl-0671088-0.040.pkl'),       # fid = 1.33
    'edm2-img512-xs-guid-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.045.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.045.pkl', guidance=1.40), # fid = 2.91
    'edm2-img512-xs-guid-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.150.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.150.pkl', guidance=1.70), # fd_dinov2 = 79.94
    'edm2-img512-s-guid-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.025.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.025.pkl', guidance=1.40), # fid = 2.23
    'edm2-img512-s-guid-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.085.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.085.pkl', guidance=1.90), # fd_dinov2 = 52.32
    'edm2-img512-m-guid-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.030.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.20), # fid = 2.01
    'edm2-img512-m-guid-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.015.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=2.00), # fd_dinov2 = 41.98
    'edm2-img512-l-guid-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.015.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.20), # fid = 1.88
    'edm2-img512-l-guid-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.035.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.035.pkl', guidance=1.70), # fd_dinov2 = 38.20
    'edm2-img512-xl-guid-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.020.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.020.pkl', guidance=1.20), # fid = 1.85
    'edm2-img512-xl-guid-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.030.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.70), # fd_dinov2 = 35.67
    'edm2-img512-xxl-guid-fid':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',      gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.20), # fid = 1.81
    'edm2-img512-xxl-guid-dino':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',      gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.70), # fd_dinov2 = 33.09
    'edm2-img512-s-autog-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.070.pkl',        gnet=f'{model_root}/edm2-img512-xs-0134217-0.125.pkl',        guidance=2.10), # fid = 1.34
    'edm2-img512-s-autog-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.120.pkl',        gnet=f'{model_root}/edm2-img512-xs-0134217-0.165.pkl',        guidance=2.45), # fd_dinov2 = 36.67
    'edm2-img512-xxl-autog-fid':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.075.pkl',      gnet=f'{model_root}/edm2-img512-m-0268435-0.155.pkl',         guidance=2.05), # fid = 1.25
    'edm2-img512-xxl-autog-dino':      dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.130.pkl',      gnet=f'{model_root}/edm2-img512-m-0268435-0.205.pkl',         guidance=2.30), # fd_dinov2 = 24.18
    'edm2-img512-s-uncond-autog-fid':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-uncond-2147483-0.070.pkl', gnet=f'{model_root}/edm2-img512-xs-uncond-0134217-0.110.pkl', guidance=2.85), # fid = 3.86
    'edm2-img512-s-uncond-autog-dino': dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-uncond-2147483-0.090.pkl', gnet=f'{model_root}/edm2-img512-xs-uncond-0134217-0.125.pkl', guidance=2.90), # fd_dinov2 = 90.39
    'edm2-img64-s-autog-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.045.pkl',         gnet=f'{model_root}/edm2-img64-xs-0134217-0.110.pkl',         guidance=1.70), # fid = 1.01
    'edm2-img64-s-autog-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.105.pkl',         gnet=f'{model_root}/edm2-img64-xs-0134217-0.175.pkl',         guidance=2.20), # fd_dinov2 = 31.85
}

# ============================================================
# 0. EDM sampler (from EDM2, supports soft labels)
# ============================================================

def edm_sampler(
    net, latents, class_labels=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    init_at_sigma=True,
):
    """EDM sampler compatible with BaseInterpolator.denoise_path.
    
    Args:
        net: model with forward(x, sigma, labels) → x₀_pred
        latents: [B, C, H, W] noisy latents (at sigma_max if init_at_sigma=True)
        class_labels: [B] integer, [B, label_dim] one-hot/soft, or None
        init_at_sigma: if True, latents are already at sigma_max level
    """
    dtype = torch.float32
    randn_like = torch.randn_like

    step_indices = torch.arange(num_steps, dtype=dtype, device=latents.device)
    t_steps = (sigma_max ** (1/rho) + step_indices / (num_steps - 1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    if init_at_sigma:
        x_next = latents.to(dtype)
    else:
        x_next = latents.to(dtype) * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        denoised = net(x_hat, t_hat, class_labels).to(dtype)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(dtype)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


# ============================================================
# 1. EDM2 Model Wrapper
# ============================================================

class EDM2ModelWrapper(torch.nn.Module):
    """Wraps EDM2 network to match BaseInterpolator's expected interface.
    
    Handles:
        - Integer class labels → one-hot conversion
        - Per-frame soft label interpolation (transparent to BaseInterpolator)
        - Classifier-free guidance via guidance network
    """

    def __init__(
        self,
        net,
        gnet=None,
        guidance: float = 1.0,
        label_dim: int = 1000,
    ):
        super().__init__()
        self.net = net
        self.gnet = gnet if gnet is not None else net
        self.guidance = guidance
        self.label_dim = label_dim
        self.device = next(net.parameters()).device

        # --- Interpolation state ---
        # When set, forward() ignores class_labels arg and uses these instead
        self._interp_labels = None  # [n_inner, label_dim] soft one-hot for inner frames
        self._n_inner = None

    def _to_onehot(self, class_labels: torch.Tensor) -> torch.Tensor:
        """Convert integer class labels to one-hot."""
        if class_labels is None:
            return None
        if class_labels.dim() == 2:
            return class_labels  # already one-hot or soft
        return F.one_hot(class_labels.long(), self.label_dim).float()

    def set_interpolation_labels(self, classA: int, classB: int, num_steps: int):
        """Precompute soft one-hot labels for cross-class interpolation.
        
        Creates n_inner = num_steps - 2 interpolated labels for inner frames.
        When active, forward() auto-assigns per-frame labels based on batch structure.
        """
        n_inner = num_steps - 2
        onehot_A = F.one_hot(torch.tensor(classA), self.label_dim).float().to(self.device)
        onehot_B = F.one_hot(torch.tensor(classB), self.label_dim).float().to(self.device)

        # Inner frames: α = 1/(num_steps-1), 2/(num_steps-1), ..., (num_steps-2)/(num_steps-1)
        alphas = torch.linspace(
            1.0 / (num_steps - 1),
            (num_steps - 2) / (num_steps - 1),
            n_inner,
            device=self.device,
        )
        self._interp_labels = torch.stack([
            (1 - a) * onehot_A + a * onehot_B for a in alphas
        ])  # [n_inner, label_dim]
        self._n_inner = n_inner

    def clear_interpolation_labels(self):
        """Clear interpolation state, revert to normal label behavior."""
        self._interp_labels = None
        self._n_inner = None

    def get_soft_label(self, alpha: float) -> torch.Tensor:
        """Get soft one-hot label for a given interpolation alpha ∈ [0, 1].
        Returns [1, label_dim]."""
        onehot_A = F.one_hot(torch.tensor(0), self.label_dim).float().to(self.device)
        onehot_B = F.one_hot(torch.tensor(0), self.label_dim).float().to(self.device)
        # This is a fallback; prefer set_interpolation_labels + _interp_labels
        if self._interp_labels is not None:
            idx = int(alpha * (len(self._interp_labels) - 1) + 0.5)
            idx = max(0, min(idx, len(self._interp_labels) - 1))
            return self._interp_labels[idx].unsqueeze(0)
        return None

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        sigma,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass matching BaseInterpolator's interface.
        
        When _interp_labels is set:
            - Batch is assumed to be n_inner frames (or multiple of n_inner)
            - Each frame gets its own interpolated soft label
            - class_labels arg is IGNORED
            
        When _interp_labels is not set:
            - class_labels (integer) are converted to one-hot normally
        """
        B = x.shape[0]

        # --- Resolve labels ---
        if self._interp_labels is not None and self._n_inner is not None:
            # Interpolation mode: assign per-frame soft labels
            if B == self._n_inner:
                labels = self._interp_labels  # [n_inner, label_dim]
            elif B % self._n_inner == 0:
                repeats = B // self._n_inner
                labels = self._interp_labels.repeat(repeats, 1)  # [B, label_dim]
            else:
                # Fallback: use midpoint label
                mid_idx = self._n_inner // 2
                labels = self._interp_labels[mid_idx].unsqueeze(0).expand(B, -1)
        else:
            # Normal mode: convert integer to one-hot
            labels = self._to_onehot(class_labels) if class_labels is not None else None

        # --- Guided denoising ---
        Dx = self.net(x, sigma, labels).to(torch.float32)
        if self.guidance != 1.0 and self.gnet is not self.net:
            ref_Dx = self.gnet(x, sigma, labels).to(torch.float32)
            Dx = ref_Dx.lerp(Dx, self.guidance)

        return Dx

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        batch_seeds: Optional[torch.Tensor] = None,
        num_steps: int = 32,
        sigma_max: float = 80,
    ) -> torch.Tensor:
        """Generate samples (compatible with shape experiment interface).
        
        Args:
            cond: [B] integer class labels
            batch_seeds: [B] random seeds per sample
        Returns:
            clean_latents: [B, C, H, W]
        """
        B = cond.shape[0]
        labels = self._to_onehot(cond)
        img_channels = self.net.img_channels
        img_resolution = self.net.img_resolution

        if batch_seeds is not None:
            noise = torch.stack([
                torch.randn(img_channels, img_resolution, img_resolution,
                            generator=torch.Generator(self.device).manual_seed(int(s) % (1 << 32)),
                            device=self.device)
                for s in batch_seeds
            ])
        else:
            noise = torch.randn(B, img_channels, img_resolution, img_resolution,
                                device=self.device)

        # Temporarily disable interpolation mode for sampling
        saved_interp = self._interp_labels
        saved_n = self._n_inner
        self._interp_labels = None
        self._n_inner = None

        # Direct EDM sampling (no wrapper overhead)
        def denoise(x_hat, t_hat):
            Dx = self.net(x_hat, t_hat, labels).to(torch.float32)
            if self.guidance != 1.0 and self.gnet is not self.net:
                ref_Dx = self.gnet(x_hat, t_hat, labels).to(torch.float32)
                Dx = ref_Dx.lerp(Dx, self.guidance)
            return Dx

        rho = 7
        sigma_min = 0.002
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=self.device)
        t_steps = (sigma_max ** (1/rho) + step_indices / (num_steps - 1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        x_next = noise * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            d_cur = (x_next - denoise(x_next, t_cur)) / t_cur
            x_prime = x_next + (t_next - t_cur) * d_cur
            if i < num_steps - 1:
                d_prime = (x_prime - denoise(x_prime, t_next)) / t_next
                x_next = x_next + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
            else:
                x_next = x_prime

        # Restore interpolation state
        self._interp_labels = saved_interp
        self._n_inner = saved_n

        return x_next.float()

    def eval(self):
        self.net.eval()
        if self.gnet is not self.net:
            self.gnet.eval()
        return self

    def parameters(self):
        return self.net.parameters()


# ============================================================
# 2. VAE Autoencoder
# ============================================================

class EDM2Autoencoder:
    """SD VAE wrapper for EDM2 latent diffusion (img512 models)."""

    def __init__(self, vae, device: str = "cuda"):
        self.vae = vae
        self.device = device
        self.scaling_factor = 0.18215

    def encode(self, image: Image.Image, resolution: int = 512) -> torch.Tensor:
        img = image.convert("RGB").resize((resolution, resolution), Image.LANCZOS)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = 2.0 * arr - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=self.vae.dtype)
        with torch.no_grad():
            latent = self.vae.encode(tensor).latent_dist.mean
        return latent * self.scaling_factor

    def decode(self, latent: torch.Tensor) -> Image.Image:
        with torch.no_grad():
            img_tensor = self.vae.decode(latent.to(self.vae.dtype) / self.scaling_factor).sample
        arr = img_tensor.squeeze(0).float().permute(1, 2, 0).cpu().numpy()
        arr = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))

    def eval(self):
        pass


# ============================================================
# 3. Pipeline
# ============================================================

class EDM2GFMPipeline:
    """GFM pipeline on EDM2 (ImageNet-512 latent diffusion).
    
    Same EDM convention as shape experiment → BaseInterpolator works unmodified.
    """

    def __init__(self, model: EDM2ModelWrapper, autoencoder: EDM2Autoencoder, device: str = "cuda"):
        self.model = model
        self.autoencoder = autoencoder
        self.device = torch.device(device)

    @classmethod
    def load(
        cls,
        preset: Optional[str] = None,
        net_path: Optional[str] = None,
        gnet_path: Optional[str] = None,
        guidance: float = 1.0,
        vae_type: str = "mse",
        device: str = "cuda",
    ) -> "EDM2GFMPipeline":
        # Resolve preset
        if preset is not None:
            if preset not in config_presets:
                raise ValueError(f"Unknown preset: {preset}")
            cfg = config_presets[preset]
            net_path = net_path or cfg.get('net')
            gnet_path = gnet_path or cfg.get('gnet')
            guidance = cfg.get('guidance', guidance)

        if net_path is None:
            raise ValueError("Must specify --preset or --net")

        # Load main network
        print(f"Loading main network from {net_path}...")
        with dnnlib.util.open_url(net_path) as f:
            data = pickle.load(f)
        net = data['ema'].to(device)
        net.eval()

        # Load guidance network
        gnet = None
        if gnet_path is not None and guidance != 1.0:
            print(f"Loading guidance network from {gnet_path}...")
            with dnnlib.util.open_url(gnet_path) as f:
                gnet = pickle.load(f)['ema'].to(device)
            gnet.eval()

        print(f"  Guidance: {guidance}")
        print(f"  Net: channels={net.img_channels}, resolution={net.img_resolution}, label_dim={net.label_dim}")

        model = EDM2ModelWrapper(net, gnet=gnet, guidance=guidance, label_dim=net.label_dim)
        model.eval()

        # Load VAE
        print(f"Loading VAE (sd-vae-ft-{vae_type})...")
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_type}").to(device)
        vae.eval()
        vae.requires_grad_(False)

        autoencoder = EDM2Autoencoder(vae, device=device)
        return cls(model, autoencoder, device=device)

    def sample(self, class_id: int, seed: int = 0, num_steps: int = 32) -> Tuple[torch.Tensor, Image.Image]:
        cond = torch.tensor([class_id], device=self.device)
        batch_seeds = torch.tensor([seed], device=self.device)
        latent = self.model.sample(cond=cond, batch_seeds=batch_seeds, num_steps=num_steps)
        image = self.autoencoder.decode(latent)
        return latent, image


# ============================================================
# 4. Per-frame denoise with interpolated labels
# ============================================================

def denoise_path_cross_class(
    net,
    gnet,
    guidance: float,
    path: torch.Tensor,
    classA: int,
    classB: int,
    sigma_start: float,
    num_denoise_steps: int = 18,
    label_dim: int = 1000,
) -> torch.Tensor:
    """Denoise each frame with its own interpolated soft label.
    
    Args:
        net: raw EDM2 network (not wrapper)
        gnet: guidance network (or same as net)
        guidance: guidance strength
        path: [B, num_steps, C, H, W] noisy path
        classA, classB: endpoint class indices
        sigma_start: noise level to denoise from
        
    Returns:
        clean_path: [B, num_steps, C, H, W]
    """
    device = path.device
    B, num_steps = path.shape[:2]

    onehot_A = F.one_hot(torch.tensor(classA), label_dim).float().to(device)
    onehot_B = F.one_hot(torch.tensor(classB), label_dim).float().to(device)

    clean_frames = []

    for t_idx in tqdm(range(num_steps), desc="Denoising frames"):
        alpha = t_idx / max(num_steps - 1, 1)
        soft_label = ((1 - alpha) * onehot_A + alpha * onehot_B).unsqueeze(0)  # [1, 1000]
        soft_label = soft_label.expand(B, -1)  # [B, 1000]

        frame_noisy = path[:, t_idx]  # [B, C, H, W]

        # Guided denoiser for this frame
        def denoise_fn(x, sigma, _labels=None):
            Dx = net(x, sigma, soft_label).to(torch.float32)
            if guidance != 1.0 and gnet is not net:
                ref_Dx = gnet(x, sigma, soft_label).to(torch.float32)
                Dx = ref_Dx.lerp(Dx, guidance)
            return Dx

        # Run EDM sampler for this single frame
        frame_clean = edm_sampler(
            net=denoise_fn,
            latents=frame_noisy,
            class_labels=None,  # labels handled inside denoise_fn
            sigma_max=sigma_start,
            init_at_sigma=True,
            num_steps=num_denoise_steps,
        )
        clean_frames.append(frame_clean)

    return torch.stack(clean_frames, dim=1).float()  # [B, num_steps, C, H, W]


# ============================================================
# 5. GFM Interpolation
# ============================================================

def run_gfm_interpolation(
    pipe: EDM2GFMPipeline,
    classA: int,
    classB: int,
    noise_level: float = 0.5,
    num_steps: int = 10,
    lam: float = 1.0,
    lr: float = 0.01,
    max_iters: int = 800,
    seedA: int = 0,
    seedB: int = 1,
    interpolator_type: str = "el",
    num_denoise_steps: int = 18,
) -> Dict[str, Any]:
    """End-to-end GFM interpolation between two ImageNet classes."""
    from gfm.path.geodesic_interpolation import (
        ELInterpolator,
        SeqELInterpolator,
        SphericalInterpolator,
    )

    device = pipe.device

    # 1. Sample endpoints
    print(f"Sampling endpoint A (class={classA}, seed={seedA})...")
    latA, imgA = pipe.sample(classA, seed=seedA)
    print(f"Sampling endpoint B (class={classB}, seed={seedB})...")
    latB, imgB = pipe.sample(classB, seed=seedB)
    print(f"  Latent shape: {latA.shape}")

    # 2. Activate per-frame soft label interpolation in model wrapper
    #    This makes compute_force use interpolated labels transparently
    pipe.model.set_interpolation_labels(classA, classB, num_steps)

    # 3. Select interpolator (BaseInterpolator, unmodified)
    if interpolator_type == "spherical":
        interpolator = SphericalInterpolator(pipe.model, pipe.autoencoder, device=str(device))
    elif interpolator_type == "el":
        interpolator = ELInterpolator(pipe.model, pipe.autoencoder, device=str(device))
    elif interpolator_type == "seq_el":
        interpolator = SeqELInterpolator(pipe.model, pipe.autoencoder, device=str(device))
    else:
        raise ValueError(f"Unknown interpolator: {interpolator_type}")

    # 4. Add noise (EDM convention: x_σ = x₀ + σ·ε)
    print(f"Adding noise (sigma={noise_level})...")
    noisy_latents = interpolator.add_noise(
        torch.cat([latA, latB], dim=0), sigma=noise_level
    )

    # 5. Optimize path
    #    class_label is passed but IGNORED by wrapper (soft labels are active)
    print(f"Running GFM ({interpolator_type}, steps={num_steps}, λ={lam}, iters={max_iters})...")
    dummy_label = torch.tensor([classA], device=device)  # ignored when interp labels active

    if interpolator_type == "spherical":
        path = interpolator.optimize_path(
            start_latent=noisy_latents[0:1],
            end_latent=noisy_latents[1:2],
            init_with_slerp=True,
            num_steps=num_steps,
        )
        info = {}
    else:
        path, info = interpolator.optimize_path(
            start_latent=noisy_latents[0:1],
            end_latent=noisy_latents[1:2],
            num_steps=num_steps,
            sigma=noise_level,
            lr=lr,
            max_iters=max_iters,
            lam=lam,
            class_label=dummy_label,
            verbose=True,
        )

    if isinstance(path, tuple):
        path = path[0]

    # 6. Denoise: per-frame with interpolated soft labels
    #    Bypass BaseInterpolator.denoise_path — use our cross-class version
    pipe.model.clear_interpolation_labels()  # disable interp mode for denoise

    print(f"Denoising path ({num_denoise_steps} steps, per-frame soft labels)...")
    clean_path = denoise_path_cross_class(
        net=pipe.model.net,
        gnet=pipe.model.gnet,
        guidance=pipe.model.guidance,
        path=path,
        classA=classA,
        classB=classB,
        sigma_start=noise_level,
        num_denoise_steps=num_denoise_steps,
        label_dim=pipe.model.label_dim,
    )

    # 7. Decode to images
    print("Decoding to images...")
    images = []
    for i in range(clean_path.shape[1]):
        images.append(pipe.autoencoder.decode(clean_path[0, i:i+1]))

    return {
        "images": images,
        "latents_noisy": path,
        "latents_clean": clean_path,
        "losses": info.get("losses", []) if isinstance(info, dict) else [],
        "classA": classA,
        "classB": classB,
        "imgA": imgA,
        "imgB": imgB,
    }


# ============================================================
# 6. Visualization
# ============================================================

def save_interpolation_strip(images, save_path, size=(256, 256), padding=5):
    n = len(images)
    w, h = size
    pw, ph = w + 2 * padding, h + 2 * padding
    strip = Image.new("RGB", (pw * n, ph), (255, 255, 255))
    for i, img in enumerate(images):
        strip.paste(img.resize(size, Image.LANCZOS), (i * pw + padding, padding))
    strip.save(save_path)
    print(f"Saved strip to {save_path}")


def save_frames(images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(output_dir, f"frame_{i:03d}.png"))
    print(f"Saved {len(images)} frames to {output_dir}")


# ============================================================
# 7. Entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GFM with EDM2")
    parser.add_argument("--preset", type=str, default="edm2-img512-s-guid-dino")
    parser.add_argument("--net", type=str, default=None)
    parser.add_argument("--gnet", type=str, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--classA", type=int, default=207,
                        help="ImageNet class A (207=golden retriever)")
    parser.add_argument("--classB", type=int, default=281,
                        help="ImageNet class B (281=tabby cat)")
    parser.add_argument("--seedA", type=int, default=0)
    parser.add_argument("--seedB", type=int, default=1)
    parser.add_argument("--noise_level", type=float, default=0.5)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max_iters", type=int, default=800)
    parser.add_argument("--vae", type=str, default="mse", choices=["mse", "ema"])
    parser.add_argument("--output_dir", type=str, default="./output/edm2_interp")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--interpolator", type=str, default="el",
                        choices=["spherical", "el", "seq_el"])
    parser.add_argument("--num_denoise_steps", type=int, default=18)
    args = parser.parse_args()

    extra = {}
    if args.guidance is not None:
        extra['guidance'] = args.guidance

    print(f"Loading EDM2 pipeline (preset={args.preset})...")
    pipe = EDM2GFMPipeline.load(
        preset=args.preset,
        net_path=args.net,
        gnet_path=args.gnet,
        vae_type=args.vae,
        device=args.device,
        **extra,
    )

    results = run_gfm_interpolation(
        pipe,
        classA=args.classA,
        classB=args.classB,
        noise_level=args.noise_level,
        num_steps=args.num_steps,
        lam=args.lam,
        lr=args.lr,
        max_iters=args.max_iters,
        seedA=args.seedA,
        seedB=args.seedB,
        interpolator_type=args.interpolator,
        num_denoise_steps=args.num_denoise_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_interpolation_strip(results["images"], os.path.join(args.output_dir, "strip.png"))
    save_frames(results["images"], os.path.join(args.output_dir, "frames"))
    print(f"\nClasses: {args.classA} -> {args.classB}")
    if results["losses"]:
        print(f"Final GFM loss: {results['losses'][-1]:.6f}")