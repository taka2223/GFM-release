"""
SD3 Rectified Flow Wrapper for Geodesic Force Matching (GFM)
Uses StableDiffusion3Pipeline with MMDiT (2B params, ~6x faster than Flux).

Same flow matching math as Flux:
    z_t = (1-t)·x₀ + t·ε,  model predicts v = ε - x₀,  x₀ = z_t - t·v

Key simplifications vs Flux:
    - Transformer takes [B,C,H,W] directly (no manual packing/unpacking)
    - No img_ids / txt_ids positional IDs
    - No guidance embedding (standard CFG via double forward pass)
    - No dynamic_shifting (fixed shift=3.0, no mu needed)
    - T5 can be dropped entirely to save VRAM (text_encoder_3=None)
"""

import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any
from diffusers import StableDiffusion3Pipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm


# ============================================================
# 0. BLIP Captioner
# ============================================================


def piecewise_slerp_resample(path: torch.Tensor, output_steps: int) -> torch.Tensor:
    """Resample a control-point path with equal-time spherical arc segments."""
    if path.ndim < 3:
        raise ValueError("path must have shape [batch, steps, ...]")
    control_steps = path.shape[1]
    if control_steps < 2:
        raise ValueError("path must contain at least two control points")
    if output_steps < 2:
        raise ValueError("output_steps must be at least two")
    if output_steps == control_steps:
        return path

    positions = torch.linspace(
        0,
        control_steps - 1,
        output_steps,
        device=path.device,
        dtype=path.dtype,
    )
    left = positions.floor().long().clamp(max=control_steps - 2)
    alpha = (positions - left.to(positions.dtype)).view(
        1, output_steps, *([1] * (path.ndim - 2))
    )

    point_left = path[:, left]
    point_right = path[:, left + 1]
    flat_left = point_left.flatten(start_dim=2)
    flat_right = point_right.flatten(start_dim=2)
    norm_left = flat_left.norm(dim=-1, keepdim=True)
    norm_right = flat_right.norm(dim=-1, keepdim=True)
    cosine = (
        (flat_left * flat_right).sum(dim=-1, keepdim=True)
        / (norm_left * norm_right + 1e-8)
    ).clamp(-1 + 1e-7, 1 - 1e-7)
    omega = torch.acos(cosine)
    sin_omega = torch.sin(omega)

    alpha_flat = alpha.flatten(start_dim=2)
    weight_left = torch.sin((1 - alpha_flat) * omega) / (sin_omega + 1e-8)
    weight_right = torch.sin(alpha_flat * omega) / (sin_omega + 1e-8)
    small_angle = sin_omega.abs() < 1e-6
    weight_left = torch.where(small_angle, 1 - alpha_flat, weight_left)
    weight_right = torch.where(small_angle, alpha_flat, weight_right)
    output_flat = weight_left * flat_left + weight_right * flat_right
    return output_flat.view(
        path.shape[0], output_steps, *path.shape[2:]
    )


class BLIPCaptioner:
    def __init__(
        self,
        model_id: str = "Salesforce/blip-image-captioning-large",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        prefix: str = "a detailed photograph of",
        num_beams: int = 3,
    ):
        self.device = device
        self.prefix = prefix
        self.num_beams = num_beams
        self.processor = BlipProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_id, cache_dir=cache_dir, torch_dtype=torch.float16
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def caption(self, image: Image.Image, max_new_tokens: int = 40) -> str:
        image = image.convert("RGB")
        if self.prefix:
            inputs = self.processor(image, text=self.prefix, return_tensors="pt").to(
                self.device, torch.float16
            )
        else:
            inputs = self.processor(image, return_tensors="pt").to(
                self.device, torch.float16
            )
        output_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, num_beams=self.num_beams
        )
        return self.processor.decode(output_ids[0], skip_special_tokens=True)

    def unload(self):
        self.model.cpu()
        torch.cuda.empty_cache()


# ============================================================
# 1. SD3Model: v-prediction wrapper for GFM
# ============================================================


class SD3Model(torch.nn.Module):
    """
    Wraps SD3 MMDiT transformer for GFM's model interface.

    Much simpler than Flux:
        - Transformer handles packing/unpacking internally
        - No img_ids, txt_ids, or guidance embedding
        - Standard CFG via double forward pass
    """

    def __init__(self, pipe: StableDiffusion3Pipeline, mode: str = "nfsd"):
        super().__init__()
        self.cfg_scale = 0.0
        self.transformer = pipe.transformer
        self.device = pipe.device
        self.mode = mode
        self.pipe = pipe

        # --- Precompute fixed embeddings ---
        self._embeds_uncond, self._pooled_uncond = self._encode_prompt("")
        self._embeds_neg, self._pooled_neg = self._encode_prompt(
            "A doubling image, unrealistic, artifacts, distortions, "
            "unnatural blending, ghosting effects, overlapping edges, "
            "harsh transitions, motion blur, poor resolution, low detail"
        )

        # --- Conditioning state ---
        self._embeds_cond = None
        self._pooled_cond = None
        self._sigma = None

        # --- Dynamic interpolation state ---
        self._embedA = None
        self._embedB = None
        self._pooledA = None
        self._pooledB = None
        self._waypoint_fractions = None

    def _encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt with SD3's text encoders (CLIP-L + OpenCLIP-G, optionally T5).

        Returns:
            prompt_embeds:  [1, seq_len, dim]  concatenated text features
            pooled_embeds:  [1, dim]           concatenated pooled features
        """
        prompt_embeds, _, pooled_embeds, _ = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            device=self.device,
            do_classifier_free_guidance=False,
        )
        return prompt_embeds, pooled_embeds

    def set_conditioning(
        self, embeds: torch.Tensor, pooled: torch.Tensor, sigma: float
    ):
        """Set static text conditioning and noise level."""
        self._embeds_cond = embeds
        self._pooled_cond = pooled
        self._sigma = sigma
        self._embedA = None

    def set_cfg_params(self, cfg_scale: float):
        self.cfg_scale = cfg_scale

    def set_endpoints(
        self,
        embedA,
        pooledA,
        embedB,
        pooledB,
        sigma: float,
        num_steps: Optional[int] = None,
        waypoint_indices: Optional[List[int]] = None,
        denominator: Optional[int] = None,
    ):
        """Set endpoints and exact full-path indices used by the solver.

        ``num_steps`` remains accepted for older callers.  New optimization
        code passes ``waypoint_indices`` explicitly so a frozen-anchor C2F
        stage cannot silently fall back to midpoint conditioning.
        """
        if waypoint_indices is None:
            if num_steps is None:
                raise ValueError("num_steps or waypoint_indices is required")
            waypoint_indices = list(range(1, num_steps - 1))
            denominator = num_steps - 1
        if denominator is None or denominator < 1:
            raise ValueError("conditioning denominator must be positive")
        if (
            not waypoint_indices
            or waypoint_indices != sorted(set(waypoint_indices))
            or any(index <= 0 or index >= denominator for index in waypoint_indices)
        ):
            raise ValueError(
                "waypoint_indices must be unique, increasing interior indices"
            )
        self._embedA = embedA
        self._pooledA = pooledA
        self._embedB = embedB
        self._pooledB = pooledB
        self._sigma = sigma
        self._waypoint_fractions = (
            torch.tensor(
                waypoint_indices,
                device=self.device,
                dtype=embedA.dtype,
            )
            / float(denominator)
        )
        self._embeds_cond = None

    def _v_pred(
        self,
        latent: torch.Tensor,
        t_val: float,
        embeds: torch.Tensor,
        pooled: torch.Tensor,
    ) -> torch.Tensor:
        B = latent.shape[0]
        # SD3 期望 timestep 为 0~1000
        timestep = torch.full(
            (B,), t_val * 1000.0, device=self.device, dtype=self.pipe.transformer.dtype
        )

        # 确保输入 transformer 的 latent 精度匹配模型（通常是 float16）
        latent_input = latent.to(self.pipe.transformer.dtype)

        output = self.transformer(
            hidden_states=latent_input,
            timestep=timestep,
            encoder_hidden_states=embeds.to(latent_input.dtype),
            pooled_projections=pooled.to(latent_input.dtype),
            return_dict=False,
        )[0]

        # 立即转回 float32，避免后续在 Interpolator 里做平方运算时溢出
        return output.float()

    @torch.no_grad()
    def forward(
        self, latent: torch.Tensor, sigma: torch.Tensor, class_labels=None
    ) -> torch.Tensor:
        """GFM-compatible forward: returns denoised x_pred."""
        batch_size = latent.shape[0]
        latent = latent.float()
        sigma_val = sigma[0].item() if sigma.dim() > 0 else sigma.item()

        # --- Resolve conditioning ---
        if self._embedA is not None:
            waypoint_count = int(self._waypoint_fractions.numel())
            if batch_size % waypoint_count:
                raise ValueError(
                    "model batch does not match explicit waypoint conditioning: "
                    f"batch={batch_size}, waypoint_count={waypoint_count}"
                )
            s = self._waypoint_fractions.repeat(batch_size // waypoint_count)
            s_emb = s.view(-1, 1, 1)
            embeds = (1.0 - s_emb) * self._embedA + s_emb * self._embedB
            s_pool = s.view(-1, 1)
            pooled = (1.0 - s_pool) * self._pooledA + s_pool * self._pooledB
        elif self._embeds_cond is not None:
            embeds = self._embeds_cond.expand(batch_size, -1, -1)
            pooled = self._pooled_cond.expand(batch_size, -1)
        else:
            raise ValueError("No conditioning set.")

        # --- Mode-specific prediction ---
        if self.mode == "denoise":
            v_cond = self._v_pred(latent, sigma_val, embeds, pooled)

            if self.cfg_scale > 0:
                uncond_embeds = self._embeds_uncond.expand(batch_size, -1, -1)
                uncond_pooled = self._pooled_uncond.expand(batch_size, -1)
                v_uncond = self._v_pred(latent, sigma_val, uncond_embeds, uncond_pooled)
                # 组合 Velocity
                v = v_uncond + self.cfg_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            return latent - sigma_val * v
        elif self.mode == "nfsd":
            embeds_neg = self._embeds_neg.expand(batch_size, -1, -1)
            pooled_neg = self._pooled_neg.expand(batch_size, -1)
            with torch.autocast(
                device_type=str(self.device).split(":")[0], dtype=torch.float16
            ):
                v_cond = self._v_pred(latent, sigma_val, embeds, pooled)
                v_neg = self._v_pred(latent, sigma_val, embeds_neg, pooled_neg)
            grad_v = 0.5 * (-v_cond.float() + v_neg.float())
            return latent + sigma_val * grad_v

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# ============================================================
# 2. SD3Autoencoder
# ============================================================


class SD3Autoencoder:
    """SD3 VAE wrapper. Same 16-channel VAE with shift_factor as Flux."""

    def __init__(self, pipe: StableDiffusion3Pipeline, resolution: int = 1024):
        self.vae = pipe.vae
        self.device = pipe.device
        self.scaling_factor = pipe.vae.config.scaling_factor
        self.shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0)
        self.resolution = resolution

    def encode(self, image: Image.Image) -> torch.Tensor:
        img_tensor = self._preprocess(image)
        latent = self.vae.encode(img_tensor)["latent_dist"].mean
        latent = (latent - self.shift_factor) * self.scaling_factor
        return latent

    def decode(self, latent: torch.Tensor) -> Image.Image:
        latent = latent.to(self.vae.dtype)
        latent = latent / self.scaling_factor + self.shift_factor
        img_tensor = self.vae.decode(latent)["sample"]
        return self._postprocess(img_tensor)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize(
            (self.resolution, self.resolution), Image.LANCZOS
        )
        arr = np.array(image).astype(np.float32) / 255.0
        arr = 2.0 * arr - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device, dtype=self.vae.dtype)

    def _postprocess(self, tensor: torch.Tensor) -> Image.Image:
        arr = tensor.squeeze(0).float().permute(1, 2, 0).cpu().numpy()
        arr = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))

    def eval(self):
        pass


# ============================================================
# 3. SD3GFMPipeline: Rectified Flow ODE Integrator
# ============================================================


class SD3GFMPipeline:
    """
    GFM pipeline on SD3 Medium (2B params).

    Same ODE math as Flux wrapper, but ~6x faster.
    No dynamic_shifting, no packing, no guidance embedding.
    """

    def __init__(
        self,
        pipe: StableDiffusion3Pipeline,
        n_inference_steps: int = 50,
        captioner: Optional[BLIPCaptioner] = None,
        resolution: int = 1024,
        mode: str = "nfsd",
    ):
        self.pipe = pipe
        self.device = pipe.device
        self.n_inference_steps = n_inference_steps
        self.captioner = captioner
        self.resolution = resolution

        pipe.transformer.requires_grad_(False)
        if pipe.text_encoder is not None:
            pipe.text_encoder.requires_grad_(False)
        if pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.requires_grad_(False)
        if getattr(pipe, "text_encoder_3", None) is not None:
            pipe.text_encoder_3.requires_grad_(False)
        pipe.vae.requires_grad_(False)

        self.model = SD3Model(pipe, mode=mode)
        self.autoencoder = SD3Autoencoder(pipe, resolution=resolution)

        # Precompute unconditional embeddings for CFG
        self._embeds_uncond, self._pooled_uncond = self.model._encode_prompt("")

    @classmethod
    def load(
        cls,
        model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        device: str = "cuda",
        dtype=torch.float16,
        cache_dir: Optional[str] = None,
        load_blip: bool = True,
        caption_model_id: str = "Salesforce/blip-image-captioning-large",
        resolution: int = 1024,
        drop_t5: bool = True,
        n_inference_steps: int = 50,
    ) -> "SD3GFMPipeline":
        """Load SD3 pipeline.

        Args:
            drop_t5: If True, skip T5-xxl encoder to save ~10GB VRAM.
                     Two CLIP encoders are sufficient for GFM.
        """
        extra_kwargs = {}
        if drop_t5:
            extra_kwargs["text_encoder_3"] = None
            extra_kwargs["tokenizer_3"] = None

        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, torch_dtype=dtype, cache_dir=cache_dir, **extra_kwargs
        )
        pipe.to(device)
        captioner = (
            BLIPCaptioner(
                model_id=caption_model_id,
                device=device,
                cache_dir=cache_dir,
            )
            if load_blip
            else None
        )
        return cls(
            pipe,
            n_inference_steps=n_inference_steps,
            captioner=captioner,
            resolution=resolution,
        )

    def auto_caption(self, image: Image.Image) -> str:
        return self.captioner.caption(image)

    def unload_captioner(self):
        if self.captioner is not None:
            self.captioner.unload()

    def encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt → (embeds, pooled). No txt_ids needed for SD3."""
        return self.model._encode_prompt(prompt)

    # ------ Sigma / timestep ------

    def get_sigma_from_noise_level(self, noise_level: float) -> float:
        """Identity: noise_level = flow time t."""
        return float(np.clip(noise_level, 0.0, 1.0))

    def get_timestep(self, noise_level: float) -> float:
        return self.get_sigma_from_noise_level(noise_level)

    def get_sigma_from_timestep(self, t) -> float:
        if isinstance(t, torch.Tensor):
            return t.item()
        return float(t)

    # ------ ODE schedule ------

    def _get_flow_schedule(
        self, t_start: float, t_end: float, num_steps: int
    ) -> torch.Tensor:
        """Karras polynomial schedule for ODE integration."""
        rho = 7.0
        s = torch.linspace(0, 1, num_steps, device=self.device)
        ts = (
            t_start ** (1 / rho) + s * (t_end ** (1 / rho) - t_start ** (1 / rho))
        ) ** rho
        return ts

    # ------ Velocity computation ------

    def _get_velocity(
        self,
        z: torch.Tensor,
        t_val: float,
        embeds: torch.Tensor,
        pooled: torch.Tensor,
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        """Compute v_θ with optional CFG (standard double forward pass)."""
        if cfg_scale > 0:
            v_uncond = self.model._v_pred(
                z,
                t_val,
                self._embeds_uncond.expand(z.shape[0], -1, -1),
                self._pooled_uncond.expand(z.shape[0], -1),
            )
            v_cond = self.model._v_pred(z, t_val, embeds, pooled)
            return v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            return self.model._v_pred(z, t_val, embeds, pooled)

    # ------ ODE integrators ------

    @torch.no_grad()
    def flow_forward(
        self,
        latent_clean: torch.Tensor,
        embeds: torch.Tensor,
        pooled: torch.Tensor,
        noise_level: float,
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        """Forward inversion: x₀ → z_τ (cfg forced to 0)."""
        if noise_level == 0:
            return latent_clean
        cfg_scale = 0.0
        t_min, t_max = 1e-3, self.get_sigma_from_noise_level(noise_level)
        ts = self._get_flow_schedule(t_min, t_max, self.n_inference_steps)
        z = latent_clean.clone()
        for i in range(len(ts) - 1):
            dt = (ts[i + 1] - ts[i]).item()
            v = self._get_velocity(z, ts[i].item(), embeds, pooled, cfg_scale)
            z = z + dt * v
        return z

    @torch.no_grad()
    def flow_backward(
        self,
        latent_noisy: torch.Tensor,
        embeds: torch.Tensor,
        pooled: torch.Tensor,
        noise_level: float,
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        """Generation: z_τ → x₀."""
        if noise_level == 0:
            return latent_noisy
        t_min, t_max = 1e-3, self.get_sigma_from_noise_level(noise_level)
        ts = torch.flip(
            self._get_flow_schedule(t_min, t_max, self.n_inference_steps), [0]
        )
        z = latent_noisy.clone()
        for i in range(len(ts) - 1):
            dt = (ts[i + 1] - ts[i]).item()
            v = self._get_velocity(z, ts[i].item(), embeds, pooled, cfg_scale)
            z = z + dt * v
        t_final = ts[-1].item()
        v_final = self._get_velocity(z, t_final, embeds, pooled, cfg_scale)
        return z - t_final * v_final

    ddim_forward = flow_forward
    ddim_backward = flow_backward

    # ------ Sphere projection ------

    @staticmethod
    def project_to_sphere(x: torch.Tensor, radius: float) -> torch.Tensor:
        x_flat = x.view(x.shape[0], -1)
        norms = x_flat.norm(dim=-1, keepdim=True)
        x_flat = x_flat * (radius / (norms + 1e-8))
        return x_flat.view_as(x)

    # ------ Text inversion ------

    def text_inversion(
        self, prompt: str, latent: torch.Tensor, steps: int = 500, lr: float = 0.001
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimize text embeddings. Returns (embeds, pooled)."""
        embeds, pooled = self.encode_prompt(prompt)
        embed_opt = embeds.clone().requires_grad_(True)
        pooled_fixed = pooled.detach()
        optimizer = torch.optim.AdamW([embed_opt], lr=lr)

        for _ in tqdm(range(steps), desc="Text Inversion (SD3)"):
            optimizer.zero_grad()
            noise = torch.randn_like(latent)
            t = torch.rand(1, device=self.device).clamp(0.01, 0.99)
            z_t = (1 - t) * latent + t * noise
            v_target = noise - latent
            with torch.autocast(
                device_type=str(self.device).split(":")[0], dtype=torch.float16
            ):
                v_pred = self.model._v_pred(z_t, t.item(), embed_opt, pooled_fixed)
                loss = F.mse_loss(v_pred.float(), v_target.float())
            loss.backward()
            optimizer.step()

        return embed_opt.detach(), pooled_fixed


# ============================================================
# 4. GFM Interpolation Pipeline
# ============================================================


def run_gfm_interpolation(
    sd3_pipe: SD3GFMPipeline,
    imgA: Image.Image,
    imgB: Image.Image,
    promptA: Optional[str] = None,
    promptB: Optional[str] = None,
    noise_level: float = 0.6,
    cfg_scale: float = 0.0,
    num_steps: int = 10,
    lam: float = 1.0,
    lr: float = 0.01,
    max_iters: int = 400,
    use_text_inversion: bool = False,
    output_frames: int = 10,
    mode: str = "nfsd",
    snapshot_iters: Optional[List[int]] = None,
    method: str = "gfm",
    cond_seed: int = 20260727,
    opt_num_steps: Optional[int] = None,
    early_stop_patience: int = 0,
    early_stop_rel_delta: float = 1e-4,
    early_stop_warmup: int = 0,
    c2f_steps: Optional[List[int]] = None,
    c2f_iters: Optional[List[int]] = None,
    c2f_lrs: Optional[List[float]] = None,
    c2f_freeze_anchors: bool = False,
) -> Dict[str, Any]:
    from gfm.path.geodesic_interpolation import (
        ELInterpolator,
        SeqELInterpolator,
        SphericalInterpolator,
        EL2Interpolator,
    )

    device = sd3_pipe.device
    output_frames = num_steps
    is_coarse_to_fine = method == "gfm_c2f"
    if is_coarse_to_fine:
        c2f_steps = c2f_steps or [3, 5, 9, output_frames]
        c2f_iters = c2f_iters or [400, 300, 200, 100]
        c2f_lrs = c2f_lrs or [0.002, 0.001, 0.0005, 0.00025]
        if not (len(c2f_steps) == len(c2f_iters) == len(c2f_lrs)):
            raise ValueError("c2f_steps, c2f_iters, and c2f_lrs must have equal length")
        if not c2f_steps:
            raise ValueError("coarse-to-fine schedule must contain at least one stage")
        if c2f_steps != sorted(set(c2f_steps)):
            raise ValueError("c2f_steps must be strictly increasing")
        if c2f_steps[0] < 3 or c2f_steps[-1] > output_frames:
            raise ValueError("c2f_steps must lie between three and the output frame count")
        if any(iterations < 1 for iterations in c2f_iters):
            raise ValueError("every coarse-to-fine stage needs at least one iteration")
        if any(stage_lr <= 0 for stage_lr in c2f_lrs):
            raise ValueError("every coarse-to-fine learning rate must be positive")
        if c2f_freeze_anchors:
            for previous_steps, current_steps in zip(
                c2f_steps[:-1], c2f_steps[1:]
            ):
                if (current_steps - 1) % (previous_steps - 1) != 0:
                    raise ValueError(
                        "anchor-frozen C2F requires nested grids: each "
                        "(current_steps - 1) must be divisible by "
                        "(previous_steps - 1)"
                    )
        optimization_steps = c2f_steps[-1]
    else:
        optimization_steps = opt_num_steps or num_steps
    if optimization_steps < 3:
        raise ValueError("opt_num_steps must be at least three")
    if optimization_steps > output_frames:
        raise ValueError("opt_num_steps cannot exceed the number of output frames")
    sd3_pipe.model.mode = mode

    # 1. Caption
    if promptA is None:
        promptA = sd3_pipe.auto_caption(imgA)
        print(f'  Caption A: "{promptA}"')
    if promptB is None:
        promptB = sd3_pipe.auto_caption(imgB)
        print(f'  Caption B: "{promptB}"')
    sd3_pipe.unload_captioner()

    # 2. Encode images
    print("Encoding images...")
    latA = sd3_pipe.autoencoder.encode(imgA)
    latB = sd3_pipe.autoencoder.encode(imgB)
    print(f"  Latent shape: {latA.shape}")

    # 3. Text conditioning
    print("Encoding text...")
    if use_text_inversion:
        embedA, pooledA = sd3_pipe.text_inversion(promptA, latA)
        embedB, pooledB = sd3_pipe.text_inversion(promptB, latB)
    else:
        embedA, pooledA = sd3_pipe.encode_prompt(promptA)
        embedB, pooledB = sd3_pipe.encode_prompt(promptB)

    if method == "cond_interp":
        # Pure condition-interpolation baseline.  Every frame starts from the
        # same Gaussian latent, so the only quantity changing across the strip
        # is the linearly interpolated text conditioning.
        fractions = torch.linspace(
            0,
            1,
            num_steps,
            device=device,
            dtype=embedA.dtype,
        )
        embed_weights = fractions.view(num_steps, 1, 1)
        pooled_weights = fractions.view(num_steps, 1)
        frame_embeds = (1 - embed_weights) * embedA + embed_weights * embedB
        frame_pooled = (1 - pooled_weights) * pooledA + pooled_weights * pooledB

        generator = torch.Generator(device=device).manual_seed(cond_seed)
        shared_noise = torch.randn(
            latA.shape,
            generator=generator,
            device=device,
            dtype=latA.dtype,
        )
        frame_noise = shared_noise.repeat(num_steps, 1, 1, 1)
        print(
            f"Generating from interpolated condition embeddings "
            f"(shared noise seed={cond_seed}, CFG={cfg_scale})..."
        )
        frame_clean = sd3_pipe.flow_backward(
            frame_noise,
            frame_embeds,
            frame_pooled,
            noise_level=1.0,
            cfg_scale=cfg_scale,
        )
        images = [
            sd3_pipe.autoencoder.decode(frame_clean[index : index + 1])
            for index in tqdm(range(num_steps), desc="Decoding[cond_interp]")
        ]
        return {
            "images": images,
            "latents_noisy": frame_noise.unsqueeze(0),
            "latents_clean": frame_clean,
            "losses": [],
            "acc_norms": [],
            "f_norms": [],
            "lam": lam,
            "sphere_radius": shared_noise.flatten().norm().item(),
            "sigma_eff": 1.0,
            "promptA": promptA,
            "promptB": promptB,
            "snapshots": {},
            "iterations_run": 0,
            "early_stopped": False,
            "best_iteration": None,
            "best_loss": None,
        }

    # 4. Forward inversion
    print(f"Flow forward inversion (noise_level={noise_level})...")
    # Both endpoints use the same ODE schedule, so invert them in one batch.
    endpoint_noisy = sd3_pipe.flow_forward(
        torch.cat([latA, latB], dim=0),
        torch.cat([embedA, embedB], dim=0),
        torch.cat([pooledA, pooledB], dim=0),
        noise_level,
        cfg_scale,
    )
    latA_noisy = endpoint_noisy[0:1]
    latB_noisy = endpoint_noisy[1:2]

    normA = latA_noisy.view(-1).norm().item()
    normB = latB_noisy.view(-1).norm().item()
    sphere_radius = 0.5 * (normA + normB)
    print(f"Sphere radius: {sphere_radius:.2f}")

    # 5. Configure GFM
    sigma_eff = sd3_pipe.get_sigma_from_noise_level(noise_level)
    print(f"Effective flow time σ = {sigma_eff:.4f}")

    if not is_coarse_to_fine:
        sd3_pipe.model.set_endpoints(
            embedA,
            pooledA,
            embedB,
            pooledB,
            sigma_eff,
            waypoint_indices=list(range(1, optimization_steps - 1)),
            denominator=optimization_steps - 1,
        )
    sd3_pipe.model.set_cfg_params(cfg_scale)

    # from gfm.path.diag import ScoreDiagnostics
    # diag = ScoreDiagnostics(sd3_pipe)
    # diag.run_all(latA_noisy, latB_noisy, sigma_eff)
    # 6. Path construction (GFM optimization or SLERP baseline)
    optimize_start = latA_noisy
    optimize_end = latB_noisy
    optimize_kwargs: Dict[str, Any] = {}

    if method == "slerp":
        print(
            f"Running SLERP baseline (control steps={optimization_steps}, "
            f"output frames={output_frames}, no optimization)..."
        )
        interpolator = SphericalInterpolator(
            diffusion_model=sd3_pipe.model,
            autoencoder=sd3_pipe.autoencoder,
            device=str(device),
        )
    elif method == "gfm_c2f":
        schedule_text = ", ".join(
            f"{steps}x{iterations}@{stage_lr:g}"
            for steps, iterations, stage_lr in zip(
                c2f_steps, c2f_iters, c2f_lrs
            )
        )
        print(
            f"Running {'anchor-frozen ' if c2f_freeze_anchors else ''}"
            f"coarse-to-fine ambient GFM ({schedule_text}, "
            f"output frames={output_frames}, λ={lam})..."
        )
        interpolator = EL2Interpolator(
            diffusion_model=sd3_pipe.model,
            autoencoder=sd3_pipe.autoencoder,
            device=str(device),
        )
    elif method == "gfm_sphere":
        # Use a genuine spherical solve: equal-radius endpoints, tangent
        # acceleration/force, tangent gradients, and per-update retraction.
        optimize_start = SD3GFMPipeline.project_to_sphere(
            latA_noisy, sphere_radius
        )
        optimize_end = SD3GFMPipeline.project_to_sphere(
            latB_noisy, sphere_radius
        )
        optimize_kwargs = {
            "retract_to_sphere": True,
            "sphere_radius": sphere_radius,
        }
        print(
            f"Running spherical GFM (control steps={optimization_steps}, "
            f"output frames={output_frames}, λ={lam}, "
            f"iters={max_iters}, radius={sphere_radius:.2f})..."
        )
        interpolator = ELInterpolator(
            diffusion_model=sd3_pipe.model,
            autoencoder=sd3_pipe.autoencoder,
            device=str(device),
        )
    else:
        print(
            f"Running ambient GFM (control steps={optimization_steps}, "
            f"output frames={output_frames}, λ={lam}, "
            f"iters={max_iters})..."
        )
        interpolator = EL2Interpolator(
            diffusion_model=sd3_pipe.model,
            autoencoder=sd3_pipe.autoencoder,
            device=str(device),
        )

    # interpolator = SteinScoreInterpolator(
    #     diffusion_model=sd3_pipe.model,
    #     autoencoder=sd3_pipe.autoencoder,
    #     device=str(device),
    # )

    def _project_control_path(path: torch.Tensor) -> torch.Tensor:
        projected = path.clone()
        for path_index in range(projected.shape[1]):
            projected[:, path_index] = SD3GFMPipeline.project_to_sphere(
                projected[:, path_index], sphere_radius
            )
        return projected

    if is_coarse_to_fine:
        stage_path: Optional[torch.Tensor] = None
        stage_summaries: List[Dict[str, Any]] = []
        combined_losses: List[float] = []
        combined_acc_norms: List[float] = []
        combined_f_norms: List[float] = []
        final_stage_info: Dict[str, Any] = {}
        previous_stage_steps: Optional[int] = None

        for stage_index, (stage_steps, stage_iters, stage_lr) in enumerate(
            zip(c2f_steps, c2f_iters, c2f_lrs), start=1
        ):
            initial_path = None
            anchor_indices: List[int] = [0, stage_steps - 1]
            trainable_indices: Optional[List[int]] = None
            if stage_path is not None:
                initial_path = piecewise_slerp_resample(
                    _project_control_path(stage_path),
                    stage_steps,
                )
                if c2f_freeze_anchors:
                    refinement = (stage_steps - 1) // (
                        previous_stage_steps - 1
                    )
                    anchor_indices = [
                        index * refinement
                        for index in range(previous_stage_steps)
                    ]
                    trainable_indices = [
                        index
                        for index in range(1, stage_steps - 1)
                        if index not in anchor_indices
                    ]
            active_indices = (
                list(range(1, stage_steps - 1))
                if trainable_indices is None
                else trainable_indices
            )
            sd3_pipe.model.set_endpoints(
                embedA,
                pooledA,
                embedB,
                pooledB,
                sigma_eff,
                waypoint_indices=active_indices,
                denominator=stage_steps - 1,
            )
            print(
                f"C2F stage {stage_index}/{len(c2f_steps)}: "
                f"steps={stage_steps}, iters={stage_iters}, lr={stage_lr:g}, "
                f"trainable={stage_steps - 2 if trainable_indices is None else len(trainable_indices)}, "
                f"anchors={len(anchor_indices)}"
            )
            stage_path, stage_info = interpolator.optimize_path(
                start_latent=optimize_start,
                end_latent=optimize_end,
                num_steps=stage_steps,
                sigma=sigma_eff,
                lr=stage_lr,
                max_iters=stage_iters,
                lam=lam,
                verbose=True,
                initial_path=initial_path,
                restore_best=True,
                early_stop_patience=0,
                trainable_indices=trainable_indices,
            )
            anchor_drift = 0.0
            if c2f_freeze_anchors and initial_path is not None:
                expected_path = initial_path.clone()
                expected_path[:, 0] = optimize_start
                expected_path[:, -1] = optimize_end
                anchor_drift = (
                    stage_path[:, anchor_indices]
                    - expected_path[:, anchor_indices]
                ).abs().max().item()
                if anchor_drift != 0.0:
                    raise RuntimeError(
                        "Frozen C2F anchors changed during optimization: "
                        f"max absolute drift={anchor_drift:.6g}"
                    )
            stage_losses = stage_info.get("losses", [])
            combined_losses.extend(stage_losses)
            combined_acc_norms.extend(stage_info.get("acc_norms", []))
            combined_f_norms.extend(stage_info.get("f_norms", []))
            stage_summaries.append(
                {
                    "stage": stage_index,
                    "num_steps": stage_steps,
                    "max_iters": stage_iters,
                    "lr": stage_lr,
                    "iterations_run": stage_info.get("iterations", stage_iters),
                    "initial_loss": stage_losses[0] if stage_losses else None,
                    "last_loss": stage_losses[-1] if stage_losses else None,
                    "best_iteration": stage_info.get("best_iteration"),
                    "best_loss": stage_info.get("best_loss"),
                    "trainable_indices": stage_info.get("trainable_indices"),
                    "conditioning_fractions": [
                        index / (stage_steps - 1) for index in active_indices
                    ],
                    "anchor_indices": anchor_indices,
                    "max_anchor_drift": anchor_drift,
                }
            )
            final_stage_info = stage_info
            previous_stage_steps = stage_steps

        optimized_path = stage_path
        info = {
            "losses": combined_losses,
            "acc_norms": combined_acc_norms,
            "f_norms": combined_f_norms,
            "method": "euler_lagrange2_coarse_to_fine",
            "sigma": sigma_eff,
            "lambda": lam,
            "iterations": sum(
                summary["iterations_run"] for summary in stage_summaries
            ),
            "early_stopped": False,
            "best_iteration": final_stage_info.get("best_iteration"),
            "best_loss": final_stage_info.get("best_loss"),
            "snapshots": {},
            "c2f_stages": stage_summaries,
            "c2f_freeze_anchors": c2f_freeze_anchors,
        }
    else:
        optimized_path, info = interpolator.optimize_path(
            start_latent=optimize_start,
            end_latent=optimize_end,
            num_steps=optimization_steps,
            sigma=sigma_eff,
            lr=lr,
            max_iters=max_iters,
            lam=lam,
            verbose=True,
            init_with_slerp=False,
            snapshot_iters=snapshot_iters,
            early_stop_patience=early_stop_patience,
            early_stop_rel_delta=early_stop_rel_delta,
            early_stop_warmup=early_stop_warmup,
            **optimize_kwargs,
        )

    def _project_and_decode(path: torch.Tensor, label: str):
        path = _project_control_path(path)

        if output_frames != path.shape[1]:
            latents_noisy = piecewise_slerp_resample(path, output_frames)
        else:
            latents_noisy = path

        n_frames = latents_noisy.shape[1]
        print(f"Flow backward [{label}] ({n_frames} frames, batched)...")
        latents_clean = [latA]
        images = [imgA]

        n_inner = n_frames - 2
        if n_inner > 0:
            fractions = torch.linspace(
                1.0 / (n_frames - 1),
                (n_frames - 2.0) / (n_frames - 1),
                n_inner,
                device=device,
                dtype=embedA.dtype,
            )
            embed_weights = fractions.view(n_inner, 1, 1)
            pooled_weights = fractions.view(n_inner, 1)
            inner_embeds = (1 - embed_weights) * embedA + embed_weights * embedB
            inner_pooled = (1 - pooled_weights) * pooledA + pooled_weights * pooledB
            inner_noisy = latents_noisy[:, 1:-1].reshape(
                n_inner, *latents_noisy.shape[2:]
            )
            inner_clean = sd3_pipe.flow_backward(
                inner_noisy,
                inner_embeds,
                inner_pooled,
                noise_level,
                cfg_scale,
            )
            for index in tqdm(range(n_inner), desc=f"Decoding[{label}]"):
                latent = inner_clean[index : index + 1]
                latents_clean.append(latent)
                images.append(sd3_pipe.autoencoder.decode(latent))

        images.append(imgB)
        latents_clean.append(latB)
        return images, latents_noisy, latents_clean

    final_iteration = info.get("iterations", max_iters)
    output_images, output_latents_noisy, output_latents_clean = _project_and_decode(
        optimized_path, f"iter{final_iteration}"
    )

    snapshots_out: Dict[int, List[Image.Image]] = {}
    raw_snapshots = info.get("snapshots", {}) or {}
    for it in sorted(raw_snapshots.keys()):
        if it == final_iteration:
            snapshots_out[it] = output_images
            continue
        imgs, _, _ = _project_and_decode(raw_snapshots[it], f"iter{it}")
        snapshots_out[it] = imgs

    return {
        "images": output_images,
        "latents_noisy": output_latents_noisy,
        "latents_clean": torch.cat(output_latents_clean, dim=0),
        "losses": info.get("losses", []),
        "acc_norms": info.get("acc_norms", []),
        "f_norms": info.get("f_norms", []),
        "lam": lam,
        "sphere_radius": sphere_radius,
        "sigma_eff": sigma_eff,
        "promptA": promptA,
        "promptB": promptB,
        "snapshots": snapshots_out,
        "iterations_run": final_iteration,
        "early_stopped": info.get("early_stopped", False),
        "best_iteration": info.get("best_iteration"),
        "best_loss": info.get("best_loss"),
        "c2f_stages": info.get("c2f_stages"),
        "c2f_freeze_anchors": info.get("c2f_freeze_anchors", False),
    }


# ============================================================
# 5. Visualization
# ============================================================


def save_interpolation_strip(
    images: List[Image.Image], save_path: str, size=(512, 512), padding=10
):
    n = len(images)
    w, h = size
    pw, ph = w + 2 * padding, h + 2 * padding
    strip = Image.new("RGB", (pw * n, ph), (255, 255, 255))
    for i, img in enumerate(images):
        strip.paste(img.resize(size), (i * pw + padding, padding))
    strip.save(save_path)
    print(f"Saved strip to {save_path}")


def save_frames(images: List[Image.Image], output_dir: str):
    import os

    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(output_dir, f"frame_{i:03d}.png"))
    print(f"Saved {len(images)} frames to {output_dir}")


def save_optimization_curves(
    losses: List[float],
    acc_norms: List[float],
    f_norms: List[float],
    lam: float,
    output_dir: str,
    snapshot_iters: Optional[List[int]] = None,
):
    import os
    import csv

    os.makedirs(output_dir, exist_ok=True)
    n = min(len(losses), len(acc_norms), len(f_norms))
    if n == 0:
        return

    csv_path = os.path.join(output_dir, "opt_curves.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "loss", "acc_rms", "f_rms", "lam_f_rms", "ratio"])
        for i in range(n):
            lam_f = lam * f_norms[i]
            ratio = lam_f / (acc_norms[i] + 1e-12)
            w.writerow([i, losses[i], acc_norms[i], f_norms[i], lam_f, ratio])
    print(f"Saved {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    iters = list(range(n))
    lam_f = [lam * x for x in f_norms[:n]]
    ratios = [lam_f[i] / (acc_norms[i] + 1e-12) for i in range(n)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(iters, losses[:n], color="black")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("iter")
    axes[0].set_ylabel("loss")
    axes[0].set_title("loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iters, acc_norms[:n], label="‖acc‖", color="C0")
    axes[1].plot(iters, lam_f, label=f"‖λf‖ (λ={lam:g})", color="C3")
    axes[1].plot(iters, f_norms[:n], label="‖f‖", color="C1", linestyle="--", alpha=0.7)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("iter")
    axes[1].set_ylabel("RMS")
    axes[1].set_title("force vs acceleration")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(iters, ratios, color="C2")
    axes[2].axhline(1.0, color="gray", linestyle=":", alpha=0.7)
    axes[2].set_xlabel("iter")
    axes[2].set_ylabel("‖λf‖ / ‖acc‖")
    axes[2].set_title("balance ratio")
    axes[2].grid(True, alpha=0.3)

    for it in snapshot_iters or []:
        if it < n:
            for ax in axes:
                ax.axvline(it, color="k", linestyle=":", alpha=0.4)

    fig.tight_layout()
    png_path = os.path.join(output_dir, "opt_curves.png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"Saved {png_path}")


# ============================================================
# 6. Entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GFM with SD3 Medium")
    parser.add_argument("--imgA", type=str, required=True)
    parser.add_argument("--imgB", type=str, required=True)
    parser.add_argument("--promptA", type=str, default=None)
    parser.add_argument("--promptB", type=str, default=None)
    parser.add_argument("--noise_level", type=float, default=0.6)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max_iters", type=int, default=800)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--n_inference_steps", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="./output/sd3_interp")
    parser.add_argument("--use_text_inv", action="store_true")
    parser.add_argument("--no_blip", action="store_true")
    parser.add_argument(
        "--keep_t5", action="store_true", help="Keep T5-xxl (uses ~10GB more VRAM)"
    )
    parser.add_argument("--cache_dir", type=str, default="/cns/USERS/zzhixuan/weights")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
    )
    parser.add_argument("--mode", type=str, default="nfsd", choices=["nfsd", "denoise"])
    parser.add_argument(
        "--snapshot_iters",
        type=str,
        default="",
        help="Comma-separated iters to snapshot mid-optimization, e.g. '200,400,600'. "
        "Each snapshot is denoised and saved to <output_dir>/snap_iter_<K>/.",
    )
    args = parser.parse_args()

    snapshot_iters: List[int] = []
    if args.snapshot_iters.strip():
        snapshot_iters = sorted(
            {int(s) for s in args.snapshot_iters.split(",") if s.strip()}
        )

    if args.no_blip and (args.promptA is None or args.promptB is None):
        parser.error("--no_blip requires both --promptA and --promptB")

    print(f"Loading SD3 from {args.model_id}...")
    sd3_pipe = SD3GFMPipeline.load(
        model_id=args.model_id,
        device=args.device,
        cache_dir=args.cache_dir,
        load_blip=not args.no_blip,
        resolution=args.resolution,
        drop_t5=not args.keep_t5,
        n_inference_steps=args.n_inference_steps,
    )

    imgA = Image.open(args.imgA)
    imgB = Image.open(args.imgB)

    results = run_gfm_interpolation(
        sd3_pipe,
        imgA,
        imgB,
        promptA=args.promptA,
        promptB=args.promptB,
        noise_level=args.noise_level,
        cfg_scale=args.cfg_scale,
        num_steps=args.num_steps,
        lam=args.lam,
        lr=args.lr,
        max_iters=args.max_iters,
        use_text_inversion=args.use_text_inv,
        mode=args.mode,
        snapshot_iters=snapshot_iters,
    )

    import os

    os.makedirs(args.output_dir, exist_ok=True)
    save_interpolation_strip(
        results["images"], os.path.join(args.output_dir, "strip.png")
    )
    save_frames(results["images"], os.path.join(args.output_dir, "frames"))

    for it, imgs in results.get("snapshots", {}).items():
        if it == args.max_iters:
            continue
        snap_dir = os.path.join(args.output_dir, f"snap_iter_{it}")
        os.makedirs(snap_dir, exist_ok=True)
        save_interpolation_strip(imgs, os.path.join(snap_dir, "strip.png"))
        save_frames(imgs, os.path.join(snap_dir, "frames"))

    save_optimization_curves(
        losses=results.get("losses", []),
        acc_norms=results.get("acc_norms", []),
        f_norms=results.get("f_norms", []),
        lam=results.get("lam", 1.0),
        output_dir=args.output_dir,
        snapshot_iters=snapshot_iters,
    )
    print(
        f"\nσ_eff={results['sigma_eff']:.4f}, "
        f"sphere_r={results['sphere_radius']:.2f}"
    )
    if results["losses"]:
        print(f"Final GFM loss: {results['losses'][-1]:.6f}")
