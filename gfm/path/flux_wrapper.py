"""
Flux.1 Rectified Flow Wrapper for Geodesic Force Matching (GFM)
Replaces SD2.1 EDM/DDIM with Flux.1 flow matching ODE integration.

=== Key Mathematical Differences from SD2.1 ===

SD2.1 (ε-prediction, VP schedule):
    z_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε       (VP noising)
    model predicts ε                          (noise prediction)
    score ∇ log p_t ≈ -ε/σ                   (where σ = √((1-ᾱ)/ᾱ))
    Requires VP ↔ EDM space conversions (c_in, c_out, etc.)

Flux.1 (v-prediction, rectified flow):
    z_t = (1-t) · x₀ + t · ε,  t ∈ [0,1]    (linear interpolation)
    model predicts v = ε - x₀                 (velocity / flow direction)
    x₀ = z_t - t · v                          (denoised prediction)
    score ∇ log p_t ≈ -v / t                  (from ε = z_t + (1-t)·v)
    ODE: dz/dt = v_θ(z_t, t)                  (no space conversions needed)

=== NFSD Adaptation (ε → v) ===

SD:   grad_ε = ½(-ε_c + ε_n),   x_pred = z + σ² · grad_ε
Flux: grad_v = ½(-v_c + v_n),   x_pred = z_t + t · grad_v

Derivation:
    force = -(z_t - x_pred) / t²             (GFM force definition)
    NFSD score ≈ ½(score_c - score_n) = ½(-v_c + v_n) / t
    ⟹ force = grad_v / t
    ⟹ x_pred = z_t + t · grad_v             (σ² in SD becomes t in FM)
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any
from diffusers import FluxPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm


# ============================================================
# 0. BLIP Captioner (Unchanged)
# ============================================================

class BLIPCaptioner:
    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-base",
                 device: str = "cuda", cache_dir: Optional[str] = None):
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_id, cache_dir=cache_dir
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def caption(self, image: Image.Image, max_new_tokens: int = 50) -> str:
        inputs = self.processor(image.convert("RGB"), return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(output_ids[0], skip_special_tokens=True)

    def unload(self):
        self.model.cpu()
        torch.cuda.empty_cache()


# ============================================================
# 1. FluxModel: v-prediction wrapper for GFM
# ============================================================

class FluxModel(torch.nn.Module):
    """
    Wraps Flux.1 transformer for GFM's model interface.

    GFM calls: x_pred = model(z_t, sigma_tensor, class_labels)
    Then:       force = -(z_t - x_pred) / sigma²

    In rectified flow:
        x₀ = z_t - t · v_θ(z_t, t)
        force = v_θ / t ≈ -∇ log p_t(z_t)
    """

    def __init__(self, pipe: FluxPipeline, mode: str = 'nfsd',
                 guidance_scale: float = 3.5):
        super().__init__()
        self.transformer = pipe.transformer
        self.device = pipe.device
        self.mode = mode
        self.pipe = pipe
        self.guidance_scale = guidance_scale

        # VAE geometry
        self._vae_scale_factor = pipe.vae_scale_factor  # typically 8
        self._num_channels = pipe.transformer.config.in_channels  # 64 packed
        # Flux VAE has 16 latent channels, packed 2×2 → 64-dim tokens
        self._latent_channels = self._num_channels // 4  # 16

        # --- Precompute fixed embeddings ---
        # Unconditional (empty prompt)
        self._prompt_embeds_uncond, self._pooled_uncond, self._text_ids_uncond = (
            self._encode_prompt("")
        )
        # Negative prompt for NFSD
        self._prompt_embeds_neg, self._pooled_neg, self._text_ids_neg = (
            self._encode_prompt(
                "A doubling image, unrealistic, artifacts, distortions, "
                "unnatural blending, ghosting effects, overlapping edges, "
                "harsh transitions, motion blur, poor resolution, low detail"
            )
        )

        # --- Conditioning state (set before forward) ---
        self._prompt_embeds_cond = None
        self._pooled_cond = None
        self._text_ids_cond = None
        self._sigma = None

        # --- Dynamic interpolation state ---
        self._embedA = None
        self._embedB = None
        self._pooledA = None
        self._pooledB = None
        self._text_ids_interp = None
        self._num_steps = None

    # ------ Text encoding ------

    def _encode_prompt(self, prompt: str):
        """Encode a prompt with Flux's dual encoders (CLIP + T5).

        Returns:
            prompt_embeds:  [1, seq_len, dim]  T5 hidden states
            pooled_embeds:  [1, dim]           CLIP pooled
            text_ids:       [seq_len, 3]       positional IDs (2D, no batch dim)
        """
        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self.device,
        )
        # Newer diffusers expects txt_ids as 2D [seq_len, 3], not 3D
        if text_ids.dim() == 3:
            text_ids = text_ids.squeeze(0)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    # ------ Conditioning interface ------

    def set_conditioning(self, prompt_embeds, pooled_embeds, text_ids, sigma: float):
        """Set static text conditioning and noise level."""
        self._prompt_embeds_cond = prompt_embeds
        self._pooled_cond = pooled_embeds
        self._text_ids_cond = text_ids
        self._sigma = sigma
        # Clear dynamic state
        self._embedA = None

    def set_endpoints(self, embedA, pooledA, text_idsA,
                      embedB, pooledB, text_idsB,
                      sigma: float, num_steps: int):
        """Set endpoints for GFM dynamic embedding interpolation."""
        self._embedA = embedA
        self._pooledA = pooledA
        self._embedB = embedB
        self._pooledB = pooledB
        # Use A's text_ids as template (positions are the same)
        self._text_ids_interp = text_idsA
        self._sigma = sigma
        self._num_steps = num_steps
        # Clear static state
        self._prompt_embeds_cond = None

    # ------ Latent packing / unpacking ------

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] → [B, (H/2)·(W/2), C·4]  (2×2 patch packing)."""
        B, C, H, W = latents.shape
        latents = latents.view(B, C, H // 2, 2, W // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)              # [B, H/2, W/2, C, 2, 2]
        latents = latents.reshape(B, (H // 2) * (W // 2), C * 4) # [B, S, D]
        return latents

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """[B, S, D] → [B, C, H, W]  (inverse of packing)."""
        B = latents.shape[0]
        C = latents.shape[-1] // 4
        latents = latents.reshape(B, H // 2, W // 2, C, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)  # [B, C, H/2, 2, W/2, 2]
        latents = latents.reshape(B, C, H, W)
        return latents

    # ------ Positional IDs ------

    def _prepare_img_ids(self, batch_size: int, H: int, W: int) -> torch.Tensor:
        """Prepare image positional IDs for the transformer.

        H, W are latent spatial dims (before packing).
        Returns 2D tensor [num_patches, 3] (no batch dim).
        """
        latent_h = H // 2  # packed height
        latent_w = W // 2  # packed width
        img_ids = self.pipe._prepare_latent_image_ids(
            batch_size, latent_h, latent_w, self.device, torch.float32
        )
        # Ensure 2D [num_patches, 3] — newer diffusers expects no batch dim
        if img_ids.dim() == 3:
            img_ids = img_ids.squeeze(0)
        return img_ids

    # ------ Core v-prediction ------

    def _v_pred(self, latent: torch.Tensor, t_val: float,
                prompt_embeds: torch.Tensor, pooled_embeds: torch.Tensor,
                text_ids: torch.Tensor) -> torch.Tensor:
        """
        Single forward pass through Flux transformer → velocity prediction.

        Args:
            latent:        [B, C, H, W] noisy latent at flow time t
            t_val:         scalar t ∈ [0, 1]
            prompt_embeds: [B, seq, dim] T5 embeddings
            pooled_embeds: [B, dim]      CLIP pooled embeddings
            text_ids:      [seq, 3]      text positional IDs (2D, no batch dim)

        Returns:
            v_pred: [B, C, H, W] predicted velocity (ε - x₀)
        """
        B, C, H, W = latent.shape

        # Pack latents for the transformer
        packed = self._pack_latents(latent)  # [B, S, D]

        # Flux scheduler convention: timestep = t / 1000? 
        # Actually in diffusers FluxPipeline, sigma is passed directly as timestep
        # and the transformer internally does 1000 * timestep for positional encoding.
        # So we pass the raw t ∈ [0,1].
        timestep = torch.full((B,), t_val, device=self.device, dtype=latent.dtype)

        # Image positional IDs
        img_ids = self._prepare_img_ids(B, H, W)

        # Guidance embedding (Flux-dev only; ignored by schnell)
        guidance = torch.full((B,), self.guidance_scale,
                              device=self.device, dtype=latent.dtype)

        # Expand embeddings to batch if needed
        if prompt_embeds.shape[0] == 1 and B > 1:
            prompt_embeds = prompt_embeds.expand(B, -1, -1)
        if pooled_embeds.shape[0] == 1 and B > 1:
            pooled_embeds = pooled_embeds.expand(B, -1)
        # text_ids is 2D [seq_len, 3] — no batch expansion needed,
        # the transformer broadcasts it internally

        # --- Transformer forward ---
        output = self.transformer(
            hidden_states=packed,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            img_ids=img_ids,
            txt_ids=text_ids,
            return_dict=False,
        )[0]  # [B, S, D]

        # Unpack back to spatial
        v_pred = self._unpack_latents(output, H, W)  # [B, C, H, W]
        return v_pred

    # ------ GFM interface: forward → x_pred ------

    @torch.no_grad()
    def forward(self, latent: torch.Tensor, sigma: torch.Tensor,
                class_labels=None) -> torch.Tensor:
        """
        GFM-compatible forward: returns denoised x_pred from noisy z_t.

        Rectified flow: x₀ = z_t − t · v_θ(z_t, t)
        """
        batch_size = latent.shape[0]
        sigma_val = sigma[0].item() if sigma.dim() > 0 else sigma.item()

        # =====================================================
        # Resolve conditioning (static vs dynamic interpolation)
        # =====================================================
        if self._embedA is not None:
            # Dynamic interpolation mode for GFM inner points
            n_inner = self._num_steps - 2
            if n_inner > 0 and batch_size % n_inner == 0:
                B = batch_size // n_inner
                s = torch.linspace(
                    1.0 / (self._num_steps - 1),
                    float(n_inner) / (self._num_steps - 1),
                    n_inner, device=self.device
                )
                s = s.repeat(B)  # [B*n_inner]

                # Interpolate T5 embeddings  [B*n_inner, seq, dim]
                s_t5 = s.view(-1, 1, 1)
                prompt_embeds = (1.0 - s_t5) * self._embedA + s_t5 * self._embedB

                # Interpolate CLIP pooled  [B*n_inner, dim]
                s_pool = s.view(-1, 1)
                pooled = (1.0 - s_pool) * self._pooledA + s_pool * self._pooledB

                # text_ids is 2D [seq_len, 3] — shared across batch
                text_ids = self._text_ids_interp
            else:
                # Fallback: average
                prompt_embeds = (0.5 * (self._embedA + self._embedB)).expand(batch_size, -1, -1)
                pooled = (0.5 * (self._pooledA + self._pooledB)).expand(batch_size, -1)
                text_ids = self._text_ids_interp

        elif self._prompt_embeds_cond is not None:
            prompt_embeds = self._prompt_embeds_cond.expand(batch_size, -1, -1)
            pooled = self._pooled_cond.expand(batch_size, -1)
            text_ids = self._text_ids_cond
        else:
            raise ValueError("No conditioning set. Call set_conditioning() or set_endpoints().")

        # =====================================================
        # Mode-specific x_pred computation
        # =====================================================
        if self.mode == 'denoise':
            v = self._v_pred(latent, sigma_val, prompt_embeds, pooled, text_ids)
            # Rectified flow denoising: x₀ = z_t - t · v
            x_pred = latent - sigma_val * v
            return x_pred

        elif self.mode == 'nfsd':
            # NFSD: combine conditional and negative predictions
            prompt_neg = self._prompt_embeds_neg.expand(batch_size, -1, -1)
            pooled_neg = self._pooled_neg.expand(batch_size, -1)
            tid_neg = self._text_ids_neg  # 2D [seq_len, 3], no batch expand

            with torch.autocast(device_type=str(self.device).split(':')[0],
                                dtype=torch.bfloat16):
                v_cond = self._v_pred(latent, sigma_val, prompt_embeds, pooled, text_ids)
                v_neg = self._v_pred(latent, sigma_val, prompt_neg, pooled_neg, tid_neg)

            # NFSD gradient in v-space  (see module docstring for derivation)
            # grad_v = ½(-v_c + v_n)
            # x_pred = z_t + t · grad_v
            grad_v = 0.5 * (-v_cond.float() + v_neg.float())
            x_pred = latent + sigma_val * grad_v
            return x_pred

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# ============================================================
# 2. FluxAutoencoder: handles Flux VAE (16-ch + shift_factor)
# ============================================================

class FluxAutoencoder:
    """
    Flux VAE wrapper.

    Flux's VAE differs from SD in two ways:
        1. 16 latent channels (vs 4 in SD)
        2. shift_factor: latent = (raw - shift) * scale   (encode)
                         raw = latent / scale + shift      (decode)
    """

    def __init__(self, pipe: FluxPipeline, resolution: int = 1024):
        self.vae = pipe.vae
        self.device = pipe.device
        self.scaling_factor = pipe.vae.config.scaling_factor
        self.shift_factor = getattr(pipe.vae.config, 'shift_factor', 0.0)
        self.resolution = resolution

    def encode(self, image: Image.Image) -> torch.Tensor:
        img_tensor = self._preprocess(image)
        latent_dist = self.vae.encode(img_tensor)['latent_dist']
        latent = latent_dist.mean  # deterministic
        # Apply Flux normalization: (raw - shift) * scale
        latent = (latent - self.shift_factor) * self.scaling_factor
        return latent

    def decode(self, latent: torch.Tensor) -> Image.Image:
        # Invert Flux normalization: raw = latent / scale + shift
        latent = latent / self.scaling_factor + self.shift_factor
        img_tensor = self.vae.decode(latent)['sample']
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
        """Compatibility stub for GFM code."""
        pass


# ============================================================
# 3. FluxGFMPipeline: Rectified Flow ODE Integrator
# ============================================================

class FluxGFMPipeline:
    """
    GFM pipeline built on Flux.1.

    Forward/backward are now simple ODE integrations:
        dz/dt = v_θ(z_t, t)

    Forward inversion:  z₀ = x₀  →  zτ   (integrate t: 0 → τ)
    Backward generation: zτ       →  z₀   (integrate t: τ → 0)

    No VP↔EDM conversions, no alphas_cumprod, no c_in/c_out.
    """

    def __init__(self, pipe: FluxPipeline, n_inference_steps: int = 50,
                 captioner: Optional[BLIPCaptioner] = None,
                 resolution: int = 1024,
                 guidance_scale: float = 3.5):
        self.pipe = pipe
        self.device = pipe.device
        self.n_inference_steps = n_inference_steps
        self.captioner = captioner
        self.resolution = resolution

        # Freeze everything
        pipe.transformer.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.requires_grad_(False)
        pipe.vae.requires_grad_(False)

        self.model = FluxModel(pipe, mode='nfsd', guidance_scale=guidance_scale)
        self.autoencoder = FluxAutoencoder(pipe, resolution=resolution)

        # Precompute unconditional embeddings for CFG
        self._prompt_embeds_uncond, self._pooled_uncond, self._text_ids_uncond = (
            self.model._encode_prompt("")
        )

    @classmethod
    def load(cls, model_id: str = "black-forest-labs/FLUX.1-dev",
             device: str = "cuda", dtype=torch.bfloat16,
             cache_dir: Optional[str] = None,
             load_blip: bool = True,
             resolution: int = 1024,
             guidance_scale: float = 3.5) -> "FluxGFMPipeline":
        pipe = FluxPipeline.from_pretrained(
            model_id, torch_dtype=dtype, cache_dir=cache_dir
        )
        pipe.to(device)
        captioner = BLIPCaptioner(device=device, cache_dir=cache_dir) if load_blip else None
        return cls(pipe, captioner=captioner, resolution=resolution,
                   guidance_scale=guidance_scale)

    def auto_caption(self, image: Image.Image) -> str:
        return self.captioner.caption(image)

    def unload_captioner(self):
        if self.captioner is not None:
            self.captioner.unload()

    def encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode prompt → (T5_embeds, CLIP_pooled, text_ids)."""
        return self.model._encode_prompt(prompt)

    # ------ Sigma / timestep utilities ------

    def get_sigma_from_noise_level(self, noise_level: float) -> float:
        """
        Map noise_level ∈ [0, 1] → flow time t ∈ [0, 1].

        In rectified flow, t IS the noise level directly:
            z_t = (1-t)·x₀ + t·ε
            t=0 → clean, t=1 → pure noise

        We apply Flux's time-shift formula to match the model's
        internal schedule without needing the scheduler (which requires
        `mu` for dynamic shifting and image resolution awareness).

        Time shift (from Flux/SD3 paper):
            t_shifted = shift / (shift + (1/t - 1))
        where shift ≈ 3.0 for Flux-dev at 1024×1024.
        This concentrates sampling steps at higher noise levels where
        the model has more to learn.
        """
        if noise_level <= 0:
            return 0.0
        if noise_level >= 1:
            return 1.0

        # Flux-dev time shift parameter (resolution-dependent)
        # For 1024×1024: shift ≈ 3.0; for 512×512: shift ≈ 1.5
        # This matches what the scheduler would compute internally.
        image_seq_len = (self.resolution // self.pipe.vae_scale_factor // 2) ** 2
        base_shift = 0.5
        max_shift = 1.15
        # mu = base_shift + (max_shift - base_shift) * (image_seq_len / max_seq_len)
        # For Flux, max_seq_len is typically 4096 (1024×1024 packed)
        mu = base_shift + (max_shift - base_shift) * min(image_seq_len / 4096.0, 1.0)
        shift = math.exp(mu)

        # Apply time shift: t_shifted = shift / (shift + (1/t - 1))
        t_shifted = shift / (shift + (1.0 / noise_level - 1.0))
        return float(t_shifted)

    # Compatibility aliases
    def get_timestep(self, noise_level: float) -> float:
        """Return sigma (= flow time t) for a given noise_level."""
        return self.get_sigma_from_noise_level(noise_level)

    def get_sigma_from_timestep(self, t) -> float:
        """Identity: in flow matching, sigma ≡ t."""
        if isinstance(t, torch.Tensor):
            return t.item()
        return float(t)

    # ------ ODE schedule ------

    def _get_flow_schedule(self, t_start: float, t_end: float,
                           num_steps: int) -> torch.Tensor:
        """
        Karras-like polynomial schedule for smooth ODE integration.

        For flow matching, t ∈ [0, 1] directly (no sigma conversion).
        """
        rho = 7.0
        s = torch.linspace(0, 1, num_steps, device=self.device)
        ts = (t_start ** (1 / rho) + s * (t_end ** (1 / rho) - t_start ** (1 / rho))) ** rho
        return ts

    # ------ Core velocity computation ------

    def _get_velocity(self, z: torch.Tensor, t_val: float,
                      prompt_embeds: torch.Tensor,
                      pooled_embeds: torch.Tensor,
                      text_ids: torch.Tensor,
                      cfg_scale: float = 0.0) -> torch.Tensor:
        """
        Compute v_θ(z_t, t) with optional classifier-free guidance.

        In Flux-dev, the model has an internal guidance embedding, so
        traditional CFG (double forward pass) is optional.

        If cfg_scale > 0: v = v_uncond + cfg · (v_cond - v_uncond)
        If cfg_scale = 0: v = v_cond  (rely on internal guidance embedding)
        """
        if cfg_scale > 0:
            # Double forward pass for traditional CFG
            v_uncond = self.model._v_pred(
                z, t_val,
                self._prompt_embeds_uncond.expand(z.shape[0], -1, -1),
                self._pooled_uncond.expand(z.shape[0], -1),
                self._text_ids_uncond,  # 2D [seq_len, 3], no batch expand
            )
            v_cond = self.model._v_pred(z, t_val, prompt_embeds, pooled_embeds, text_ids)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            # Single pass (Flux-dev internal guidance handles quality)
            v = self.model._v_pred(z, t_val, prompt_embeds, pooled_embeds, text_ids)
        return v

    # ------ ODE Integrators ------

    @torch.no_grad()
    def flow_forward(self, latent_clean: torch.Tensor,
                     prompt_embeds: torch.Tensor,
                     pooled_embeds: torch.Tensor,
                     text_ids: torch.Tensor,
                     noise_level: float,
                     cfg_scale: float = 0.0) -> torch.Tensor:
        """
        Forward inversion: x₀ → z_τ via probability flow ODE (Euler).

        dz/dt = v_θ(z_t, t),  integrate t: t_min → τ

        CRITICAL: cfg_scale forced to 0 for stable inversion
        (same rationale as DDIM inversion in SD).
        """
        if noise_level == 0:
            return latent_clean

        cfg_scale = 0.0  # Force unconditional for stable inversion

        sigma_max = self.get_sigma_from_noise_level(noise_level)
        t_min = 1e-3  # Avoid numerical issues at t=0
        t_max = sigma_max

        # Build schedule: t_min → t_max (forward in time = adding noise)
        ts = self._get_flow_schedule(t_min, t_max, self.n_inference_steps)

        z = latent_clean.clone()  # z₀ ≈ x₀ (at t ≈ 0)

        for i in range(len(ts) - 1):
            t_i = ts[i].item()
            dt = (ts[i + 1] - ts[i]).item()
            v = self._get_velocity(z, t_i, prompt_embeds, pooled_embeds,
                                   text_ids, cfg_scale)
            z = z + dt * v  # Forward Euler

        return z

    @torch.no_grad()
    def flow_backward(self, latent_noisy: torch.Tensor,
                      prompt_embeds: torch.Tensor,
                      pooled_embeds: torch.Tensor,
                      text_ids: torch.Tensor,
                      noise_level: float,
                      cfg_scale: float = 0.0) -> torch.Tensor:
        """
        Generation: z_τ → x₀ via probability flow ODE (Euler, reverse).

        dz/dt = v_θ(z_t, t),  integrate t: τ → t_min

        With Flux-dev's internal guidance, cfg_scale=0 is usually fine.
        Set cfg_scale > 0 for additional quality via traditional CFG.
        """
        if noise_level == 0:
            return latent_noisy

        sigma_max = self.get_sigma_from_noise_level(noise_level)
        t_min = 1e-3
        t_max = sigma_max

        # Build schedule: t_max → t_min (backward = denoising)
        ts = self._get_flow_schedule(t_min, t_max, self.n_inference_steps)
        ts = torch.flip(ts, [0])  # Reverse: large t → small t

        z = latent_noisy.clone()

        for i in range(len(ts) - 1):
            t_i = ts[i].item()
            dt = (ts[i + 1] - ts[i]).item()  # Negative (going backward)
            v = self._get_velocity(z, t_i, prompt_embeds, pooled_embeds,
                                   text_ids, cfg_scale)
            z = z + dt * v  # Euler step (dt < 0 → moves toward clean)

        # Final projection: at t_min, extract clean estimate
        t_final = ts[-1].item()
        v_final = self._get_velocity(z, t_final, prompt_embeds, pooled_embeds,
                                     text_ids, cfg_scale)
        x_clean = z - t_final * v_final  # x₀ = z_t - t · v
        return x_clean

    # Aliases for compatibility with existing GFM test scripts
    ddim_forward = flow_forward
    ddim_backward = flow_backward

    # ------ Sphere projection (unchanged) ------

    @staticmethod
    def project_to_sphere(x: torch.Tensor, radius: float) -> torch.Tensor:
        x_flat = x.view(x.shape[0], -1)
        norms = x_flat.norm(dim=-1, keepdim=True)
        x_flat = x_flat * (radius / (norms + 1e-8))
        return x_flat.view_as(x)

    # ------ Text inversion (adapted for Flux) ------

    def text_inversion(self, prompt: str, latent: torch.Tensor,
                       steps: int = 500, lr: float = 0.005
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optimize text embeddings to reconstruct a specific latent.

        In Flux, we optimize the T5 embeddings (higher capacity than CLIP).
        Returns (prompt_embeds, pooled_embeds, text_ids).
        """
        prompt_embeds, pooled_embeds, text_ids = self.encode_prompt(prompt)
        # Only optimize T5 embeddings; freeze CLIP pooled
        embed_opt = prompt_embeds.clone().requires_grad_(True)
        pooled_fixed = pooled_embeds.detach()

        optimizer = torch.optim.AdamW([embed_opt], lr=lr)

        for _ in tqdm(range(steps), desc="Text Inversion (Flux)"):
            optimizer.zero_grad()
            noise = torch.randn_like(latent)
            # Sample random flow time
            t = torch.rand(1, device=self.device).clamp(0.01, 0.99)
            # Construct noisy sample: z_t = (1-t)·x₀ + t·ε
            z_t = (1 - t) * latent + t * noise
            # Target velocity: v_target = ε - x₀
            v_target = noise - latent

            # Predict velocity
            with torch.autocast(device_type=str(self.device).split(':')[0],
                                dtype=torch.bfloat16):
                v_pred = self.model._v_pred(
                    z_t, t.item(), embed_opt, pooled_fixed, text_ids
                )
                loss = F.mse_loss(v_pred.float(), v_target.float())

            loss.backward()
            optimizer.step()

        return embed_opt.detach(), pooled_fixed, text_ids


# ============================================================
# 4. GFM Interpolation Pipeline (adapted for Flux)
# ============================================================

def run_gfm_interpolation(
    flux_pipe: FluxGFMPipeline,
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
) -> Dict[str, Any]:
    """
    Full GFM interpolation pipeline using Flux.1.

    Pipeline:
      1. Auto-caption images with BLIP (if prompts not given)
      2. Encode images via Flux VAE (16 channels)
      3. Forward inversion via flow ODE: x₀ → z_τ
      4. GFM optimization with sphere constraint
      5. Backward flow ODE: z_τ → x₀
      6. Flux VAE decode to images

    Args:
        flux_pipe:   Loaded FluxGFMPipeline
        imgA, imgB:  Input PIL images (endpoints)
        promptA/B:   Text prompts (auto-captioned if None)
        noise_level: Flow time τ ∈ (0, 1); 0.6 recommended
        cfg_scale:   CFG scale (0 = use Flux internal guidance only)
        num_steps:   Number of GFM waypoints
        lam:         GFM potential weight λ
        lr:          GFM learning rate
        max_iters:   GFM optimization iterations
        use_text_inversion: Run text inversion (slow)
        output_frames: Number of output frames

    Returns:
        Dict with 'images', 'latents_noisy', 'latents_clean',
        'losses', 'promptA', 'promptB', etc.
    """
    from gfm.path.geodesic_interpolation import ELInterpolator

    device = flux_pipe.device
    output_frames = num_steps

    # --- Step 1: Auto-caption ---
    if promptA is None:
        print("Auto-captioning image A with BLIP...")
        promptA = flux_pipe.auto_caption(imgA)
        print(f'  Caption A: "{promptA}"')
    if promptB is None:
        print("Auto-captioning image B with BLIP...")
        promptB = flux_pipe.auto_caption(imgB)
        print(f'  Caption B: "{promptB}"')

    flux_pipe.unload_captioner()

    # --- Step 2: Encode images (Flux VAE: 16 channels) ---
    print("Encoding images with Flux VAE...")
    latA = flux_pipe.autoencoder.encode(imgA)  # [1, 16, H/8, W/8]
    latB = flux_pipe.autoencoder.encode(imgB)
    print(f"  Latent shape: {latA.shape}")

    # --- Step 3: Text conditioning (dual encoder: T5 + CLIP) ---
    print("Encoding text with T5 + CLIP...")
    if use_text_inversion:
        embedA, pooledA, tidA = flux_pipe.text_inversion(promptA, latA)
        embedB, pooledB, tidB = flux_pipe.text_inversion(promptB, latB)
    else:
        embedA, pooledA, tidA = flux_pipe.encode_prompt(promptA)
        embedB, pooledB, tidB = flux_pipe.encode_prompt(promptB)

    # --- Step 4: Forward inversion (flow ODE: x₀ → z_τ) ---
    print(f"Flow forward inversion (noise_level={noise_level})...")
    latA_noisy = flux_pipe.flow_forward(latA, embedA, pooledA, tidA, noise_level, cfg_scale)
    latB_noisy = flux_pipe.flow_forward(latB, embedB, pooledB, tidB, noise_level, cfg_scale)

    # Compute sphere radius
    normA = latA_noisy.view(-1).norm().item()
    normB = latB_noisy.view(-1).norm().item()
    sphere_radius = 0.5 * (normA + normB)
    print(f"Sphere radius: {sphere_radius:.2f} (normA={normA:.2f}, normB={normB:.2f})")

    latA_noisy = FluxGFMPipeline.project_to_sphere(latA_noisy, sphere_radius)
    latB_noisy = FluxGFMPipeline.project_to_sphere(latB_noisy, sphere_radius)

    # --- Step 5: Configure GFM model ---
    sigma_eff = flux_pipe.get_sigma_from_noise_level(noise_level)
    print(f"Effective flow time σ = {sigma_eff:.4f}")

    flux_pipe.model.set_endpoints(
        embedA, pooledA, tidA,
        embedB, pooledB, tidB,
        sigma_eff, num_steps
    )

    # --- Step 6: Run GFM ---
    print(f"Running GFM (steps={num_steps}, λ={lam}, iters={max_iters})...")
    interpolator = ELInterpolator(
        diffusion_model=flux_pipe.model,
        autoencoder=flux_pipe.autoencoder,
        device=str(device)
    )

    optimized_path, info = interpolator.optimize_path(
        start_latent=latA_noisy,
        end_latent=latB_noisy,
        num_steps=num_steps,
        sigma=sigma_eff,
        lr=lr,
        max_iters=max_iters,
        lam=lam,
        verbose=True
    )

    # Project all waypoints to sphere
    B, T = optimized_path.shape[:2]
    for t in range(T):
        optimized_path[:, t] = FluxGFMPipeline.project_to_sphere(
            optimized_path[:, t], sphere_radius
        )

    # --- Step 7: Resample if needed ---
    if output_frames != num_steps:
        t_out = torch.linspace(0, 1, output_frames, device=device)
        t_orig = torch.linspace(0, 1, num_steps, device=device)
        indices = torch.searchsorted(t_orig, t_out).clamp(0, num_steps - 1)
        output_latents_noisy = optimized_path[:, indices]
    else:
        output_latents_noisy = optimized_path

    n_frames = output_latents_noisy.shape[1]

    # --- Step 8: Flow backward (denoise) ---
    print(f"Flow backward denoising ({n_frames} frames)...")
    output_latents_clean = []
    output_images = []

    for t_idx in tqdm(range(n_frames), desc="Denoising frames"):
        t_frac = t_idx / (n_frames - 1)
        # Interpolate text embeddings for this frame
        embed_t = (1 - t_frac) * embedA + t_frac * embedB
        pooled_t = (1 - t_frac) * pooledA + t_frac * pooledB
        tid_t = tidA  # Text IDs are positional, use A's

        lat_noisy = output_latents_noisy[:, t_idx]

        # Endpoints: use originals for higher quality
        if t_idx == 0:
            output_images.append(imgA)
            output_latents_clean.append(latA)
            continue
        elif t_idx == n_frames - 1:
            output_images.append(imgB)
            output_latents_clean.append(latB)
            continue

        lat_clean = flux_pipe.flow_backward(
            lat_noisy, embed_t, pooled_t, tid_t, noise_level, cfg_scale
        )
        img = flux_pipe.autoencoder.decode(lat_clean)

        output_latents_clean.append(lat_clean)
        output_images.append(img)

    print("Done!")

    return {
        'images': output_images,
        'latents_noisy': output_latents_noisy,
        'latents_clean': torch.cat(output_latents_clean, dim=0),
        'losses': info.get('losses', []),
        'sphere_radius': sphere_radius,
        'sigma_eff': sigma_eff,
        'promptA': promptA,
        'promptB': promptB,
    }


# ============================================================
# 5. Visualization Utilities (unchanged)
# ============================================================

def save_interpolation_strip(images: List[Image.Image], save_path: str,
                             size=(512, 512), padding=10):
    n = len(images)
    w, h = size
    pw, ph = w + 2 * padding, h + 2 * padding
    strip = Image.new("RGB", (pw * n, ph), (255, 255, 255))
    for i, img in enumerate(images):
        img_resized = img.resize(size)
        strip.paste(img_resized, (i * pw + padding, padding))
    strip.save(save_path)
    print(f"Saved interpolation strip to {save_path}")


def save_frames(images: List[Image.Image], output_dir: str):
    import os
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(output_dir, f"frame_{i:03d}.png"))
    print(f"Saved {len(images)} frames to {output_dir}")


# ============================================================
# 6. Entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GFM Image Interpolation with Flux.1"
    )
    parser.add_argument("--imgA", type=str, required=True)
    parser.add_argument("--imgB", type=str, required=True)
    parser.add_argument("--promptA", type=str, default=None)
    parser.add_argument("--promptB", type=str, default=None)
    parser.add_argument("--noise_level", type=float, default=0.6,
                        help="Flow time τ (0.6 recommended)")
    parser.add_argument("--cfg_scale", type=float, default=0.0,
                        help="CFG scale (0 = Flux internal guidance only)")
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                        help="Flux-dev internal guidance scale")
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max_iters", type=int, default=800)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--n_inference_steps", type=int, default=50,
                        help="ODE integration steps for forward/backward")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Image resolution (Flux native: 1024)")
    parser.add_argument("--output_dir", type=str, default="./output/flux_interp")
    parser.add_argument("--use_text_inv", action="store_true")
    parser.add_argument("--no_blip", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_id", type=str,
                        default="black-forest-labs/FLUX.1-dev")
    args = parser.parse_args()

    if args.no_blip and (args.promptA is None or args.promptB is None):
        parser.error("--no_blip requires both --promptA and --promptB")

    print(f"Loading Flux.1 from {args.model_id}...")
    flux_pipe = FluxGFMPipeline.load(
        model_id=args.model_id,
        device=args.device,
        cache_dir=args.cache_dir,
        load_blip=not args.no_blip,
        resolution=args.resolution,
        guidance_scale=args.guidance_scale,
    )

    imgA = Image.open(args.imgA)
    imgB = Image.open(args.imgB)

    results = run_gfm_interpolation(
        flux_pipe, imgA, imgB,
        promptA=args.promptA,
        promptB=args.promptB,
        noise_level=args.noise_level,
        cfg_scale=args.cfg_scale,
        num_steps=args.num_steps,
        lam=args.lam,
        lr=args.lr,
        max_iters=args.max_iters,
        use_text_inversion=args.use_text_inv,
    )

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    save_interpolation_strip(
        results['images'],
        os.path.join(args.output_dir, "strip.png")
    )
    save_frames(results['images'], os.path.join(args.output_dir, "frames"))

    print(f"\nResults saved to {args.output_dir}")
    print(f'Caption A: "{results["promptA"]}"')
    print(f'Caption B: "{results["promptB"]}"')
    print(f"Sphere radius: {results['sphere_radius']:.2f}")
    print(f"Effective σ (flow time): {results['sigma_eff']:.4f}")
    if results['losses']:
        print(f"Final GFM loss: {results['losses'][-1]:.6f}")