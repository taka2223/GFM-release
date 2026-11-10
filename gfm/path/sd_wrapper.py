"""Production Stable Diffusion 2.1-base wrapper for GFM interpolation.

Stable Diffusion uses variance-preserving (VP) latents ``x_t`` while the GFM
solver expects variance-exploding/EDM coordinates.  This module keeps that
boundary explicit:

    z_t = x_t / sqrt(alpha_t)
    sigma_eff = sqrt((1 - alpha_t) / alpha_t)
    x0_pred = z_t - sigma_eff * epsilon_theta(x_t, t)

DDIM inversion and denoising always operate on ``x_t``.  GFM and SLERP always
operate on ``z_t``.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor


DEFAULT_SD21_MODEL_ID = "sd2-community/stable-diffusion-2-1-base"
DEFAULT_SD21_REVISION = "4e63672c03103b6c636b8fb4119ba982469b2955"


def _as_alpha(alpha: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(alpha, device=reference.device, dtype=reference.dtype)


def vp_to_edm(x_t: torch.Tensor, alpha_t: torch.Tensor | float) -> torch.Tensor:
    """Convert a VP latent ``x_t`` to the GFM/EDM coordinate ``z_t``."""
    alpha = _as_alpha(alpha_t, x_t)
    return x_t / alpha.sqrt()


def edm_to_vp(z_t: torch.Tensor, alpha_t: torch.Tensor | float) -> torch.Tensor:
    """Convert a GFM/EDM coordinate ``z_t`` back to a VP latent ``x_t``."""
    alpha = _as_alpha(alpha_t, z_t)
    return z_t * alpha.sqrt()


def sigma_eff_from_alpha(alpha_t: torch.Tensor | float) -> torch.Tensor:
    """Return ``sqrt((1-alpha_t)/alpha_t)`` without changing tensor placement."""
    alpha = torch.as_tensor(alpha_t)
    return ((1.0 - alpha) / alpha).sqrt()


def epsilon_to_x0(
    z_t: torch.Tensor,
    epsilon: torch.Tensor,
    sigma_eff: torch.Tensor | float,
) -> torch.Tensor:
    """Convert an epsilon prediction to an EDM-coordinate clean prediction."""
    sigma = torch.as_tensor(
        sigma_eff,
        device=z_t.device,
        dtype=z_t.dtype,
    )
    return z_t - sigma * epsilon.to(z_t.dtype)


def conditioning_fractions(
    waypoint_indices: Sequence[int],
    denominator: int,
    *,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return explicit path fractions for the supplied full-path indices."""
    if denominator < 1:
        raise ValueError("conditioning denominator must be positive")
    indices = [int(index) for index in waypoint_indices]
    if not indices:
        raise ValueError("waypoint_indices must not be empty")
    if indices != sorted(set(indices)):
        raise ValueError("waypoint_indices must be strictly increasing and unique")
    if any(index <= 0 or index >= denominator for index in indices):
        raise ValueError("waypoint indices must be interior full-path indices")
    return torch.tensor(indices, device=device, dtype=dtype) / float(denominator)


def piecewise_slerp_resample(path: torch.Tensor, output_steps: int) -> torch.Tensor:
    """Resample a control path with equal-time spherical arc segments."""
    if path.ndim < 3 or path.shape[1] < 2:
        raise ValueError("path must have shape [batch, steps>=2, ...]")
    if output_steps < 2:
        raise ValueError("output_steps must be at least two")
    if output_steps == path.shape[1]:
        return path

    control_steps = path.shape[1]
    positions = torch.linspace(
        0,
        control_steps - 1,
        output_steps,
        device=path.device,
        dtype=path.dtype,
    )
    left = positions.floor().long().clamp(max=control_steps - 2)
    fraction = positions - left.to(positions.dtype)
    point_left = path[:, left].flatten(start_dim=2)
    point_right = path[:, left + 1].flatten(start_dim=2)
    norm_left = point_left.norm(dim=-1, keepdim=True)
    norm_right = point_right.norm(dim=-1, keepdim=True)
    cosine = (
        (point_left * point_right).sum(dim=-1, keepdim=True)
        / (norm_left * norm_right + 1e-8)
    ).clamp(-1 + 1e-7, 1 - 1e-7)
    omega = torch.acos(cosine)
    sin_omega = torch.sin(omega)
    fraction = fraction.view(1, output_steps, 1)
    left_weight = torch.sin((1 - fraction) * omega) / (sin_omega + 1e-8)
    right_weight = torch.sin(fraction * omega) / (sin_omega + 1e-8)
    small = sin_omega.abs() < 1e-6
    left_weight = torch.where(small, 1 - fraction, left_weight)
    right_weight = torch.where(small, fraction, right_weight)
    result = left_weight * point_left + right_weight * point_right
    return result.view(path.shape[0], output_steps, *path.shape[2:])


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
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def caption(self, image: Image.Image, max_new_tokens: int = 40) -> str:
        inputs = self.processor(
            image.convert("RGB"),
            text=self.prefix or None,
            return_tensors="pt",
        ).to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=self.num_beams,
        )
        return self.processor.decode(output_ids[0], skip_special_tokens=True)

    def unload(self) -> None:
        self.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SD21Autoencoder:
    """Stable Diffusion 2.1 VAE adapter."""

    def __init__(self, pipe: StableDiffusionPipeline, resolution: int = 512):
        self.vae = pipe.vae
        self.device = pipe.device
        self.resolution = resolution
        self.scaling_factor = float(pipe.vae.config.scaling_factor)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize(
            (self.resolution, self.resolution),
            Image.Resampling.LANCZOS,
        )
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return (
            torch.from_numpy(array)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, dtype=self.vae.dtype)
        )

    def _postprocess(self, tensor: torch.Tensor) -> Image.Image:
        array = (
            tensor.detach()
            .float()
            .squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        array = np.clip((array + 1.0) * 127.5, 0, 255).round().astype(np.uint8)
        return Image.fromarray(array)

    @torch.no_grad()
    def encode(self, image: Image.Image) -> torch.Tensor:
        distribution = self.vae.encode(self._preprocess(image)).latent_dist
        return distribution.mean * self.scaling_factor

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> Image.Image:
        decoded = self.vae.decode(
            latent.to(self.vae.dtype) / self.scaling_factor,
            return_dict=False,
        )[0]
        return self._postprocess(decoded)

    def eval(self) -> "SD21Autoencoder":
        self.vae.eval()
        return self


class SD21Model(torch.nn.Module):
    """Expose SD2.1 epsilon prediction as a GFM denoised-point model."""

    def __init__(self, pipeline: "SD21GFMPipeline"):
        super().__init__()
        self.pipeline = pipeline
        self.unet = pipeline.pipe.unet
        self.device = pipeline.device
        self.mode = "denoise"
        self.cfg_scale = 0.0
        self._embed_a: Optional[torch.Tensor] = None
        self._embed_b: Optional[torch.Tensor] = None
        self._fractions: Optional[torch.Tensor] = None
        self._static_embed: Optional[torch.Tensor] = None
        self._timestep: Optional[int] = None
        self._sigma_eff: Optional[float] = None

    def set_cfg_params(self, cfg_scale: float) -> None:
        self.cfg_scale = float(cfg_scale)

    def set_conditioning(
        self,
        embed: torch.Tensor,
        timestep: int,
        sigma_eff: Optional[float] = None,
    ) -> None:
        self._static_embed = embed
        self._embed_a = None
        self._fractions = None
        self._timestep = int(timestep)
        self._sigma_eff = (
            self.pipeline.get_sigma_from_timestep(timestep)
            if sigma_eff is None
            else float(sigma_eff)
        )

    def set_endpoints(
        self,
        embed_a: torch.Tensor,
        embed_b: torch.Tensor,
        sigma_eff: float,
        timestep: int,
        waypoint_indices: Sequence[int],
        denominator: int,
    ) -> None:
        """Set endpoint prompts and the exact path points used by the solver."""
        self._embed_a = embed_a
        self._embed_b = embed_b
        self._fractions = conditioning_fractions(
            waypoint_indices,
            denominator,
            device=self.device,
            dtype=embed_a.dtype,
        )
        self._static_embed = None
        self._timestep = int(timestep)
        self._sigma_eff = float(sigma_eff)

    def conditioning_for_batch(self, batch_size: int) -> torch.Tensor:
        if self._embed_a is None:
            if self._static_embed is None:
                raise RuntimeError("SD2.1 conditioning has not been configured")
            return self._static_embed.expand(batch_size, -1, -1)
        assert self._embed_b is not None and self._fractions is not None
        count = int(self._fractions.numel())
        if batch_size % count != 0:
            raise ValueError(
                "model batch does not match explicit waypoint conditioning: "
                f"batch={batch_size}, waypoint_count={count}"
            )
        fractions = self._fractions.repeat(batch_size // count)
        weights = fractions.view(-1, 1, 1)
        return (1 - weights) * self._embed_a + weights * self._embed_b

    @torch.no_grad()
    def forward(
        self,
        z_t: torch.Tensor,
        sigma: torch.Tensor,
        class_labels=None,
    ) -> torch.Tensor:
        if self._timestep is None or self._sigma_eff is None:
            raise RuntimeError("SD2.1 timestep has not been configured")
        conditioning = self.conditioning_for_batch(z_t.shape[0])
        x_t = edm_to_vp(
            z_t,
            self.pipeline.alpha_at(self._timestep),
        )
        epsilon = self.pipeline.predict_epsilon(
            x_t,
            self._timestep,
            conditioning,
            self.cfg_scale,
        )
        # ``nfsd`` is retained as a runner-compatible name.  In the production
        # epsilon-prediction wrapper both modes use the correctly scaled x0
        # prediction; no negative-prompt pseudo-force is injected.
        if self.mode not in {"denoise", "nfsd"}:
            raise ValueError(f"unsupported SD2.1 GFM mode: {self.mode}")
        return epsilon_to_x0(z_t, epsilon, self._sigma_eff)


class SD21GFMPipeline:
    """Stable Diffusion 2.1-base pipeline aligned with ``SD3GFMPipeline``."""

    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        n_inference_steps: int = 50,
        captioner: Optional[BLIPCaptioner] = None,
        resolution: int = 512,
        *,
        model_id: str = DEFAULT_SD21_MODEL_ID,
        revision: Optional[str] = DEFAULT_SD21_REVISION,
    ):
        self.pipe = pipe
        self.device = pipe.device
        self.n_inference_steps = int(n_inference_steps)
        self.captioner = captioner
        self.resolution = int(resolution)
        self.model_id = model_id
        self.requested_revision = revision
        self.resolved_revision = (
            getattr(pipe.config, "_commit_hash", None)
            or getattr(pipe.unet.config, "_commit_hash", None)
            or revision
        )

        self._validate_checkpoint()
        self.pipe.scheduler.set_timesteps(self.n_inference_steps, device=self.device)
        self.pipe.unet.requires_grad_(False).eval()
        self.pipe.text_encoder.requires_grad_(False).eval()
        self.pipe.vae.requires_grad_(False).eval()
        self._embed_uncond = self._encode_prompt("")
        self.autoencoder = SD21Autoencoder(pipe, resolution=self.resolution)
        self.model = SD21Model(self)

    @classmethod
    def load(
        cls,
        model_id: str = DEFAULT_SD21_MODEL_ID,
        device: str = "cuda",
        dtype=torch.float16,
        cache_dir: Optional[str] = None,
        load_blip: bool = True,
        caption_model_id: str = "Salesforce/blip-image-captioning-large",
        resolution: int = 512,
        n_inference_steps: int = 50,
        revision: Optional[str] = DEFAULT_SD21_REVISION,
        **_unused,
    ) -> "SD21GFMPipeline":
        if resolution != 512:
            raise ValueError("SD2.1-base is a 512px checkpoint; --resolution must be 512")
        source = StableDiffusionPipeline.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            safety_checker=None,
            requires_safety_checker=False,
        )
        source.scheduler = DDIMScheduler.from_config(source.scheduler.config)
        source.to(device)
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
            source,
            n_inference_steps=n_inference_steps,
            captioner=captioner,
            resolution=resolution,
            model_id=model_id,
            revision=revision,
        )

    def _validate_checkpoint(self) -> None:
        prediction_type = str(self.pipe.scheduler.config.prediction_type)
        if prediction_type != "epsilon":
            raise ValueError(
                "SD2.1-base must use epsilon prediction; got "
                f"prediction_type={prediction_type!r}. The 768px SD2.1 checkpoint "
                "is v-prediction and is not compatible with this wrapper."
            )
        sample_size = int(self.pipe.unet.config.sample_size)
        vae_scale = int(getattr(self.pipe, "vae_scale_factor", 8))
        native_resolution = sample_size * vae_scale
        if sample_size != 64 or native_resolution != 512 or self.resolution != 512:
            raise ValueError(
                "Expected the 512px SD2.1-base UNet "
                f"(sample_size=64, VAE scale=8); got sample_size={sample_size}, "
                f"VAE scale={vae_scale}, resolution={self.resolution}."
            )

    @property
    def model_metadata(self) -> Dict[str, Any]:
        return {
            "backbone": "sd21",
            "model_id": self.model_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "resolution": self.resolution,
            "prediction_type": str(self.pipe.scheduler.config.prediction_type),
            "scheduler_class": type(self.pipe.scheduler).__name__,
            "n_inference_steps": self.n_inference_steps,
        }

    def auto_caption(self, image: Image.Image) -> str:
        if self.captioner is None:
            raise RuntimeError("BLIP is not loaded; provide endpoint prompts")
        return self.captioner.caption(image)

    def unload_captioner(self) -> None:
        if self.captioner is not None:
            self.captioner.unload()

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        tokens = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return self.pipe.text_encoder(tokens, return_dict=False)[0]

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        return self._encode_prompt(prompt)

    def alpha_at(self, timestep: int) -> torch.Tensor:
        return self.pipe.scheduler.alphas_cumprod[int(timestep)].to(self.device)

    def get_timestep(self, noise_level: float) -> int:
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError("noise_level must lie in [0, 1]")
        if noise_level == 0:
            return 0
        timesteps = self.pipe.scheduler.timesteps
        target = noise_level * (self.pipe.scheduler.config.num_train_timesteps - 1)
        index = (timesteps.float() - target).abs().argmin()
        return int(timesteps[index].item())

    def get_sigma_from_timestep(self, timestep: int) -> float:
        return float(sigma_eff_from_alpha(self.alpha_at(timestep)).item())

    def get_sigma_from_noise_level(self, noise_level: float) -> float:
        return self.get_sigma_from_timestep(self.get_timestep(noise_level))

    def _denoise_timesteps(self, noise_level: float) -> List[int]:
        if noise_level == 0:
            return []
        target = self.get_timestep(noise_level)
        values = [int(value) for value in self.pipe.scheduler.timesteps.tolist()]
        start = values.index(target)
        return values[start:]

    @torch.no_grad()
    def predict_epsilon(
        self,
        x_t: torch.Tensor,
        timestep: int | torch.Tensor,
        embed_cond: torch.Tensor,
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        """Predict epsilon; CFG=0 is explicitly conditional-only."""
        batch = x_t.shape[0]
        embed_cond = embed_cond.expand(batch, -1, -1)
        model_dtype = self.pipe.unet.dtype
        timestep_tensor = torch.as_tensor(timestep, device=self.device)
        if cfg_scale > 0:
            x_input = torch.cat([x_t, x_t]).to(model_dtype)
            embeddings = torch.cat(
                [
                    self._embed_uncond.expand(batch, -1, -1),
                    embed_cond,
                ]
            ).to(model_dtype)
            prediction = self.pipe.unet(
                x_input,
                timestep_tensor,
                encoder_hidden_states=embeddings,
                return_dict=False,
            )[0].float()
            epsilon_uncond, epsilon_cond = prediction.chunk(2)
            return epsilon_uncond + cfg_scale * (epsilon_cond - epsilon_uncond)
        return self.pipe.unet(
            x_t.to(model_dtype),
            timestep_tensor,
            encoder_hidden_states=embed_cond.to(model_dtype),
            return_dict=False,
        )[0].float()

    @torch.no_grad()
    def ddim_forward(
        self,
        latent_clean: torch.Tensor,
        embed_cond: torch.Tensor,
        noise_level: float,
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        """Deterministic DDIM inversion in VP coordinates."""
        descending = self._denoise_timesteps(noise_level)
        if not descending:
            return latent_clean
        ascending = list(reversed(descending))
        # Move from the mathematical clean state (alpha=1) to the scheduler's
        # first discrete state before traversing the remaining inversion grid.
        first_t = ascending[0]
        first_alpha = self.alpha_at(first_t).to(latent_clean)
        first_epsilon = self.predict_epsilon(
            latent_clean.float(),
            first_t,
            embed_cond,
            cfg_scale,
        )
        x_t = (
            first_alpha.sqrt() * latent_clean.float()
            + (1 - first_alpha).sqrt() * first_epsilon
        )
        for current_t, next_t in zip(ascending[:-1], ascending[1:]):
            alpha_current = self.alpha_at(current_t).to(x_t)
            alpha_next = self.alpha_at(next_t).to(x_t)
            epsilon = self.predict_epsilon(
                x_t,
                current_t,
                embed_cond,
                cfg_scale,
            )
            pred_x0 = (
                x_t - (1 - alpha_current).sqrt() * epsilon
            ) / alpha_current.sqrt()
            x_t = (
                alpha_next.sqrt() * pred_x0
                + (1 - alpha_next).sqrt() * epsilon
            )
        return x_t

    @torch.no_grad()
    def ddim_backward(
        self,
        latent_noisy: torch.Tensor,
        embed_cond: torch.Tensor,
        noise_level: float,
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        """Deterministic DDIM denoising in VP coordinates."""
        descending = self._denoise_timesteps(noise_level)
        if not descending:
            return latent_noisy
        x_t = latent_noisy.float()
        for index, current_t in enumerate(descending):
            alpha_current = self.alpha_at(current_t).to(x_t)
            if index + 1 < len(descending):
                alpha_previous = self.alpha_at(descending[index + 1]).to(x_t)
            else:
                alpha_previous = torch.ones_like(alpha_current)
            epsilon = self.predict_epsilon(
                x_t,
                current_t,
                embed_cond,
                cfg_scale,
            )
            pred_x0 = (
                x_t - (1 - alpha_current).sqrt() * epsilon
            ) / alpha_current.sqrt()
            x_t = (
                alpha_previous.sqrt() * pred_x0
                + (1 - alpha_previous).sqrt() * epsilon
            )
        return x_t

    @staticmethod
    def project_to_sphere(x: torch.Tensor, radius: float) -> torch.Tensor:
        flat = x.flatten(start_dim=1)
        flat = flat * (radius / (flat.norm(dim=-1, keepdim=True) + 1e-8))
        return flat.view_as(x)


# Backwards-compatible names for older callers.
SDPipeline = SD21GFMPipeline
SDModel = SD21Model
SDAutoencoder = SD21Autoencoder


def _validate_path_configuration(
    method: str,
    output_frames: int,
    opt_num_steps: Optional[int],
    c2f_steps: Optional[List[int]],
    c2f_iters: Optional[List[int]],
    c2f_lrs: Optional[List[float]],
    c2f_freeze_anchors: bool,
) -> Tuple[int, List[int], List[int], List[float]]:
    if output_frames < 3:
        raise ValueError("num_steps must be at least three")
    if method == "gfm_c2f":
        steps = c2f_steps or [3, 5, 9, output_frames]
        iterations = c2f_iters or [400, 300, 200, 100]
        learning_rates = c2f_lrs or [0.002, 0.001, 0.0005, 0.00025]
        if not (len(steps) == len(iterations) == len(learning_rates)):
            raise ValueError("c2f_steps, c2f_iters, and c2f_lrs must have equal length")
        if steps != sorted(set(steps)) or steps[0] < 3 or steps[-1] > output_frames:
            raise ValueError("c2f_steps must be increasing within [3, num_steps]")
        if any(value < 1 for value in iterations):
            raise ValueError("every C2F stage needs at least one iteration")
        if any(value <= 0 for value in learning_rates):
            raise ValueError("every C2F learning rate must be positive")
        if c2f_freeze_anchors:
            for previous, current in zip(steps[:-1], steps[1:]):
                if (current - 1) % (previous - 1):
                    raise ValueError("frozen C2F requires nested waypoint grids")
        return steps[-1], steps, iterations, learning_rates
    optimization_steps = opt_num_steps or output_frames
    if not 3 <= optimization_steps <= output_frames:
        raise ValueError("opt_num_steps must lie in [3, num_steps]")
    return optimization_steps, [], [], []


def run_gfm_interpolation(
    sd_pipe: SD21GFMPipeline,
    imgA: Image.Image,
    imgB: Image.Image,
    promptA: Optional[str] = None,
    promptB: Optional[str] = None,
    noise_level: float = 0.6,
    cfg_scale: float = 0.0,
    num_steps: int = 17,
    lam: float = 1.0,
    lr: float = 0.001,
    max_iters: int = 400,
    use_text_inversion: bool = False,
    output_frames: Optional[int] = None,
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
    """Run GFM, frozen-anchor C2F, or SLERP under one SD2.1 protocol."""
    from gfm.path.geodesic_interpolation import EL2Interpolator, SphericalInterpolator

    if use_text_inversion:
        raise NotImplementedError("text inversion is not part of the SD2.1 smoke protocol")
    if method not in {"gfm", "gfm_c2f", "slerp"}:
        raise ValueError(f"SD2.1 wrapper does not support method={method!r}")
    output_frames = num_steps if output_frames is None else int(output_frames)
    if output_frames != num_steps:
        raise ValueError("the production runner uses num_steps as the output frame count")
    optimization_steps, c2f_steps, c2f_iters, c2f_lrs = _validate_path_configuration(
        method,
        output_frames,
        opt_num_steps,
        c2f_steps,
        c2f_iters,
        c2f_lrs,
        c2f_freeze_anchors,
    )
    sd_pipe.model.mode = mode
    sd_pipe.model.set_cfg_params(cfg_scale)

    if promptA is None:
        promptA = sd_pipe.auto_caption(imgA)
    if promptB is None:
        promptB = sd_pipe.auto_caption(imgB)
    sd_pipe.unload_captioner()

    print("Encoding SD2.1 endpoint images...", flush=True)
    latent_a = sd_pipe.autoencoder.encode(imgA)
    latent_b = sd_pipe.autoencoder.encode(imgB)
    embed_a = sd_pipe.encode_prompt(promptA)
    embed_b = sd_pipe.encode_prompt(promptB)

    print(f"DDIM inversion (noise_level={noise_level:g})...", flush=True)
    # The same captions, scheduler, step count, and CFG definition are shared
    # by every method.  CFG=0 remains conditional-only in both directions.
    x_a = sd_pipe.ddim_forward(latent_a, embed_a, noise_level, cfg_scale)
    x_b = sd_pipe.ddim_forward(latent_b, embed_b, noise_level, cfg_scale)
    timestep = sd_pipe.get_timestep(noise_level)
    alpha_t = sd_pipe.alpha_at(timestep)
    sigma_eff = sd_pipe.get_sigma_from_timestep(timestep)
    z_a = vp_to_edm(x_a, alpha_t)
    z_b = vp_to_edm(x_b, alpha_t)
    radius = 0.5 * (
        z_a.flatten().norm().item() + z_b.flatten().norm().item()
    )
    print(
        f"SD2.1 timestep={timestep}, alpha={alpha_t.item():.6g}, "
        f"sigma_eff={sigma_eff:.6g}, sphere_radius={radius:.4f}",
        flush=True,
    )

    interpolator = (
        SphericalInterpolator(sd_pipe.model, sd_pipe.autoencoder, str(sd_pipe.device))
        if method == "slerp"
        else EL2Interpolator(sd_pipe.model, sd_pipe.autoencoder, str(sd_pipe.device))
    )

    def project_path(path: torch.Tensor) -> torch.Tensor:
        result = path.clone()
        for index in range(result.shape[1]):
            result[:, index] = sd_pipe.project_to_sphere(result[:, index], radius)
        return result

    stage_summaries: Optional[List[Dict[str, Any]]] = None
    if method == "gfm_c2f":
        stage_path: Optional[torch.Tensor] = None
        stage_summaries = []
        losses: List[float] = []
        acc_norms: List[float] = []
        f_norms: List[float] = []
        previous_steps: Optional[int] = None
        final_info: Dict[str, Any] = {}

        for stage_number, (stage_steps, stage_iters, stage_lr) in enumerate(
            zip(c2f_steps, c2f_iters, c2f_lrs),
            start=1,
        ):
            initial_path = None
            anchor_indices = [0, stage_steps - 1]
            trainable_indices: Optional[List[int]] = None
            if stage_path is not None:
                initial_path = piecewise_slerp_resample(
                    project_path(stage_path),
                    stage_steps,
                )
                if c2f_freeze_anchors:
                    assert previous_steps is not None
                    refinement = (stage_steps - 1) // (previous_steps - 1)
                    anchor_indices = [
                        index * refinement for index in range(previous_steps)
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
            # This is deliberately explicit.  For the final frozen 17-frame
            # stage the conditioning is [1,3,...,15]/16, never eight midpoints.
            sd_pipe.model.set_endpoints(
                embed_a,
                embed_b,
                sigma_eff,
                timestep,
                active_indices,
                stage_steps - 1,
            )
            print(
                f"C2F stage {stage_number}/{len(c2f_steps)}: "
                f"steps={stage_steps}, active={active_indices}, "
                f"iters={stage_iters}, lr={stage_lr:g}",
                flush=True,
            )
            stage_path, stage_info = interpolator.optimize_path(
                start_latent=z_a,
                end_latent=z_b,
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
                expected = initial_path.clone()
                expected[:, 0], expected[:, -1] = z_a, z_b
                anchor_drift = (
                    stage_path[:, anchor_indices] - expected[:, anchor_indices]
                ).abs().max().item()
                if anchor_drift != 0.0:
                    raise RuntimeError(
                        f"frozen C2F anchors drifted by {anchor_drift:.6g}"
                    )
            stage_losses = stage_info.get("losses", [])
            losses.extend(stage_losses)
            acc_norms.extend(stage_info.get("acc_norms", []))
            f_norms.extend(stage_info.get("f_norms", []))
            stage_summaries.append(
                {
                    "stage": stage_number,
                    "num_steps": stage_steps,
                    "max_iters": stage_iters,
                    "lr": stage_lr,
                    "iterations_run": stage_info.get("iterations", stage_iters),
                    "initial_loss": stage_losses[0] if stage_losses else None,
                    "last_loss": stage_losses[-1] if stage_losses else None,
                    "best_iteration": stage_info.get("best_iteration"),
                    "best_loss": stage_info.get("best_loss"),
                    "trainable_indices": active_indices,
                    "conditioning_fractions": [
                        index / (stage_steps - 1) for index in active_indices
                    ],
                    "anchor_indices": anchor_indices,
                    "max_anchor_drift": anchor_drift,
                }
            )
            final_info = stage_info
            previous_steps = stage_steps
        assert stage_path is not None
        optimized_path = stage_path
        info = {
            "losses": losses,
            "acc_norms": acc_norms,
            "f_norms": f_norms,
            "iterations": sum(
                int(stage["iterations_run"]) for stage in stage_summaries
            ),
            "best_iteration": final_info.get("best_iteration"),
            "best_loss": final_info.get("best_loss"),
            "early_stopped": False,
            "snapshots": {},
        }
    else:
        active_indices = list(range(1, optimization_steps - 1))
        sd_pipe.model.set_endpoints(
            embed_a,
            embed_b,
            sigma_eff,
            timestep,
            active_indices,
            optimization_steps - 1,
        )
        optimized_path, info = interpolator.optimize_path(
            start_latent=z_a,
            end_latent=z_b,
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
        )

    z_path = project_path(optimized_path)
    if z_path.shape[1] != output_frames:
        z_path = piecewise_slerp_resample(z_path, output_frames)
    x_path = edm_to_vp(z_path, alpha_t)

    images: List[Image.Image] = [imgA]
    clean_latents: List[torch.Tensor] = [latent_a]
    inner_count = output_frames - 2
    if inner_count:
        fractions = torch.arange(
            1,
            output_frames - 1,
            device=embed_a.device,
            dtype=embed_a.dtype,
        ) / float(output_frames - 1)
        inner_embed = (
            (1 - fractions.view(-1, 1, 1)) * embed_a
            + fractions.view(-1, 1, 1) * embed_b
        )
        print(f"DDIM denoising {inner_count} interior frames...", flush=True)
        inner_clean = sd_pipe.ddim_backward(
            x_path[:, 1:-1].reshape(inner_count, *x_path.shape[2:]),
            inner_embed,
            noise_level,
            cfg_scale,
        )
        for index in tqdm(range(inner_count), desc="Decoding[sd21]"):
            latent = inner_clean[index : index + 1]
            clean_latents.append(latent)
            images.append(sd_pipe.autoencoder.decode(latent))
    images.append(imgB)
    clean_latents.append(latent_b)

    losses = info.get("losses", [])
    return {
        "images": images,
        "latents_noisy": z_path,
        "latents_vp": x_path,
        "latents_clean": torch.cat(clean_latents),
        "losses": losses,
        "acc_norms": info.get("acc_norms", []),
        "f_norms": info.get("f_norms", []),
        "lam": lam,
        "sphere_radius": radius,
        "sigma_eff": sigma_eff,
        "alpha_t": float(alpha_t.item()),
        "timestep": timestep,
        "promptA": promptA,
        "promptB": promptB,
        "snapshots": {},
        "iterations_run": info.get("iterations", len(losses)),
        "early_stopped": info.get("early_stopped", False),
        "best_iteration": info.get("best_iteration"),
        "best_loss": info.get("best_loss"),
        "c2f_stages": stage_summaries,
        "c2f_freeze_anchors": c2f_freeze_anchors,
        "model": sd_pipe.model_metadata,
        "conditioning_protocol": {
            "output_fractions": [
                index / (output_frames - 1)
                for index in range(1, output_frames - 1)
            ],
            "explicit_waypoint_indices": True,
        },
    }


def save_interpolation_strip(
    images: List[Image.Image],
    save_path: str,
    size: Tuple[int, int] = (512, 512),
    padding: int = 10,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    width, height = size
    cell_width, cell_height = width + 2 * padding, height + 2 * padding
    strip = Image.new(
        "RGB",
        (cell_width * len(images), cell_height),
        (255, 255, 255),
    )
    for index, image in enumerate(images):
        strip.paste(
            image.convert("RGB").resize(size),
            (index * cell_width + padding, padding),
        )
    strip.save(save_path)


def save_frames(images: List[Image.Image], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for index, image in enumerate(images):
        image.save(os.path.join(output_dir, f"frame_{index:03d}.png"))


def save_optimization_curves(
    losses: List[float],
    acc_norms: List[float],
    f_norms: List[float],
    lam: float,
    output_dir: str,
    snapshot_iters: Optional[List[int]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    count = min(len(losses), len(acc_norms), len(f_norms))
    if count == 0:
        return
    with open(
        os.path.join(output_dir, "opt_curves.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["iter", "loss", "acc_rms", "f_rms", "lam_f_rms", "ratio"])
        for index in range(count):
            lam_force = lam * f_norms[index]
            writer.writerow(
                [
                    index,
                    losses[index],
                    acc_norms[index],
                    f_norms[index],
                    lam_force,
                    lam_force / (acc_norms[index] + 1e-12),
                ]
            )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    iterations = list(range(count))
    lam_force = [lam * value for value in f_norms[:count]]
    ratios = [
        lam_force[index] / (acc_norms[index] + 1e-12)
        for index in range(count)
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(iterations, losses[:count], color="black")
    axes[0].set_yscale("log")
    axes[1].plot(iterations, acc_norms[:count], label="acc")
    axes[1].plot(iterations, lam_force, label=f"lambda*force ({lam:g})")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[2].plot(iterations, ratios)
    axes[2].axhline(1.0, color="gray", linestyle=":")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("iteration")
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "opt_curves.png"), dpi=120)
    plt.close(figure)
