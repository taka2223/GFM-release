"""
SD2.1 Wrapper for Geodesic Force Matching (GFM)

Adapts Stable Diffusion 2.1 to work with the GFM interpolation framework.
Follows the pipeline from Yu et al. (2025) "Probability Density Geodesics":
  1. Encode images to VAE latent
  2. Auto-caption images with BLIP (or use provided prompts)
  3. DDIM forward inversion to noise level τ  
  4. GFM optimization at noise level τ (uses sphere constraint)
  5. DDIM backward to denoise
  6. VAE decode to images

Key design decisions:
  - Uses NFSD-style score (positive + negative prompt) following Yu et al.
  - BLIP auto-captioning removes the need for manual prompts
  - Operates at a FIXED diffusion timestep τ, not continuous σ
  - Sphere constraint: latents are projected to hypersphere after each update
  - Text inversion is OPTIONAL (skipped by default for efficiency)

Usage:
    from sd_wrapper import SDPipeline, run_gfm_interpolation
    
    pipe = SDPipeline.load("stabilityai/stable-diffusion-2-1-base")
    results = run_gfm_interpolation(pipe, imgA, imgB)
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm


# ============================================================
# 0. BLIP Captioner
# ============================================================

class BLIPCaptioner:
    """
    Auto-caption images using BLIP.
    
    Caches the model so it's only loaded once. Call .unload() to free
    GPU memory after captioning is done (before running diffusion).
    """
    
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
        """Generate a short caption for a single image."""
        inputs = self.processor(image.convert("RGB"), return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(output_ids[0], skip_special_tokens=True)
    
    def caption_batch(self, images: List[Image.Image], 
                      max_new_tokens: int = 50) -> List[str]:
        """Generate captions for multiple images."""
        return [self.caption(img, max_new_tokens) for img in images]

    def unload(self):
        """Free GPU memory by moving model to CPU and clearing cache."""
        self.model.cpu()
        torch.cuda.empty_cache()


# ============================================================
# 1. SDModel: Wraps UNet to match GFM's model interface
# ============================================================

class SDModel(torch.nn.Module):
    """
    Wraps SD2.1 UNet to provide EDM-style interface for GFM.
    
    GFM's compute_force expects:
        x_pred = model(latent, sigma, class_labels)
        force = -(latent - x_pred) / sigma^2
    
    This wrapper converts SD's noise prediction to x_pred (denoised estimate).
    
    Two modes:
      - 'denoise': Standard x_pred via noise prediction (like Tweedie's formula)
      - 'nfsd': Noise-Free Score Distillation gradient (Yu et al. 2025)
    """
    
    def __init__(self, pipe: StableDiffusionPipeline, mode='nfsd'):
        super().__init__()
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler
        self.device = pipe.device
        self.mode = mode
        
        # Pre-compute embeddings
        self._embed_uncond = self._encode_prompt(pipe, "")
        self._embed_neg = self._encode_prompt(
            pipe,
            "A doubling image, unrealistic, artifacts, distortions, "
            "unnatural blending, ghosting effects, overlapping edges, "
            "harsh transitions, motion blur, poor resolution, low detail"
        )
        
        # Default: will be set per-call
        self._embed_cond = None
        self._timestep = None
    
    @staticmethod
    def _encode_prompt(pipe, prompt: str) -> torch.Tensor:
        """Encode text prompt to CLIP embedding."""
        tokens = pipe.tokenizer(
            prompt,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        ).input_ids
        return pipe.text_encoder(tokens.to(pipe.device))[0]  # [1, 77, 768]
    
    def set_conditioning(self, embed_cond: torch.Tensor, timestep: int):
        """Set the text conditioning and diffusion timestep for subsequent calls."""
        self._embed_cond = embed_cond
        self._timestep = timestep
    
    def _noise_pred(self, latent, t, embed):
        """Raw UNet noise prediction."""
        return self.unet(latent, t, encoder_hidden_states=embed).sample
    
    def _noise_pred_cfg(self, latent, t, embed_cond, guidance_scale=1.0):
        """Noise prediction with classifier-free guidance."""
        latent_in = torch.cat([latent] * 2)
        embed_in = torch.cat([self._embed_uncond.expand(latent.shape[0], -1, -1),
                              embed_cond])
        noise_pred = self.unet(latent_in, t, encoder_hidden_states=embed_in).sample
        noise_uncond, noise_cond = noise_pred.chunk(2)
        return noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    
    @torch.no_grad()
    def forward(self, latent: torch.Tensor, sigma: torch.Tensor, 
                class_labels=None) -> torch.Tensor:
        """
        Return denoised x_pred from noisy latent.
        
        For GFM compatibility:
            force = -(latent - x_pred) / sigma^2
        
        Args:
            latent: [B, 4, 64, 64] noisy latent codes
            sigma: [B] noise levels (used to find closest timestep)
            class_labels: ignored (conditioning set via set_conditioning)
        
        Returns:
            x_pred: [B, 4, 64, 64] denoised prediction
        """
        t = self._timestep
        batch_size = latent.shape[0]
        
        if self.mode == 'denoise':
            # Standard denoising: x_pred = (x_t - sqrt(1-α_t) * ε) / sqrt(α_t)
            embed = self._embed_cond.expand(batch_size, -1, -1)
            noise_pred = self._noise_pred(latent, t, embed)
            alpha_t = self.scheduler.alphas_cumprod[t]
            x_pred = (latent - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            return x_pred
            
        elif self.mode == 'nfsd':
            # NFSD: return latent + sigma^2 * score_direction
            # So that compute_force gives: -(latent - x_pred)/sigma^2 = score_direction
            embed_cond = self._embed_cond.expand(batch_size, -1, -1)
            embed_neg = self._embed_neg.expand(batch_size, -1, -1)
            
            with torch.autocast(device_type=str(self.device).split(':')[0], dtype=torch.float16):
                ep_cond = self._noise_pred(latent, t, embed_cond)
                ep_neg = self._noise_pred(latent, t, embed_neg)
            
            # NFSD direction: positive direction + negative direction
            # grad_c = -(noise conditioned) → points toward conditioned distribution  
            # grad_d = +(noise negative) → pushes away from negative distribution
            grad = 0.5 * (-ep_cond.float() + ep_neg.float())
            
            # Convert to x_pred so that compute_force recovers the gradient
            # force = -(latent - x_pred) / sigma^2 = grad
            # → x_pred = latent + sigma^2 * grad
            sigma_val = sigma[0].item() if sigma.dim() > 0 else sigma.item()
            x_pred = latent + (sigma_val ** 2) * grad
            return x_pred
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# ============================================================
# 2. SDAutoencoder: Wraps VAE for GFM compatibility
# ============================================================

class SDAutoencoder:
    """Wraps SD2.1 VAE encoder/decoder."""
    
    def __init__(self, pipe: StableDiffusionPipeline):
        self.vae = pipe.vae
        self.device = pipe.device
        self.scaling_factor = pipe.vae.config.scaling_factor  # 0.18215
    
    def encode(self, image: Image.Image) -> torch.Tensor:
        """PIL Image → VAE latent [1, 4, 64, 64]."""
        img_tensor = self._preprocess(image)
        latent = self.vae.encode(img_tensor)['latent_dist'].mean
        return latent * self.scaling_factor
    
    def decode(self, latent: torch.Tensor) -> Image.Image:
        """VAE latent [1, 4, 64, 64] → PIL Image."""
        latent = latent / self.scaling_factor
        img_tensor = self.vae.decode(latent)['sample']
        return self._postprocess(img_tensor)
    
    def decode_batch(self, latents: torch.Tensor) -> List[Image.Image]:
        """Decode a batch of latents one by one (to avoid VAE batch artifacts)."""
        images = []
        for i in range(latents.shape[0]):
            images.append(self.decode(latents[i:i+1]))
        return images
    
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL Image → normalized tensor [1, 3, 512, 512]."""
        image = image.convert("RGB").resize((512, 512))
        arr = np.array(image).astype(np.float32) / 255.0
        arr = 2.0 * arr - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    def _postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """Tensor [1, 3, 512, 512] → PIL Image."""
        arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        arr = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))
    
    def eval(self):
        pass


# ============================================================
# 3. SDPipeline: Full pipeline with DDIM forward/backward
# ============================================================

class SDPipeline:
    """
    Complete SD2.1 pipeline for GFM interpolation.
    
    Handles:
      - Model loading (+ optional BLIP captioner)
      - Auto-captioning via BLIP
      - Text prompt encoding (with optional text inversion)
      - DDIM forward inversion (image → noisy latent at timestep τ)
      - DDIM backward (noisy latent → clean image)
      - Sphere projection utilities
    """
    
    def __init__(self, pipe: StableDiffusionPipeline, n_inference_steps: int = 50,
                 captioner: Optional[BLIPCaptioner] = None):
        self.pipe = pipe
        self.device = pipe.device
        self.n_inference_steps = n_inference_steps
        self.captioner = captioner
        
        # Set up scheduler
        pipe.scheduler.set_timesteps(n_inference_steps)
        
        # Freeze all parameters
        pipe.unet.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.vae.requires_grad_(False)
        
        # Create sub-components
        self.model = SDModel(pipe, mode='nfsd')
        self.autoencoder = SDAutoencoder(pipe)
        
        # Unconditional embedding (for CFG in DDIM)
        self.embed_uncond = SDModel._encode_prompt(pipe, "")
    
    @classmethod
    def load(cls, model_id: str = "sd2-community/stable-diffusion-2-1",
             device: str = "cuda", dtype=torch.float32,
             cache_dir: Optional[str] = '/cns/USERS/zzhixuan/weights',
             blip_model_id: str = "Salesforce/blip-image-captioning-base",
             load_blip: bool = True) -> "SDPipeline":
        """
        Load pretrained SD2.1 pipeline with optional BLIP captioner.
        
        Args:
            model_id: HuggingFace model ID for SD2.1
            device: torch device
            dtype: model dtype
            cache_dir: shared weight cache directory
            blip_model_id: HuggingFace model ID for BLIP
            load_blip: whether to load BLIP captioner
        """
        scheduler = DDIMScheduler.from_pretrained(
            model_id, subfolder='scheduler', cache_dir=cache_dir
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, scheduler=scheduler, torch_dtype=dtype,
            cache_dir=cache_dir
        )
        pipe.to(device)
        
        captioner = None
        if load_blip:
            print("Loading BLIP captioner...")
            captioner = BLIPCaptioner(
                model_id=blip_model_id, device=device, cache_dir=cache_dir
            )
        
        return cls(pipe, captioner=captioner)
    
    def auto_caption(self, image: Image.Image) -> str:
        """
        Generate a caption for an image using BLIP.
        Raises RuntimeError if BLIP was not loaded.
        """
        if self.captioner is None:
            raise RuntimeError(
                "BLIP captioner not loaded. Pass load_blip=True to SDPipeline.load() "
                "or provide prompts manually."
            )
        return self.captioner.caption(image)
    
    def unload_captioner(self):
        """Free BLIP GPU memory after captioning is done."""
        if self.captioner is not None:
            self.captioner.unload()
            print("BLIP captioner unloaded from GPU.")
    
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Text → CLIP embedding [1, 77, 768]."""
        return SDModel._encode_prompt(self.pipe, prompt)
    
    def get_timestep(self, noise_level: float) -> int:
        """
        Convert noise_level ∈ (0, 1] to a discrete diffusion timestep.
        noise_level=0.6 means 60% of the way through the diffusion process.
        """
        timesteps = self.pipe.scheduler.timesteps
        idx = max(int(len(timesteps) * noise_level), 1)
        return timesteps[-idx]
    
    def get_sigma_from_timestep(self, t: int) -> float:
        """
        Convert discrete timestep to an effective sigma for GFM.
        σ_eff = sqrt((1 - α_t) / α_t), the SNR-based sigma.
        """
        alpha_t = self.pipe.scheduler.alphas_cumprod[t]
        return ((1 - alpha_t) / alpha_t).sqrt().item()
    
    # --- DDIM Forward Inversion ---
    
    @torch.no_grad()
    def ddim_forward(self, latent: torch.Tensor, embed_cond: torch.Tensor,
                     noise_level: float, cfg_scale: float = 0.5) -> torch.Tensor:
        """
        DDIM forward inversion: clean latent → noisy latent at timestep τ.
        
        Args:
            latent: [1, 4, 64, 64] clean VAE latent
            embed_cond: [1, 77, 768] text embedding
            noise_level: fraction of diffusion process (0.0 = clean, 1.0 = pure noise)
            cfg_scale: classifier-free guidance scale for inversion
        
        Returns:
            noisy_latent: [1, 4, 64, 64] inverted latent at noise_level
        """
        if noise_level == 0:
            return latent
        
        timesteps = self.pipe.scheduler.timesteps
        target_idx = max(int(len(timesteps) * noise_level), 1)
        inv_timesteps = list(reversed(timesteps[-target_idx:].tolist()))
        
        if cfg_scale > 0:
            prompt_cfg = torch.cat([self.embed_uncond, embed_cond])
        
        step_size = self.pipe.scheduler.config.num_train_timesteps // \
                    self.pipe.scheduler.num_inference_steps
        for t in inv_timesteps:
            t_prev = max(t - step_size, 0)
            t_tensor = torch.tensor(t, device=self.device)

            alpha_prev = self.pipe.scheduler.alphas_cumprod[t_prev]
            alpha_t = self.pipe.scheduler.alphas_cumprod[t]

            if cfg_scale > 0:
                latent_in = torch.cat([latent] * 2)
                noise_pred = self.pipe.unet(latent_in, t_tensor,
                                           encoder_hidden_states=prompt_cfg).sample
                noise_uncond, noise_cond = noise_pred.chunk(2)
                epsilon = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                epsilon = self.pipe.unet(latent, t_tensor,
                                        encoder_hidden_states=self.embed_uncond).sample

            x0 = (latent - (1 - alpha_prev).sqrt() * epsilon) / alpha_prev.sqrt()
            latent = alpha_t.sqrt() * x0 + (1 - alpha_t).sqrt() * epsilon
        
        return latent
    
    # --- DDIM Backward (Denoising) ---
    
    @torch.no_grad()
    def ddim_backward(self, latent: torch.Tensor, embed_cond: torch.Tensor,
                      noise_level: float, cfg_scale: float = 0.5) -> torch.Tensor:
        """
        DDIM backward: noisy latent → clean latent.
        
        Args:
            latent: [1, 4, 64, 64] noisy latent
            embed_cond: [1, 77, 768] text embedding  
            noise_level: fraction of diffusion process
            cfg_scale: classifier-free guidance scale
        
        Returns:
            clean_latent: [1, 4, 64, 64] denoised latent
        """
        if noise_level == 0:
            return latent
        
        timesteps = self.pipe.scheduler.timesteps
        target_idx = max(int(len(timesteps) * noise_level), 1)
        denoise_timesteps = timesteps[-target_idx:]
        
        if cfg_scale > 0:
            prompt_cfg = torch.cat([self.embed_uncond, embed_cond])
        
        for t in denoise_timesteps:
            if cfg_scale > 0:
                latent_in = torch.cat([latent] * 2)
                noise_pred = self.pipe.unet(latent_in, t,
                                           encoder_hidden_states=prompt_cfg).sample
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = self.pipe.unet(latent, t,
                                           encoder_hidden_states=self.embed_uncond).sample
            
            latent = self.pipe.scheduler.step(noise_pred, t, latent).prev_sample
        
        return latent
    
    def ddim_backward_batch(self, latents: torch.Tensor, embed_cond: torch.Tensor,
                            noise_level: float, cfg_scale: float = 0.5,
                            batch_size: int = 4) -> torch.Tensor:
        """Denoise a batch of latents, processing batch_size at a time."""
        results = []
        for i in range(0, latents.shape[0], batch_size):
            batch = latents[i:i+batch_size]
            embed_batch = embed_cond.expand(batch.shape[0], -1, -1)
            
            if cfg_scale > 0:
                prompt_cfg = torch.cat([
                    self.embed_uncond.expand(batch.shape[0], -1, -1),
                    embed_batch
                ])
            
            timesteps = self.pipe.scheduler.timesteps
            target_idx = max(int(len(timesteps) * noise_level), 1)
            
            lat = batch
            for t in timesteps[-target_idx:]:
                if cfg_scale > 0:
                    lat_in = torch.cat([lat] * 2)
                    noise_pred = self.pipe.unet(lat_in, t,
                                               encoder_hidden_states=prompt_cfg).sample
                    noise_uncond, noise_cond = noise_pred.chunk(2)
                    noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
                else:
                    noise_pred = self.pipe.unet(lat, t,
                                               encoder_hidden_states=self.embed_uncond).sample
                lat = self.pipe.scheduler.step(noise_pred, t, lat).prev_sample
            
            results.append(lat)
        
        return torch.cat(results, dim=0)
    
    # --- Text Inversion (Optional) ---
    
    def text_inversion(self, prompt: str, latent: torch.Tensor,
                       steps: int = 500, lr: float = 0.005) -> torch.Tensor:
        """
        Optimize text embedding to better reconstruct the given latent.
        Optional — skip for efficiency, marginal quality difference.
        """
        embed = self.encode_prompt(prompt).clone().requires_grad_(True)
        optimizer = torch.optim.AdamW([embed], lr=lr)
        
        num_train_timesteps = self.pipe.scheduler.config.num_train_timesteps
        
        for _ in tqdm(range(steps), desc="Text Inversion"):
            optimizer.zero_grad()
            noise = torch.randn_like(latent)
            t = torch.randint(num_train_timesteps, (1,), device=self.device)
            noisy = self.pipe.scheduler.add_noise(latent, noise, t)
            
            with torch.autocast(device_type=str(self.device).split(':')[0], 
                               dtype=torch.float16):
                pred = self.pipe.unet(noisy, t, encoder_hidden_states=embed).sample
                loss = F.mse_loss(pred.float(), noise.float())
            
            loss.backward()
            optimizer.step()
        
        return embed.detach()

    # --- Sphere Utilities ---
    
    @staticmethod
    def project_to_sphere(x: torch.Tensor, radius: float) -> torch.Tensor:
        """Project latent vector(s) onto hypersphere of given radius."""
        x_flat = x.view(x.shape[0], -1)
        norms = x_flat.norm(dim=-1, keepdim=True)
        x_flat = x_flat * (radius / (norms + 1e-8))
        return x_flat.view_as(x)


# ============================================================
# 4. Full GFM Interpolation Pipeline
# ============================================================

def run_gfm_interpolation(
    sd_pipe: SDPipeline,
    imgA: Image.Image,
    imgB: Image.Image,
    promptA: Optional[str] = None,
    promptB: Optional[str] = None,
    noise_level: float = 0.6,
    cfg_scale: float = 0.5,
    num_steps: int = 10,
    lam: float = 1.0,
    lr: float = 0.01,
    max_iters: int = 400,
    use_text_inversion: bool = False,
    output_frames: int = 10,
) -> Dict[str, Any]:
    """
    Full pipeline: two images → GFM interpolation → output frames.
    
    Prompts are auto-generated via BLIP if not provided.
    
    Pipeline:
      1. Auto-caption images with BLIP (if prompts not given)
      2. Encode images to VAE latent space
      3. DDIM forward inversion to noise level τ
      4. GFM optimization with sphere constraint
      5. DDIM backward to denoise
      6. VAE decode to images
    
    Args:
        sd_pipe: Loaded SDPipeline (with BLIP if prompts not provided)
        imgA, imgB: Input PIL images (endpoints)
        promptA, promptB: Text prompts (auto-generated via BLIP if None)
        noise_level: Diffusion noise level (0.6 recommended by Yu et al.)
        cfg_scale: CFG scale for DDIM forward/backward
        num_steps: Number of GFM waypoints
        lam: GFM potential weight λ
        lr: GFM learning rate
        max_iters: GFM optimization iterations
        use_text_inversion: Whether to run text inversion (slow, marginal gain)
        output_frames: Number of frames in final output
    
    Returns:
        Dictionary with 'images', 'latents_noisy', 'latents_clean', 
        'losses', 'promptA', 'promptB'
    """
    from gfm.path.geodesic_interpolation import ELInterpolator  # Your GFM
    
    device = sd_pipe.device
    output_frames = num_steps
    sd_pipe.model.mode = 'denoise'
    # --- Step 1: Auto-caption if prompts not provided ---
    if promptA is None:
        print("Auto-captioning image A with BLIP...")
        promptA = sd_pipe.auto_caption(imgA)
        print(f"  Caption A: \"{promptA}\"")
    if promptB is None:
        print("Auto-captioning image B with BLIP...")
        promptB = sd_pipe.auto_caption(imgB)
        print(f"  Caption B: \"{promptB}\"")
    
    # Free BLIP GPU memory before running diffusion
    sd_pipe.unload_captioner()
    
    # --- Step 2: Encode images ---
    print("Encoding images...")
    latA = sd_pipe.autoencoder.encode(imgA)  # [1, 4, 64, 64]
    latB = sd_pipe.autoencoder.encode(imgB)
    
    # --- Step 3: Prepare text conditioning ---
    print("Preparing text conditioning...")
    if use_text_inversion:
        embedA = sd_pipe.text_inversion(promptA, latA)
        embedB = sd_pipe.text_inversion(promptB, latB)
    else:
        embedA = sd_pipe.encode_prompt(promptA)
        embedB = sd_pipe.encode_prompt(promptB)
    
    # --- Step 4: DDIM forward inversion ---
    print(f"DDIM forward inversion (noise_level={noise_level})...")
    latA_noisy = sd_pipe.ddim_forward(latA, embedA, noise_level, cfg_scale)
    latB_noisy = sd_pipe.ddim_forward(latB, embedB, noise_level, cfg_scale)
    
    # Compute sphere radius (average of endpoint norms)
    normA = latA_noisy.view(-1).norm().item()
    normB = latB_noisy.view(-1).norm().item()
    sphere_radius = 0.5 * (normA + normB)
    print(f"Sphere radius: {sphere_radius:.2f} (normA={normA:.2f}, normB={normB:.2f})")
    
    # Project endpoints to sphere
    latA_noisy = SDPipeline.project_to_sphere(latA_noisy, sphere_radius)
    latB_noisy = SDPipeline.project_to_sphere(latB_noisy, sphere_radius)
    
    # --- Step 5: Configure GFM model ---
    timestep = sd_pipe.get_timestep(noise_level)
    sigma_eff = sd_pipe.get_sigma_from_timestep(timestep)
    print(f"Timestep τ={timestep}, effective σ={sigma_eff:.4f}")
    
    # Set up conditioning: interpolate embeddings linearly
    embed_avg = 0.5 * (embedA + embedB)
    sd_pipe.model.set_conditioning(embed_avg, timestep)
    
    # --- Step 6: Run GFM ---
    print(f"Running GFM (steps={num_steps}, λ={lam}, iters={max_iters})...")
    interpolator = ELInterpolator(
        diffusion_model=sd_pipe.model,
        autoencoder=sd_pipe.autoencoder,
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
    # optimized_path: [1, num_steps, 4, 64, 64]
    
    # Project all waypoints back to sphere
    B, T = optimized_path.shape[:2]
    for t in range(T):
        optimized_path[:, t] = SDPipeline.project_to_sphere(
            optimized_path[:, t], sphere_radius
        )
    
    # --- Step 7: Resample to output_frames if needed ---
    if output_frames != num_steps:
        t_out = torch.linspace(0, 1, output_frames, device=device)
        t_orig = torch.linspace(0, 1, num_steps, device=device)
        indices = torch.searchsorted(t_orig, t_out).clamp(0, num_steps - 1)
        output_latents_noisy = optimized_path[:, indices]
    else:
        output_latents_noisy = optimized_path
    
    n_frames = output_latents_noisy.shape[1]
    
    # --- Step 8: DDIM backward (denoise) ---
    print(f"DDIM backward denoising ({n_frames} frames)...")
    output_latents_clean = []
    output_images = []
    
    for t_idx in tqdm(range(n_frames), desc="Denoising frames"):
        t_frac = t_idx / (n_frames - 1)  # 0 to 1
        embed_t = (1 - t_frac) * embedA + t_frac * embedB
        
        lat_noisy = output_latents_noisy[:, t_idx]  # [1, 4, 64, 64]
        
        # Endpoints: use original images directly (higher quality)
        if t_idx == 0:
            output_images.append(imgA)
            output_latents_clean.append(latA)
            continue
        elif t_idx == n_frames - 1:
            output_images.append(imgB)
            output_latents_clean.append(latB)
            continue
        
        lat_clean = sd_pipe.ddim_backward(lat_noisy, embed_t, noise_level, cfg_scale)
        img = sd_pipe.autoencoder.decode(lat_clean)
        
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
        'timestep': timestep,
        'promptA': promptA,
        'promptB': promptB,
    }


# ============================================================
# 5. Visualization Utilities
# ============================================================

def save_interpolation_strip(images: List[Image.Image], save_path: str,
                              size=(512, 512), padding=10):
    """Save a horizontal strip of images."""
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
    """Save individual frames."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(output_dir, f"frame_{i:03d}.png"))
    print(f"Saved {len(images)} frames to {output_dir}")


# ============================================================
# 6. Entry point for quick testing
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GFM Image Interpolation with SD2.1")
    parser.add_argument("--imgA", type=str, required=True, help="Path to start image")
    parser.add_argument("--imgB", type=str, required=True, help="Path to end image")
    parser.add_argument("--promptA", type=str, default=None,
                        help="Prompt for start image (auto-captioned if omitted)")
    parser.add_argument("--promptB", type=str, default=None,
                        help="Prompt for end image (auto-captioned if omitted)")
    parser.add_argument("--noise_level", type=float, default=0.6)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max_iters", type=int, default=800)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./output/sd_interp")
    parser.add_argument("--use_text_inv", action="store_true")
    parser.add_argument("--no_blip", action="store_true",
                        help="Skip loading BLIP (requires --promptA and --promptB)")
    parser.add_argument("--cache_dir", type=str, default="/cns/USERS/zzhixuan/weights",
                        help="Cache directory for model weights")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Validate: if no BLIP, prompts must be provided
    if args.no_blip and (args.promptA is None or args.promptB is None):
        parser.error("--no_blip requires both --promptA and --promptB")
    
    # Load pipeline
    print("Loading SD2.1...")
    sd_pipe = SDPipeline.load(
        device=args.device,
        cache_dir=args.cache_dir,
        load_blip=not args.no_blip
    )

    
    
    # Load images
    imgA = Image.open(args.imgA)
    imgB = Image.open(args.imgB)
    
    # Run interpolation
    results = run_gfm_interpolation(
        sd_pipe, imgA, imgB,
        promptA=args.promptA,
        promptB=args.promptB,
        noise_level=args.noise_level,
        num_steps=args.num_steps,
        lam=args.lam,
        lr=args.lr,
        max_iters=args.max_iters,
        use_text_inversion=args.use_text_inv,
    )
    
    # Save outputs
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    save_interpolation_strip(results['images'], 
                             os.path.join(args.output_dir, "strip.png"))
    save_frames(results['images'], os.path.join(args.output_dir, "frames"))
    
    print(f"\nResults saved to {args.output_dir}")
    print(f"Caption A: \"{results['promptA']}\"")
    print(f"Caption B: \"{results['promptB']}\"")
    print(f"Sphere radius: {results['sphere_radius']:.2f}")
    print(f"Effective σ: {results['sigma_eff']:.4f}")
    print(f"Final GFM loss: {results['losses'][-1]:.6f}" if results['losses'] else "")