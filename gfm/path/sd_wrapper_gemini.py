"""
EDM Continuous-Time Wrapper for Geodesic Force Matching (GFM)
Replaces discrete DDIM with continuous ODE Euler integration.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any
from diffusers import StableDiffusionPipeline, DDIMScheduler
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
# 1. SDModel: Adapted for Continuous Sigma
# ============================================================

class SDModel(torch.nn.Module):
    def __init__(self, pipe: StableDiffusionPipeline, mode='nfsd'):
        super().__init__()
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler
        self.device = pipe.device
        self.mode = mode
        
        self._embed_uncond = self._encode_prompt(pipe, "")
        self._embed_neg = self._encode_prompt(
            pipe,
            "A doubling image, unrealistic, artifacts, distortions, "
            "unnatural blending, ghosting effects, overlapping edges, "
            "harsh transitions, motion blur, poor resolution, low detail"
        )
        self._embed_cond = None
        self._sigma = None # Use continuous sigma instead of discrete t
        
        # Precompute mapping for t <-> sigma
        alphas = self.scheduler.alphas_cumprod.cpu().numpy()
        self._sigmas_array = np.sqrt((1 - alphas) / alphas)
        self._ts_array = np.arange(len(alphas))

    @staticmethod
    def _encode_prompt(pipe, prompt: str) -> torch.Tensor:
        tokens = pipe.tokenizer(
            prompt, max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt", padding="max_length", truncation=True
        ).input_ids
        return pipe.text_encoder(tokens.to(pipe.device))[0]
    
    def set_conditioning(self, embed_cond: torch.Tensor, sigma: float):
        """Set text conditioning and continuous sigma."""
        self._embed_cond = embed_cond
        self._sigma = sigma
        
    def _t_from_sigma(self, sigma: float) -> float:
        """Interpolate exact continuous timestep t from sigma."""
        return float(np.interp(sigma, self._sigmas_array, self._ts_array))

    def _noise_pred(self, latent, t_tensor, embed):
        return self.unet(latent, t_tensor, encoder_hidden_states=embed).sample
    
    def set_endpoints(self, embedA: torch.Tensor, embedB: torch.Tensor, sigma: float, num_steps: int):
            """为 GFM 动态插值设置端点"""
            self._embedA = embedA
            self._embedB = embedB
            self._sigma = sigma
            self._num_steps = num_steps
            self._embed_cond = None # 清空静态 condition

    @torch.no_grad()
    def forward(self, latent: torch.Tensor, sigma: torch.Tensor, class_labels=None) -> torch.Tensor:
        batch_size = latent.shape[0]
        sigma_val = sigma[0].item() if sigma.dim() > 0 else sigma.item()
        
        # 获取连续时间步 t
        t = self._t_from_sigma(sigma_val)
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.float32)

        # ==========================================================
        # 动态 Condition 拦截逻辑
        # ==========================================================
        if hasattr(self, '_embedA') and self._embedA is not None:
            n_inner = self._num_steps - 2
            # 探测这是否是 ELInterpolator 内部发起的批处理调用
            if n_inner > 0 and batch_size % n_inner == 0:
                B = batch_size // n_inner
                
                # 构建内部点的相对进度 s (范围在 0 到 1 之间，不包含 0 和 1)
                # 例如 num_steps=10, 内部点 s = [1/9, 2/9, ..., 8/9]
                s = torch.linspace(1.0 / (self._num_steps - 1), 
                                   float(n_inner) / (self._num_steps - 1), 
                                   n_inner, device=self.device)
                
                # 适配 flatten 后的张量: [s1, s2... sn, s1, s2... sn]
                s = s.repeat(B).view(-1, 1, 1) 
                
                # 动态线性插值 Embedding
                embed_cond = (1.0 - s) * self._embedA + s * self._embedB
            else:
                # Fallback: 如果是测试单图或非插值状态
                embed_cond = 0.5 * (self._embedA + self._embedB)
                embed_cond = embed_cond.expand(batch_size, -1, -1)
        elif self._embed_cond is not None:
            embed_cond = self._embed_cond.expand(batch_size, -1, -1)
        else:
            raise ValueError("No conditioning set in SDModel.")
        # ==========================================================

        if self.mode == 'denoise':
            # 后面的逻辑完全不变，把原来的 self._embed_cond 换成局部变量 embed_cond
            noise_pred = self._noise_pred(latent, t_tensor, embed_cond)
            z = latent * math.sqrt(1.0 + sigma_val**2)
            x_pred = z - sigma_val * noise_pred
            return x_pred
            
        elif self.mode == 'nfsd':
            embed_neg = self._embed_neg.expand(batch_size, -1, -1)
            with torch.autocast(device_type=str(self.device).split(':')[0], dtype=torch.float16):
                ep_cond = self._noise_pred(latent, t_tensor, embed_cond)
                ep_neg = self._noise_pred(latent, t_tensor, embed_neg)
            
            grad = 0.5 * (-ep_cond.float() + ep_neg.float())
            x_pred = latent + (sigma_val ** 2) * grad
            return x_pred
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# ============================================================
# 2. SDAutoencoder (Unchanged)
# ============================================================

class SDAutoencoder:
    def __init__(self, pipe: StableDiffusionPipeline):
        self.vae = pipe.vae
        self.device = pipe.device
        self.scaling_factor = pipe.vae.config.scaling_factor
    
    def encode(self, image: Image.Image) -> torch.Tensor:
        img_tensor = self._preprocess(image)
        latent = self.vae.encode(img_tensor)['latent_dist'].mean
        return latent * self.scaling_factor
    
    def decode(self, latent: torch.Tensor) -> Image.Image:
        latent = latent / self.scaling_factor
        img_tensor = self.vae.decode(latent)['sample']
        return self._postprocess(img_tensor)
    
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((512, 512))
        arr = np.array(image).astype(np.float32) / 255.0
        arr = 2.0 * arr - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    def _postprocess(self, tensor: torch.Tensor) -> Image.Image:
        arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        arr = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))
    def eval(self):
        pass


# ============================================================
# 3. SDPipeline: Pure EDM Integrator
# ============================================================

class SDPipeline:
    def __init__(self, pipe: StableDiffusionPipeline, n_inference_steps: int = 50,
                 captioner: Optional[BLIPCaptioner] = None):
        self.pipe = pipe
        self.device = pipe.device
        self.n_inference_steps = n_inference_steps
        self.captioner = captioner
        
        pipe.unet.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.vae.requires_grad_(False)
        
        self.model = SDModel(pipe, mode='nfsd')
        self.autoencoder = SDAutoencoder(pipe)
        self.embed_uncond = SDModel._encode_prompt(pipe, "")

    @classmethod
    def load(cls, model_id: str = "sd2-community/stable-diffusion-2-1",
             device: str = "cuda", dtype=torch.float32,
             cache_dir: Optional[str] = None,
             load_blip: bool = True) -> "SDPipeline":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype, cache_dir=cache_dir
        )
        pipe.to(device)
        captioner = BLIPCaptioner(device=device, cache_dir=cache_dir) if load_blip else None
        return cls(pipe, captioner=captioner)

    def auto_caption(self, image: Image.Image) -> str:
        return self.captioner.caption(image)
    
    def unload_captioner(self):
        if self.captioner is not None: self.captioner.unload()

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        return SDModel._encode_prompt(self.pipe, prompt)
    
    def get_timestep(self, noise_level: float) -> int:
        timesteps = self.pipe.scheduler.timesteps
        if len(timesteps) == 0:
            self.pipe.scheduler.set_timesteps(self.n_inference_steps)
            timesteps = self.pipe.scheduler.timesteps
        idx = max(int(len(timesteps) * noise_level), 1)
        return timesteps[-idx]
    
    def get_sigma_from_timestep(self, t: int) -> float:
        alpha_t = self.pipe.scheduler.alphas_cumprod[t]
        return ((1 - alpha_t) / alpha_t).sqrt().item()

    # --- EDM Math Utilities ---
    
    def _get_edm_sigmas(self, sigma_min: float, sigma_max: float, num_steps: int) -> torch.Tensor:
        """Karras polynomial schedule for smooth ODE integration."""
        rho = 7.0
        t = torch.linspace(0, 1, num_steps, device=self.device)
        sigmas = (sigma_min**(1/rho) + t * (sigma_max**(1/rho) - sigma_min**(1/rho)))**rho
        return sigmas

    def _get_derivative(self, z: torch.Tensor, sigma: float, embed_cond: torch.Tensor, 
                        cfg_scale: float) -> torch.Tensor:
        """Compute dz/dsigma = epsilon_theta via SD UNet."""
        c_in = 1.0 / math.sqrt(sigma**2 + 1.0)
        x_in = z * c_in # Convert EDM latent to SD VP latent
        
        t = self.model._t_from_sigma(sigma)
        t_tensor = torch.full((z.shape[0],), t, device=self.device, dtype=torch.float32)

        if cfg_scale > 0:
            x_in_double = torch.cat([x_in] * 2)
            t_double = torch.cat([t_tensor] * 2)
            embed_double = torch.cat([self.embed_uncond.expand_as(embed_cond), embed_cond])
            
            noise_pred = self.pipe.unet(x_in_double, t_double, encoder_hidden_states=embed_double).sample
            noise_uncond, noise_cond = noise_pred.chunk(2)
            epsilon = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            epsilon = self.pipe.unet(x_in, t_tensor, encoder_hidden_states=embed_cond).sample
            
        return epsilon

    # --- Continuous ODE Integrators ---

    @torch.no_grad()
    def edm_forward(self, latent_clean: torch.Tensor, embed_cond: torch.Tensor,
                    noise_level: float, cfg_scale: float = 0.0) -> torch.Tensor:
        """
        Inversion via Probability Flow ODE (Euler).
        CRITICAL: cfg_scale is forced to 0.0 for stable forward inversion.
        """
        if noise_level == 0: return latent_clean
        
        # We enforce unconditional forward pass to prevent CFG divergence!
        cfg_scale = 0.0 
        
        t_target = self.get_timestep(noise_level)
        sigma_max = self.get_sigma_from_timestep(t_target)
        
        alphas = self.pipe.scheduler.alphas_cumprod
        sigma_min = max(math.sqrt((1 - alphas[0].item()) / alphas[0].item()), 0.02)
        
        # Map input VP latent to EDM z-space at starting minimum noise
        z = latent_clean * math.sqrt(sigma_min**2 + 1.0)
        sigmas = self._get_edm_sigmas(sigma_min, sigma_max, self.n_inference_steps)

        for i in range(len(sigmas) - 1):
            s_i, s_next = sigmas[i], sigmas[i+1]
            d_i = self._get_derivative(z, s_i.item(), embed_cond, cfg_scale)
            z = z + (s_next - s_i) * d_i # Forward Euler

        # Convert back to SD VP latent for GFM compatibility
        return z / math.sqrt(sigma_max**2 + 1.0)

    @torch.no_grad()
    def edm_backward(self, latent_noisy: torch.Tensor, embed_cond: torch.Tensor,
                     noise_level: float, cfg_scale: float = 0.5) -> torch.Tensor:
        """Generation via Probability Flow ODE (Euler)."""
        if noise_level == 0: return latent_noisy
        
        t_target = self.get_timestep(noise_level)
        sigma_max = self.get_sigma_from_timestep(t_target)
        
        alphas = self.pipe.scheduler.alphas_cumprod
        sigma_min = max(math.sqrt((1 - alphas[0].item()) / alphas[0].item()), 0.02)
        
        # VP -> EDM space
        z = latent_noisy * math.sqrt(sigma_max**2 + 1.0)
        
        # Flip schedule for backward generation
        sigmas = self._get_edm_sigmas(sigma_min, sigma_max, self.n_inference_steps)
        sigmas = torch.flip(sigmas, [0])

        for i in range(len(sigmas) - 1):
            s_i, s_next = sigmas[i], sigmas[i+1] # s_next is smaller
            d_i = self._get_derivative(z, s_i.item(), embed_cond, cfg_scale)
            z = z + (s_next - s_i) * d_i # Backward Euler

        # Final projection to pure signal
        s_final = sigmas[-1].item()
        d_final = self._get_derivative(z, s_final, embed_cond, cfg_scale)
        x_clean = z - s_final * d_final
        return x_clean

    # Alias to keep your test script happy!
    ddim_forward = edm_forward
    ddim_backward = edm_backward

    # --- Sphere Utilities ---
    @staticmethod
    def project_to_sphere(x: torch.Tensor, radius: float) -> torch.Tensor:
        x_flat = x.view(x.shape[0], -1)
        norms = x_flat.norm(dim=-1, keepdim=True)
        x_flat = x_flat * (radius / (norms + 1e-8))
        return x_flat.view_as(x)

    def text_inversion(self, prompt: str, latent: torch.Tensor,
                       steps: int = 500, lr: float = 0.005) -> torch.Tensor:
        embed = self.encode_prompt(prompt).clone().requires_grad_(True)
        optimizer = torch.optim.AdamW([embed], lr=lr)
        num_train_timesteps = self.pipe.scheduler.config.num_train_timesteps
        for _ in tqdm(range(steps), desc="Text Inversion"):
            optimizer.zero_grad()
            noise = torch.randn_like(latent)
            t = torch.randint(num_train_timesteps, (1,), device=self.device)
            noisy = self.pipe.scheduler.add_noise(latent, noise, t)
            with torch.autocast(device_type=str(self.device).split(':')[0], dtype=torch.float16):
                pred = self.pipe.unet(noisy, t, encoder_hidden_states=embed).sample
                loss = F.mse_loss(pred.float(), noise.float())
            loss.backward()
            optimizer.step()
        return embed.detach()

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
    # embed_avg = 0.5 * (embedA + embedB)
    # sd_pipe.model.set_conditioning(embed_avg, timestep)
    sd_pipe.model.set_endpoints(embedA, embedB, sigma_eff, num_steps)

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