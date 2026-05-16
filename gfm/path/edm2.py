"""
EDM2 GFM Interpolation Core Pipeline.
Handles model wrapping, VAE normalization (using official EDM2 encoder), and GFM execution.
"""
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm

import dnnlib
dnnlib.util.set_cache_dir('/cns/USERS/zzhixuan/weights/edm2') # 取消注释并配置你的缓存路径

# --- 预设配置字典 ---
model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions'
CONFIG_PRESETS = {
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

def edm_sampler(net, latents, class_labels=None, num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, init_at_sigma=True):
    """标准的 EDM 采样器"""
    dtype = torch.float32
    step_indices = torch.arange(num_steps, dtype=dtype, device=latents.device)
    t_steps = (sigma_max ** (1/rho) + step_indices / (num_steps - 1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    x_next = latents.to(dtype) if init_at_sigma else latents.to(dtype) * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        denoised = net(x_next, t_cur, class_labels).to(dtype)
        d_cur = (x_next - denoised) / t_cur
        x_prime = x_next + (t_next - t_cur) * d_cur

        if i < num_steps - 1:
            denoised_prime = net(x_prime, t_next, class_labels).to(dtype)
            d_prime = (x_prime - denoised_prime) / t_next
            x_next = x_next + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
        else:
            x_next = x_prime

    return x_next

class EDM2ModelWrapper(torch.nn.Module):
    """封装 EDM2 网络，处理 One-hot 转换、连续软标签以及 CFG。"""
    def __init__(self, net, gnet=None, guidance=1.0, label_dim=1000):
        super().__init__()
        self.net = net
        self.gnet = gnet if gnet is not None else net
        self.guidance = guidance
        self.label_dim = label_dim
        self.device = next(net.parameters()).device
        self._interp_labels = None
        self._n_inner = None

    def set_interpolation_labels(self, classA: int, classB: int, num_steps: int):
        n_inner = num_steps - 2
        onehot_A = F.one_hot(torch.tensor(classA), self.label_dim).float().to(self.device)
        onehot_B = F.one_hot(torch.tensor(classB), self.label_dim).float().to(self.device)
        alphas = torch.linspace(1.0 / (num_steps - 1), (num_steps - 2) / (num_steps - 1), n_inner, device=self.device)
        self._interp_labels = torch.stack([(1 - a) * onehot_A + a * onehot_B for a in alphas])
        self._n_inner = n_inner

    def clear_interpolation_labels(self):
        self._interp_labels = None
        self._n_inner = None

    @torch.no_grad()
    def forward(self, x, sigma, class_labels=None):
        B = x.shape[0]
        if self._interp_labels is not None and self._n_inner is not None:
            if B == self._n_inner:
                labels = self._interp_labels
            elif B % self._n_inner == 0:
                labels = self._interp_labels.repeat(B // self._n_inner, 1)
            else:
                labels = self._interp_labels[self._n_inner // 2].unsqueeze(0).expand(B, -1)
        else:
            labels = class_labels if (class_labels is None or class_labels.dim() == 2) else F.one_hot(class_labels.long(), self.label_dim).float()

        Dx = self.net(x, sigma, labels).to(torch.float32)
        if self.guidance != 1.0 and self.gnet is not self.net:
            ref_Dx = self.gnet(x, sigma, labels).to(torch.float32)
            Dx = ref_Dx.lerp(Dx, self.guidance)
        return Dx

    @torch.no_grad()
    def sample(self, cond, batch_seeds=None, num_steps=32, sigma_max=80):
        B = cond.shape[0]
        if batch_seeds is not None:
            noise = torch.stack([torch.randn(self.net.img_channels, self.net.img_resolution, self.net.img_resolution, 
                                             generator=torch.Generator(self.device).manual_seed(int(s) % (1<<32)), device=self.device) for s in batch_seeds])
        else:
            noise = torch.randn(B, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=self.device)
        
        saved_interp, saved_n = self._interp_labels, self._n_inner
        self.clear_interpolation_labels()
        x_clean = edm_sampler(self, noise, class_labels=cond, num_steps=num_steps, sigma_max=sigma_max, init_at_sigma=False)
        self._interp_labels, self._n_inner = saved_interp, saved_n
        return x_clean.float()

class EDM2Autoencoder:
    """基于 EDM2 官方 StabilityVAEEncoder 的精确封装"""
    def __init__(self, encoder, device="cuda"):
        self.encoder = encoder
        self.device = device

    def encode(self, image: Image.Image, resolution: int = 512) -> torch.Tensor:
        # 1. 裁剪和缩放
        w, h = image.size
        crop = min(w, h)
        left, top = (w - crop) // 2, (h - crop) // 2
        img = image.crop((left, top, left + crop, top + crop)).resize((resolution, resolution), Image.LANCZOS)
        
        # 2. 【关键修复】：官方 encode_pixels 要求输入为 0~255 的 tensor
        arr = np.array(img) # uint8, [H, W, 3]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32)
        
        with torch.no_grad():
            # 3. 【关键修复】：手动剥离官方 encode_latents 里的随机性 (randn_like)
            # 先调用 encode_pixels 拿到 8通道的 raw latents
            raw_latents = self.encoder.encode_pixels(tensor) 
            mean, std = raw_latents.chunk(2, dim=1) # 切分为均值和方差
            
            # 提取官方的缩放因子并转换为 Tensor
            scale = torch.tensor(self.encoder.scale, device=self.device).reshape(1, -1, 1, 1)
            bias = torch.tensor(self.encoder.bias, device=self.device).reshape(1, -1, 1, 1)
            
            # 直接使用 mean 进行确定的仿射变换（抛弃 std 带来的随机性）
            latent = mean * scale + bias
            
        return latent

    def decode(self, latent: torch.Tensor) -> Image.Image:
        with torch.no_grad():
            # 官方 decode 会自动执行逆向仿射变换 -> VAE decode -> 映射并 clamp 到 0~255 (返回 uint8)
            img_tensor = self.encoder.decode(latent)
            
        arr = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr, 'RGB')

    def eval(self):
        pass

class EDM2GFMPipeline:
    def __init__(self, model, autoencoder, device="cuda"):
        self.model = model
        self.autoencoder = autoencoder
        self.device = torch.device(device)

    @classmethod
    def load(cls, preset=None, net_path=None, gnet_path=None, guidance=1.0, device="cuda"):
        if preset in CONFIG_PRESETS:
            cfg = CONFIG_PRESETS[preset]
            net_path = net_path or cfg.get('net')
            gnet_path = gnet_path or cfg.get('gnet')
            guidance = cfg.get('guidance', guidance)

        print(f"Loading Main Network & Official Encoder from {net_path}...")
        with dnnlib.util.open_url(net_path) as f:
            data = pickle.load(f)
        
        net = data['ema'].to(device).eval()
        encoder = data['encoder'] # 直接提取官方 Encoder 解决 Normalization 问题
        encoder.init(device)

        gnet = None
        if gnet_path and guidance != 1.0:
            print(f"Loading Guidance Network from {gnet_path}...")
            with dnnlib.util.open_url(gnet_path) as f:
                gnet = pickle.load(f)['ema'].to(device).eval()

        model = EDM2ModelWrapper(net, gnet=gnet, guidance=guidance, label_dim=net.label_dim).eval()
        autoencoder = EDM2Autoencoder(encoder, device=device)
        return cls(model, autoencoder, device=device)

def denoise_path_cross_class(model, path, classA, classB, sigma_start, num_denoise_steps):
    device = path.device
    B, num_steps = path.shape[:2]
    onehot_A = F.one_hot(torch.tensor(classA), model.label_dim).float().to(device)
    onehot_B = F.one_hot(torch.tensor(classB), model.label_dim).float().to(device)

    clean_frames = []
    for t_idx in tqdm(range(num_steps), desc="Denoising frames"):
        alpha = t_idx / max(num_steps - 1, 1)
        soft_label = ((1 - alpha) * onehot_A + alpha * onehot_B).unsqueeze(0).expand(B, -1)
        
        def denoise_fn(x, sigma, _labels=None):
            return model(x, sigma, class_labels=soft_label)

        frame_clean = edm_sampler(denoise_fn, path[:, t_idx], sigma_max=sigma_start, init_at_sigma=True, num_steps=num_denoise_steps)
        clean_frames.append(frame_clean)
    return torch.stack(clean_frames, dim=1).float()

# ---------- GFM 核心执行函数 ----------

def run_gfm_interpolation(pipe, latA, latB, classA, classB, imgA=None, imgB=None, args=None):
    """通用的 GFM 路径优化与解码函数"""
    from gfm.path.geodesic_interpolation import ELInterpolator, SeqELInterpolator, SphericalInterpolator
    
    device = pipe.device
    pipe.model.set_interpolation_labels(classA, classB, args.num_steps)

    if args.interpolator == "spherical":
        interpolator = SphericalInterpolator(pipe.model, pipe.autoencoder, device=str(device))
    elif args.interpolator == "el":
        interpolator = ELInterpolator(pipe.model, pipe.autoencoder, device=str(device))
    else:
        interpolator = SeqELInterpolator(pipe.model, pipe.autoencoder, device=str(device))

    print(f"Adding noise (sigma={args.noise_level})...")
    noisy_latents = interpolator.add_noise(torch.cat([latA, latB], dim=0), sigma=args.noise_level)

    print(f"Running GFM ({args.interpolator}, steps={args.num_steps}, iters={args.max_iters})...")
    dummy_label = torch.tensor([classA], device=device)
    
    if args.interpolator == "spherical":
        path = interpolator.optimize_path(start_latent=noisy_latents[0:1], end_latent=noisy_latents[1:2], init_with_slerp=False, num_steps=args.num_steps)
        info = {}
    else:
        path, info = interpolator.optimize_path(start_latent=noisy_latents[0:1], end_latent=noisy_latents[1:2], num_steps=args.num_steps, 
                                                sigma=args.noise_level, lr=args.lr, max_iters=args.max_iters, lam=args.lam, class_label=dummy_label, verbose=True)
    if isinstance(path, tuple): path = path[0]

    pipe.model.clear_interpolation_labels()
    clean_path = denoise_path_cross_class(pipe.model, path, classA, classB, args.noise_level, args.num_denoise_steps)

    print("Decoding to images...")
    images = []
    # 首尾帧如果是真实图片，直接使用原图以防 VAE 画质损失
    images.append(imgA.resize((512, 512), Image.LANCZOS) if imgA else pipe.autoencoder.decode(clean_path[0, 0:1]))
    for i in range(1, clean_path.shape[1] - 1):
        images.append(pipe.autoencoder.decode(clean_path[0, i:i+1]))
    images.append(imgB.resize((512, 512), Image.LANCZOS) if imgB else pipe.autoencoder.decode(clean_path[0, -1:]))

    return {"images": images, "latents_clean": clean_path, "losses": info.get("losses", []) if isinstance(info, dict) else []}

# ---------- 辅助保存工具 ----------
def save_results(images, output_dir, prefix=""):
    os.makedirs(output_dir, exist_ok=True)
    strip_path = os.path.join(output_dir, f"{prefix}strip.png")
    
    n, w, h, p = len(images), 256, 256, 5
    strip = Image.new("RGB", ((w + 2*p) * n, h + 2*p), (255, 255, 255))
    for i, img in enumerate(images):
        strip.paste(img.resize((w, h), Image.LANCZOS), (i * (w + 2*p) + p, p))
        img.save(os.path.join(output_dir, f"{prefix}frame_{i:03d}.png"))
    strip.save(strip_path)
    print(f"Results saved to {output_dir}")