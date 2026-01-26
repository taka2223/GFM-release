"""
Minimal test for noise-denoise consistency using GeodesicInterpolator.
"""
from ast import arg
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import argparse
import numpy as np
import mcubes
import torch
import trimesh
from pathlib import Path

from gfm.models import models_class_cond, models_ae
from gfm.path.geodesic_interpolation import GeodesicInterpolator,RBFKernelInterpolator, LandInterpolator, ScoreBasedInterpolator, ELInterpolator, SteinScoreInterpolator, SphericalInterpolator,EL2Interpolator
from gfm.helper.help import decode_and_save, load_model_from_path

def main(args):
    SIGMA = args.noise_level
    category = args.category
    seed = args.seed
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    root = os.path.join(args.output_dir, args.dataset, args.approximator)
    os.makedirs(root, exist_ok=True)
    output_dir = os.path.join(root,
        f"test_denoise_sigma{SIGMA}_category{category}_seed{seed}")
    os.makedirs(output_dir, exist_ok=True)
    
    
    # Load models
    ae_name = args.ae_name
    ae_path = args.ae_path
    ae = load_model_from_path(
        model_name=ae_name,
        ckpt_path=ae_path,
        registry=models_ae.__dict__,
        device=device
    )
    
    dm_name = args.dm_name
    dm_path = args.dm_path
    dm = load_model_from_path(
        model_name=dm_name,
        ckpt_path=dm_path,
        registry=models_class_cond.__dict__,
        device=device
    )
    
    if args.approximator == 'rbf':
        interpolator = RBFKernelInterpolator(dm, ae, device=device)
        interpolator.load(args.h_path)
    elif args.approximator == 'land':
        interpolator = LandInterpolator(dm, ae, device=device)
        interpolator.load(args.h_path)
    elif args.approximator == 'score':
        interpolator = ScoreBasedInterpolator(dm, ae, device=device)
    elif args.approximator == 'el':
        interpolator = ELInterpolator(dm, ae, device=device)
    elif args.approximator == 'stein':
        interpolator = SteinScoreInterpolator(dm, ae, device=device)
    elif args.approximator == 'spherical' or args.approximator == 'linear':
        interpolator = SphericalInterpolator(dm, ae, device=device)
    elif args.approximator == 'el2':
        interpolator = EL2Interpolator(dm, ae, device=device)
    else:
        raise ValueError
    
    ## sampling endpoints
    print(f"\n[Step 1] Sampling 2 clean latents (category={ category})...")
    with torch.no_grad():
        class_labels = torch.tensor([category, category], device=device).long()
        batch_seeds = torch.tensor([seed, seed + 1], device=device)
        clean_latents = dm.sample(cond=class_labels,
                                batch_seeds=batch_seeds).float()

    print(f"Clean latents: shape={clean_latents.shape}, "
        f"mean={clean_latents.mean():.4f}, std={clean_latents.std():.4f}")
    
    print(f"\n[Step 2] Decode clean latents...")
    for i, latent in enumerate(clean_latents):
        decode_and_save(ae=ae,
                        latent=latent.unsqueeze(0),
                        save_path=os.path.join(output_dir, f"clean_{i}.obj"),
                        device=device)
    
    print(f'\n[Step 3] Interpolation between init latents...')

    if args.approximator != 'rbf' and args.approximator != 'land':
        # add noise
        single_class_label = class_labels[:1]
        noisy_latents = interpolator.add_noise(clean_latents, sigma=SIGMA)
        print(f"Noisy latents: shape={noisy_latents.shape}, "
            f"mean={noisy_latents.mean():.4f}, std={noisy_latents.std():.4f}")
        # noisy_latents = clean_latents
        if args.approximator == 'linear':
            path = interpolator.optimize_path(start_latent=noisy_latents[0].unsqueeze(0),
                                            end_latent=noisy_latents[1].unsqueeze(0),init_with_slerp=False,
                                            num_steps=10,)
        elif args.approximator == 'spherical':
            path = interpolator.optimize_path(start_latent=noisy_latents[0].unsqueeze(0),
                                            end_latent=noisy_latents[1].unsqueeze(0),init_with_slerp=True,
                                            num_steps=10,) # [b,n_steps, *]
        # optimization
        else:
            path, info = interpolator.optimize_path(start_latent=noisy_latents[0].unsqueeze(0),
                                                end_latent=noisy_latents[1].unsqueeze(0),
                                                num_steps=10, lr=args.lr, sigma=SIGMA,
                                                class_label=single_class_label,max_iters=800) # [b,n_steps, *]
        # Denoise
        path = interpolator.denoise_path(path, sigma_start=SIGMA, class_label=single_class_label, num_denoise_steps=18)
        print(f"Denoised latents: shape={path.shape}, "
            f"mean={path.mean():.4f}, std={path.std():.4f}")
    else:
        path, info = interpolator.optimize_path(start_latent=clean_latents[0].unsqueeze(0),
                                            end_latent=clean_latents[1].unsqueeze(0),
                                            num_steps=10, lr=args.lr) # [b,n_steps, *]
    for i, latent in enumerate(path[0]): # only single pair, batch_dim is dummy
        decode_and_save(ae=ae,
                        latent=latent.unsqueeze(0),
                        save_path=os.path.join(output_dir, f"{args.approximator}_step{i}.obj"),
                        device=device)
        
        

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--dataset', type=str, default='shapenet')
    parser.add_argument('--approximator', type=str, default='rbf',
                        help='approximator type')
    parser.add_argument('--h_path', type=str, default='output/kernel/shapenet/rbf/h.pth',
                        help='path to pre-trained kernel (RBF or Land)')
    parser.add_argument('--ae_name', type=str, default='kl_d512_m512_l8')
    parser.add_argument('--ae_path', type=str, default='output/ae/kl_d512_m512_l8/checkpoint-199.pth')
    parser.add_argument('--dm_name', type=str, default='kl_d512_m512_l8_d24_edm')
    parser.add_argument('--dm_path', type=str, default='output/class_cond_dm/kl_d512_m512_l8_d24_edm/checkpoint-499.pth')
    parser.add_argument('--noise_level', type=float, default=0.50)
    parser.add_argument('--category', type=int, default=5)
    parser.add_argument('--seed', type=int, default=9999)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--lr', type=float, default=0.01)
    
    args = parser.parse_args()
    main(args)
    
    

