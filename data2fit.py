import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import argparse
import math

import numpy as np

import mcubes

import torch

import trimesh

from gfm.models import models_class_cond, models_ae

from pathlib import Path

if __name__ == "__main__":

    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--dm', type=str,
                        required=True)  # 'kl_d512_m512_l16_edm'
    parser.add_argument(
        '--dm-pth', type=str, required=True
    )  # 'output/uncond_dm/kl_d512_m512_l16_edm/checkpoint-999.pth'
    args = parser.parse_args()
    print(args)

    Path("class_cond_obj/{}".format(args.dm)).mkdir(parents=True,
                                                    exist_ok=True)

    device = torch.device('cuda:0')

    model = models_class_cond.__dict__[args.dm]()
    model.eval()

    model.load_state_dict(torch.load(args.dm_pth, weights_only=False)['model'])
    model.to(device)

    total = 5000
    iters = 1000

    latents = torch.empty((total, 512, 8), dtype=torch.float32, device=device)
    category_id = 10
    conds = torch.Tensor([category_id] * iters).long().to(device)
    with torch.no_grad():
        for i in range(total // iters):
            latents[i * iters:(i + 1) * iters] = model.sample(
                cond=conds,
                batch_seeds=torch.arange(i * iters, (i + 1) * iters,
                                         device=device)).float()
            print(latents.shape, latents.max(), latents.min(), latents.mean(),
                  latents.std())
    latents = latents.detach().cpu()
    torch.save(latents, f"class_cond_obj/{args.dm}/latents_{category_id}.pth")
