import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from gfm.path.metrics import ConformalMetric, GradDiagonalMetric, h_diag_RBF, load_metric, h_diag_Land
import argparse
import math

import numpy as np
import torch
import random


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        if args.dataset == 'shapenet':
            latents = torch.load(
                args.latents_pth)
        lt_size = latents.shape[1:]
        if args.approximator == 'rbf':
            args_rbf = {
                "n_centers": args.rbf_center,
                "latent_size": lt_size,
                "ambiant_size": lt_size,
                "kappa": args.rbf_kappa
            }
            torch.manual_seed(1)
            np.random.seed(1)
            random.seed(1)
            data_to_fit_latent = latents.view(-1, np.prod(lt_size))
            data_to_fit_ambiant = latents.view(-1, np.prod(lt_size))
            h = h_diag_RBF(**args_rbf, data_to_fit_latent=data_to_fit_latent, data_to_fit_ambiant=data_to_fit_ambiant).to(device)

            h.normalize(data_to_fit_latent.to(device))
            del data_to_fit_latent
            del data_to_fit_ambiant

            a = 1
        elif args.approximator == 'land':
            data_ref = latents.view(-1, np.prod(lt_size))[:1000].to(device)
            data_to_fit = latents.view(-1, np.prod(lt_size))[1000:].to(device)
            with torch.no_grad():
                sample_dists = torch.cdist(data_ref[:200], data_ref[:200])
                mask = ~torch.eye(200, dtype=bool, device=device)
                gamma = sample_dists[mask].median().item() / 3
                print(f"Using data-driven gamma: {gamma:.2f}")
            h = h_diag_Land(data_ref, gamma=gamma).to(device)
            h.normalize(data_to_fit)
        else:
            raise ValueError

        print(f'approximator: {args.approximator}-- min={args.min_h} max={args.max_h}')
        
        if args.approximator is not None:
            path_to_save = os.path.join(args.save_root, args.dataset, args.approximator)
            os.makedirs(path_to_save, exist_ok=True)
            torch.save(args, path_to_save + '/param.config')
            torch.save(h, path_to_save + '/h.pth')
    # metric = load_metric(args.metric_type, args.approximator, h)           

if __name__ == '__main__':
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--approximator', type=str, default='rbf',
                        help='approximator type')
    parser.add_argument('--dataset', type=str, default='shapenet')
    parser.add_argument('--latents_pth', type=str, default='class_cond_obj/kl_d512_m512_l8_d24_edm/latents_5.pth',
                        help='path to latents')
    parser.add_argument('--save_root', type=str, default='output/kernel',
                        help='path to save kernel')
    parser.add_argument('--min_h', type=float, default=0.0,
                        help='min value for h')
    parser.add_argument('--max_h', type=float, default=1e3,
                        help='max value for h')
    parser.add_argument('--rbf_center', type=int, default=2000,
                        help='number of centers for rbf approximator')
    parser.add_argument('--rbf_kappa', type=float, default=2.0,
                        help='kappa for rbf approximator')
    parser.add_argument('--land_gamma', type=float, default=100,
                        help='gamma for land approximator')
    args = parser.parse_args()
    main(args)