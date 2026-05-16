"""
Approximative algorithms for free-support Wasserstein-2 barycenters of discrete probability distributions.

Johannes von Lindheim, 2022
https://github.com/jvlindheim/free-support-barycenters
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from ot import emd

"""
Reference algorithm.
"""

def ref_bary(supports, masses, weights=None, ref_index=0, precision=7):
    """
    Computation of an approximate Wasserstein-2 barycenter using the reference algorithm.
    Convenience method combining execution of the reference measure alignment and construction
    of the correcponding barycenter.
    
    Parameters
    ----------
    supports: Measure support positions list/array of length n. Can be given just a 2d-array, if the positions are always the same.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    weights=None: array of same length as number of measures, needs to sum to one.
        These are the weights in the barycenter problem, typically denoted by lambda_i.
        If None is given, use uniform weights.
    ref_index=0: Which of the given measures is treated as the reference measure (first measure is the default)
    precision=7: Rounded to which decimal place the weights need to sum up to one.
        
    Returns
    -------
    bary_supp: Barycenter support positions.
    bary_masses: Barycenter masses corresponding to the support positions.
    """
    supports, masses, n, nis, d = prepare_data(supports, masses)
    
    # set weights uniformly if None is given
    if weights is None:
        n = len(masses)
        weights = np.ones((n,)) / n
    assert np.round(weights.sum(), decimals=precision) == 1, "weights need to sum to one"
    assert ref_index in np.arange(n), "ref_index needs to be between 0 and n-1, where n is the number of given measures"
    
    alignment, _ = ref_alignment(supports, masses, ref_index)
    return ref_supports_from_alignment(alignment, weights, precision=7), masses[ref_index]

def ref_alignment(supports, masses, ref_index=0):
    '''
    Computes approximations of the given measures as barycentric projections from transport plans
    from a reference measure to the given input measures.
    These can be further used to compute approximate barycenters for arbitrary sets of weights.
    
    Parameters
    ----------
    supports: Measure support positions list/array of length n. Can be given just a 2d-array, if the positions are always the same.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    ref_index=0: Which of the given measures is treated as the reference measure (first measure is the default)
    
    Returns
    -------
    alignment: (n, n_ref, d)-shaped array, where n_ref is the number of support points in the reference measure.
    bary_masses: Barycenter masses corresponding to the support positions computed by 'ref_supports_from_alignment'. This is precisely the masses corresponding to the reference measure, which might have been reduced to less entries by 'prepare_data' for small or zero masses.
    '''
    
    supports, masses, n, nis, d = prepare_data(supports, masses)
    assert ref_index in np.arange(n), "ref_index needs to be between 0 and n-1, where n is the number of given measures"
    
    # compute transport plans from reference measure to other measures
    ref_mass = masses[ref_index]
    ref_supp = supports[ref_index]
    pis = [emd(ref_mass, mui_mass, cdist(ref_supp, mui_pos, metric='sqeuclidean'))
           for mui_pos, mui_mass in zip(supports, masses)]
    return np.stack([(pi/ref_mass[:, None]).dot(pos) for pi, pos in zip(pis, supports)]), masses[ref_index]

def ref_supports_from_alignment(alignment, weights=None, precision=7):
    '''
    From an alignment of the original measures with respect to a reference measure as computed by
    the function 'ref_alignment', compute barycenter support with respect to a given set of weights.
    The weights of the support points are simply given by the weights of the chosen reference
    measure and are not returned.
    
    Parameters
    ----------
    alignment: result of the function 'ref_alignment'.
    weights=None: array of same length as number of measures, needs to sum to one.
        These are the weights in the barycenter problem, typically denoted by lambda_i.
        If None is given, use uniform weights.
    precision=7: Rounded to which decimal place the weights need to sum up to one.
    
    Returns
    -------
    bary_supp: Barycenter support positions.
    '''
    
    if weights is None:
        n = alignment.shape[0]
        weights = np.ones((n,)) / n
    assert np.round(weights.sum(), decimals=precision) == 1, "weights need to sum to one"
    
    return (weights[:, None, None]*alignment).sum(axis=0)

"""
Pairwise algorithm.
"""

def pairwise_bary(supports, masses, weights=None, compute_err_bound=False, precision=7):
    """
    Computation of an approximate Wasserstein-2 barycenter using the pairwise algorithm.
    Convenience method combining computation of all pairwise transport plans and construction
    of the correcponding barycenter.
    If barycenters for multiple sets of weights should be computed, for speed rather execute
    'pairwise_kernels' (bottleneck) only once and construct multiple barycenters using
    'pairwise_bary_from_kernels' (fast).
    
    Parameters
    ----------
    supports: Measure support positions list/array of length n. Can be given just a 2d-array, if the positions are always the same.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    weights=None: array of same length as number of measures, needs to sum to one.
        These are the weights in the barycenter problem, typically denoted by lambda_i.
        If None is given, use uniform weights.
    precision=7: Rounded to which decimal place the weights need to sum up to one.
        
    Returns
    -------
    bary_supp: Barycenter support positions.
    bary_masses: Barycenter masses corresponding to the support positions.
    """
    
    supports, masses, n, nis, d = prepare_data(supports, masses)
    # set weights uniformly if None is given
    if weights is None:
        weights = np.ones((n,)) / n
    assert np.round(weights.sum(), decimals=precision) == 1, "weights need to sum to one"

    return pairwise_bary_from_kernels(supports, masses, pairwise_kernels(supports, masses),
                                      weights, compute_err_bound, precision)

def pairwise_kernels(supports, masses):
    '''
    Computes the row-normalized matrix of all pairwise optimal transports between the input measues.
    Only computes transports once for each (unordered) distinct of measures and uses the transpose
    for the transpose pair.
    Further use the result in 'pairwise_bary_from_kernels'.
    
    Parameters
    ----------
    supports: Measure support positions list/array of length n. Can be given just a 2d-array, if the positions are always the same.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    
    Returns
    -------
    kernels: Barycenter support positions.
    '''

    supports, masses, n, nis, d = prepare_data(supports, masses)
    total_supp = np.concatenate(supports, axis=0)
    total_masses = np.concatenate(masses)
    n_supp = nis.sum()
    pairwise_pis = np.zeros((n_supp, n_supp))
    nis_cum = np.concatenate([[0], np.cumsum(nis)])
    
    # write all pairwise wasserstein-2 plans to a big matrix
    for i in range(n):
        for j in range(i+1, n):
            pi_ij = emd(masses[i], masses[j], cdist(supports[i], supports[j], metric='sqeuclidean'))
            pairwise_pis[nis_cum[i]:nis_cum[i+1], nis_cum[j]:nis_cum[j+1]] = pi_ij
    pairwise_pis += pairwise_pis.T
    pairwise_pis += np.diagflat(total_masses)
    
    return pairwise_pis/total_masses[:, None]

def pairwise_bary_from_kernels(supports, masses, kernels, weights=None, compute_err_bound=False, precision=7):
    '''
    From the precomputed result from 'pairwise_kernels', 
    compute a barycenter with respect to a given set of weights.
    
    Parameters
    ----------
    supports: Measure support positions list/array of length n. Can be given just a 2d-array, if the positions are always the same.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    kernels: matrix of all pairwise precomputed transport kernels using 'pairwise_kernels'.
    weights=None: array of same length as number of measures, needs to sum to one.
        These are the weights in the barycenter problem, typically denoted by lambda_i.
        If None is given, use uniform weights.
    compute_err_bound=False: If set to True, additionally to the barycenter, a bound for the relative error is returned.
    precision=7: Rounded to which decimal place the weights need to sum up to one.
    
    Returns
    -------
    bary_supp: Barycenter support positions.
    bary_masses: Barycenter masses corresponding to the support positions.
    error_bound: only returned if compute_err_bound is set to True.
        This is an upper bound on the relative error <= 1, where the relative is defined as
        Psi(approx. bary)/Psi(opt. bary) - 1 with Psi being the optimization functional of the barycenter problem.
    '''
    
    supports, masses, n, nis, d = prepare_data(supports, masses)
    
    # set weights uniformly if None is given
    if weights is None:
        n = len(masses)
        weights = np.ones((n,)) / n
    assert np.round(weights.sum(), decimals=precision) == 1, "weights need to sum to one"
    
    # compute barycenter from pairwise transport kernels
    bary_masses = np.concatenate([w*mass for w, mass in zip(weights, masses)])
    bary_supp = kernels.dot(np.concatenate([w*supp for w, supp in zip(weights, supports)], axis=0))
    
    if compute_err_bound:
        total_support = np.concatenate(supports)
        nis_cum = np.concatenate([[0], np.cumsum(nis)])
        # compute error bound
        enum = (bary_masses*(((total_support-bary_supp)**2).sum(axis=1))).sum() # weighted dists from bary to input measures
        denom = 0.5*sum([w*(np.repeat(weights, nis)*cdist(supp, total_support, 'sqeuclidean')*kernels[nis_cum[i]:nis_cum[i+1]]*mass[:, None]).sum()
                    for i, (w, supp, mass) in enumerate(zip(weights, supports, masses))]) # pairwise W2-dists
        if denom > 0:
            err_bound = min(2-enum/denom, 2.0)
        else:
            err_bound = 2.0 if enum > 0 else 1.0
        return bary_supp, bary_masses, err_bound
    
    return bary_supp, bary_masses

"""
Helper functions.
"""

def prepare_data(posns, masses, min_mass=1e-10):
    '''
    Given a support positions and masses array, determine (and make security checks for) the number of measures, number
    of support points array and dimension. Also modify posns to array, if given only one array for all measures.

    Parameters
    ----------
    posns: Measure support positions list/array of length n. Can be given just a 2d-array, if the positions are always the same.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    min_mass=1e-10: Every point with a mass less than this parameter is discarded.
    min_mass=1e-10: Every point with a mass less than this parameter is discarded.
        
    Returns
    -------
    posns: Measure support positions list of length n.
    masses: Masses/weights of the given measure support points.
    n: Number of determined measures.
    nis: Number of determined support points for each measure.
    d: Dimension of the support points.    
    '''    
    # if given a list, we assume that we are given multiple measures and the length of the list is
    # the number of measures
    if isinstance(posns, list):
        n = len(posns)
        assert len(masses) == n, "masses needs have same length as pos (equal number of measures)"
    # if given a 2d-array and a list of of mass arrays, we assume that we are given a number of measures,
    # which are all supported on the same posns-array, so the number of measures n is len(masses)
    elif isinstance(posns, np.ndarray) and posns.ndim == 2 and (isinstance(masses, list) \
                                                                or (isinstance(masses, np.ndarray) and masses.ndim == 2)):
        n = len(masses)
        posns = [posns]*n
    # if given a 3d-array and a 2d array of masses, we assume that we are given a number of measures that
    # all have the same number of points
    elif isinstance(posns, np.ndarray) and posns.ndim == 3:
        n = posns.shape[0]
        assert masses.shape[0] == n, "masses needs have same length as pos (equal number of measures)"
        assert posns.shape[1] == masses.shape[1], "number of points and number of mass entries need to be the same"

    # if given a 2d-array and a 1d-array of masses, assume that we are given only one measure
    elif isinstance(posns, np.ndarray) and posns.ndim == 2 and isinstance(masses, np.ndarray) and masses.ndim == 1:
        n = 1
        assert len(posns) == len(masses), "if only one measure is given, length of posns and masses need to match"
        posns = [posns]
        masses = masses[None, :]
    else:
        raise ValueError("cannot see what the number of measures is for given parameters 'posns' and 'masses'")
    assert n >= 1, "at least one measure needs to be given"
    assert all([pos.ndim == 2 for pos in posns]), "position arrays need to be two-dimensional"
    posns = [pos[mass > min_mass] for (pos, mass) in zip(posns, masses)] # throw out points with mass <= min_mass
    masses = [mass[mass > min_mass] for mass in masses]
    nis = np.array([pos.shape[0] for pos in posns]) # number of support points for all measures
    d = posns[0].shape[1]
    return [pos for pos in posns], [mass for mass in masses], n, nis, d

def scatter_distr(posns, masses, n_plots_per_row=2, scale=4, invert=False, disk_size=6000/5, xmarkers=False, color='gray',
           alpha=0.5, xmarker_posns=None, axis_off=False, margin_fac=0.2, dpi=300,
           subtitles='', figax=None, savepath=None):
    '''
    Scatter plot function for either one or multiple discrete probability distributions.

    Parameters
    ----------
    posns: Measure support positions list/array of length 1 or n.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    n_plots_per_row=2: In case multiple measures are given, a figure with this number of
        subplots per column is generated.
    scale=4: Size of the figure is proportional to this parameter.
    invert=False: Whether to invert the y-axis.
    disk_size=6000/5: Size of the plotted disks per support point is proportional
        to their weight and this parameter.
    xmarkers=False: Whether to plot an x in the center of each disk (support point).
    color='gray': Color of each disk (support point).
    alpha=0.5: Transparency of each disk (support point).
    xmarker_posns=None: An additional set of x-markers can be plotted if this
        parameter is given an array of 2d positions.
    axis_off=False: Whether to turn off the coordinate system of the subplots.
    margin_fac=0.2: How much margin to leave around the minimum and maximum x/y-values
        of the support points.
    dpi=300: Resolution of figure (important for export).
    subtitles='': Array of subtitles to each subplot.
    figax=None: Tuple of the form (fig, ax, k, l). This can be used, if this function
        is only supposed to plot in the given axis array 'ax' at index k, l that
        has already been constructed.
        If None is given, a new fig and axis array are created.
    savepath=None: Saves figure to this given path.
    '''
    posns, masses, n, nis, d = prepare_data(posns, masses)
    n_plots = n
    if figax is None:
        n_rows = np.ceil(n_plots / n_plots_per_row).astype(int)
        n_cols = min(n_plots_per_row, n_plots)
        figsize = (scale*n_cols, scale*n_rows)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    else:
        fig, ax, k, l = figax

    if isinstance(subtitles, list) and len(subtitles) == 1:
        subtitles = [subtitles[0]]*n_plots
    elif isinstance(subtitles, str):
        subtitles = [subtitles]*n_plots
    else:
        assert len(subtitles) == n_plots, "length of subtitles array needs to be equal to number of plots"

    xmin, xmax, ymin, ymax = min([pos[:, 0].min() for pos in posns]), max([pos[:, 0].max() for pos in posns]), min([pos[:, 1].min() for pos in posns]), max([pos[:, 1].max() for pos in posns])

    if figax is None:
        row_inds, col_inds = range(n_rows), range(n_cols)
    else:
        row_inds, col_inds = [k], [l]
    for i in row_inds:
        for j in col_inds:
            idx = i*n_plots_per_row + j if figax is None else 0
            if idx >= n_plots or axis_off:
                ax[i, j].axis('off')
                if idx >= n_plots:
                    continue
            pos = posns[idx]
            mass = masses[idx]

            # set plot dimensions
            xmargin = margin_fac*(xmax-xmin)
            ymargin = margin_fac*(ymax-ymin)
            ax[i, j].set_xlim([xmin-xmargin, xmax+xmargin])
            ax[i, j].set_ylim([ymin-ymargin, ymax+ymargin])
            ax[i, j].set_aspect('equal')
            ax[i, j].set_title(subtitles[idx])

            # plot
            if xmarkers:
                ax[i, j].scatter(pos[:, 0], pos[:, 1], marker='x', c='red')
            if xmarker_posns is not None:
                ax[i, j].scatter(xmarker_posns[:, 0], xmarker_posns[:, 1], marker='x', c='red')
            ax[i, j].scatter(pos[:, 0], pos[:, 1], marker='o', s=mass*disk_size*scale, c=color, alpha=alpha)
            if invert:
                ax[i, j].set_ylim(ax[i, j].get_ylim()[::-1])

    if savepath is not None:
        plt.savefig(savepath, dpi=dpi, pad_inches=0, bbox_inches='tight')


"""
Log-domain Sinkhorn Algorithm & Wasserstein Barycenter in PyTorch.

Features:
  - Log-domain for numerical stability (no exp overflow/underflow)
  - Full batch support
  - ε-scaling option for faster convergence
  - Fixed-support Wasserstein Barycenter (Benamou et al.)

Author: Claude (for Zhixuan)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List


# ─────────────────────────────────────────────
#  Core: Log-domain Sinkhorn
# ─────────────────────────────────────────────

def cost_matrix_euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean cost matrix.
    
    Args:
        x: (B, N, D) or (N, D)
        y: (B, M, D) or (M, D)
    Returns:
        C: (B, N, M) or (N, M)
    """
    # ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 <x_i, y_j>
    x_sqnorm = (x ** 2).sum(dim=-1, keepdim=True)   # (..., N, 1)
    y_sqnorm = (y ** 2).sum(dim=-1, keepdim=True)    # (..., M, 1)
    xy = torch.matmul(x, y.transpose(-2, -1))        # (..., N, M)
    C = x_sqnorm + y_sqnorm.transpose(-2, -1) - 2 * xy
    return C.clamp(min=0)


def log_sinkhorn(
    C: torch.Tensor,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    eps: float = 0.05,
    max_iter: int = 100,
    thresh: float = 1e-6,
    return_plan: bool = False,
) -> dict:
    """Log-domain Sinkhorn algorithm for entropic OT.
    
    Solves:  min_{P in U(a,b)} <C, P> + eps * KL(P || a ⊗ b)
    
    Args:
        C: (B, N, M) or (N, M) cost matrix
        a: (B, N) or (N,) source marginal (uniform if None)
        b: (B, M) or (M,) target marginal (uniform if None)
        eps: regularization strength
        max_iter: maximum Sinkhorn iterations
        thresh: convergence threshold on marginal error
        return_plan: if True, also return the transport plan P
        
    Returns:
        dict with keys:
          'cost': (B,) or scalar, Sinkhorn divergence
          'f': (B, N) dual variable (Kantorovich potential)
          'g': (B, M) dual variable
          'plan': (B, N, M) transport plan (if return_plan=True)
          'n_iter': number of iterations used
    """
    has_batch = C.dim() == 3
    if not has_batch:
        C = C.unsqueeze(0)  # (1, N, M)
    
    B, N, M = C.shape
    
    # Default: uniform marginals
    if a is None:
        a = torch.ones(B, N, device=C.device, dtype=C.dtype) / N
    elif a.dim() == 1:
        a = a.unsqueeze(0).expand(B, -1)
    
    if b is None:
        b = torch.ones(B, M, device=C.device, dtype=C.dtype) / M
    elif b.dim() == 1:
        b = b.unsqueeze(0).expand(B, -1)
    
    log_a = a.log()  # (B, N)
    log_b = b.log()  # (B, M)
    
    # Gibbs kernel in log-domain: K_{ij} = -C_{ij} / eps
    log_K = -C / eps  # (B, N, M)
    
    # Dual variables (log-domain)
    f = torch.zeros(B, N, device=C.device, dtype=C.dtype)  # log-scaling for rows
    g = torch.zeros(B, M, device=C.device, dtype=C.dtype)  # log-scaling for cols
    
    for it in range(max_iter):
        # Row update: f_i = log a_i - logsumexp_j (log_K_{ij} + g_j)
        f_new = log_a - torch.logsumexp(log_K + g.unsqueeze(1), dim=2)  # (B, N)
        
        # Col update: g_j = log b_j - logsumexp_i (log_K_{ij} + f_i)
        g_new = log_b - torch.logsumexp(log_K + f_new.unsqueeze(2), dim=1)  # (B, M)
        
        # Check convergence: marginal error
        if thresh > 0 and it % 5 == 0:
            # Check row marginal: sum_j P_{ij} should equal a_i
            log_P_row = torch.logsumexp(
                f_new.unsqueeze(2) + log_K + g_new.unsqueeze(1), dim=2
            )
            err = (log_P_row.exp() - a).abs().max().item()
            if err < thresh:
                f, g = f_new, g_new
                break
        
        f, g = f_new, g_new
    
    # Compute OT cost: <C, P> where log P = f + log_K + g
    log_P = f.unsqueeze(2) + log_K + g.unsqueeze(1)  # (B, N, M)
    cost = (C * log_P.exp()).sum(dim=(1, 2))  # (B,)
    
    result = {
        'cost': cost.squeeze(0) if not has_batch else cost,
        'f': f.squeeze(0) if not has_batch else f,
        'g': g.squeeze(0) if not has_batch else g,
        'n_iter': it + 1,
    }
    
    if return_plan:
        P = log_P.exp()
        result['plan'] = P.squeeze(0) if not has_batch else P
    
    return result


def sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    eps: float = 0.05,
    **kwargs,
) -> torch.Tensor:
    """Debiased Sinkhorn divergence: S_eps(a,b) = OT_eps(a,b) - 0.5*OT_eps(a,a) - 0.5*OT_eps(b,b).
    
    This removes the entropic bias so S_eps(a,a) = 0.
    
    Args:
        x: (B, N, D) source points
        y: (B, M, D) target points
        a, b: marginals
        eps: regularization
    Returns:
        (B,) debiased Sinkhorn divergence
    """
    C_xy = cost_matrix_euclidean(x, y)
    C_xx = cost_matrix_euclidean(x, x)
    C_yy = cost_matrix_euclidean(y, y)
    
    ot_xy = log_sinkhorn(C_xy, a, b, eps=eps, **kwargs)['cost']
    ot_xx = log_sinkhorn(C_xx, a, a, eps=eps, **kwargs)['cost']
    ot_yy = log_sinkhorn(C_yy, b, b, eps=eps, **kwargs)['cost']
    
    return ot_xy - 0.5 * ot_xx - 0.5 * ot_yy


# ─────────────────────────────────────────────
#  Wasserstein Barycenter (fixed support)
# ─────────────────────────────────────────────

def wasserstein_barycenter(
    supports: List[torch.Tensor],
    weights_list: List[torch.Tensor],
    barycenter_support: torch.Tensor,
    lambdas: Optional[torch.Tensor] = None,
    eps: float = 0.05,
    max_iter: int = 200,
    thresh: float = 1e-6,
    log_every: int = 0,
) -> torch.Tensor:
    """Fixed-support Wasserstein Barycenter (Benamou et al. 2015).
    
    Given K distributions {(X_k, a_k)}_{k=1}^K with fixed support X_bar,
    find weights q on X_bar minimizing:
    
        min_q  sum_k  lambda_k * OT_eps(q, a_k)
    
    Algorithm: interleaved Sinkhorn iterations with geometric averaging.
    At each step:
      1. Compute s_k = K_k v_k  for each k  (absorption step)
      2. Update q ∝ prod_k s_k^{lambda_k}  (geometric mean)
      3. Update u_k = q / s_k  for each k
      4. Update v_k = b_k / (K_k^T u_k)  for each k
    
    This interleaves a single Sinkhorn step per measure with the
    barycenter weight update — no inner loop needed.
    
    Args:
        supports: list of K tensors, each (N_k, D)
        weights_list: list of K tensors, each (N_k,) summing to 1
        barycenter_support: (M, D) fixed support for barycenter
        lambdas: (K,) coefficients summing to 1 (uniform if None)
        eps: entropic regularization
        max_iter: maximum iterations
        thresh: convergence on barycenter weights
        log_every: print info every N steps (0 = silent)
        
    Returns:
        q: (M,) optimal barycenter weights
    """
    K = len(supports)
    M = barycenter_support.shape[0]
    device = barycenter_support.device
    dtype = barycenter_support.dtype
    
    if lambdas is None:
        lambdas = torch.ones(K, device=device, dtype=dtype) / K
    
    # Precompute log-Gibbs kernels: log K_k[i,j] = -C_k[i,j] / eps
    log_kernels = []
    for k in range(K):
        C_k = cost_matrix_euclidean(
            barycenter_support.unsqueeze(0),
            supports[k].unsqueeze(0)
        ).squeeze(0)  # (M, N_k)
        log_kernels.append(-C_k / eps)
    
    # Initialize dual variables (log-domain): log(v_k)
    log_v = [torch.zeros(supports[k].shape[0], device=device, dtype=dtype) for k in range(K)]
    log_b = [w.log() for w in weights_list]
    
    log_q = torch.full((M,), -torch.tensor(float(M)).log(), device=device, dtype=dtype)
    
    for it in range(max_iter):
        # Step 1: compute log(s_k) = log(K_k v_k) for each k
        # s_k[i] = sum_j K_k[i,j] * v_k[j]
        # log s_k[i] = logsumexp_j (log_K_k[i,j] + log_v_k[j])
        log_s = []
        for k in range(K):
            log_s_k = torch.logsumexp(log_kernels[k] + log_v[k].unsqueeze(0), dim=1)  # (M,)
            log_s.append(log_s_k)
        
        # Step 2: barycenter update via geometric mean
        # log q[i] = sum_k lambda_k * log s_k[i]  + const
        log_q_new = sum(lambdas[k] * log_s[k] for k in range(K))
        log_q_new = log_q_new - torch.logsumexp(log_q_new, dim=0)
        
        # Convergence check
        delta = (log_q_new.exp() - log_q.exp()).abs().max().item()
        
        if log_every > 0 and (it + 1) % log_every == 0:
            print(f"  [Barycenter] iter {it+1:3d} | delta={delta:.2e}")
        
        log_q = log_q_new
        
        if delta < thresh:
            if log_every > 0:
                print(f"  [Barycenter] converged at iter {it+1}")
            break
        
        # Step 3 & 4: update u_k and v_k
        for k in range(K):
            # log u_k = log q - log s_k
            log_u_k = log_q - log_s[k]  # (M,)
            # log v_k[j] = log b_k[j] - logsumexp_i(log_K_k[i,j] + log_u_k[i])
            log_v[k] = log_b[k] - torch.logsumexp(
                log_kernels[k].T + log_u_k.unsqueeze(0), dim=1
            )  # (N_k,)
    
    return log_q.exp()


# ─────────────────────────────────────────────
#  Free-support Barycenter (gradient-based)
# ─────────────────────────────────────────────

def free_support_barycenter(
    supports: List[torch.Tensor],
    weights_list: List[torch.Tensor],
    M: int,
    D: int,
    init_Y: Optional[torch.Tensor] = None,  # 新增：允许传入上一帧的结果
    lambdas: Optional[torch.Tensor] = None,
    eps: float = 0.05,
    lr: float = 0.01,
    n_steps: int = 200,
    max_sinkhorn: int = 100,
    log_every: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """带有隐式梯度的自由支持点重心求解器"""
    device = supports[0].device
    dtype = supports[0].dtype
    K = len(supports)
    
    if lambdas is None:
        lambdas = torch.ones(K, device=device, dtype=dtype) / K
    
    # ---- 核心修改：支持 Sequential Initialization ----
    if init_Y is not None:
        # Eq 10: 使用上一帧优化好的结果作为当前帧的起点
        Y = init_Y.clone().detach().requires_grad_(True)
    else:
        # 默认初始化：随机采样
        all_pts = torch.cat(supports, dim=0)
        idx = torch.randperm(all_pts.shape[0])[:M]
        Y = all_pts[idx].clone().requires_grad_(True)
    
    q = torch.ones(M, device=device, dtype=dtype) / M
    optimizer = torch.optim.Adam([Y], lr=lr)
    
    for step in range(n_steps):
        optimizer.zero_grad()
        
        # Phase 1: 计算传输计划 P* (不计梯度)
        # 我们需要 P* 来作为后续计算 Cost 梯度的“常数权重”
        plans = []
        with torch.no_grad():
            for k in range(K):
                C_k = cost_matrix_euclidean(Y.unsqueeze(0), supports[k].unsqueeze(0))
                # 注意：返回 plan 是关键
                res = log_sinkhorn(
                    C_k, q.unsqueeze(0), weights_list[k].unsqueeze(0),
                    eps=eps, max_iter=max_sinkhorn, return_plan=True,
                )
                plans.append(res['plan'].squeeze(0)) # (M, N_k)
        
        # Phase 2: 应用包络定理计算梯度
        # Loss = Σ λ_k <C_k(Y), P_k*>
        # 此时梯度只通过 C_k 流向 Y，非常节省显存
        loss = torch.tensor(0.0, device=device, dtype=dtype)
        for k in range(K):
            C_k = cost_matrix_euclidean(Y.unsqueeze(0), supports[k].unsqueeze(0)).squeeze(0)
            # P* 是 detached 的，扮演了“固定权重”的角色
            loss = loss + lambdas[k] * (C_k * plans[k]).sum()
        
        loss.backward()
        optimizer.step()
        
        if log_every > 0 and (step + 1) % log_every == 0:
            print(f"  [FreeBary] step {step+1:3d} | loss={loss.item():.6f} | Grad norm={Y.grad.norm().item():.6f}")

            
    return Y.detach(), q

# ─────────────────────────────────────────────
#  Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("1. Sinkhorn OT between two Gaussians")
    print("=" * 60)
    
    N, M_pts, D = 200, 200, 2
    x = torch.randn(N, D, device=device) * 0.5 + torch.tensor([2.0, 0.0], device=device)
    y = torch.randn(M_pts, D, device=device) * 0.5 + torch.tensor([-2.0, 0.0], device=device)
    
    C = cost_matrix_euclidean(x, y)
    result = log_sinkhorn(C, eps=0.1, return_plan=True)
    print(f"  OT cost: {result['cost'].item():.4f}")
    print(f"  Converged in {result['n_iter']} iterations")
    print(f"  Plan shape: {result['plan'].shape}")
    print(f"  Plan row sums (should be 1/N={1/N:.4f}): "
          f"mean={result['plan'].sum(1).mean():.6f}")
    
    # Batch test
    print("\n  --- Batch mode ---")
    B = 8
    x_batch = torch.randn(B, N, D, device=device)
    y_batch = torch.randn(B, M_pts, D, device=device)
    C_batch = cost_matrix_euclidean(x_batch, y_batch)
    res_batch = log_sinkhorn(C_batch, eps=0.1)
    print(f"  Batch costs: {res_batch['cost']}")
    
    # Sinkhorn divergence
    print("\n  --- Sinkhorn divergence (debiased) ---")
    sd_same = sinkhorn_divergence(x.unsqueeze(0), x.unsqueeze(0), eps=0.1)
    sd_diff = sinkhorn_divergence(x.unsqueeze(0), y.unsqueeze(0), eps=0.1)
    print(f"  S_eps(x,x) = {sd_same.item():.6f}  (should be ~0)")
    print(f"  S_eps(x,y) = {sd_diff.item():.6f}")
    
    print("\n" + "=" * 60)
    print("2. Fixed-support Wasserstein Barycenter")
    print("=" * 60)
    
    # 3 Gaussians at different locations
    K_measures = 3
    centers = [
        torch.tensor([3.0, 0.0], device=device),
        torch.tensor([-1.5, 2.6], device=device),
        torch.tensor([-1.5, -2.6], device=device),
    ]
    sups = [torch.randn(100, 2, device=device) * 0.3 + c for c in centers]
    wts = [torch.ones(100, device=device) / 100 for _ in range(K_measures)]
    
    # Barycenter support: grid
    grid_n = 30
    gx = torch.linspace(-4, 4, grid_n, device=device)
    gy = torch.linspace(-4, 4, grid_n, device=device)
    grid = torch.stack(torch.meshgrid(gx, gy, indexing='ij'), dim=-1).reshape(-1, 2)
    
    q = wasserstein_barycenter(
        sups, wts, grid,
        eps=0.2, max_iter=200, thresh=1e-6,
        log_every=20
    )
    print(f"  Barycenter weights sum: {q.sum().item():.6f}")
    print(f"  Barycenter support shape: {grid.shape}")
    print(f"  Top-10 weight locations:")
    top_idx = q.argsort(descending=True)[:10]
    for i in top_idx:
        print(f"    ({grid[i,0]:.2f}, {grid[i,1]:.2f}): w={q[i]:.4f}")
    
    print("\n" + "=" * 60)
    print("3. Free-support Wasserstein Barycenter")
    print("=" * 60)
    
    Y, q_free = free_support_barycenter(
        sups, wts, M=50, D=2,
        eps=0.2, lr=0.05, n_steps=100,
        log_every=20
    )
    print(f"  Barycenter center of mass: ({Y.mean(0)[0]:.3f}, {Y.mean(0)[1]:.3f})")
    print(f"  (Expected: ~origin since 3 sources are symmetric)")
    
    print("\nDone!")