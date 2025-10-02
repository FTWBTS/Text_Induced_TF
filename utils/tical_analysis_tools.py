
"""
TICaL Analysis & Heatmap Toolkit
================================
Drop-in utilities to visualize and evaluate iTransformer+TICaL outputs.

Functions expect numpy arrays. If you have torch.Tensors, convert with .detach().cpu().numpy().
All charts are plotted with matplotlib (single chart per figure, no seaborn).

Provided functions
------------------
- plot_kappa_heatmap(kappa, sample=0, savepath=None)
- plot_pi_heatmap(Pi, sample=0, savepath=None)
- compute_early_rate(Pi, mu) -> (mean_rate, per_batch[B])
- compute_band_mass(Pi, mu, bandwidth) -> (mean_rate, per_batch[B])
- plot_baseline_vs_tical(y_res, y_hat, y_true=None, sample=0, var=0, savepath=None)
- plot_gate_hist(gate, savepath=None)
- save_alignment_summary_csv(Pi, mu, bandwidth, csv_path)

Also includes a synthetic demo in the __main__ block to showcase outputs.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_Tir_line(kappa, epoch, setting, save_path):
    """
    kappa: The kappa matrix containing the values for each text kernel (shape: [num_kernels, num_time_steps])
    epoch: Current epoch number for naming
    setting: Configuration setting for naming
    save_path: Path to save the figure
    """
    num_kernels = kappa.shape[1]
    plt.figure(figsize=(10, 6))
    
    batch_num = 0
    
    # Sum all the kernel values for each time step
    summed_kappa = np.sum(kappa[batch_num], axis=0)

    # Plot the summed line
    plt.plot(summed_kappa, label=f'{batch_num}_Y_tir', color='blue')

    plt.title(f"y tir Curve for Epoch {epoch + 1} ({setting[:100]})")
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Save the figure
    plt.savefig(save_path)
    plt.close()


def plot_kappa_multiple_lines(kappa, epoch, setting, save_path):
    """
    kappa: The kappa matrix containing the values for each text kernel (shape: [num_kernels, num_time_steps])
    epoch: Current epoch number for naming
    setting: Configuration setting for naming
    save_path: Path to save the figure
    """
    num_kernels = kappa.shape[1]
    # print(kappa.shape)
    # print("num_kernels:", num_kernels)
    plt.figure(figsize=(10, 6))
    
    batch_num = 0
    
    # Iterate over each kernel and plot its curve
    for kernel_idx in range(num_kernels):
        plt.plot(kappa[batch_num][kernel_idx], label=f'Kernel {kernel_idx + 1}')
    
    plt.title(f"Kappa Curves for Epoch {epoch + 1} ({setting[:100]})")
    plt.xlabel('Time Steps')
    plt.ylabel('Kappa Value')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Save the figure
    plt.savefig(save_path)
    plt.close()


# --------------------- Utilities ---------------------

def _as_np(x):
    """Convert torch or numpy to numpy without importing torch."""
    if x is None:
        return None
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# --------------------- Plots ---------------------

def plot_kappa_heatmap(kappa_bkh: np.ndarray, sample: int = 0, savepath: Optional[str] = None):
    """
    Plot heatmap for kappa (K x H).
    Args:
        kappa_bkh: [B,K,H]
    """
    kappa = _as_np(kappa_bkh)
    if kappa.ndim != 3:
        raise ValueError("kappa must be [B,K,H].")
    K = kappa.shape[1]
    H = kappa.shape[2]
    fig = plt.figure(figsize=(max(6, H/4), max(3, K/2)))
    plt.imshow(kappa[sample], aspect="auto", interpolation="nearest")
    plt.xlabel("Horizon step (1..H)")
    plt.ylabel("Kernel index (1..K)")
    plt.title(f"kappa heatmap (sample={sample})  shape=({K}x{H})")
    plt.colorbar()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


def plot_pi_heatmap(Pi_bkh: np.ndarray, sample: int = 0, savepath: Optional[str] = None):
    """
    Plot heatmap for OT plan Π (K x H).
    Args:
        Pi_bkh: [B,K,H]
    """
    Pi = _as_np(Pi_bkh)
    if Pi.ndim != 3:
        raise ValueError("Pi must be [B,K,H].")
    K = Pi.shape[1]
    H = Pi.shape[2]
    fig = plt.figure(figsize=(max(6, H/4), max(3, K/2)))
    plt.imshow(Pi[sample], aspect="auto", interpolation="nearest")
    plt.xlabel("Horizon step (1..H)")
    plt.ylabel("Kernel index (1..K)")
    plt.title(f"OT plan Π heatmap (sample={sample})  shape=({K}x{H})")
    plt.colorbar()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


def plot_baseline_vs_tical(y_res_bhd: np.ndarray, y_hat_bhd: np.ndarray, y_true_bhd: Optional[np.ndarray] = None,
                           sample: int = 0, var: int = 0, savepath: Optional[str] = None):
    """
    Compare baseline (y_res) vs TICaL (y_hat) vs ground truth (optional) for one (sample,var).
    Inputs: [B,H,D]
    """
    y_res = _as_np(y_res_bhd)
    y_hat = _as_np(y_hat_bhd)
    y_true = _as_np(y_true_bhd) if y_true_bhd is not None else None

    series = {
        "Baseline y_res": y_res[sample, :, var],
        "TICaL   y_hat": y_hat[sample, :, var],
    }
    if y_true is not None:
        series["Ground truth"] = y_true[sample, :, var]

    fig = plt.figure(figsize=(8, 4))
    for name, arr in series.items():
        plt.plot(arr, label=name)
    plt.xlabel("Horizon step")
    plt.ylabel("Value")
    plt.title(f"Baseline vs TICaL (sample={sample}, var={var})")
    plt.legend()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


def plot_gate_hist(gate_b1d: np.ndarray, savepath: Optional[str] = None):
    """
    Histogram of gate across variables (averaged over batch if needed).
    gate: [B,1,D] or [B,D]
    """
    gate = _as_np(gate_b1d)
    if gate.ndim == 3:
        gate = gate[:, 0, :]  # [B,D]
    if gate.ndim != 2:
        raise ValueError("gate must be [B,1,D] or [B,D].")

    gate_mean = gate.mean(axis=0)  # [D]
    fig = plt.figure(figsize=(8, 4))
    plt.bar(np.arange(gate_mean.shape[0]), gate_mean)
    plt.xlabel("Variable index (d)")
    plt.ylabel("Gate value (0..1)")
    plt.title("Average gate per variable")
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()
    return gate_mean


# --------------------- Metrics ---------------------

def compute_early_rate(Pi_bkh: np.ndarray, mu_bk: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Early-match rate = mass where h < mu_k divided by total mass.
    Returns: (scalar_rate, per_batch_rate[B])
    """
    Pi = _as_np(Pi_bkh)
    mu = _as_np(mu_bk)
    if Pi.ndim != 3 or mu.ndim != 2:
        raise ValueError("Pi must be [B,K,H] and mu must be [B,K].")

    B, K, H = Pi.shape
    idx = np.arange(H)[None, None, :] + 1.0  # 1..H
    mu_exp = mu[..., None]  # [B,K,1]

    early_mask = (idx < mu_exp).astype(Pi.dtype)  # [B,K,H]
    early_mass = (Pi * early_mask).sum(axis=(1, 2))  # [B]
    total_mass = Pi.sum(axis=(1, 2)) + 1e-12
    per_batch = (early_mass / total_mass)
    return float(per_batch.mean()), per_batch


def compute_band_mass(Pi_bkh: np.ndarray, mu_bk: np.ndarray, bandwidth: float) -> Tuple[float, np.ndarray]:
    """
    Band mass = mass where |h - mu_k| <= bandwidth divided by total mass.
    Returns: (scalar_rate, per_batch_rate[B])
    """
    Pi = _as_np(Pi_bkh)
    mu = _as_np(mu_bk)
    B, K, H = Pi.shape
    idx = np.arange(H)[None, None, :] + 1.0
    mu_exp = mu[..., None]
    band_mask = (np.abs(idx - mu_exp) <= bandwidth).astype(Pi.dtype)
    band_mass = (Pi * band_mask).sum(axis=(1, 2))
    total_mass = Pi.sum(axis=(1, 2)) + 1e-12
    per_batch = (band_mass / total_mass)
    return float(per_batch.mean()), per_batch


def save_alignment_summary_csv(Pi_bkh: np.ndarray, mu_bk: np.ndarray, bandwidth: float, csv_path: str):
    """Save early-rate and band-mass per batch to CSV."""
    mean_early, early_pb = compute_early_rate(Pi_bkh, mu_bk)
    mean_band, band_pb = compute_band_mass(Pi_bkh, mu_bk, bandwidth)
    df = pd.DataFrame({
        "batch": np.arange(len(early_pb)),
        "early_rate": early_pb,
        "band_mass": band_pb
    })
    df.to_csv(csv_path, index=False)
    return mean_early, mean_band, df


# --------------------- Synthetic Demo ---------------------

@dataclass
class DemoPack:
    kappa: np.ndarray   # [B,K,H]
    Pi: np.ndarray      # [B,K,H]
    mu: np.ndarray      # [B,K]
    gate: np.ndarray    # [B,1,D]
    y_res: np.ndarray   # [B,H,D]
    y_hat: np.ndarray   # [B,H,D]
    y_true: np.ndarray  # [B,H,D]


def _build_bank(l, mu, sigma, tau):
    # Shapes: [B,K,H]
    gauss  = np.exp(-0.5 * ((l - mu) / (sigma + 1e-9))**2)
    expdc  = np.exp(-(np.maximum(l - mu, 0.0)) / (tau + 1e-9))
    step   = (l >= mu).astype(float)
    bipeak = np.exp(-0.5 * ((l - (mu - sigma)) / (sigma + 1e-9))**2) \
           + np.exp(-0.5 * ((l - (mu + sigma)) / (sigma + 1e-9))**2)
    # Equal weight mix for demo
    return (gauss + expdc + step + bipeak) / 4.0


def make_synthetic_demo(B=3, K=4, H=24, D=5, rng_seed=7) -> DemoPack:
    rng = np.random.default_rng(rng_seed)
    # Kernels
    mu = rng.uniform(4, H-4, size=(B, K))
    sigma = rng.uniform(1.5, 4.0, size=(B, K))
    tau = rng.uniform(2.0, 6.0, size=(B, K))
    a = rng.uniform(0.5, 1.5, size=(B, K))

    l = np.arange(1, H+1)[None, None, :]  # [1,1,H]
    bank = _build_bank(l, mu[..., None], sigma[..., None], tau[..., None])  # [B,K,H]
    kappa = bank / (bank.sum(axis=-1, keepdims=True) + 1e-9)
    kappa = a[..., None] * kappa

    # Construct an OT-like plan Pi around mu with a band
    Pi = np.zeros((B, K, H))
    bw = 3.0
    for b in range(B):
        for k in range(K):
            h_grid = np.arange(1, H+1)
            weights = np.exp(-0.5 * ((h_grid - mu[b, k]) / (sigma[b, k] + 1e-9))**2)
            # band
            mask = (np.abs(h_grid - mu[b, k]) <= bw).astype(float)
            weights = weights * mask
            if weights.sum() < 1e-9:
                weights[int(round(min(max(mu[b, k]-1, 0), H-1)))] = 1.0
            Pi[b, k] = weights / (weights.sum() + 1e-9)
    # Row-scale Pi to respect kernel mass r proportional to kappa.sum(-1)
    r = kappa.sum(-1)  # [B,K]
    for b in range(B):
        Pi[b] = (Pi[b].T * (r[b] / (Pi[b].sum(axis=1) + 1e-9))).T

    # Build time series: baseline y_res and TICaL y_hat
    y_true = rng.normal(0.0, 0.5, size=(B, H, D)).cumsum(axis=1)  # some smooth ground truth
    # inject text-induced effect on var 0 and 1 using kappa sum over K
    tir_scalar = kappa.sum(axis=1)  # [B,H]
    # expand to D with variable sensitivities
    var_sens = np.linspace(1.0, 0.3, D)[None, None, :]  # higher on small index vars
    y_tir = (tir_scalar[..., None]) * var_sens  # [B,H,D]
    # baseline misses the effect
    y_res = y_true - 0.5 * y_tir + rng.normal(0, 0.1, size=(B, H, D))
    # TICaL corrects toward the effect
    y_hat = y_res + 0.8 * (y_tir - (y_res - y_true))

    # gate (higher for early vars)
    gate = np.clip(var_sens * 0.9, 0, 1.0)  # [1,1,D]; tile over batch
    gate = np.tile(gate, (B, 1, 1))

    return DemoPack(kappa=kappa, Pi=Pi, mu=mu, gate=gate, y_res=y_res, y_hat=y_hat, y_true=y_true)


# --------------------- Demo Run ---------------------

if __name__ == "__main__":
    demo = make_synthetic_demo()

    # 1) kappa heatmap
    plot_kappa_heatmap(demo.kappa, sample=0, savepath="fig_kappa_sample0.png")

    # 2) OT plan Π heatmap
    plot_pi_heatmap(demo.Pi, sample=0, savepath="fig_Pi_sample0.png")

    # 3) Early-match rate & Band-mass table
    mean_early, early_pb = compute_early_rate(demo.Pi, demo.mu)
    mean_band, band_pb = compute_band_mass(demo.Pi, demo.mu, bandwidth=3.0)
    df = pd.DataFrame({
        "batch": np.arange(len(early_pb)),
        "early_rate": early_pb,
        "band_mass@3": band_pb
    })
    df.to_csv("alignment_summary.csv", index=False)

    # 4) Baseline vs TICaL curves (one var)
    plot_baseline_vs_tical(demo.y_res, demo.y_hat, demo.y_true, sample=0, var=0, savepath="fig_curve_s0_v0.png")

    # 5) Gate histogram
    plot_gate_hist(demo.gate, savepath="fig_gate_hist.png")

    print("Saved demo figures:")
    print("- fig_kappa_sample0.png")
    print("- fig_Pi_sample0.png")
    print("- fig_curve_s0_v0.png")
    print("- fig_gate_hist.png")
    print("Saved metrics CSV: alignment_summary.csv")
