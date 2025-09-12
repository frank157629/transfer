#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convergence1.py

1. This file takes the trajectories from a dataset and filters the converging
    datasets from the non-converging, and saves a number of the converging
    datasets under "../lhs_sampling" for further processing. A convergence
    map would also be saved inside the lhs_sampling folder.
2. Configuration for the error and standard deviation tolerance can be tuned under
    section <Configurations>.
3. For the training, the datasets under "../data/GFL_2nd_order" would be used.
    (e.g. "dataset_v10")
"""

import numpy as np, pickle, matplotlib.pyplot as plt
from collections import defaultdict
import os

# ======== configurations ========
dataset_id      = 9
data_path       = f"../data/GFL_2nd_order/dataset_v{dataset_id}.pkl"

tail_window     = 100          # tail window
delta_offset    = 0.137        # Δ₀：offset from 2*pi because of error of Rahul's Model
delta_tol       = 0.001        # Error tolerance in tail window ±tol (rad)

# sampling ratio
sampling_ratio = dict(region_0 = 1,      # δ → 0+Δ₀
                      region_2kpi = 0,   # δ → 2kπ+Δ₀, k≠0
                      non_converged= 0)
num_points_to_save = 10000              # Total trajectories
# =========================

pi = np.pi
# ---------- Tools ----------
def load_dataset(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return np.array(pickle.load(f))     # shape: (N, 3, T)

def classify_delta(data: np.ndarray,tail: int,offset: float,tol: float):
    """
    Convergence of δ ：
      • center = k·2π + offset
      • For all δ ∈ [center ± tol] in tail window -> allocate in k
    return dict{k: [traj_id,…]}
    """
    groups = defaultdict(list)
    N      = data.shape[0]

    for idx in range(N):
        delta_tail = data[idx, 1, -tail:]                    # shape (tail,)
        mean_tail  = np.mean(delta_tail)
        k          = int(np.round((mean_tail - offset) / (2*pi)))
        center     = k * 2 * pi + offset

        # Check all deltas whether under tolerance or not.
        if np.all(np.abs(delta_tail - center) <= tol):
            groups[k].append(idx)

    return groups

def plot_ic_scatter(groups, data):
    """Scatter points plot：region_0 green；other 2kπ blue；non-convergent red"""
    region0, regionK, nonconv = [], [], []

    all_ids = set(range(data.shape[0]))
    conv_ids = set()

    for k, lst in groups.items():
        conv_ids.update(lst)
        for idx in lst:
            delta0, omega0 = data[idx, 1, 0], data[idx, 2, 0]
            (region0 if k == 0 else regionK).append((delta0, omega0))

    for idx in all_ids - conv_ids:
        delta0, omega0 = data[idx, 1, 0], data[idx, 2, 0]
        nonconv.append((delta0, omega0))

    def arr(x): return np.array(x) if x else np.empty((0, 2))
    A, B, C = map(arr, (region0, regionK, nonconv))

    plt.figure(figsize=(8, 6))
    if A.size: plt.scatter(A[:, 0], A[:, 1], s=1, c='green', label=f"[0,2π): {len(A)}")
    if B.size: plt.scatter(B[:, 0], B[:, 1], s=1, c='blue',  label=f"2kπ: {len(B)}")
    if C.size: plt.scatter(C[:, 0], C[:, 1], s=1, c='red',   label=f"Non-conv: {len(C)}")
    plt.xlabel("Initial δ₀"), plt.ylabel("Initial ω₀"), plt.title("Trajectory Convergence Classification")
    plt.grid(True), plt.legend(), plt.tight_layout()
    plt.savefig("convergence_map.pdf")
    plt.close()
    return A, B, C

def quick_plot(traj_ids, title):
    """draw δ & ω trajectories"""
    if not traj_ids:
        print(f"[{title}] empty set, skip.")
        return
    plt.figure(figsize=(8, 8))

    # δ
    ax1 = plt.subplot(2, 1, 1)
    for idx in traj_ids:
        t = data[idx, 0, :]
        ax1.plot(t, data[idx, 1], lw=.8)
    ax1.set_ylabel("delta (rad)")
    ax1.set_title(title); ax1.grid(alpha=.3)

    # ω
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    for idx in traj_ids:
        t = data[idx, 0, :]
        ax2.plot(t, data[idx, 2], lw=.8)
    ax2.set_ylabel("omega")
    ax2.set_xlabel("time (s)"); ax2.grid(alpha=.3)

    plt.tight_layout(); plt.show()



# ---------- Main function ----------
if __name__ == "__main__":
    print(" Loading dataset…")
    data = load_dataset(data_path)                    # shape (N,3,T)

    print(" Classifying trajectories by δ tail behaviour…")
    groups = classify_delta(data, tail_window, delta_offset, delta_tol)

    # ---- statistic ----
    total_conv = sum(len(v) for v in groups.values())
    print(f" Converged trajectories: {total_conv}/{data.shape[0]}\n")
    for k in sorted(groups):
        print(f"  k = {k:2d} ([{2*k}π+Δ₀]): {len(groups[k])} trajectories")

    # ---- visualisation ----
    A, B, C = plot_ic_scatter(groups, data)

    # ---- Sampling inside Trajectory set----
    k0_ids  = groups.get(0, [])
    kN_ids  = [idx for k,v in groups.items() if k!=0 for idx in v]
    non_ids = list(set(range(data.shape[0])) - set(k0_ids) - set(kN_ids))

    n_total = num_points_to_save
    n_k0 = min(len(k0_ids),  int(n_total*sampling_ratio["region_0"]))
    n_kN = min(len(kN_ids),  int(n_total*sampling_ratio["region_2kpi"]))
    n_non= min(len(non_ids), n_total - n_k0 - n_kN)

    rng = np.random.default_rng(42)
    sample_ids = np.concatenate([
        rng.choice(k0_ids,  n_k0,  replace=False) if n_k0  else [],
        rng.choice(kN_ids,  n_kN,  replace=False) if n_kN  else [],
        rng.choice(non_ids, n_non, replace=False) if n_non else []
    ])
    rng.shuffle(sample_ids)
    sample_ids = sample_ids.astype(np.int64)

    # ---------------------------------------------
    quick_plot(k0_ids, "Class A  (δ→0·2π+Δ₀)")
    quick_plot(kN_ids, "Class B  (δ→k·2π+Δ₀, k≠0)")
    quick_plot(non_ids, "Class C  (Non-converged)")

    # ---- save dataset----
    final_ds   = data[sample_ids]
    out_dir    = "../data/GFL_2nd_order"
    os.makedirs(out_dir, exist_ok=True)
    save_path  = f"{out_dir}/dataset_v{dataset_id}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(final_ds, f)
    print(f"✅ Saved {len(sample_ids)} trajectories to → {save_path}")

    # ---- IC map PDF ----
    delta0_all, omega0_all = data[sample_ids, 1, 0], data[sample_ids, 2, 0]
    plt.figure(figsize=(8,6))
    plt.scatter(delta0_all, omega0_all, s=1, c='black', alpha=.6)
    plt.xlabel("Initial δ₀"), plt.ylabel("Initial ω₀")
    plt.title("IC of Saved Trajectories"), plt.grid(True), plt.tight_layout()
    pdf_path = f"{out_dir}/ic_map_dataset_v{dataset_id}_mixed_k{len(sample_ids)}.pdf"
    plt.savefig(pdf_path), plt.close()
    print(f"✅ IC map saved to → {pdf_path}")