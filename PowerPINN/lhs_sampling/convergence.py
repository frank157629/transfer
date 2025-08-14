import numpy as np
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt


'''
1. This file takes the trajectories from a dataset and filters the converging 
    datasets from the non-converging, and saves a number of the converging 
    datasets under the name "converged_all_zones.pkl" under "../data/GFL_2nd_order" 
    for further processing. A convergence map would be saved inside the lhs_sampling folder.
    Trajectories are considered to be convergent under following criteria in a tail time window of 100ms:
        -Standard deviation should be less than a certain tolerance.
        -Final value of delta should lie within a certain tolerance near multiples of 2π.
        -Final value of omega should lie within a certain tolerance near zero.
2. Configuration for the error and standard deviation tolerance can be tuned under 
    section <Configuration parameters>.
3. For the training, you should change the name "converged_all_zones.pkl" into conventional 
    names that can be recognized. (e.g. "dataset_v10")
'''
# ======== Configuration parameters ========
dataset = "dataset_v" + str(8)
data_path = f"../data/GFL_2nd_order/{dataset}.pkl"

# --- Threshold settings ---
# --- Threshold settings ---
omega_std_thresh   = 0.01     # RMS 波动阈值   (Hz)
omega_ptp_thresh   = 0.05       # 峰-峰摆幅阈值 (Hz)
omega_abs_thresh   = 1.26       # 尾窗内 |ω| 最大允许值 (Hz)

delta_std_thresh   = 0.05       # δ RMS 波动 (rad)
delta_final_thresh = 0.14       # δ 收敛到 2kπ±0.14 (rad)
tail_window        = 1       # 建议覆盖≥4-6个周期
# omega_std_thresh     = 0.01        # Oscillation tolerance for ω (std, <1 % of 2π * 0.2)
# omega_mean_thresh    = 1.26       # Final value tolerance for ω (≈ 2π * 0.2 Hz, European standard)
#
# delta_std_thresh     = 0.05       # Oscillation tolerance for δ (in rad, <1 % of 2π)
# delta_final_thresh   = 0.14        # Convergence tolerance for δ to 0 or 2π·n (≈ 8°, IEEE Std 1547-2018)
#
# tail_window = 40                  # Number of tail time steps to check convergence
pi = np.pi

# --- Sampling ratio for A / B / C ---
sampling_ratio = {
    "region_0": 1,        # [0, 2π)
    "region_2kpi": 0,     # other 2kπ
    "non_converged": 0   # red
}

num_points_to_save = 10000
# =========================================


def load_dataset(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.array(data)  # shape = (N, 3, T)
#New version with abs
def analyze_convergence(data,
                        ω_std_thr,  ω_ptp_thr,  ω_abs_thr,
                        δ_std_thr,  δ_final_thr,
                        tail_window):
    """
    1. ω 在尾窗内 —— 绝对值、RMS、峰-峰值 都小于阈值
    2. δ 在尾窗内 —— RMS 足够小 & 收敛到 2kπ
    """
    N = data.shape[0]
    groups = defaultdict(list)
    pi = np.pi

    for i in range(N):
        δ_series = data[i, 1, :]
        ω_series = data[i, 2, :]

        δ_tail = δ_series[-tail_window:]
        ω_tail = ω_series[-tail_window:]

        # ---------- ω 判据 ----------
        ω_abs_ok  = np.max(np.abs(ω_tail)) < ω_abs_thr
        ω_std_ok  = np.std(ω_tail)         < ω_std_thr
        ω_ptp_ok  = np.ptp(ω_tail)         < ω_ptp_thr

        # ---------- δ 判据 ----------
        δ_std_ok  = np.std(δ_tail)         < δ_std_thr
        δ_final   = δ_series[-1]
        δ_mod     = np.abs((δ_final + pi) % (2*pi) - pi)   # 距离最近的 2kπ
        δ_conv_ok = δ_mod < δ_final_thr

        if ω_abs_ok and ω_std_ok and ω_ptp_ok and δ_std_ok and δ_conv_ok:
        # if ω_abs_ok and ω_ptp_ok and δ_std_ok and δ_conv_ok:
            k = int(np.floor(δ_final / (2*pi)))            # 收敛区间索引
            groups[k].append((
                i,                      # 轨迹 ID
                δ_series[0],            # δ0
                ω_series[0],            # ω0
                δ_final,                # δ_final
                ω_series[-1])           # ω_final
            )
    return groups
#old version
# def analyze_convergence(data, omega_std_thr, omega_mean_thr,
#                         delta_std_thr, delta_final_thr, tail_window):
#     """
#     Check convergence:
#     - ω: low oscillation (std) and mean close to 0
#     - δ: low oscillation (std) and final value close to n·2π
#     """
#     N = data.shape[0]
#     groups = defaultdict(list)
#
#     for i in range(N):
#         delta_series = data[i, 1, :]
#         omega_series = data[i, 2, :]
#
#         delta_tail = delta_series[-tail_window:]
#         omega_tail = omega_series[-tail_window:]
#
#         # ω convergence criteria
#         omega_std = np.std(omega_tail)
#         omega_mean = np.abs(np.mean(omega_tail))
#
#         # δ convergence criteria
#         delta_std = np.std(delta_tail)
#         final_delta = delta_series[-1]
#         delta_mod = np.abs((final_delta + pi) % (2 * pi) - pi)  # Normalize to [-π, π]
#         delta_residual = np.abs(delta_mod)  # Distance to n·2π
#
#         if (omega_std  < omega_std_thr and
#             omega_mean < omega_mean_thr and
#             delta_std  < delta_std_thr and
#             delta_residual < delta_final_thr):
#
#             delta0 = delta_series[0]
#             omega0 = omega_series[0]
#             final_omega = omega_series[-1]
#             k = int(np.floor(final_delta / (2 * pi)))  # Index of convergence zone
#             groups[k].append((i, delta0, omega0, final_delta, final_omega))
#
#     return groups

def print_report(groups):
    total = sum(len(v) for v in groups.values())
    print(f"📊 Total converged trajectories: {total}\n")
    print("📦 Grouped by δ ∈ [2kπ, 2(k+1)π):")
    for k in sorted(groups.keys()):
        group = groups[k]
        print(f"  Interval [{2*k}π, {2*(k+1)}π): {len(group)} trajectories")
        for entry in group:
            i, delta0, omega0, deltaf, omegaf = entry
            print(f"    ID {i:5d} | δ₀ = {delta0:+.3f}, ω₀ = {omega0:+.3f} → δf = {deltaf:+.3f}, ωf = {omegaf:+.3f}")
        print()

def plot_scatter(groups, data):
    group_A = []  # Converged to [0, 2π)
    group_B = []  # Converged to other 2kπ intervals
    group_C = []  # Non-converged

    for k, trajs in groups.items():
        for entry in trajs:
            _, delta0, omega0, deltaf, _ = entry
            if 0 <= deltaf < 2 * pi:
                group_A.append((delta0, omega0))
            else:
                group_B.append((delta0, omega0))

    all_ids = set(range(data.shape[0]))
    converged_ids = set(i for group in groups.values() for i, *_ in group)
    non_converged_ids = all_ids - converged_ids
    for i in non_converged_ids:
        delta0 = data[i, 1, 0]
        omega0 = data[i, 2, 0]
        group_C.append((delta0, omega0))

    A = np.array(group_A)
    B = np.array(group_B)
    C = np.array(group_C)

    plt.figure(figsize=(8, 6))
    if len(A) > 0:
        plt.scatter(A[:, 0], A[:, 1], color='green', label=f"[0, 2π): {len(A)}", s=1)
    if len(B) > 0:
        plt.scatter(B[:, 0], B[:, 1], color='blue', label=f"Other 2kπ: {len(B)}", s=1)
    if len(C) > 0:
        plt.scatter(C[:, 0], C[:, 1], color='red', label=f"Non-converged: {len(C)}", s=1)

    plt.xlabel("Initial δ₀")
    plt.ylabel("Initial ω₀")
    plt.title("Trajectory Convergence Classification")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("convergence_map.pdf")
    plt.show()
    return A, B, C

if __name__ == "__main__":
    print("📥 Loading dataset...")
    data = load_dataset(data_path)

    print("🧮 Analyzing convergence based on tail behavior...")
    # grouped = analyze_convergence(data,
    #                               omega_std_thresh,
    #                               omega_mean_thresh,
    #                               delta_std_thresh,
    #                               delta_final_thresh,
    #                               tail_window)
    grouped = analyze_convergence(data,
                                  omega_std_thresh,
                                  omega_ptp_thresh,
                                  omega_abs_thresh,
                                  delta_std_thresh,
                                  delta_final_thresh,
                                  tail_window)

    print("📋 Generating report...")
    print_report(grouped)

    print("🎯 Plotting classification map...")
    A, B, C = plot_scatter(grouped, data)

    print("🎯 Sampling from three regions by ratio...")

    # region_0: green (label 1)
    # region_2kpi: blue (label 2)
    # non_converged: red (label 0)

    # === Step 1: 分出三类 ID ===
    region_0_ids = [entry[0] for group in grouped.values() for entry in group if 0 <= entry[3] < 2 * pi]
    region_2kpi_ids = [entry[0] for group in grouped.values() for entry in group if not (0 <= entry[3] < 2 * pi)]
    all_ids = set(range(data.shape[0]))
    converged_ids = set(region_0_ids + region_2kpi_ids)
    non_converged_ids = list(all_ids - converged_ids)

    # === Step 2: 按比例确定采样数量 ===
    n_total = num_points_to_save
    n_r0 = min(len(region_0_ids), int(n_total * sampling_ratio["region_0"]))
    n_r2 = min(len(region_2kpi_ids), int(n_total * sampling_ratio["region_2kpi"]))
    n_nc = min(len(non_converged_ids), n_total - n_r0 - n_r2)
    if n_r0 < int(n_total * sampling_ratio["region_0"]):
        print(f"⚠️  Warning: region_0 only has {len(region_0_ids)} samples, using all of them.")
    if n_r2 < int(n_total * sampling_ratio["region_2kpi"]):
        print(f"⚠️  Warning: region_2kpi only has {len(region_2kpi_ids)} samples, using all of them.")
    if n_nc < int(n_total * sampling_ratio["non_converged"]):
        print(f"⚠️  Warning: non-converged only has {len(non_converged_ids)} samples, using all of them.")

    # === Step 3: 随机抽样 ===
    np.random.seed(42)
    r0_sample = np.random.choice(region_0_ids, n_r0, replace=False)
    r2_sample = np.random.choice(region_2kpi_ids, n_r2, replace=False)
    nc_sample = np.random.choice(non_converged_ids, n_nc, replace=False)

    # === Step 4: 合并样本 ===
    final_indices = np.concatenate([r0_sample, r2_sample, nc_sample]).astype(int)
    np.random.shuffle(final_indices)
    # === Step 5: 保存数据 ===
    final_dataset = data[final_indices]
    save_path = f"../lhs_sampling/{dataset}_mixed_k{n_total}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(final_dataset, f)

    print(f"✅ Saved {len(final_indices)} mixed-sample trajectories to → {save_path}")

    import matplotlib.pyplot as plt

    print("📊 Generating IC scatter plot for saved trajectories...")

    # === Step 6: 画出 final_indices 轨迹的初始点 ===
    delta0_all = data[final_indices, 1, 0]
    omega0_all = data[final_indices, 2, 0]

    plt.figure(figsize=(8, 6))
    plt.scatter(delta0_all, omega0_all, c='black', s=1, alpha=0.6)
    plt.xlabel("Initial δ₀")
    plt.ylabel("Initial ω₀")
    plt.title("Initial Conditions of Saved Trajectories")
    plt.grid(True)
    plt.tight_layout()

    # Save PDF
    pdf_path = f"../lhs_sampling/ic_map_{dataset}_mixed_k{n_total}.pdf"
    plt.savefig(pdf_path)
    plt.close()

    print(f"✅ IC map saved to → {pdf_path}")