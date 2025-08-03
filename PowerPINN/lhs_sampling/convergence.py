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
data_path = "../data/GFL_2nd_order/dataset_v19.pkl"

# --- Threshold settings ---
omega_std_thresh     = 1e-2        # Oscillation tolerance for ω (std, <1 % of 2π * 0.2)
omega_mean_thresh    = 1.26        # Final value tolerance for ω (≈ 2π * 0.2 Hz, European standard)

delta_std_thresh     = 0.05        # Oscillation tolerance for δ (in rad, <1 % of 2π)
delta_final_thresh   = 0.14        # Convergence tolerance for δ to 0 or 2π·n (≈ 8°, IEEE Std 1547-2018)

tail_window = 100                  # Number of tail time steps to check convergence
pi = np.pi

num_points_to_save = 10000
# =========================================

def load_dataset(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.array(data)  # shape = (N, 3, T)

def analyze_convergence(data, omega_std_thr, omega_mean_thr,
                        delta_std_thr, delta_final_thr, tail_window):
    """
    Check convergence:
    - ω: low oscillation (std) and mean close to 0
    - δ: low oscillation (std) and final value close to n·2π
    """
    N = data.shape[0]
    groups = defaultdict(list)

    for i in range(N):
        delta_series = data[i, 1, :]
        omega_series = data[i, 2, :]

        delta_tail = delta_series[-tail_window:]
        omega_tail = omega_series[-tail_window:]

        # ω convergence criteria
        omega_std = np.std(omega_tail)
        omega_mean = np.abs(np.mean(omega_tail))

        # δ convergence criteria
        delta_std = np.std(delta_tail)
        final_delta = delta_series[-1]
        delta_mod = np.abs((final_delta + pi) % (2 * pi) - pi)  # Normalize to [-π, π]
        delta_residual = np.abs(delta_mod)  # Distance to n·2π

        if (omega_std  < omega_std_thr and
            omega_mean < omega_mean_thr and
            delta_std  < delta_std_thr and
            delta_residual < delta_final_thr):

            delta0 = delta_series[0]
            omega0 = omega_series[0]
            final_omega = omega_series[-1]
            k = int(np.floor(final_delta / (2 * pi)))  # Index of convergence zone
            groups[k].append((i, delta0, omega0, final_delta, final_omega))

    return groups

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
    grouped = analyze_convergence(data,
                                  omega_std_thresh,
                                  omega_mean_thresh,
                                  delta_std_thresh,
                                  delta_final_thresh,
                                  tail_window)

    print("📋 Generating report...")
    print_report(grouped)

    print("🎯 Plotting classification map...")
    A, B, C = plot_scatter(grouped, data)

    print("💾 Saving all converged trajectories (any 2kπ zone)...")
    # Get all converged trajectory indices
    all_converged_entries = [entry for group in grouped.values() for entry in group]
    all_converged_ids = [entry[0] for entry in all_converged_entries]
    all_converged_trajectories = data[all_converged_ids]

    # 💾 只保存前 k 条收敛轨迹并覆盖文件
    print(f"💾 Saving first {num_points_to_save} converged trajectories (overwriting)...")
    k_converged_ids = all_converged_ids[:num_points_to_save]
    k_converged_trajectories = data[k_converged_ids]

    save_path = f"../lhs_sampling/dataset_v19_converged_k{num_points_to_save}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(k_converged_trajectories, f)

    print(f"✅ Done. Saved {len(k_converged_ids)} converged trajectories to → {save_path}")
