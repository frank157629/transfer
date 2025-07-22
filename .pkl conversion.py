import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("./data/GFM/dataset_v13.pkl", "rb") as f:
    sol = np.asarray(pickle.load(f))
print("sol shape:", sol.shape)  # (n_traj, 14, 1000)

if sol.shape[1] != 14:
    raise ValueError(f"Expected 14 states, but found {sol.shape[1]}!") #13 states + t


var_names14 = [
    't', 'xi_d', 'xi_q', 'vfd',
    'vfq', 'ifd', 'ifq', 'itd',
    'itq', 'sigma_d', 'sigma_q', 'gamma_q',
    'theta_gfm', 'theta_grid'
]

# === Plot 1st trajectory ===
idx = 0
traj = sol[idx]
t = traj[0]

fig, axes = plt.subplots(14, 1, figsize=(10, 28), sharex=True)
if not hasattr(axes, '__len__'):
    axes = [axes]
print(f"Number of axis: {len(axes)}")
for row in range(14):
    axes[row].plot(t, traj[row], label=f'traj {idx}')
    axes[row].set_ylabel(var_names14[row])
    axes[row].grid(True, ls='--', alpha=.3)
axes[-1].set_xlabel('time (s)')
axes[1].legend(loc='upper right')
plt.tight_layout()
plt.show()

# === Print state variables at t=0 and t=1 ===
for i, traj in enumerate(sol):
    time = traj[0]
    idx_t0 = (np.abs(time - 0)).argmin()
    idx_t1 = (np.abs(time - 1)).argmin()

    print(f"\n--- Trajectory {i} ---")
    print(f"Initial condition: t = {time[idx_t0]:.6f}")
    for name, value in zip(var_names14, traj[:, idx_t0]):
        print(f"{name} @ t=0: {value:.6f}")

    print(f"\nEnd of simulation: t = {time[idx_t1]:.6f}")
    for name, value in zip(var_names14, traj[:, idx_t1]):
        print(f"{name} @ t=1: {value:.6f}")

    # === Compute p and q at t=1 ===
    vfd = traj[3, idx_t1]
    vfq = traj[4, idx_t1]
    itd = traj[7, idx_t1]
    itq = traj[8, idx_t1]

    p = vfd * itd + vfq * itq
    q = vfq * itd - vfd * itq

    print(f"\n[Computed] Active power p @ t=1: {p:.6f}")
    print(f"[Computed] Reactive power q @ t=1: {q:.6f}")

