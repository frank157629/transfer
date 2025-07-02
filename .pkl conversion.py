import pickle
import numpy as np
import matplotlib.pyplot as plt

# Upload dataset generation file
# Insert the name of the dataset you want to convert es. dataset_v2 (directory: data-->GFM)
# Close plot first if you want to visualize printed values on terminal

with open("./data/GFM/dataset_v1.pkl", "rb") as f:
    sol = np.asarray(pickle.load(f))
print("sol shape:", sol.shape)  # (n_traj, 14, 1000)

if sol.shape[1] != 14:
    raise ValueError(f"Expected 14 states, but found {sol.shape[1]}!")

# Variables names
var_names14 = [
    't', 'xi_d', 'xi_q', 'vfd',
    'vfq', 'ifd', 'ifq', 'itd',
    'itq', 'sigma_d', 'sigma_q', 'gamma_q',
    'theta_gfm', 'theta_grid'
]

# === Plot 1st trajectory ===
idx = 0
traj = sol[idx]         # (14, 1000)
t = traj[0]             # tempo

fig, axes = plt.subplots(14, 1, figsize=(10, 28), sharex=True)
if not hasattr(axes, '__len__'):
    axes = [axes]
print(f"Numero assi: {len(axes)}")
for row in range(14):
    axes[row].plot(t, traj[row], label=f'traj {idx}')
    axes[row].set_ylabel(var_names14[row])
    axes[row].grid(True, ls='--', alpha=.3)
axes[-1].set_xlabel('time (s)')
axes[1].legend(loc='upper right')
plt.tight_layout()
plt.show()

# === Print state variables value at t=0 (initial condition) and t=1 (end simulation) ===
for i, traj in enumerate(sol):
    time = traj[0]  # vector time for this trajectory
    idx_t0 = (np.abs(time - 0)).argmin()
    idx_t1 = (np.abs(time - 1)).argmin()

    print(f"\n--- Trajectory {i} ---")
    print(f"Initial condition: t = {time[idx_t0]:.6f}")
    for name, value in zip(var_names14, traj[:, idx_t0]):
        print(f"{name} @ t=0: {value:.6f}")

    print(f"\nEnd of simulation: t = {time[idx_t1]:.6f}")
    for name, value in zip(var_names14, traj[:, idx_t1]):
        print(f"{name} @ t=1: {value:.6f}")
