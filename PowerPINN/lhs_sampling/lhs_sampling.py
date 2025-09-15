import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.stats import qmc
'''
1. This script generates initial condition (IC) sets for δ and ω within a 
   rectangular region using Latin Hypercube Sampling (LHS). Parameters can be 
   tuned under "Configurable parameters".
2. The ICs will be saved as "lhs_init_conditions.pkl".
3. A plot of the sampled points will be saved as "lhs_region_plot.pdf".
4. The next step is handled in "create_dataset_d1.py", which uses the generated 
   ICs to simulate trajectories.
'''
# ======== Configurable parameters ========
n_samples = 100000                      # Number of sample points
delta_range = (-np.pi, np.pi)       # Range for δ (x-axis)
omega_range = (-60, 60)             # Range for ω (y-axis)
base_height = 400
save_path = "../lhs_init_conditions.pkl"
plot_path = "../lhs_region_plot.pdf"
# ========================================

def generate_lhs_points_and_plot(n_samples, delta_range, omega_range, save_path, plot_path):
    # LHS sampling
    sampler = qmc.LatinHypercube(d=2)
    sample_unit = sampler.random(n=n_samples)
    sample_scaled = qmc.scale(sample_unit, [delta_range[0], omega_range[0]], [delta_range[1], omega_range[1]])

    # Print first 5 samples for inspection
    print("🔍 Example sampled points:")
    for i in range(5):
        print(f"{i+1}: δ={sample_scaled[i,0]:.3f}, ω={sample_scaled[i,1]:.3f}")

    # Save as .pkl file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(sample_scaled.tolist(), f)
    print(f"Saved {n_samples} LHS points to {save_path}")

    # Compute aspect ratio
    delta_width = delta_range[1] - delta_range[0]
    omega_height = omega_range[1] - omega_range[0]
    aspect_ratio = delta_width / omega_height
    fig_width = base_height * aspect_ratio
    fig_height = base_height

    # Plotting
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.scatter(sample_scaled[:, 0], sample_scaled[:, 1],
               s=5, color='red', label='LHS Samples', alpha=0.8)

    ax.set_xlim(delta_range)
    ax.set_ylim(omega_range)
    ax.set_aspect('equal')  # Equal aspect ratio for both axes

    ax.set_xlabel(r"$\delta_0$ (rad)")
    ax.set_ylabel(r"$\omega_0$ (rad/s)")
    ax.set_title(f"LHS Samples ({n_samples}) in Rectangular Region")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")
    # plt.show()

    return sample_scaled

# ======= Execution entry point =======
if __name__ == "__main__":
    generate_lhs_points_and_plot(n_samples, delta_range, omega_range, save_path, plot_path)
