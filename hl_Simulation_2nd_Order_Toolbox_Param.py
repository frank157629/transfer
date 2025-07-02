import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# === 1. Parameters exactly as in the paper (Equations (3) and (4)) ===
#
#   Equation (4) defines (using the same variable names as the paper):
#     M   = 1 − k_p · L_g · i_d^c
#     T_m = k_i · (r_Lg · i_q^c + L_g · i_d^c · ω_g)
#     T_e = k_i · v_g · sin(δ)
#     D   = k_p · Vg · cos(δ) − k_i · L_g · i_d^c
#
#     k_p     → k_p
#     k_i     → k_i
#     L_g     → L_g
#     r_Lg    → r_Lg
#     i_d^c   → i_d_c
#     i_q^c   → i_q_c
#     v_g     → v_g
#     ω_g     → omega_g
#
#   Equation (3) then reads:
#     δ̇      = ω
#     M · ω̇  = T_m − T_e − D·ω
#
#   so that:
#     ω̇ = (T_m − T_e − D·ω) / M
#

# 1.1 GFL_2nd_order gains (SRF-GFL_2nd_order gains, first number = k_p, second = k_i) and nominal angular frequency
k_p = 0.025         # proportional gain of SRF-GFL_2nd_order (KP in the paper)
k_i = 1.5           # integral gain of SRF-GFL_2nd_order    (KI in the paper)
f_0 = 50.0                           # f_0: nominal grid frequency (Hz)
omega_g = 2 * np.pi * f_0                # ω_g: 2π·50 rad/s


# 1.2 Grid-side inductor parameters (X/R = 16.3 → choose r_Lg = 1.0 pu, then X = 16.3 pu)
#     L_g in pu = X / ω_g
L_g = 16.3 / omega_g   # per‐unit inductance, since X_pu = 16.3 and ω_g = 2π·f0
r_Lg = 1.0                      # per‐unit resistance of grid‐side inductor

# 1.3 Pre-fault (converter) currents, denoted i_d^c and i_q^c in the paper
i_d_c = 1.0      # i_d^c (d‐axis current before fault) in pu
i_q_c = -0.1      # i_q^c (q‐axis current before fault) in pu

# 1.4 Grid voltage
v_g     = 690.0                          # v_g: nominal grid voltage (peak, V)

# 1.5 Compute M and T_m from Equation (4)  (both are constants)
M   = 1.0 - k_p * L_g * i_d_c
T_m = k_i * (r_Lg * i_q_c + L_g * i_d_c * omega_g)

# === 2. Simulation time settings ===
t_start = 0.0
t_end   = 1.0
t_eval  = np.linspace(t_start, t_end, 1001)

# === 3. Define the reduced second-order GFL_2nd_order ODE (Equation (3) & (4)) ===
# State vector x = [δ, ω]
def pll_second_order(t, x):
    delta, omega = x

    # T_e = k_i · v_g · sin(δ)
    T_e = k_i * v_g * np.sin(delta)

    # D = k_p · r_Lg · cos(δ) − k_i · L_g · i_d_c
    D = k_p * v_g * np.cos(delta) - k_i * L_g * i_d_c

    # ω̇ = (T_m − T_e − D·ω) / M
    domega_dt = (T_m - T_e - D * omega) / M

    # δ̇ = ω
    ddelta_dt = omega

    return [ddelta_dt, domega_dt]

# === 4. Generate 10 different initial conditions ===
np.random.seed(42)  # for reproducibility

initial_conditions = []
for _ in range(10):
    # δ₀ ∈ [–π/2, +π/2];  ω₀ ∈ [–10, +10]
    delta0 = np.random.uniform(-np.pi/2, np.pi/2)
    omega0 = np.random.uniform(-10.0, 10.0)
    initial_conditions.append((delta0, omega0))

# === 5. Solve the ODE for each initial condition ===
results = []

for run_index, (d0, w0) in enumerate(initial_conditions, start=1):
    # run_index is simply the counter (1 through 10) from enumerate(...)
    sol = solve_ivp(
        fun=pll_second_order,
        t_span=(t_start, t_end),
        y0=[d0, w0],
        t_eval=t_eval,
        method='RK45'
    )
    if not sol.success:
        print(f"Run {run_index} failed: {sol.message}")
        continue

    # Build a DataFrame for this run
    df = pd.DataFrame({
        'run': run_index,
        't': sol.t,
        'delta': sol.y[0],
        'omega': sol.y[1]
    })
    results.append(df)
    print(f"Run {run_index} succeeded: δ₀={d0:.3f}, ω₀={w0:.3f}")

# === 6. (Optional) Plot δ(t) for the first three runs ===
plt.figure(figsize=(8, 5))
for i in range(min(10, len(results))):
    plt.plot(results[i]['t'], results[i]['delta'], label=f"Run {i+1}")
plt.xlabel("Time (s)")
plt.ylabel("δ (rad)")
plt.title("δ(t) trajectories (first 10 runs)")
plt.legend()
plt.grid(True)
plt.show()
plt.close()


# === 7. Concatenate all runs and save as CSV on Desktop ===
all_results_df = pd.concat(results, ignore_index=True)

# Construct a Desktop path that works on macOS/Linux/Windows:
desktop = Path.home() / "Desktop"
csv_path = desktop / "pll_results.csv"

all_results_df.to_csv(csv_path, index=False)
print(f"Simulation data has been saved to: {csv_path}")