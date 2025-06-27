import pickle, random
import numpy as np
import matplotlib.pyplot as plt

# ---------- 读取并转成 ndarray ----------
with open("./data/GFL_7th_order/dataset_v1.pkl", "rb") as f:
    sol = np.asarray(pickle.load(f))        # sol → shape: (2187, 8, 1000)

print("sol shape:", sol.shape)

# ---------- 选 3 条轨迹 ----------
indices = random.sample(range(sol.shape[0]), 3)
print("抽到的索引:", indices)

var_names = ['t', 'gamma', 'delta', 'theta_pll',
             'i_gd_g', 'i_gq_g', 'v_od_g', 'v_oq_g']

fig, axes = plt.subplots(8, 1, figsize=(9, 18), sharex=True)

for idx in indices:
    traj = sol[idx]                 # (8, 1000)
    t    = traj[0]                  # 时间行
    for row in range(0, 8):         # 7 个状态量
        axes[row].plot(t, traj[row], label=f'traj {idx}')
        axes[row].set_ylabel(var_names[row])
        axes[row].grid(True, ls='--', alpha=.3)

axes[-1].set_xlabel('time (s)')
axes[1].legend(loc='upper right')
plt.tight_layout()
plt.show()