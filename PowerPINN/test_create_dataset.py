import pickle, random
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from sympy.categories.baseclasses import Class
from src.ode.gfl_models_d import calculate_frequency


#Haitian, plot the selected samples of the solved trajectories
cfg = OmegaConf.load('src/conf/config.yaml')
if cfg.theme == "GFL":
    cfg_gfl = OmegaConf.load('./src/conf/setup_dataset_gfl.yaml')
    print("model name", cfg_gfl.model.model_flag)
    print("🐍  YAML for create :", "setup_dataset_gfl.yaml")  # 生成时用的
    print("📊  YAML for test   :", "setup_dataset_gfl.yaml")  # 让两边一致

    with open("./data/" + str(cfg_gfl.model.model_flag) + "/dataset_v" + str(cfg_gfl.model.model_num) + ".pkl", "rb") as f:
        sol = np.asarray(pickle.load(f))        # sol → shape: (num_init_conds, 4, 1000)
        print("sol shape:", sol.shape)

        #  randomly choose k trajectories for a test plotting
        indices = random.sample(range(sol.shape[0]), 25)
        print("How many indices:", len(indices))
        print("Chosen indices:", indices)
        if cfg_gfl.model.model_flag == "GFL_2nd_order":
            var_names = ["delta", "omega", "f"]

            fig, axes = plt.subplots(2, 1, figsize=(9, 18), sharex=True)

            for idx in indices:
                traj = sol[idx]
                t = traj[0]  # time column
                init_states = traj[1:, 0]  # get delta, omega
                print(f"🔹 Initial condition at t=0 for traj {idx}: {tuple(init_states.tolist())}")
                print(f"⏱️ Traj {idx} time range: {t[0]:.2f} → {t[-1]:.2f}")

                for row, var_idx in enumerate([1]):
                    axes[row].plot(t, traj[var_idx])
                    axes[row].set_ylabel(var_names[row])
                    axes[row].grid(True, ls='--', alpha=.3)

                delta_omega = traj[2]
                w_g = np.pi * 100
                f = calculate_frequency(delta_omega, w_g)
                axes[1].plot(t, f, label=f'traj {idx}')
                axes[1].set_ylabel("Frequency")
                axes[1].grid(True, ls='--', alpha=.3)

            axes[-1].set_xlabel('time (s)')
            axes[1].legend(loc='upper right')
            plt.tight_layout()
            plt.show()
        if cfg_gfl.model.model_flag == "GFL_4th_order":
            var_names = ["delta", "delta_omega", "delta_Id", "delta_Id_dt","f"]

            fig, axes = plt.subplots(5, 1, figsize=(9, 18), sharex=True)

            for idx in indices:
                traj = sol[idx]
                t = traj[0]  # time column
                init_states = traj[1:, 0]  # get delta, omega, Id, Id_dt at t=0
                print(f"🔹 Initial condition at t=0 for traj {idx}: {tuple(init_states.tolist())}")
                print(f"⏱️ Traj {idx} time range: {t[0]:.2f} → {t[-1]:.2f}")

                for row, var_idx in enumerate([1, 4]):
                    axes[row].plot(t, traj[var_idx], label=f'traj {idx}')
                    axes[row].set_ylabel(var_names[row])
                    axes[row].grid(True, ls='--', alpha=.3)

                delta_omega = traj[1]
                w_g = np.pi * 100
                f = calculate_frequency(delta_omega, w_g)
                axes[4].plot(t, f, label=f'traj {idx}')
                axes[4].set_ylabel("Frequency")
                axes[4].grid(True, ls='--', alpha=.3)

            axes[-1].set_xlabel('time (s)')
            axes[1].legend(loc='upper right')
            plt.tight_layout()
            plt.show()
elif cfg.theme == "SM":
    # Add your specifications here...
    raise NotImplementedError

elif cfg.theme == "GFM":
    # Add your specifications here...
    raise NotImplementedError

else:
    raise NotImplementedError

#check the sampled the correctness of generating initial condition


