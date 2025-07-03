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
        indices = random.sample(range(sol.shape[0]), 2)
        print("Chosen indices:", indices)
        if cfg_gfl.model.model_flag == "GFL_4th_order":
            var_names = ["t","delta", "delta_omega", "delta_Id", "delta_Id_dt"]

            fig, axes = plt.subplots(4, 1, figsize=(9, 18), sharex=True)

            for idx in indices:
                traj = sol[idx]  # (4, 1000)
                t = traj[0]  # time column
                init_states = traj[1:, 0]  # get delta, omega, Id, Id_dt at t=0
                print(f"🔹 Initial condition at t=0 for traj {idx}: {tuple(init_states.tolist())}")
                print(f"⏱️ Traj {idx} time range: {t[0]:.2f} → {t[-1]:.2f}")

                for row in range(4):  # 4 states
                    axes[row].plot(t, traj[row+1], label=f'traj {idx}')
                    axes[row].set_ylabel(var_names[row])
                    axes[row].grid(True, ls='--', alpha=.3)
                delta_omega = traj[1]
                w_g = np.pi * 100

                f = calculate_frequency(delta_omega, w_g)
                plt.plot(t, f, label='frequency')

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


