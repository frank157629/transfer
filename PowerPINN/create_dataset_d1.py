"PowerPINN/create_data_d1.py"
"""
Solve GFL model trajectories from LHS-sampled initial conditions.

Steps:
1. Load δ₀–ω₀ samples from lhs_init_conditions.pkl (LHS over region).
2. Use GridFollowingConverterModels to solve ODEs for each IC.
3. Save full time-domain trajectories as dataset_vXX.pkl under PowerPINN/data/GFL_2nd_order.
4. Use convergence.py later to filter valid (converging) trajectories.

"""

from src.functions import *
import torch
import wandb
import hydra
from src.dataset.create_dataset_functions import ODE_modelling
from src.ode.gfl_models_d import GridFollowingConverterModels
from omegaconf import OmegaConf
import pickle
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["HYDRA_FULL_ERROR"]="1"

# Use hydra to configure the dataset creation along with the setup_dataset_sm.yaml file
#@hydra.main(config_path="src/conf", config_name="setup_dataset_sm.yaml",version_base=None)
@hydra.main(config_path="src/conf", config_name="config", version_base=None)
def main(config):

    theme = config.theme

    if theme == "GFL":
        cfg = OmegaConf.load("src/conf/setup_dataset_gfl.yaml")
        cfg.theme = "GFL"
        run = wandb.init(project=cfg.wandb.project)
        log_data_metrics_to_wandb(run, cfg)
        print("Is cuda available?", torch.cuda.is_available())
        GFL_model = ODE_modelling(cfg)
        with open("lhs_init_conditions.pkl", "rb") as f:
            lhs_dataset = pickle.load(f)
        init_conditions = np.array(lhs_dataset)[:,:2]  # Extract [δ₀, ω₀] samples
        modelling_full = GridFollowingConverterModels(cfg) # Instantiate GFL model
        flag_for_time = True  # Output full time-domain trajectory
        solution = GFL_model.solve_model(init_conditions, modelling_full,flag_for_time)  # Solve ODEs for each IC
        GFL_model.save_dataset(solution)  # Save dataset to disk
    else:
        raise NotImplementedError

    return None

if __name__ == "__main__":
    main()

