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

'''
1. This file is for solving the ICs that are newly generated. 
2. The generated ICs come from lhs_init_conditions.pkl, 
    which are generated with LHS sampling inside a rectangular 
    region of 'delta x omega'.
3. The solution are trajectories, which would be saved under 
    the regular naming convention.(e.g. dataset_v10.pkl)
4. The next step would be loading dataset_v10.pkl into convergence.pkl 
    to filter the non-converging trajectories and use a part of 
    the converging points as dataset.
5. If you are using datasets directly from the whole region for training, 
    change the number under section dataset inside "setup_dataset_pinn_gfl.yaml" 
    or "setup_dataset_vanilla_gfl.yaml as 10. "
'''

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
        init_conditions = np.array(lhs_dataset)[:,:2]  # 只取 [δ₀, ω₀]
        modelling_full = GridFollowingConverterModels(cfg) # Define the GridFollowingConverterModels model to be used
        flag_for_time = True  # we expect solution of each timestep
        solution = GFL_model.solve_model(init_conditions, modelling_full,flag_for_time)  # Solve the model for the various initial conditions
        GFL_model.save_dataset(solution)  # Save the dataset
    else:
        raise NotImplementedError

    return None

if __name__ == "__main__":
    main()

