"PowerPINN/create_dataset_d.py"

"""
Entry point for generating the GFL dataset.

- Loads configuration via Hydra.
- Initializes Weights & Biases logging.
- Builds initial conditions and solves the Grid-Following Converter model.
- Saves the generated dataset to disk.

"""

from src.functions import *
import torch
import wandb
import hydra
from src.dataset.create_dataset_functions import ODE_modelling
from src.ode.gfl_models_d import GridFollowingConverterModels
from omegaconf import OmegaConf
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["HYDRA_FULL_ERROR"]="1"


# Use hydra to configure the dataset creation along with the setup_dataset_gfl.yaml file
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
        init_conditions = GFL_model.create_init_conditions_set3()  # Define the initial conditions of the system
        modelling_full = GridFollowingConverterModels(cfg) # Instantiate GFL model
        flag_for_time = True  # we expect solution of each timestep
        solution = GFL_model.solve_model(init_conditions, modelling_full,flag_for_time)  # Solve the model for the various initial conditions
        GFL_model.save_dataset(solution)  # Save the dataset
    else:
        raise NotImplementedError

    return None

if __name__ == "__main__":
    main()

