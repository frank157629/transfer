# Toolbox/create_dataset_d.py
from src.functions import *
import torch
import wandb
import hydra
from src.dataset.create_dataset_functions import ODE_modelling
#from src.ode.sm_models_d import SynchronousMachineModels
from src.ode.GFM_model import GFM
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


os.environ["HYDRA_FULL_ERROR"]="1"

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Use hydra to configure the dataset creation along with the setup_dataset.yaml file
@hydra.main(config_path="src/conf", config_name="setup_dataset.yaml", version_base=None)
def main(config):

    # Initialize wandb and log the dataset creation
    wandb.login(key=config.wandb.api_key)
    run = wandb.init(project=config.wandb.project)
    log_data_metrics_to_wandb(run, config)
    print("Is cuda available?", torch.cuda.is_available())

    gfm_model = ODE_modelling(config)
    init_conditions = gfm_model.create_init_conditions_set3()
    modelling_full = GFM(config)
    flag_for_time = True
    solution = gfm_model.solve_GFM_model(init_conditions, modelling_full, flag_for_time)  # Solve the model for the various initial conditions
    gfm_model.save_dataset(solution)  # Save the dataset
    # plotting_solution(solution[0],flag_for_time)
    # plotting_solution_gridspec_original_all_7th(solution, modelling_full, show=True)
    return None

if __name__ == "__main__":
    main()
