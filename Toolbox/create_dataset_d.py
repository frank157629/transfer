# Toolbox/create_dataset_d.py
from src.functions import *
import torch
import wandb
import hydra
import numpy as np
from src.dataset.create_dataset_functions import ODE_modelling
#from src.ode.sm_models_d import SynchronousMachineModels
from src.ode.GFL import GFL
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


os.environ["HYDRA_FULL_ERROR"]="1"


# Use hydra to configure the dataset creation along with the setup_dataset.yaml file
@hydra.main(config_path="src/conf", config_name="setup_dataset.yaml",version_base=None)
def main(config):
    #wandb.login(key=config.wandb.api_key)
    # Initialize wandb and log the dataset creation
    run = wandb.init(project=config.wandb.project)
    log_data_metrics_to_wandb(run, config)
    print("Is cuda available?", torch.cuda.is_available())
    # SM_model=ODE_modelling(config) # Create an instance of the class ODE_modelling that creates the initial conditions and generates the dataset
    # init_conditions=SM_model.create_init_conditions_set3() # Define the initial conditions of the system
    # modelling_full=SynchronousMachineModels(config) # Define the model to be used
    # flag_for_time = True
    #  solution = SM_model.solve_sm_model(init_conditions, modelling_full, flag_for_time)  # Solve the model for the various initial conditions
    # SM_model.save_dataset(solution)  # Save the dataset

    PLL_model=ODE_modelling(config)
    init_conditions=PLL_model.create_init_conditions_set3() # Define the initial conditions of the system
    modelling_full = GFL(config)
    flag_for_time = True # we expect solution of each timestep
    solution = PLL_model.solve_pll_model(init_conditions, modelling_full,flag_for_time)  # Solve the model for the various initial conditions
    PLL_model.save_dataset(solution)  # Save the dataset
    plotting_solution(solution[0],flag_for_time)
    # plotting_solution_gridspec_original_all_7th(solution, modelling_full, show=True)
    return None

if __name__ == "__main__":
    main()



