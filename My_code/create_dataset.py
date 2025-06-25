import wandb
import hydra
import os
from src.functions import *
import torch
from src.dataset.create_dataset_functions import ODE_modelling
from src.ode.ode import ODE

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


os.environ["HYDRA_FULL_ERROR"]="1"

# Use hydra to configure the dataset creation along with the setup_dataset.yaml file
@hydra.main(config_path="src/conf",
            config_name="setup_dataset.yaml",
            version_base=None)
def main(config):
    run = wandb.init(project=config.wandb.project)
    log_data_metrics_to_wandb(run, config)

    print("Is cuda available?", torch.cuda.is_available())

    Model = ODE_modelling(config)
    init_cond = Model.create_init_cond()
    modelling_full = ODE(config)
    flag_for_time = True
    solution = Model.solve_model(init_cond, modelling_full, flag_for_time)
    Model.save_dataset(solution)

if __name__ == "__main__":
    main()