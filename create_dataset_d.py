# Toolbox/create_dataset_d.py
from src.functions import *
import torch
import wandb
import hydra
from src.dataset.create_dataset_functions import ODE_modelling
from src.ode.GFM_model import GFM
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(config_path="src/conf", config_name="setup_dataset.yaml", version_base=None)
def main(config):

    wandb.login(key=config.wandb.api_key)
    run = wandb.init(project=config.wandb.project)
    log_data_metrics_to_wandb(run, config)
    print("Is cuda available?", torch.cuda.is_available())

    all_solutions = []

    flag_for_time = True

    # Generate 2x set of trajectories
    for _ in range(1):
        modelling_full = GFM(config)  # nuovo ratio p/q ogni volta
        gfm_model = ODE_modelling(config)  # nuova istanza (evita riferimenti interni statici)
        init_conditions = gfm_model.create_init_conditions_set3()
        solution = gfm_model.solve_GFM_model(init_conditions, modelling_full, flag_for_time)
        all_solutions.extend(solution)

    # Salva il dataset completo
    gfm_model.save_dataset(all_solutions)

    return None

if __name__ == "__main__":
    main()

