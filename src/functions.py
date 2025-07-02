import numpy as np
import matplotlib.pyplot as plt
import inspect
import torch


# Set time
def set_time(end_time, interval_points):
    t_span = (0, end_time)
    t_eval = np.linspace(0, end_time, interval_points)
    return t_span, t_eval


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # set in hydra
    np.random.seed(random_seed)
    return


def plotting_solution_custom(sol, model, show=True):
    # sol expected to be a dict with 't' and 'y' or list of such dicts
    if isinstance(sol, dict) and 'message' in sol:
        solutions = [sol]
    else:
        solutions = sol if isinstance(sol, list) else [sol]

    var_names = ["xi_d", "xi_q", "vfd", "vfq", "ifd", "ifq", "itd", "itq",
                 "sigma_d", "sigma_q", "gamma_q", "theta_gfm", "theta_grid"]

    n_vars = len(var_names)
    fig, axs = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars), sharex=True)

    for solution in solutions:
        for i in range(n_vars):
            axs[i].plot(solution.t, solution.y[i], label=f'{var_names[i]}')

    for i in range(n_vars):
        axs[i].set_ylabel(var_names[i])
        axs[i].grid(True)
        axs[i].legend(loc='upper right')

    axs[-1].set_xlabel('Time (s)')
    axs[0].set_title(f"Machine number {model}")

    if show:
        plt.tight_layout()
        plt.show()


# Find missing parameters
def find_missing_params(func, parameters_):
    for i in inspect.getfullargspec(func)[0]:
        if i not in parameters_ and i not in ["x", "y", "t", "theta", "omega", "E_d_dash", "E_q_dash"]:
            print(i)


def checkflag(not_ib_flag, avr_flag, gov_flag):
    # check if more than two flags are true
    if not_ib_flag and avr_flag:
        print("Please select only one model")
        return False
    if not_ib_flag and gov_flag:
        print("Please select only one model")
        return False
    if avr_flag and gov_flag:
        print("Please select only one model")
        return False
    return True


def log_data_metrics_to_wandb(run, config):
    run.log({"Total time of simulation": config.time})
    run.log({"Number of samples": config.num_of_points})
    run.log({"Test with model": config.model.model_flag})
    run.log({"Machine number": config.model.machine_num})
    run.log({"Initial conditions": config.model.init_condition_bounds})
    run.log({"Sampling method for initial points": config.model.sampling})
    return


def log_pinn_metrics_to_wandb(run, config):
    run.log({"Number of hidden layers": config.nn.hidden_layers})
    run.log({"Number of hidden dimensions": config.nn.hidden_dim})
    run.log({"Loss criterion": config.nn.loss_criterion})
    run.log({"Optimizer": config.nn.optimizer})
    run.log({"Weight initialization": config.nn.weight_init})
    run.log({"Learning rate": config.nn.lr})
    run.log({"Learning rate scheduler": config.nn.lr_scheduler})
    run.log({"Number of epochs": config.nn.num_epochs})
    run.log({"Batch size": config.nn.batch_size})
    run.log({"Shuffle": config.dataset.shuffle})
    return


def log_losses_and_weights_to_wandb(run, epoch, loss_data, loss_dt, loss_pinn, loss_total, weight_data, weight_dt,
                                    weight_pinn):
    if run is not None:
        run.log({"Loss_data": loss_data.item(), 'epoch': epoch})
        run.log({"Loss_dt": loss_dt, 'epoch': epoch})
        run.log({"Loss_pinn": loss_pinn, 'epoch': epoch})
        run.log({"Loss_total": loss_total.item(), 'epoch': epoch})
        run.log({"Weight_data": weight_data, 'epoch': epoch})
        run.log({"Weight_dt": weight_dt, 'epoch': epoch})
        run.log({"Weight_pinn": weight_pinn, 'epoch': epoch})
    return


def plotting_solution_gridspec(sol, modelling, model, num_of_points):
    var_names = ["xi_d", "xi_q", "vfd", "vfq", "ifd", "ifq", "itd", "itq",
                 "sigma_d", "sigma_q", "gamma_q", "theta_gfm", "theta_grid"]
    n_vars = len(var_names)
    fig, axs = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars), sharex=True)

    for solution in sol:
        # solution[0] assumed to be time array
        # solution[1:] assumed to be states in order matching var_names
        for i in range(n_vars):
            axs[i].plot(solution[0][:num_of_points], solution[i + 1][:num_of_points], label=var_names[i])

    axs[0].set_title(f"{modelling} for model num : {model}")
    for i in range(n_vars):
        axs[i].set_ylabel(var_names[i])
        axs[i].grid(True)
        axs[i].legend(loc='upper right')

    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    return None


def plotting_solution_gridspec_dt(network, sol, modelling, model, num_of_points):
    var_names = ["xi_d", "xi_q", "vfd", "vfq", "ifd", "ifq", "itd", "itq",
                 "sigma_d", "sigma_q", "gamma_q", "theta_gfm", "theta_grid"]
    n_vars = len(var_names)

    fig, axs = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars), sharex=True)

    for solution in sol:
        solution_tensor = torch.tensor(solution, dtype=torch.float32).T
        time = solution_tensor[:num_of_points, 0].numpy()
        states = solution_tensor[:num_of_points, 1:]
        dt = network.calculate_from_ode(states)
        dt = dt.to('cpu').detach().numpy()

        for i in range(n_vars):
            axs[i].plot(time, dt[:, i], label=f'd{var_names[i]}')

    axs[0].set_title(modelling)
    for i in range(n_vars):
        axs[i].set_ylabel(f'd{var_names[i]}')
        axs[i].grid(True)
        axs[i].legend(loc='upper right')

    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    return None


def plotting_target_gridspec_dt(network, target, time, modelling, model, num_of_points):
    var_names = ["xi_d", "xi_q", "vfd", "vfq", "ifd", "ifq", "itd", "itq",
                 "sigma_d", "sigma_q", "gamma_q", "theta_gfm", "theta_grid"]
    n_vars = len(var_names)

    target_tensor = torch.tensor(target, dtype=torch.float32).T
    target_states = target_tensor[:num_of_points, 1:]
    dt = network.calculate_from_ode(target_states)
    dt = dt.to('cpu').detach().numpy()

    fig, axs = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars), sharex=True)

    for i in range(n_vars):
        axs[i].plot(time, dt[:, i], label=f'd{var_names[i]}')
        axs[i].set_ylabel(f'd{var_names[i]}')
        axs[i].grid(True)
        axs[i].legend(loc='upper right')

    axs[0].set_title(f"{modelling} for model num : {model}")
    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    return None
