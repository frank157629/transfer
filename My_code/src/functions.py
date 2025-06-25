import numpy as np

def log_data_metrics_to_wandb(run, config):
    run.log({"Total time of simulation ": config.time})
    run.log({"Number of samples ": config.num_of_points})
    run.log({"Test with model ": config.model.model_flag})
    run.log({"Model number ": config.model.model_num })
    run.log({"Initial conditions ": config.model.init_condition_bounds})
    run.log({"Sampling method for initial points ": config.model.sampling})
    return

# Set time
def set_time(end_time, interval_points):
    t_span = (0, end_time)
    t_eval = np.linspace(0, end_time, interval_points)
    return t_span, t_eval