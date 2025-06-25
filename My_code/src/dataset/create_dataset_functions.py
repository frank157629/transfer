from src.functions import *
from omegaconf import OmegaConf
import os
import wandb
import pickle
import numpy as np
from pyDOE import lhs
from scipy.integrate import solve_ivp
import time

class ODE_modelling:
    def __init__(self, config):
        self.config = config
        self.modelling_method = config.modelling_method
        self.model_flag = config.model.model_flag
        self.time = config.time
        self.num_of_points = config.num_of_points
        self.init_condition_bounds = config.model.init_condition_bounds
        self.sampling = config.model.sampling
        self.init_conditions_dir = config.dirs.init_conditions_dir
        self.dataset_dir = config.dirs.dataset_dir
        self.torch = config.model.torch
        self.seed = config.model.seed

    def append_element_set(self, value_set, Value_range, num_ranges):
        if self.sampling=="Random":
            points = np.random.uniform(0, 1, num_ranges)
            points = points.reshape(-1, 1)
        elif self.sampling=="Linear":
            points = np.linspace(0, 1, num_ranges)
            points = points.reshape(-1, 1)
        elif self.sampling=="Lhs":
            points = lhs(n=1, samples=num_ranges)
        else:
            raise Exception("Sampling method not implemented")

        new_value_set = []
        iterations = len(value_set) if len(value_set) > 1 else 1
        for j in range(iterations):
            if len(value_set) < 1:
                values = (
                    Value_range[0] + points * (Value_range[1] - Value_range[0]) if num_ranges > 1 else Value_range[0])
                # new_value_set = [values][0].tolist()
                new_value_set = [values][0].tolist() if isinstance(values, np.ndarray) else [[values]]

            else:
                for i in points:
                    if isinstance(i, np.ndarray):
                        i = i.item()
                    value = (
                        Value_range[0] + i * (Value_range[1] - Value_range[0]) if num_ranges > 1 else Value_range[0])
                    new_state = value_set[j].copy()
                    if isinstance(new_state, np.ndarray):
                        new_state = new_state.tolist()
                    new_state.extend([value])
                    new_value_set.append(new_state)
        return new_value_set

    def create_init_cond(self):
        init_conditions_path = os.path.join(self.init_conditions_dir, self.model_flag,"init_cond" + str(self.init_condition_bounds) + ".yaml")
        init_conditions = OmegaConf.load(init_conditions_path)

        modeling_guide_path = os.path.join(self.init_conditions_dir, "modellings_guide.yaml")
        modeling_guide = OmegaConf.load(modeling_guide_path)

        for model in modeling_guide:
            model_name = model.get("name")
            if model_name == self.model_flag:
                keys = model.get("keys")
                self.fullname = model.get("fullname")
        if not keys:
            raise ValueError(f"Model {self.model_flag} not found in modeling guide")

        for i in range(len(init_conditions)):
            name = init_conditions[i].get("name")
            if name not in keys:
                raise ValueError(f"Variable {name} not found in modeling guide")
            if name != keys[i]: #same order as in the modeling guide
                raise ValueError(f"Variable {name} does not match the modeling variable {keys[i]}")
            #check if iterations are always a number:
            iterations = init_conditions[i].get("iterations")
            if not isinstance(iterations, int):
                raise ValueError(f"Variable {name} iterations must be an integer")

        for i in range(len(init_conditions)):
            if len(init_conditions[i]["range"]) == 1: # if unique value then set iterations to 1
                init_conditions[i]["iterations"] = 1

        # Initialize the set_of_values and iterations lists and variables
        set_of_values = []
        iterations = []
        variables = []

        # Extract values, iterations and variables from init_conditions
        for condition in init_conditions:
            set_of_values.append(condition['range'])
            iterations.append(condition['iterations'])
            variables.append(condition['name'])

        # Calculate the number of different initial conditions (All combinations of the initial conditions)
        number_of_conditions = 1
        for it in iterations:
            number_of_conditions *= it

        print("Number of different initial conditions: ", number_of_conditions)
        wandb.log({"Number of different initial conditions: ": number_of_conditions})

        print(variables, "Variables")
        print(set_of_values, "Set of values for init conditions")
        print(iterations, "Iterations per value")

        init_condition_table = []
        for k in range(len(set_of_values)):
            init_condition_table = self.append_element_set(init_condition_table, set_of_values[k], iterations[k])
        return init_condition_table

    def solve_model(self, init_conditions, modelling_full, flag_time=False):
        self.t_span, self.t_eval = set_time(self.time, self.num_of_points)
        solution_all = []
        if flag_time:
            start_time = time.time()
            start_per_iteration = start_time
            time_list = []
        for i in range(len(init_conditions)):
            solution = solve_ivp(modelling_full.odequation_hl, self.t_span, init_conditions[i], t_eval=self.t_eval)
            solution_all.append(solution)
            if flag_time:
                end_per_iteration = time.time()
                time_list.append(end_per_iteration - start_per_iteration)
                start_per_iteration = end_per_iteration

        if flag_time:
            end_time = time.time()
            # print mean and std of time per iteration
            print("Mean time per iteration: ", np.mean(time_list), " and std: ", np.std(time_list))
            print(f"Time taken to solve the model for {len(init_conditions)} initial conditions: {end_time - start_time} seconds.")
        return solution_all

    def save_dataset(self, solution):
        """
        Create and save dataset for the model.

        Args:
            solution (list): The solution of the differential equations of the synchronous machine.

        Returns:
            list: The dataset of the synchronous machine.
        """
        dataset = []
        for i in range(len(solution)):
            r = [solution[i].t]  # append time to directory
            for j in range(len(solution[i].y)):
                r.append(solution[i].y[j])  # append the solution at each time step
            dataset.append(r)

        # check if folder exists if not create it
        if not os.path.exists(os.path.join(self.dataset_dir, self.model_flag)):
            os.makedirs(os.path.join(self.dataset_dir, self.model_flag))

        # count the number of files in the directory
        num_files = len([f for f in os.listdir(os.path.join(self.dataset_dir, self.model_flag)) if os.path.isfile(os.path.join(self.dataset_dir, self.model_flag, f))])
        print("Number of files in the directory: ", num_files)
        print(f'Saved dataset "{self.model_flag, "dataset_v" + str(num_files + 1)}".')
        wandb.log({"Dataset saved": f'Saved dataset "{self.model_flag, "dataset_v" + str(num_files + 1)}".'})
        # save the dataset as pickle in the dataset directory
        dataset_path = os.path.join(self.dataset_dir, self.model_flag, "dataset_v" + str(num_files + 1) + ".pkl")
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)

        return dataset


