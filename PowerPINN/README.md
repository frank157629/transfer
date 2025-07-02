# PowerPINN
Physics-Informed Neural Networks (PINNs) and Vanilla Neural Networks for Grid-Following and Grid-Forming Converter Models and Synchronous Machine Models.

## Overview
This repository provides a framework for generating and training **Physics-Informed Neural Networks (PINNs)** for power system components(**Here: SM models and GFL models and GFM models**). It allows users to define Ordinary Differential Equations (ODEs), generate datasets, and train PINNs or Vanilla-NNs to approximate system dynamics efficiently. 

## Features
- Define and integrate new sets of **ODEs** for different power system components.
- Configure initial conditions and variable ranges.
- Automatically generate datasets using numerical solvers.
- Train and test PINNs using **WandB** for tracking.
- Fully parameterized using YAML configuration files.
- Modular design for easy extension.

## Installation
### Prerequisites
Ensure you have Python installed (>=3.8). Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### 1. Define ODEs
ODEs are stored in `src/ode/gfl_models_d.py`(Example for GFL models). You can add any new ODE model in this directory (src/ode/NameOfYourModel_models_d.py). 

### 2. Configure Variables
The independent variables should be defined in `modellings_guide_gfl.yaml` (Example for GFL models), ensuring they are in the same order as in the ODEs.

### 3. Set Initial Conditions
Initial condition values and ranges should be specified in respective YAML files, located in the `src/conf/initial_conditions/` folder under the corresponding ODE name and init_cond number(e.g., `GFL_4th_order/init_cond1.yaml`).

### 4. Define Machine Parameters
Different machine parameters can be configured in the `src/conf/params/` folder.

### 5. Generate Dataset
To generate the dataset for PINN training, use:
```bash
python create_dataset_d.py
```
**Configuration file:** `setup_dataset.yaml`
- `time`: Total simulation time.
- `num_of_points`: Number of data points per trajectory.
- `modelling_method`: Defines how state variables evolve.
- `model`: Specifies which ODE model to use (e.g., `SM_AVR_GOV`).
- `sampling`: Type of sampling for initial conditions (`Lhs`, `Linear`, `Random`).
- `dirs`: Paths for storing parameters, initial conditions, and dataset.
- `PowerPINN/test_create_dataset.py`: For a quick check of your generated dataset as a plot 
### 6. Train & Test a PINN
Train the PINN model with:
```bash
python test_sweep.py
```

**Configuration file:** `setup_dataset_nn_gfl.yaml`(Example for training GFL models)
- `time`, `num_of_points`, `modelling_method`: Same as dataset setup.
- `seed`: Ensures reproducibility.
- `dataset`: Defines data usage (`shuffle`, `split_ratio`, `transform_input/output`).
- `nn` (Neural Network Configuration):
  - `type`: Network architecture (`DynamicNN`, `StaticNN`, etc.).
  - `hidden_layers`, `hidden_dim`: Network depth and width.
  - `optimizer`: (`Adam`, `LBFGS`, etc.).
  - `weighting`: Adjusts loss function balance.
- `dirs`: Paths for dataset, model, and training parameters.

## Configuration Files(Example for training GFL models)
| File                        | Purpose |
|-----------------------------|--------------------------------------------------|
| `setup_dataset_gfl.yaml`    | Defines parameters for dataset generation |
| `setup_dataset_nn_gfl.yaml` | Defines parameters for neural network training |
| `modellings_guide_gfl.yaml` | Lists different ODE models and their variables |
| `initial_conditions/`       | Folder containing initial conditions for each ODE |
| `params/`                   | Folder containing different synchronous machine parameters |

### 7. Extension of Your Own Model
- Follow steps 1 to 6 to define your own model
- Global search in the PowerPINN folder, adding your own model specifications using following command to navigate:(For windows)

```bash
- Get-ChildItem -Recurse -Include *.py | Select-String -Pattern "#Add your specifications here..."
```
- For mac
```bash
grep -Rn --color --include="*.py" "#Add your specifications here..." .
```
- Remove the `raise NotImplementedError`

### 8.To Find Changes of Haitian Based on Toolbox's Original Implementation
- To see functions that had been modified or implemented for the GFL, please give the following command within the PowerPINN folder:(for windows)
```bash
Get-ChildItem -Recurse -Include *.py | Select-String -Pattern "#Haitian"
```
- for mac
```bash
grep -Rn --color --include="*.py" "#Haitian" .
```


## Citation
If you use this repository in your research, please cite the following paper:

**Ioannis Karampinis, Petros Ellinas, Ignasi Ventura Nadal, Rahul Nellikkath, Spyros Chatzivasileiadis**, *Toolbox for Developing Physics-Informed Neural Networks for Power System Components*, DTU.

## License
This project is licensed under the MIT License.

