# BusEnv CL MARLlib

This repository contains a plugin developed to integrate the **MARLlib** framework and the **BusEnv** environment, adding advanced support for **Continual Learning (CL)** training applied to multiple tasks.

## 🎯 Objective

The main objective of this project is to transform the standard MARLlib and BusEnv installations, which natively do not support continual training across multiple tasks, into an ecosystem capable of learning and retaining knowledge over time.

In addition to enabling Continual Learning, the plugin deeply modifies the database structure used by BusEnv (the **sunt** base), focused on public transport vehicles in the city of Salvador, Bahia. This modification allows researchers and developers to manipulate essential variables of the simulation environment, such as:

* Passenger **Demand**
* Vehicle **Occupancy**
* Road **Traffic**

## 🧠 Supported Algorithms

The implementation was designed to allow cross-training combining different Reinforcement Learning (RL) algorithms with Continual Learning (CL) memory strategies.

### Training Algorithms (RL)
* **MAPPO** (Multi-Agent Proximal Policy Optimization)
* **IPPO** (Independent Proximal Policy Optimization)
* **MAA2C** (Multi-Agent Advantage Actor-Critic)
* **IA2C** (Independent Advantage Actor-Critic)
* **HAPPO** (Heterogeneous-Agent Proximal Policy Optimization)

### Memory Algorithms (CL)
* **A-GEM** (Averaged Gradient Episodic Memory)
* **EWC** (Elastic Weight Consolidation)
* **DER++** (Dark Experience Replay)
* **ER** (Experience Replay)
* **None/Baseline** (Continual training without memory retention algorithm/Fine-tuning)

## 🚀 Installation

For the plugin to work correctly, it is **mandatory** that both MARLlib and BusEnv are already installed in your environment.

1. **Install Base Dependencies:**
   Follow the official installation instructions in the BusEnv repository:
   [https://github.com/LabIA-UFBA/BusEnv](https://github.com/LabIA-UFBA/BusEnv)

2. **Preparing the Plugin:**
   Download the files from this repository and place this folder **inside** your root MARLlib directory.

3. **Installing the CL Plugin:**
   Navigate to the plugin folder and run one of the commands below according to your preference:

   To install CL mode locally in developer mode:
   ```bash
   python install.py -dev
   ```
   
   To install CL mode in the MARLLib files directly in conda or in the installation directory:
   ```bash
   python install.py
   ```

## ⚙️ Training Configuration (`training_settings.py`)

All training orchestration, algorithm choices, and task definitions happen in the `training_settings.py` file. Below are the main configuration points:

### 1. Main Arguments
You can define via command line or by changing the `default` in the file which algorithms will be used:
```python
parser.add_argument("--algo", type=str, default="ippo", choices=["mappo", "ippo", "maa2c", "ia2c", "happo"])
parser.add_argument("--cl-method", type=str, default="derpp", choices=["none", "agem", "ewc", "derpp", "er"])
```

### 2. Automatic Task Generation
The system allows generating random tasks.
```python
parser.add_argument("--random-tasks", type=int, default=0, help="Number of random tasks")
```
* The `default` value defines the number of tasks that will be automatically created. It ranges from `0` to any integer.
* **⚠️ Warning:** Be careful with memory usage. If you set a very high number of tasks, your computer's RAM/VRAM may not be able to handle it.

### 3. Manual Task Configuration
You can configure multipliers and specific characteristics for each task (e.g., increasing traffic or demand):
```python
{
    "map_name": "Task_2",
    "traffic_mult": 1.2,   
    "demand_mult": 1.1,    
    "occupancy_add": 0.5,  
    "stochastic_traffic": False
}
```

### 4. Algorithm Parameters (RL and CL)
The file also contains hyperparameter dictionaries for the Reinforcement Learning and Continual Learning algorithms:
```python

algo_args_dict = {}
if args.algo in ["mappo", "happo", "ippo"]:
    algo_args_dict = {"batch_episode": 20, "lr": 0.0005, "num_sgd_iter": 10, "use_gae": True, "batch_mode": "truncate_episodes", "clip_param": 0.3, "vf_clip_param": 10.0, "lambda": 0.95, "kl_coeff": 0.1, "vf_loss_coeff": 1.0, "entropy_coeff": 0.05}
elif args.algo in ["maa2c", "ia2c"]:
    algo_args_dict = {"batch_episode": 20, "lr": 0.0005, "use_gae": True, "batch_mode": "truncate_episodes", "entropy_coeff": 0.01, "vf_loss_coeff": 0.5}


cl_args_dict = {
    "cl_method": args.cl_method,
    "episodic_memory_size": 200, 
    "ewc_lambda": 1000.0,        
    "derpp_alpha": 0.5           
}
algo_args_dict.update(cl_args_dict)
```
Finally, the `exp_params`, `env_params`, and `run_params` dictionaries control the neural network architecture (`mlp`, *hidden states*), the environment's episode limits, and hardware usage (*workers* and GPUs), respectively.

## 🤝 Contributions

Contributions are welcome! Feel free to open *Issues* reporting bugs or *Pull Requests* with improvements.