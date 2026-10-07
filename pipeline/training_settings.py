import os
import argparse
import marllib
from marllib import marl
from marllib.marl.algos.run_cl import run_cl
from marllib.marl.algos.core.CL.mcrl import build_cl_policy, build_cl_trainer
from pipelines.sunt_bus_cl import RLlibSuntBusContinual, build_task_definitions

def setup_marllib_bypasses():
    marl.ENV_REGISTRY["sunt_bus_cl"] = RLlibSuntBusContinual
    marl.COOP_ENV_REGISTRY["sunt_bus_cl"] = RLlibSuntBusContinual

    marl_dir = os.path.dirname(marllib.__file__)
    yaml_path = os.path.join(marl_dir, "envs", "base_env", "config", "sunt_bus_cl.yaml")
    
    yaml_content = """env: sunt_bus_cl
env_args:
  map_name: "sunt_bus_cl"
  num_agents: 5
  episode_limit: 1000
  traffic_mult: 1.0
  demand_mult: 1.0
  occupancy_add: 0.0
  stochastic_traffic: False
mask_flag: False
global_state_flag: False
episode_limit: 1000"""
    try:
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w") as f: f.write(yaml_content)
    except Exception as e: print(f"[Warning] Could not create YAML.: {e}")

def get_base_rllib_classes(algo_name):
    algo_name = algo_name.lower()
    if algo_name in ["mappo", "ppo", "ippo", "happo"]:
        from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
        from ray.rllib.agents.ppo.ppo import PPOTrainer
        return PPOTorchPolicy, PPOTrainer
    elif algo_name in ["maa2c", "ia2c"]:
        from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
        from ray.rllib.agents.a3c.a2c import A2CTrainer
        return A3CTorchPolicy, A2CTrainer
    else:
        raise ValueError(f"Algoritmo {algo_name} não suportado.")

def main():
    parser = argparse.ArgumentParser(description="Universal MACRL in BusEnv")
    parser.add_argument("--exp-name", type=str, default="BusEnv_CL", help="Base experiment name")
    parser.add_argument("--random-tasks", type=int, default=0, help="Qtd de tasks aleatórias")
    
    # SELETORES UNIVERSAIS
    parser.add_argument("--algo", type=str, default="mappo", choices=["mappo", "ippo", "maa2c", "ia2c", "happo"])
    parser.add_argument("--cl-method", type=str, default="derpp", choices=["none", "agem", "ewc", "derpp", "er"])
    args = parser.parse_args()

    setup_marllib_bypasses()

    explicit_tasks = [
        {
            "map_name": "Task_1",
            "traffic_mult": 1.0, 
            "demand_mult": 1.0, 
            "occupancy_add": 0.0, 
            "stochastic_traffic": False
        },
        {
            "map_name": "Task_2",
            "traffic_mult": 1.2,   
            "demand_mult": 1.1,    
            "occupancy_add": 0.5,  
            "stochastic_traffic": False
        },
        {
            "map_name": "Task_3",
            "traffic_mult": 1.6,   
            "demand_mult": 1.5,    
            "occupancy_add": 0.7,  
            "stochastic_traffic": False
        },
        {
            "map_name": "Task_4",
            "traffic_mult": 1.2,   
            "demand_mult": 1.3,    
            "occupancy_add": 0.15,  
            "stochastic_traffic": False
        },
        {
            "map_name": "Task_5",
            "traffic_mult": 1.6,   
            "demand_mult": 1.5,    
            "occupancy_add": 0.7,  
            "stochastic_traffic": False
        },
        {
            "map_name": "Task_6",
            "traffic_mult": 1.45,   
            "demand_mult": 1.4,    
            "occupancy_add": 0.22,  
            "stochastic_traffic": False
        }
    ]
    task_definitions = build_task_definitions(explicit_tasks=explicit_tasks, num_random_tasks=args.random_tasks)


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

    exp_params = {
        "algorithm": args.algo, 
        "framework": "torch",
        "checkpoint_freq": 100,
        "local_dir": "./exp_results",
        "eval_episodes": 10,
        "model_arch_args": {"core_arch": "mlp", "fc_layer": 2, "out_dim_fc_0": 128, "out_dim_fc_1": 128, "hidden_state_size": 128},
        "opp_action_in_cc": True, 
        "global_state_flag": False,
        "mask_flag": False,
        "algo_args": algo_args_dict
    }
    
    env_params = {"environment_name": "sunt_bus_cl", "force_coop": True, "episode_limit": 1000, "stop": {"timesteps_total": 1000000}}
    run_params = {"num_workers": 5, "num_gpus": 0}

    dummy_env_tuple = marl.make_env(environment_name="sunt_bus_cl", map_name="dummy_init", force_coop=True, num_agents=5)
    algo_ctor = getattr(marl.algos, exp_params["algorithm"])
    algo_instance = algo_ctor(hyperparam_source="common")
    
    model_config_dict = {"core_arch": "mlp", "encode_layer": "128-128", "custom_model_config": {"opp_action_in_cc": True, "global_state_flag": False, "mask_flag": False}}
    dummy_model_output = marl.build_model(dummy_env_tuple, algo_instance, model_config_dict)
    
    ModelClass = None
    if isinstance(dummy_model_output, tuple):
        for item in dummy_model_output:
            if isinstance(item, type): ModelClass = item; break
        if ModelClass is None: ModelClass = dummy_model_output[0] 
    elif isinstance(dummy_model_output, type): ModelClass = dummy_model_output
    else: ModelClass = type(dummy_model_output)

    BasePolicy, BaseTrainer = get_base_rllib_classes(args.algo)
    CLPolicy = build_cl_policy(BasePolicy)
    CL_Trainer_Class = build_cl_trainer(BaseTrainer, CLPolicy, f"MCRL_{args.algo.upper()}_{args.cl_method.upper()}")

    run_cl(
        TrainerClass=CL_Trainer_Class,
        task_definitions=task_definitions,
        model=ModelClass,
        exp=exp_params,
        env=env_params,
        run=run_params,
        stop=env_params["stop"],
        cl_method=args.cl_method, 
        experiment_name=f"{args.exp_name}_{args.algo}_{args.cl_method.upper()}"
    )

if __name__ == "__main__":
    main()