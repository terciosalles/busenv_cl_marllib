import ray
import os
import pickle
import shutil
import gc
import time
import traceback
import torch
import numpy as np 
from ray import tune
from ray.tune import CLIReporter
from marllib.marl.common import recursive_dict_update
from marllib.marl.algos.utils.log_dir_util import available_local_dir
from torch.utils.tensorboard import SummaryWriter
from marllib import marl 
from marllib.marl.algos.scripts.mcrl import setup_mcrl_config 

def calculate_weight_drift(checkpoint_path_old, checkpoint_path_new):
    def load_ckpt(path):
        try:
            with open(path, "rb") as f: return pickle.load(f)
        except: return None
        
    def extract_weights_recursively(obj, depth=0):
        if depth > 5: return {}
        if isinstance(obj, bytes):
            try: obj = pickle.loads(obj)
            except: pass
        if not isinstance(obj, dict): return {}
        keys = list(obj.keys())
        if len(keys) > 0 and isinstance(keys[0], str):
            val = obj[keys[0]]
            if hasattr(val, 'shape') or isinstance(val, np.ndarray) or "torch" in str(type(val)): return obj
        priority_keys = ["worker", "weights", "state", "default_policy", "model", "state_dict"]
        for key in priority_keys:
            if key in obj:
                res = extract_weights_recursively(obj[key], depth+1)
                if res: return res
        for key, val in obj.items():
            if key not in priority_keys and isinstance(val, (dict, bytes)):
                res = extract_weights_recursively(val, depth+1)
                if res: return res
        return {}
        
    state_old, state_new = load_ckpt(checkpoint_path_old), load_ckpt(checkpoint_path_new)
    if not state_old or not state_new: return 0.0
    w_old, w_new = extract_weights_recursively(state_old), extract_weights_recursively(state_new)
    if not w_old or not w_new: return 0.0
    
    drift, common_keys = 0.0, set(w_old.keys()) & set(w_new.keys())
    for key in common_keys:
        w1, w2 = w_old[key], w_new[key]
        if hasattr(w1, "detach"): w1 = w1.detach().cpu().numpy()
        if hasattr(w2, "detach"): w2 = w2.detach().cpu().numpy()
        w1, w2 = np.array(w1), np.array(w2)
        if w1.shape != w2.shape: continue
        try: drift += np.sum((w1 - w2) ** 2)
        except: pass 
    return np.sqrt(drift)


def evaluate_checkpoint(TrainerClass, checkpoint_path, eval_task_params, env_base, exp_base, run_base, model_base, num_episodes=30):
    env_args = env_base.copy()
    env_args.update(eval_task_params)
    
    current_env_name = env_args.pop("environment_name")
    current_map = env_args.pop("map_name")
    force_coop = env_args.pop("force_coop", False)
    episode_limit = env_args.pop("episode_limit", 100)
    env_args.pop("stop", {}) 

    print(f"    [Manual Evaluate] Preparando avaliação para {current_map} (Aguardando liberação de porta TCP...)")
    time.sleep(2)

    mean_reward = 0.0
    env_instance = None
    eval_trainer = None

    try:
        env_instance, env_config_registered = marl.make_env(environment_name=current_env_name, map_name=current_map, force_coop=force_coop, **env_args)
        env_info = env_instance.get_env_info()
        env_info["episode_limit"] = episode_limit

        current_exp = recursive_dict_update(exp_base.copy(), {"env_args": eval_task_params})
        if "algo_args" not in current_exp: current_exp["algo_args"] = {}
        current_exp["algo_args"]["num_workers"] = 0 
        
        run_config = run_base.copy()
        run_config.pop("share_policy", None)
        run_config["multiagent"] = {"policies": {"default_policy"}, "policy_mapping_fn": (lambda agent_id, episode, **kwargs: "default_policy")}

        final_config = setup_mcrl_config(model=model_base, exp=current_exp, run=run_config, env=env_info)
        final_config["env"] = env_config_registered["env"] + "_" + env_config_registered["env_args"]["map_name"]
        final_config["env_config"] = env_config_registered["env_args"]
        final_config["explore"] = False 
        final_config["create_env_on_driver"] = False

        if "custom_model_config" not in final_config["model"]:
            final_config["model"]["custom_model_config"] = {}
        base_lr = final_config.get("lr", 5e-4) 
        if "actor_lr" not in final_config["model"]["custom_model_config"]:
            final_config["model"]["custom_model_config"]["actor_lr"] = base_lr
        if "critic_lr" not in final_config["model"]["custom_model_config"]:
            final_config["model"]["custom_model_config"]["critic_lr"] = base_lr

    
        default_keys = TrainerClass._default_config.keys()
        incompatible_keys = ["sgd_minibatch_size", "num_sgd_iter", "clip_param", "vf_clip_param", "entropy_coeff_schedule"]
        for bad_key in incompatible_keys:
            if bad_key not in default_keys and bad_key in final_config:
                final_config.pop(bad_key)
        

        eval_trainer = TrainerClass(config=final_config)
        eval_trainer.restore(checkpoint_path)
        policy = eval_trainer.get_policy("default_policy")

        print(f"    [Manual Evaluate] Iniciando {num_episodes} episódios...")
        episode_rewards = []
        
        for ep in range(num_episodes):
            obs_dict = env_instance.reset()
            done = False
            ep_reward = 0.0
            rnn_states = {agent_id: policy.get_initial_state() for agent_id in obs_dict.keys()}
            
            while not done:
                action_dict, next_rnn_states = {}, {}
                for agent_id, obs in obs_dict.items():
                    action, state_out, _ = policy.compute_single_action(obs, state=rnn_states.get(agent_id, []), explore=False)
                    action_dict[agent_id] = action
                    next_rnn_states[agent_id] = state_out
                
                next_obs_dict, rewards_dict, dones_dict, _ = env_instance.step(action_dict)
                ep_reward += sum(rewards_dict.values())
                
                if dones_dict.get("__all__", False): done = True
                obs_dict, rnn_states = next_obs_dict, next_rnn_states
                
            episode_rewards.append(ep_reward)
            print(f"      -> Ep {ep+1}/{num_episodes} concluído | Reward: {ep_reward:.2f}")
            
        mean_reward = sum(episode_rewards) / max(1, len(episode_rewards))

    except Exception as e:
        print(f"\n[ERRO CRÍTICO] Falha na simulação manual: {e}")
        traceback.print_exc()
    finally:
        if env_instance:
            try: env_instance.close()
            except: pass
        if eval_trainer:
            try: eval_trainer.stop()
            except: pass
        
        time.sleep(2)
        del eval_trainer, env_instance
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    return float(mean_reward)


def run_cl(TrainerClass, task_definitions: list, model: any, exp: dict, env: dict, run: dict, stop: dict, cl_method="none", experiment_name="Default"):
    if ray.is_initialized(): ray.shutdown()
    ray.init(local_mode=exp.get("local_mode", False), num_gpus=exp.get("num_gpus", 1))

    last_checkpoint_path, last_cl_data_path, start_checkpoint_of_task = None, None, None
    all_results, final_drift_history = [], {}
    num_tasks = len(task_definitions)
    R_matrix = np.zeros((num_tasks, num_tasks))

    base_log_dir = "exp_results/CL_Monitoring"
    current_log_dir = os.path.join(base_log_dir, experiment_name)
    if os.path.exists(current_log_dir): shutil.rmtree(current_log_dir)
    os.makedirs(current_log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=current_log_dir)
    cumulative_timesteps = 0
  
    for i, task_params in enumerate(task_definitions):
        task_name = task_params.get("map_name", f"task_{i}")
        print(f"\n{'='*80}\n=== INICIANDO TAREFA {i+1}/{num_tasks} ({experiment_name}): {task_name} ===\n{'='*80}")
        
        start_checkpoint_of_task = last_checkpoint_path
        env_args = env.copy()
        env_args.update(task_params)
        current_stop = env_args.pop("stop", {}) 
        current_env_name = env_args.pop("environment_name") 
        current_map = env_args.pop("map_name") 
        force_coop = env_args.pop("force_coop", False)
        current_episode_limit = env_args.pop("episode_limit", 100)
        
        env_instance, env_config_registered = marl.make_env(environment_name=current_env_name, map_name=current_map, force_coop=force_coop, **env_args)
        env_info = env_instance.get_env_info()
        env_instance.close() 

        current_exp = recursive_dict_update(exp.copy(), {"env_args": task_params})
        if "algo_args" not in current_exp: current_exp["algo_args"] = {}
        current_exp["algo_args"]["cl_method"] = cl_method

        current_env_dict_for_setup = env_info
        current_env_dict_for_setup["episode_limit"] = current_episode_limit
        
        run_config = run.copy()
        run_config.pop("share_policy", None)
        run_config["multiagent"] = {"policies": {"default_policy"}, "policy_mapping_fn": (lambda agent_id, episode, **kwargs: "default_policy")}
        
        stop_config = recursive_dict_update(stop.copy(), current_stop)
        if not stop_config: stop_config = {"training_iteration": 100}

        final_config = setup_mcrl_config(model=model, exp=current_exp, run=run_config, env=current_env_dict_for_setup)
        final_config["env"] = env_config_registered["env"] + "_" + env_config_registered["env_args"]["map_name"]
        final_config["env_config"] = env_config_registered["env_args"]
        final_config["cl_method"] = cl_method

    
        if "custom_model_config" not in final_config["model"]:
            final_config["model"]["custom_model_config"] = {}
        base_lr = final_config.get("lr", 5e-4)
        if "actor_lr" not in final_config["model"]["custom_model_config"]:
            final_config["model"]["custom_model_config"]["actor_lr"] = base_lr
        if "critic_lr" not in final_config["model"]["custom_model_config"]:
            final_config["model"]["custom_model_config"]["critic_lr"] = base_lr
        

        default_keys = TrainerClass._default_config.keys()
        incompatible_keys = ["sgd_minibatch_size", "num_sgd_iter", "clip_param", "vf_clip_param", "entropy_coeff_schedule"]
        for bad_key in incompatible_keys:
            if bad_key not in default_keys and bad_key in final_config:
                final_config.pop(bad_key)
        

        if last_checkpoint_path:
            if "custom_model_config" not in final_config["model"]: final_config["model"]["custom_model_config"] = {}
            final_config["model"]["custom_model_config"]["checkpoint_path_to_load"] = last_checkpoint_path

        if cl_method in ["agem", "er", "derpp"] and last_cl_data_path:
             final_config["episodic_memory_file_to_load"] = last_cl_data_path
        elif cl_method == "ewc" and last_cl_data_path:
             final_config["ewc_file_to_load"] = last_cl_data_path

        reporter = CLIReporter(metric_columns={"training_iteration": "iter", "episode_reward_mean": "reward"}, max_progress_rows=10, max_report_frequency=30)
        
        results = tune.run(
            TrainerClass, 
            name=f"Task_{i+1}_{experiment_name}", 
            config=final_config, stop=stop_config, checkpoint_at_end=True,
            checkpoint_freq=exp.get("checkpoint_freq", 999), verbose=1,
            progress_reporter=reporter,
            local_dir=available_local_dir if exp["local_dir"] == "" else exp["local_dir"],
        )
        
        all_results.append(results)
        best_trial = results.get_best_trial("training_iteration", "max", "last")
        cumulative_timesteps += best_trial.last_result.get("timesteps_total", 0)
        current_checkpoint = best_trial.checkpoint.value
        last_checkpoint_path = current_checkpoint 

        if start_checkpoint_of_task and current_checkpoint:
             drift_val = calculate_weight_drift(start_checkpoint_of_task, current_checkpoint)
             writer.add_scalar("Stability/Weight_Drift", drift_val, cumulative_timesteps)
             final_drift_history[f"Drift_Task_{i+1}"] = drift_val

        n_eval_eps = exp.get("eval_episodes", 1) 
        for j, eval_task_params in enumerate(task_definitions):
            mean_reward = evaluate_checkpoint(TrainerClass, current_checkpoint, eval_task_params, env, exp, run, model, num_episodes=n_eval_eps)
            R_matrix[i, j] = mean_reward

        AP_i = np.mean(R_matrix[i, :i+1]) 
        BWT_i = np.mean([R_matrix[i, k] - R_matrix[k, k] for k in range(i)]) if i > 0 else 0.0
        FWT_i = np.mean(R_matrix[i, i+1:]) if i < num_tasks - 1 else 0.0

        writer.add_scalar("Eval_Metrics/Average_Performance", AP_i, cumulative_timesteps)
        writer.add_scalar("Eval_Metrics/Backward_Transfer", BWT_i, cumulative_timesteps)
        writer.add_scalar("Eval_Metrics/Forward_Transfer", FWT_i, cumulative_timesteps)
        np.savetxt(os.path.join(current_log_dir, f"R_matrix_after_task_{i+1}.csv"), R_matrix, delimiter=",", fmt="%.2f")

        if cl_method != "none":
            sampling_config = final_config.copy()
            sampling_config["num_workers"] = 1 
            trainer_instance = TrainerClass(config=sampling_config) 
            trainer_instance.restore(current_checkpoint)
            
            trainer_instance.end_of_task() 
            mem_dir = os.path.join(os.path.dirname(current_checkpoint), f"{cl_method}_data")
            os.makedirs(mem_dir, exist_ok=True)
            
            if cl_method in ["agem", "er", "derpp"]:
                last_cl_data_path = os.path.join(mem_dir, "memory.pkl")
                with open(last_cl_data_path, "wb") as f: pickle.dump(trainer_instance.episodic_memory, f)
            elif cl_method == "ewc":
                last_cl_data_path = os.path.join(mem_dir, "ewc_data.pkl")
                with open(last_cl_data_path, "wb") as f: pickle.dump(getattr(trainer_instance, "ewc_data", {}), f)
            
            trainer_instance.stop()
            del trainer_instance
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        writer.flush()
        
    writer.close()
    if ray.is_initialized():
        ray.shutdown()
        
    print("\n" + "="*60)
    print("TREINAMENTO CONTÍNUO CONCLUÍDO COM SUCESSO!")
    print("="*60)
    print("Matriz R Final (Linhas = Treinamento, Colunas = Avaliação):")
    print(R_matrix)
    print("="*60 + "\n")
    
    return all_results, final_drift_history