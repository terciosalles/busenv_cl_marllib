import os
import pickle
import time
import math
import warnings
import random
import numpy as np
from gym.spaces import Dict as GymDict
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from envs.sunt_env import parallel_env
from supersuit import pad_observations_v0, pad_action_space_v0
from contextlib import nullcontext

warnings.filterwarnings("ignore", message="The observation_spaces dictionary is deprecated")
warnings.filterwarnings("ignore", message="The action_spaces dictionary is deprecated")

def build_task_definitions(explicit_tasks=None, num_random_tasks=0, seed=42):
    """
    Recebe as tasks explícitas e gera N tasks aleatórias, retornando a lista final
    para o treinamento do Continual Learning.
    """
    if explicit_tasks is None:
        explicit_tasks = []
        
    task_list = list(explicit_tasks) 
    
    if num_random_tasks > 0:
        rng = random.Random(seed)
        print(f"\n[MONTADOR] Adicionando {num_random_tasks} tarefas aleatórias à fila...")
        
        for i in range(num_random_tasks):

            rand_traffic = round(rng.uniform(0.5, 5.0), 2)
            rand_demand = round(rng.uniform(0.2, 3.0), 2)
            rand_occ = round(rng.uniform(-0.5, 0.5), 2)
            rand_stochastic = rng.choice([True, False])
            
            random_task = {
                "map_name": f"Task_Random_{i+1}",
                "traffic_mult": rand_traffic,
                "demand_mult": rand_demand,
                "occupancy_add": rand_occ,
                "stochastic_traffic": rand_stochastic
            }
            task_list.append(random_task)
            print(f" -> Criada: {random_task['map_name']} | Transito: {rand_traffic}x | Dem: {rand_demand}x | Caos: {rand_stochastic}")
            
    return task_list



class BusScenarioGenerator:
    @staticmethod
    def generate_task(travel_times, demand, occupancy, traffic_mult=1.0, demand_mult=1.0, occupancy_add=0.0, stochastic=False):
        
        safe_traffic = max(0.1, min(traffic_mult, 10.0))
        safe_demand = max(0.0, min(demand_mult, 10.0))
        safe_occ_add = max(-1.0, min(occupancy_add, 1.0))
        
        new_travel = {}
        new_demand = {}
        new_occ = {}
        rng = random.Random(42) 

   
        print_count = 0
        for edge, t in travel_times.items():
            if stochastic:
                lower_bound = min(1.0, safe_traffic)
                upper_bound = max(1.0, safe_traffic)
                mult = rng.uniform(lower_bound, upper_bound)
                new_travel[edge] = max(t * mult, 1.0) 
            else:
                new_travel[edge] = max(t * safe_traffic, 1.0)
                
            if print_count < 3:
                print(f"[DEBUG STOCHASTIC] Rua {edge} | Original: {t:.1f}s --> Novo: {new_travel[edge]:.1f}s")
                print_count += 1

    
        for node, d in demand.items():
            new_demand[node] = d * safe_demand

      
        for node, occ in occupancy.items():
            new_occ[node] = max(0.0, min(occ + safe_occ_add, 1.0))

        return new_travel, new_demand, new_occ


class RLlibSuntBusContinual(MultiAgentEnv):
    def __init__(self, env_config):
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

       
        task_name = env_config.get("map_name", "Task_0_Base")
        traffic_mult = env_config.get("traffic_mult", 1.0)
        demand_mult = env_config.get("demand_mult", 1.0)
        occupancy_add = env_config.get("occupancy_add", 0.0)
        stochastic = env_config.get("stochastic_traffic", False)

        print(f"\n[BENCHMARK CL] Iniciando {task_name}")
        print(f" -> Traffic: {traffic_mult}x | Demand: {demand_mult}x | Occ Add: {occupancy_add} | Caótico: {stochastic}")

       
        graph_path = os.path.join(BASE_DIR, "viz", "graph_gtfs_fev_2024.gpickle")
        with open(graph_path, "rb") as f:
            G = pickle.load(f)

        obs_dir = os.path.join(BASE_DIR, "training_observation")

        def load_pickle(filename):
            path = os.path.join(obs_dir, filename)
            with open(path, "rb") as f:
                return pickle.load(f)

        base_avg_travel_time = load_pickle("avg_travel_time_AB.pkl")
        base_future_demand = load_pickle("future_demand_at_B.pkl")
        base_occupancy = load_pickle("occupancy_rate.pkl")
        uptime_normalized = load_pickle("uptime_normalized.pkl")
        real_routes = load_pickle("real_routes.pkl")
        route_metadata = load_pickle("route_metadata.pkl")

     
        task_travel_time, task_demand, task_occupancy = BusScenarioGenerator.generate_task(
            base_avg_travel_time, 
            base_future_demand, 
            base_occupancy, 
            traffic_mult=traffic_mult,
            demand_mult=demand_mult,
            occupancy_add=occupancy_add,
            stochastic=stochastic
        )

        
        self.env = parallel_env(
            network=G,
            actions_amount=3,
            max_steps=1000,
            num_agents=5,
            avg_travel_time_AB=task_travel_time,
            future_demand_at_B=task_demand,
            occupancy_rate=task_occupancy,
            uptime_normalized=uptime_normalized,
            real_routes=real_routes,
            route_metadata=route_metadata,
        )

        self.env = pad_observations_v0(self.env)
        self.env = pad_action_space_v0(self.env)

        self.agents = self.env.possible_agents.copy()
        self.num_agents = len(self.agents)

        self._base_obs_space = self.env.observation_space(self.agents[0])
        self._obs_shape = tuple(getattr(self._base_obs_space, "shape", ()))
        if len(self._obs_shape) == 0:
            self._obs_shape = (1,)
        self._obs_size = int(np.prod(self._obs_shape))
        self._obs_dtype = np.float32

        self.observation_space = GymDict({"obs": self._base_obs_space})
        self.action_space = self.env.action_space(self.agents[0])
        self.action_spaces = {agent: self.env.action_space(agent) for agent in self.agents}
        self._team_done = True

    def _fix_obs(self, o):
        x = np.asarray(o, dtype=self._obs_dtype)
        if x.shape != self._obs_shape:
            flat = x.ravel()
            if flat.size < self._obs_size:
                pad = np.zeros(self._obs_size - flat.size, dtype=self._obs_dtype)
                flat = np.concatenate([flat, pad], axis=0)
            elif flat.size > self._obs_size:
                flat = flat[:self._obs_size]
            x = flat.reshape(self._obs_shape)
        return x

    def _wrap_obs_dict(self, obs_dict):
        wrapped = {}
        for agent in self.agents:
            raw = obs_dict.get(agent, np.zeros(self._obs_shape, dtype=self._obs_dtype))
            wrapped[agent] = {"obs": self._fix_obs(raw)}
        return wrapped

    def _default_action(self, agent):
        sp = self.action_spaces[agent]
        if hasattr(sp, "nvec"):
            return np.zeros_like(sp.nvec, dtype=np.int64)
        if hasattr(sp, "n"):
            return 0
        return 0

    def reset(self):
        original_obs = self.env.reset()
        self.agents = self.env.possible_agents.copy()
        for a in self.agents:
            original_obs.setdefault(a, np.zeros(self._obs_shape, dtype=self._obs_dtype))
        return self._wrap_obs_dict(original_obs)

    def step(self, action_dict):
        for a in self.agents:
            if a not in action_dict:
                action_dict[a] = self._default_action(a)
        o, r, d, info = self.env.step(action_dict)

        if self._team_done and any(d.get(a, False) for a in self.agents):
            for a in self.agents:
                d[a] = True

        for a in self.agents:
            if a not in o:
                o[a] = np.zeros(self._obs_shape, dtype=self._obs_dtype)
            if a not in r:
                r[a] = 0.0
            if a not in d:
                d[a] = False
            if a not in info:
                info[a] = {}

        obs = self._wrap_obs_dict(o)
        
        
        rewards = {a: round(math.copysign(math.log1p(abs(float(r[a]))), float(r[a])), 2) for a in self.agents}
        
        dones = {"__all__": all(d.get(a, False) for a in self.agents)}
        infos = {a: info[a] for a in self.agents}
        return obs, rewards, dones, infos

    def render(self, mode=None):
        self.env.render()
        time.sleep(0.05)
        return True

    def close(self):
        self.env.close()

    def get_env_info(self):
        return {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 1000,
            "agent_id": self.agents,
            "share_observation_space": self.observation_space,
            "policy_mapping_info": {
                "sunt_bus_cl": { 
                    "all_agents_one_policy": False,
                    "one_agent_one_policy": True,
                    "policy_map": {agent_id: f"policy_{i}" for i, agent_id in enumerate(self.agents)},
                }
            },
        }

ENV_REGISTRY["sunt_bus_cl"] = RLlibSuntBusContinual