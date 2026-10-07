import torch
import numpy as np
import pickle
import ray 
import sys
import os
from typing import Dict, Any, Type
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TrainerConfigDict
from ray.rllib.agents.ppo.ppo_torch_policy import ppo_surrogate_loss
from marllib.marl.algos.core import setup_torch_mixins
from marllib.marl.algos.utils.centralized_critic import CentralizedValueMixin

def build_cl_policy(BasePolicyClass: Type[Policy]) -> Type[Policy]:
    class CL_TorchPolicy(BasePolicyClass):
        def __init__(self, observation_space, action_space, config):
            super().__init__(observation_space, action_space, config)
            

            custom_config = config.get("model", {}).get("custom_model_config", {})
            self.is_centralized = "opp_action_in_cc" in custom_config
            if self.is_centralized:
                CentralizedValueMixin.__init__(self)

                
            try: setup_torch_mixins(self, observation_space, action_space, config)
            except Exception: pass 
            
            self.episodic_memory = [] 
            self.ewc_fisher = {}
            self.ewc_theta_star = {}
            self._load_attempted = False
            self.cl_method = config.get("cl_method", "none").lower()

            model_config = config.get("model", {})
            custom_arch = model_config.get("model_arch_args", {}).get("core_arch", "")
            self.is_rnn = "lstm" in str(custom_arch).lower() or "gru" in str(custom_arch).lower() or model_config.get("use_lstm")


        def loss(self, model, dist_class, train_batch):

            if getattr(self, "is_centralized", False) and hasattr(self.model, "central_value_function"):

                vf_saved = model.value_function
                
                custom_config = self.config.get("model", {}).get("custom_model_config", {})
                opp_action = custom_config.get("opp_action_in_cc", False)
                

                model.value_function = lambda: self.model.central_value_function(
                    train_batch["state"],
                    train_batch["opponent_actions"] if opp_action else None
                )
                self._central_value_out = model.value_function()
                

                loss_out = super().loss(model, dist_class, train_batch)
                

                model.value_function = vf_saved 
                
                return loss_out
            else:

                return super().loss(model, dist_class, train_batch)

        def _lazy_load_cl_data(self):
            if self._load_attempted: return 
            if self.cl_method in ["agem", "er", "derpp"]:
                mem_path = self.config.get("episodic_memory_file_to_load")
                if mem_path and os.path.exists(mem_path):
                    with open(mem_path, "rb") as f: self.episodic_memory = pickle.load(f)
            elif self.cl_method == "ewc":
                ewc_path = self.config.get("ewc_file_to_load")
                if ewc_path and os.path.exists(ewc_path):
                    with open(ewc_path, "rb") as f:
                        data = pickle.load(f)
                        self.ewc_fisher = data.get("fisher", {})
                        self.ewc_theta_star = data.get("theta_star", {})
            self._load_attempted = True

        def learn_on_batch(self, samples: SampleBatch) -> Dict[str, Any]:
            self._lazy_load_cl_data()
            
            # MODO FINE-TUNING 
            if self.cl_method == "none":
                grads, info = self.compute_gradients(samples)
                self.apply_gradients(grads)
                return info

            # EXPERIENCE REPLAY (ER) 
            if self.cl_method == "er":
                ref_batch = self._sample_from_memory(samples.count) if self.episodic_memory else None
                train_batch = SampleBatch.concat_samples([samples, ref_batch]) if ref_batch else samples
                grads, info = self.compute_gradients(train_batch)
                self.apply_gradients(grads)
                return info

            # A-GEM (Gradient Projection) 
            if self.cl_method == "agem":
                grads, info = self.compute_gradients(samples)
                grads = [g.clone() if g is not None else None for g in grads]
                
                if self.episodic_memory:
                    ref_batch = self._sample_from_memory(samples.count)
                    if ref_batch:
                        if self.device: ref_batch = ref_batch.to_device(self.device)
                        ref_grads, _ = self.compute_gradients(ref_batch)
                        
                        a_grads, c_grads, a_idx, c_idx = self._separate_actor_critic_grads(grads)
                        ref_a_grads, _, _, _ = self._separate_actor_critic_grads(ref_grads)
                        
                        g_vec = torch.cat([g.flatten() for g in a_grads if g is not None])
                        g_ref_vec = torch.cat([g.flatten() for g in ref_a_grads if g is not None])
                        dot_prod = torch.dot(g_vec, g_ref_vec)
                        
                        if dot_prod < 0:
                            proj_a_grads = self._project_gradients(a_grads, g_vec, g_ref_vec, dot_prod)
                            final_grads = [None] * len(grads)
                            for i, g in zip(a_idx, proj_a_grads): final_grads[i] = g 
                            for i, g in zip(c_idx, c_grads): final_grads[i] = g 
                            self.apply_gradients(final_grads)
                            return info
                self.apply_gradients(grads)
                return info

            # EWC (Elastic Weight Consolidation) 
            if self.cl_method == "ewc":
                grads, info = self.compute_gradients(samples)
                ewc_lambda = self.config.get("ewc_lambda", 1000.0)
                
                if self.ewc_fisher and self.ewc_theta_star:
                    new_grads = []
                    for i, (name, p) in enumerate(self.model.named_parameters()):
                        g = grads[i]
                        if p.requires_grad and g is not None and name in self.ewc_fisher:
                            fisher = self.ewc_fisher[name].to(p.device)
                            theta_star = self.ewc_theta_star[name].to(p.device)
                           
                            penalty = ewc_lambda * fisher * (p - theta_star)
                            new_grads.append(g + penalty)
                        else:
                            new_grads.append(g)
                    self.apply_gradients(new_grads)
                else:
                    self.apply_gradients(grads)
                return info


            if self.cl_method == "derpp":
                ref_batch = self._sample_from_memory(samples.count) if self.episodic_memory else None
                train_batch = SampleBatch.concat_samples([samples, ref_batch]) if ref_batch else samples
                grads, info = self.compute_gradients(train_batch)
                
                
                if ref_batch and "action_dist_inputs" in ref_batch:
                    derpp_alpha = self.config.get("derpp_alpha", 0.5)
                    try:

                        input_dict = {"obs": ref_batch["obs"]}
                        logits, _ = self.model(input_dict, [], None)

                        old_logits = ref_batch["action_dist_inputs"].to(logits.device)
                        distill_loss = derpp_alpha * torch.nn.functional.mse_loss(logits, old_logits)
                        distill_grads = torch.autograd.grad(distill_loss, self.model.parameters(), allow_unused=True)
                        grads = [g + dg if dg is not None else g for g, dg in zip(grads, distill_grads)]
                    except Exception: pass
                self.apply_gradients(grads)
                return info

            return {}

        def _project_gradients(self, current_grads, g_vec, g_ref_vec, dot_prod):
            ref_norm = torch.dot(g_ref_vec, g_ref_vec) + 1e-5
            scaling_factor_raw = dot_prod / ref_norm
            g_proj_vec = g_vec - (scaling_factor_raw * g_ref_vec)
            
            norm_orig = torch.norm(g_vec).item()
            norm_proj_temp = torch.norm(g_proj_vec).item()
            if norm_proj_temp > norm_orig:
                g_proj_vec = g_proj_vec * (norm_orig / (norm_proj_temp + 1e-8))

            final_grads = []
            idx = 0
            for g in current_grads:
                if g is None:
                    final_grads.append(None)
                    continue
                count = g.numel()
                final_grads.append(g_proj_vec[idx : idx + count].view(g.size()))
                idx += count
            return final_grads

        def _separate_actor_critic_grads(self, grads):
            a_grads, c_grads, a_idx, c_idx = [], [], [], []
            trainable_names = [name for name, p in self.model.named_parameters() if p.requires_grad]
            is_policy_gradient = "vf_loss_coeff" in self.config or "entropy_coeff" in self.config
            
            for i, name in enumerate(trainable_names):
                if is_policy_gradient:
                    is_critic = any(key in name.lower() for key in ["value", "vf", "critic"])
                    if not is_critic:
                        a_grads.append(grads[i])
                        a_idx.append(i)
                    else:
                        c_grads.append(grads[i])
                        c_idx.append(i)
                else:
                    a_grads.append(grads[i])
                    a_idx.append(i)
            return a_grads, c_grads, a_idx, c_idx

        def _sample_from_memory(self, batch_size: int) -> SampleBatch:
            if not self.episodic_memory: return None
            if self.is_rnn:
                num_batches = len(self.episodic_memory)
                if num_batches == 0: return None
                indices = np.random.permutation(num_batches)
                selected_batches = []
                current_count = 0
                for idx in indices:
                    batch = self.episodic_memory[idx]
                    if batch.count == 0: continue
                    selected_batches.append(batch)
                    current_count += batch.count
                    if current_count >= batch_size: break
                return SampleBatch.concat_samples(selected_batches) if selected_batches else None
            else:
                if not hasattr(self, "_memory_cache") or self._memory_cache is None:
                    try: self._memory_cache = SampleBatch.concat_samples(self.episodic_memory)
                    except: return None
                count = self._memory_cache.count
                if count == 0: return None
                safe_batch_size = min(batch_size, count)
                indices = torch.randint(0, count, (safe_batch_size,))
                new_data = {}
                for key, value in self._memory_cache.items():
                    if isinstance(value, np.ndarray): new_data[key] = value[indices.cpu().numpy()]
                    elif torch.is_tensor(value): new_data[key] = value[indices.to(value.device) if value.device != indices.device else indices]
                    else: new_data[key] = value[:safe_batch_size]
                return SampleBatch(new_data)
                
    return CL_TorchPolicy

def build_cl_trainer(BaseTrainerClass, CLPolicyClass, name="Universal_CL_Trainer"):
    class CL_Trainer(BaseTrainerClass):
        def setup(self, config):
            super().setup(config) 
            self.cl_method = config.get("cl_method", "none").lower()
            self.episodic_memory = []
            self.memory_limit = config.get("episodic_memory_size", 200)
            

            ckpt_path = config.get("model", {}).get("custom_model_config", {}).get("checkpoint_path_to_load")
            if ckpt_path:
                with open(ckpt_path, "rb") as f: state = pickle.load(f)
                raw_worker_state = state.get("worker")
                
                if raw_worker_state:
                    if isinstance(raw_worker_state, bytes):
                        worker_state = pickle.loads(raw_worker_state)
                    else:
                        worker_state = raw_worker_state


                    print("\n" + "="*55)
                    print("[AUDITORIA E CARGA DE PESOS]")
                    
                    policy_weights = {}
                    if isinstance(worker_state, dict) and "state" in worker_state:
                        for policy_id, policy_state in worker_state["state"].items():
                            print(f" -> Extraindo parâmetros da política: '{policy_id}'")
                            if "_optimizer_variables" in policy_state:
                                print(" -> [STATUS] Inércia do Adam detectada no checkpoint e descartada.")
                            

                            if "weights" in policy_state:
                                policy_weights[policy_id] = policy_state["weights"]
                    
                    if policy_weights:
                        self.workers.local_worker().set_weights(policy_weights)
                        print(" -> [SUCESSO] Pesos carregados no local_worker com otimizador limpo!")
                    else:

                        if "weights" in worker_state:
                            self.workers.local_worker().set_weights(worker_state["weights"])
                            print(" -> [SUCESSO] Pesos carregados da raiz do worker_state!")
                    print("="*55 + "\n")


                    filtros_locais = self.workers.local_worker().get_filters()
                    if hasattr(self.workers, "remote_workers") and self.workers.remote_workers():
                        import ray
                        filtros_remotos = ray.get(self.workers.remote_workers()[0].get_filters.remote())
                        print("\n" + "="*55)
                        print("[AUDITORIA DE FILTROS DE OBSERVAÇÃO]")
                        print(f" -> Filtro Local (Principal): {filtros_locais}")
                        print(f" -> Filtro Remoto (Paralelo 0): {filtros_remotos}")
                        print("="*55 + "\n")
                        
                        for remote in self.workers.remote_workers():
                            remote.sync_filters.remote(filtros_locais)
                    
                    
                    self.workers.sync_weights()




           
            if self.cl_method in ["agem", "er", "derpp"]:
                mem_path = config.get("episodic_memory_file_to_load")
                if mem_path:
                    with open(mem_path, "rb") as f: self.episodic_memory = pickle.load(f)
                    self._sync_data_to_workers("episodic_memory", self.episodic_memory)
            elif self.cl_method == "ewc":
                ewc_path = config.get("ewc_file_to_load")
                if ewc_path:
                    with open(ewc_path, "rb") as f: ewc_data = pickle.load(f)
                    self._sync_data_to_workers("ewc_fisher", ewc_data.get("fisher", {}))
                    self._sync_data_to_workers("ewc_theta_star", ewc_data.get("theta_star", {}))

        @classmethod
        def get_default_config(cls):
            if hasattr(BaseTrainerClass, "get_default_config"):
                config = BaseTrainerClass.get_default_config().copy()
            elif hasattr(BaseTrainerClass, "_default_config"):
                config = BaseTrainerClass._default_config.copy()
            else:
                config = {}
                
            config.update({
                "cl_method": "none", 
                "episodic_memory_size": 200, 
                "episodic_memory_file_to_load": None, 
                "ewc_file_to_load": None,
                "ewc_lambda": 1000.0,
                "derpp_alpha": 0.5
            })
            return config
        
        def _sync_data_to_workers(self, attr_name, data):
            if not hasattr(self, "workers") or self.workers is None or not data: return
            data_ref = ray.put(data)
            def set_data_on_policy(policy, policy_id):
                import ray
                try: 
                    setattr(policy, attr_name, ray.get(data_ref))
                   
                    if attr_name == "episodic_memory" and hasattr(policy, "_memory_cache"):
                        policy._memory_cache = None
                except Exception: pass 
            if self.workers.local_worker(): self.workers.local_worker().foreach_policy(set_data_on_policy)
            if hasattr(self.workers, "remote_workers") and self.workers.remote_workers(): self.workers.foreach_policy(set_data_on_policy)

        def end_of_task(self):
            if self.cl_method == "none": return
            if not hasattr(self, "workers") or self.workers is None: return
            import ray
            
            
            remote_workers = self.workers.remote_workers()
            use_remote = len(remote_workers) > 0
            
            if use_remote:
                target_worker = remote_workers[0]
                target_worker.foreach_env.remote(lambda env: env.reset())
            else:
                target_worker = self.workers.local_worker()
                if not target_worker: return
                target_worker.foreach_env(lambda env: env.reset())

            if self.cl_method in ["agem", "er", "derpp"]:
                print(f"\n    [CL Setup] Coletando memória episódica para o método {self.cl_method.upper()} via Worker Remoto...")
                attempts = 0
                while len(self.episodic_memory) < self.memory_limit and attempts < 50:
                    attempts += 1
                    try:
                     
                        if use_remote:
                            multi_batch = ray.get(target_worker.sample.remote())
                        else:
                            multi_batch = target_worker.sample()
                            
                        if hasattr(multi_batch, "policy_batches"):
                            for pid, batch in multi_batch.policy_batches.items():
                                if batch.count > 0: self.episodic_memory.append(batch.copy())
                        elif hasattr(multi_batch, "count") and multi_batch.count > 0:
                            self.episodic_memory.append(multi_batch.copy())
                    except Exception as e:
                        print(f"    [AVISO CL] Falha na amostragem de memória: {e}")
                        break
                
                while len(self.episodic_memory) > self.memory_limit: self.episodic_memory.pop(0)
                self._sync_data_to_workers("episodic_memory", self.episodic_memory)
                print(f"    [CL Setup] Memória salva com sucesso! (Tamanho: {len(self.episodic_memory)} amostras)\n")
                
            elif self.cl_method == "ewc":
                print(f"\n    [CL Setup] Calculando Matriz de Informação de Fisher (EWC) via Worker Remoto...")
                local_worker = self.workers.local_worker()
                policy = local_worker.get_policy("default_policy")
                fisher = {name: torch.zeros_like(p).cpu() for name, p in policy.model.named_parameters() if p.requires_grad}
                theta_star = {name: p.clone().detach().cpu() for name, p in policy.model.named_parameters() if p.requires_grad}
                
                for _ in range(5): 
                    try:
                        if use_remote:
                            batch = ray.get(target_worker.sample.remote())
                        else:
                            batch = target_worker.sample()
                            
                        target_batch = batch.policy_batches["default_policy"] if hasattr(batch, "policy_batches") else batch
                        grads, _ = policy.compute_gradients(target_batch)
                        idx = 0
                        for name, p in policy.model.named_parameters():
                            if p.requires_grad and grads[idx] is not None:
                                fisher[name] += grads[idx].detach().cpu().pow(2) / 5.0
                            if p.requires_grad: idx += 1
                    except Exception as e:
                        print(f"    [AVISO CL] Falha ao estimar matriz de Fisher: {e}")
                        break
                old_fisher = getattr(self, "ewc_data", {}).get("fisher", {})
                if old_fisher:
                    for name in fisher:
                        if name in old_fisher:
                            fisher[name] = fisher[name] + old_fisher[name].to(fisher[name].device)
                              
                self.ewc_data = {"fisher": fisher, "theta_star": theta_star}
                self._sync_data_to_workers("ewc_fisher", fisher)
                self._sync_data_to_workers("ewc_theta_star", theta_star)
                print(f"    [CL Setup] Matriz de Fisher computada e salva com sucesso!\n")

    CL_Trainer.__name__ = name
    CL_Trainer._name = name
    CL_Trainer._policy_class = CLPolicyClass
    CL_Trainer._default_config = CL_Trainer.get_default_config()
    
   
    def get_default_policy_class(self, config):
        return CLPolicyClass
        
    CL_Trainer.get_default_policy_class = get_default_policy_class

    return CL_Trainer