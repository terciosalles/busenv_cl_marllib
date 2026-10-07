from ray.tune.utils import merge_dicts
from ray.rllib.models import ModelCatalog
from marllib.marl.algos.utils.setup_utils import AlgVar
from typing import Any, Dict

def setup_mcrl_config(model: Any, exp: Dict, run: Dict, env: Dict) -> Dict:
    model_reg_name = f"{exp['algorithm']}_Custom_Model"
    ModelCatalog.register_custom_model(model_reg_name, model)

    _param = AlgVar(exp)
    train_batch_size = _param.get("batch_episode", 10) * env["episode_limit"]
    if "fixed_batch_timesteps" in exp:
        train_batch_size = exp["fixed_batch_timesteps"]
        
    sgd_minibatch_size = train_batch_size
    episode_limit = env["episode_limit"]
    while sgd_minibatch_size < episode_limit:
        sgd_minibatch_size *= 2

    back_up_config = merge_dicts(exp, env)
    algo_args = back_up_config.pop("algo_args", {})
    if "batch_episode" in algo_args: del algo_args["batch_episode"]

    config = {
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "framework": exp.get("framework", "torch"),
        "model": {
            "custom_model": model_reg_name,
            "custom_model_config": back_up_config,
        },
    }

    config.update(algo_args)
    config.update(run)

    return config