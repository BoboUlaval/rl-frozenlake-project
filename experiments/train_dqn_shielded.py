"""
Entraînement DQN avec post-shielding sur les 3 cas d'étude.

- Mêmes hyperparamètres et budgets que train_dqn.py (baseline)
- Le shield est appliqué via PostShieldWrapper : si l'agent choisit une
  action dangereuse, elle est remplacée par une action sûre avant que
  l'environnement n'exécute le pas.
- 30 graines, multiprocessing, callback de safety déjà existant
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import random
import numpy as np
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


configs = [
    {
        "name": "4x4_deterministic",
        "size": 4,
        "slippery": False,
        "total_timesteps": 15_000,
    },
    {
        "name": "4x4_stochastic",
        "size": 4,
        "slippery": True,
        "total_timesteps": 150_000,
    },
    {
        "name": "8x8_stochastic",
        "size": 8,
        "slippery": True,
        "total_timesteps": 500_000,
    },
]

seeds = list(range(30))
n_eval_episodes = 30
n_eval_points = 25
n_workers = 18

# Hyperparamètres du shield (mêmes pour les 3 cas)
SHIELD_HORIZON = 3
SHIELD_TOLERANCE = 0.01


def train_one(args):
    config, seed = args

    import torch
    torch.set_num_threads(1)

    from stable_baselines3 import DQN
    from callbacks.safety_eval_callback import SafetyEvalCallback
    from shielding.shielded_env import make_shielded_env_post

    model_path = f"models/dqn_shielded_{config['name']}_seed_{seed}.zip"
    if os.path.exists(model_path):
        return f"[skip] DQN-shield {config['name']} seed={seed}"

    random.seed(seed)
    np.random.seed(seed)

    env = make_shielded_env_post(
        size=config["size"], slippery=config["slippery"],
        horizon=SHIELD_HORIZON, tolerance=SHIELD_TOLERANCE,
    )
    env.reset(seed=seed)
    env.action_space.seed(seed)

    eval_env = make_shielded_env_post(
        size=config["size"], slippery=config["slippery"],
        horizon=SHIELD_HORIZON, tolerance=SHIELD_TOLERANCE,
    )
    eval_env.reset(seed=seed)
    eval_env.action_space.seed(seed)

    log_dir = f"./logs/dqn_shielded/{config['name']}/seed_{seed}/"
    os.makedirs(log_dir, exist_ok=True)

    eval_freq = config["total_timesteps"] // n_eval_points

    eval_callback = SafetyEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        log_path=log_dir,
        deterministic=True,
        verbose=0,
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=100,
        batch_size=32,
        gamma=0.99,
        verbose=0,
        seed=seed,
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=eval_callback,
    )

    model.save(model_path)

    # Sauvegarde des stats du shield (interventions)
    n_int = env.n_interventions
    n_st = env.n_steps
    np.savez(
        os.path.join(log_dir, "shield_stats.npz"),
        n_interventions=n_int,
        n_steps=n_st,
        intervention_rate=(n_int / n_st) if n_st > 0 else 0.0,
    )

    env.close()
    eval_env.close()

    return (
        f"[done] DQN-shield {config['name']} seed={seed} "
        f"interventions={n_int}/{n_st}"
    )


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    tasks = [(config, seed) for config in configs for seed in seeds]
    print(f"Total : {len(tasks)} runs sur {n_workers} workers")

    with Pool(n_workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(train_one, tasks), 1):
            print(f"[{i}/{len(tasks)}] {msg}")

    print("Training DQN-shielded terminé.")


if __name__ == "__main__":
    main()
