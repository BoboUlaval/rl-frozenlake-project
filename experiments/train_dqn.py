import sys
import os
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import DQN
from envs.frozenlake_envs import make_env
from callbacks.safety_eval_callback import SafetyEvalCallback

configs = [
    {"name": "4x4_deterministic", "size": 4, "slippery": False},
    {"name": "4x4_stochastic", "size": 4, "slippery": True},
    {"name": "8x8_stochastic", "size": 8, "slippery": True},
]

seeds = [0, 1, 2]

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

for config in configs:
    for seed in seeds:
        print(f"Training DQN on {config['name']} with seed {seed}")

        random.seed(seed)
        np.random.seed(seed)

        env = make_env(size=config["size"], slippery=config["slippery"])
        env.reset(seed=seed)
        env.action_space.seed(seed)

        eval_env = make_env(size=config["size"], slippery=config["slippery"])
        eval_env.reset(seed=seed)
        eval_env.action_space.seed(seed)

        log_dir = f"./logs/dqn/{config['name']}/seed_{seed}/"
        os.makedirs(log_dir, exist_ok=True)

        eval_callback = SafetyEvalCallback(
            eval_env=eval_env,
            eval_freq=500,
            n_eval_episodes=100,
            log_path=log_dir,
            deterministic=True,
            verbose=1
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
            seed=seed
        )

        model.learn(total_timesteps=5000, callback=eval_callback)

        model.save(f"models/dqn_{config['name']}_seed_{seed}")
        env.close()
        eval_env.close()

print("Training DQN terminé")