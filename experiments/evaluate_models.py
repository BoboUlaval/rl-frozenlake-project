import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import numpy as np
from stable_baselines3 import DQN, PPO
from envs.frozenlake_envs import make_env

configs = [
    {"name": "4x4_deterministic", "size": 4, "slippery": False},
    {"name": "4x4_stochastic", "size": 4, "slippery": True},
    {"name": "8x8_stochastic", "size": 8, "slippery": True},
]

models_info = [
    {"algo": "DQN", "prefix": "dqn", "loader": DQN},
    {"algo": "PPO", "prefix": "ppo", "loader": PPO},
]

os.makedirs("results", exist_ok=True)

rows = []

for config in configs:
    for model_info in models_info:
        model_path = f"models/{model_info['prefix']}_{config['name']}.zip"
        print(f"Evaluating {model_info['algo']} on {config['name']}")

        env = make_env(size=config["size"], slippery=config["slippery"])
        model = model_info["loader"].load(model_path)

        rewards = []
        lengths = []
        successes = 0

        n_eval_episodes = 100

        for _ in range(n_eval_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            ep_reward = 0
            ep_length = 0

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
                obs, reward, done, truncated, _ = env.step(action)
                ep_reward += reward
                ep_length += 1

            rewards.append(ep_reward)
            lengths.append(ep_length)
            if ep_reward > 0:
                successes += 1

        row = {
            "case": config["name"],
            "algorithm": model_info["algo"],
            "success_rate": successes / n_eval_episodes,
            "mean_reward": float(np.mean(rewards)),
            "mean_episode_length": float(np.mean(lengths)),
        }
        rows.append(row)
        env.close()

with open("results/evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["case", "algorithm", "success_rate", "mean_reward", "mean_episode_length"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("Évaluation terminée. Résultats sauvegardés dans results/evaluation_results.csv")