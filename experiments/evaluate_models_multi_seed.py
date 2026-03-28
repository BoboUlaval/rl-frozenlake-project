import sys
import os
import csv
import math
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import DQN, PPO
from envs.frozenlake_envs import make_env

configs = [
    {"name": "4x4_deterministic", "size": 4, "slippery": False},
    {"name": "4x4_stochastic", "size": 4, "slippery": True},
    {"name": "8x8_stochastic", "size": 8, "slippery": True},
]

seeds = [0, 1, 2]

models_info = [
    {"algo": "DQN", "prefix": "dqn", "loader": DQN},
    {"algo": "PPO", "prefix": "ppo", "loader": PPO},
]

os.makedirs("results", exist_ok=True)

def evaluate_one_model(model, env, n_eval_episodes=100):
    rewards = []
    lengths = []
    successes = 0
    holes = 0

    for _ in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        ep_length = 0

        final_state = None

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1

        final_state = env.unwrapped.s
        desc = env.unwrapped.desc.flatten()
        cell = desc[final_state].decode("utf-8")

        rewards.append(ep_reward)
        lengths.append(ep_length)

        if ep_reward > 0:
            successes += 1
        elif cell == "H":
            holes += 1

    return {
        "mean_reward": float(np.mean(rewards)),
        "success_rate": successes / n_eval_episodes,
        "hole_rate": holes / n_eval_episodes,
        "mean_episode_length": float(np.mean(lengths)),
    }

def summarize(values):
    arr = np.array(values, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    ic95 = 1.96 * std / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, ic95

all_seed_rows = []
summary_rows = []

for config in configs:
    for model_info in models_info:
        metric_lists = {
            "mean_reward": [],
            "success_rate": [],
            "hole_rate": [],
            "mean_episode_length": [],
        }

        for seed in seeds:
            model_path = f"models/{model_info['prefix']}_{config['name']}_seed_{seed}.zip"
            env = make_env(size=config["size"], slippery=config["slippery"])
            env.reset(seed=seed)
            env.action_space.seed(seed)

            model = model_info["loader"].load(model_path)
            metrics = evaluate_one_model(model, env, n_eval_episodes=100)
            env.close()

            all_seed_rows.append({
                "case": config["name"],
                "algorithm": model_info["algo"],
                "seed": seed,
                **metrics
            })

            for k in metric_lists:
                metric_lists[k].append(metrics[k])

        reward_mean, reward_ic = summarize(metric_lists["mean_reward"])
        success_mean, success_ic = summarize(metric_lists["success_rate"])
        hole_mean, hole_ic = summarize(metric_lists["hole_rate"])
        length_mean, length_ic = summarize(metric_lists["mean_episode_length"])

        summary_rows.append({
            "case": config["name"],
            "algorithm": model_info["algo"],
            "mean_reward": reward_mean,
            "mean_reward_ic95": reward_ic,
            "success_rate": success_mean,
            "success_rate_ic95": success_ic,
            "hole_rate": hole_mean,
            "hole_rate_ic95": hole_ic,
            "mean_episode_length": length_mean,
            "mean_episode_length_ic95": length_ic,
        })

with open("results/evaluation_results_by_seed.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["case", "algorithm", "seed", "mean_reward", "success_rate", "hole_rate", "mean_episode_length"]
    )
    writer.writeheader()
    writer.writerows(all_seed_rows)

with open("results/evaluation_summary_ic95.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "case", "algorithm",
            "mean_reward", "mean_reward_ic95",
            "success_rate", "success_rate_ic95",
            "hole_rate", "hole_rate_ic95",
            "mean_episode_length", "mean_episode_length_ic95"
        ]
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print("Évaluation multi-seed terminée.")
print("Fichiers générés :")
print("- results/evaluation_results_by_seed.csv")
print("- results/evaluation_summary_ic95.csv")