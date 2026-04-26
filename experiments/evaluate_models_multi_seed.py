"""
Évaluation finale (post-entraînement) des 4 variantes d'algorithmes sur
les 3 cas d'étude, avec 30 graines.

Variantes :
  - DQN baseline
  - PPO baseline
  - DQN + post-shield
  - PPO + pre-shield (MaskablePPO)

Génère deux CSV :
  - evaluation_results_by_seed.csv (résultats par graine)
  - evaluation_summary_ic95.csv (moyennes ± IC95 par algo et cas)
"""
import sys
import os
import csv
import math
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import DQN, PPO
from sb3_contrib import MaskablePPO
from envs.frozenlake_envs import make_env
from shielding.shielded_env import make_shielded_env_post, make_shielded_env_pre


configs = [
    {"name": "4x4_deterministic", "size": 4, "slippery": False},
    {"name": "4x4_stochastic",    "size": 4, "slippery": True},
    {"name": "8x8_stochastic",    "size": 8, "slippery": True},
]

seeds = list(range(30))
n_eval_episodes = 100

SHIELD_HORIZON = 3
SHIELD_TOLERANCE = 0.01

variants = [
    ("DQN",         "dqn",          DQN,          "baseline",   False),
    ("PPO",         "ppo",          PPO,          "baseline",   False),
    ("DQN+Shield",  "dqn_shielded", DQN,          "post_shield", False),
    ("PPO+Shield",  "ppo_shielded", MaskablePPO,  "pre_shield",  True),
]


def make_eval_env(config, env_kind):
    if env_kind == "baseline":
        return make_env(size=config["size"], slippery=config["slippery"])
    if env_kind == "post_shield":
        return make_shielded_env_post(
            size=config["size"], slippery=config["slippery"],
            horizon=SHIELD_HORIZON, tolerance=SHIELD_TOLERANCE,
        )
    if env_kind == "pre_shield":
        return make_shielded_env_pre(
            size=config["size"], slippery=config["slippery"],
            horizon=SHIELD_HORIZON, tolerance=SHIELD_TOLERANCE,
        )
    raise ValueError(env_kind)


def evaluate_one_model(model, env, requires_mask, n_eval=100):
    if requires_mask:
        from sb3_contrib.common.maskable.utils import get_action_masks
    rewards, lengths = [], []
    successes = holes = 0

    for _ in range(n_eval):
        obs, _ = env.reset()
        done = truncated = False
        ep_reward = 0.0
        ep_length = 0
        while not (done or truncated):
            if requires_mask:
                masks = get_action_masks(env)
                action, _ = model.predict(
                    obs, deterministic=True, action_masks=masks
                )
            else:
                action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            ep_length += 1

        final_state = env.unwrapped.s
        cell = env.unwrapped.desc.flatten()[final_state].decode("utf-8")

        rewards.append(ep_reward)
        lengths.append(ep_length)
        if ep_reward > 0:
            successes += 1
        elif cell == "H":
            holes += 1

    return {
        "mean_reward": float(np.mean(rewards)),
        "success_rate": successes / n_eval,
        "hole_rate": holes / n_eval,
        "mean_episode_length": float(np.mean(lengths)),
    }


def summarize(values):
    arr = np.array(values, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    ic95 = 1.96 * std / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, ic95


def main():
    os.makedirs("results", exist_ok=True)

    all_seed_rows = []
    summary_rows = []

    for config in configs:
        for label, prefix, klass, env_kind, req_mask in variants:
            metric_lists = {
                "mean_reward": [], "success_rate": [],
                "hole_rate": [], "mean_episode_length": [],
            }

            for seed in seeds:
                model_path = f"models/{prefix}_{config['name']}_seed_{seed}.zip"
                if not os.path.exists(model_path):
                    print(f"  Manquant : {model_path}")
                    continue

                env = make_eval_env(config, env_kind)
                env.reset(seed=seed)
                env.action_space.seed(seed)

                model = klass.load(model_path)
                metrics = evaluate_one_model(
                    model, env, req_mask, n_eval=n_eval_episodes
                )
                env.close()

                all_seed_rows.append({
                    "case": config["name"],
                    "algorithm": label,
                    "seed": seed,
                    **metrics,
                })

                for k in metric_lists:
                    metric_lists[k].append(metrics[k])

            r_mean, r_ic = summarize(metric_lists["mean_reward"])
            s_mean, s_ic = summarize(metric_lists["success_rate"])
            h_mean, h_ic = summarize(metric_lists["hole_rate"])
            l_mean, l_ic = summarize(metric_lists["mean_episode_length"])

            summary_rows.append({
                "case": config["name"],
                "algorithm": label,
                "mean_reward": r_mean,
                "mean_reward_ic95": r_ic,
                "success_rate": s_mean,
                "success_rate_ic95": s_ic,
                "hole_rate": h_mean,
                "hole_rate_ic95": h_ic,
                "mean_episode_length": l_mean,
                "mean_episode_length_ic95": l_ic,
            })
            print(f"{config['name']:20s} {label:12s} "
                  f"r={r_mean:.3f}±{r_ic:.3f}  h={h_mean:.3f}±{h_ic:.3f}")

    with open(
        "results/evaluation_results_by_seed.csv", "w",
        newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case", "algorithm", "seed",
            "mean_reward", "success_rate", "hole_rate", "mean_episode_length"
        ])
        writer.writeheader()
        writer.writerows(all_seed_rows)

    with open(
        "results/evaluation_summary_ic95.csv", "w",
        newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case", "algorithm",
            "mean_reward", "mean_reward_ic95",
            "success_rate", "success_rate_ic95",
            "hole_rate", "hole_rate_ic95",
            "mean_episode_length", "mean_episode_length_ic95",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nFichiers générés :")
    print("  - results/evaluation_results_by_seed.csv")
    print("  - results/evaluation_summary_ic95.csv")


if __name__ == "__main__":
    main()
