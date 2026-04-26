"""
Test de convergence pour décider du budget d'entraînement final.

Lance DQN et PPO sur les 3 cas d'étude avec des budgets volontairement larges
et 2 graines (suffisant pour identifier visuellement un plateau).

Sauvegarde les courbes dans logs_convergence_test/ pour ne pas écraser les
résultats de l'analyse préliminaire.

Usage (depuis la racine du projet) :
    python experiments/convergence_test.py
"""
import sys
import os
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import DQN, PPO
from envs.frozenlake_envs import make_env
from callbacks.safety_eval_callback import SafetyEvalCallback


# Configurations avec budgets et fréquences d'évaluation adaptés.
# eval_freq choisi pour donner ~50 points d'évaluation par run (lisible mais
# pas trop coûteux). n_eval_episodes réduit à 50 pour accélérer le test.
configs = [
    {
        "name": "4x4_deterministic",
        "size": 4,
        "slippery": False,
        "total_timesteps": 30_000,
        "eval_freq": 500,
    },
    {
        "name": "4x4_stochastic",
        "size": 4,
        "slippery": True,
        "total_timesteps": 100_000,
        "eval_freq": 2_000,
    },
    {
        "name": "8x8_stochastic",
        "size": 8,
        "slippery": True,
        "total_timesteps": 500_000,
        "eval_freq": 10_000,
    },
]

seeds = [0, 1]  # 2 graines suffisent pour repérer un plateau
n_eval_episodes = 50

algos = [
    {
        "name": "DQN",
        "klass": DQN,
        "kwargs": dict(
            learning_rate=1e-3,
            buffer_size=10_000,
            learning_starts=100,
            batch_size=32,
            gamma=0.99,
        ),
    },
    {
        "name": "PPO",
        "klass": PPO,
        "kwargs": dict(
            learning_rate=3e-4,
            n_steps=64,
            batch_size=64,
            gamma=0.99,
        ),
    },
]


def main():
    os.makedirs("logs_convergence_test", exist_ok=True)

    for config in configs:
        for algo in algos:
            for seed in seeds:
                print(
                    f"\n=== {algo['name']} | {config['name']} | "
                    f"seed={seed} | {config['total_timesteps']:,} pas ==="
                )

                random.seed(seed)
                np.random.seed(seed)

                env = make_env(size=config["size"], slippery=config["slippery"])
                env.reset(seed=seed)
                env.action_space.seed(seed)

                eval_env = make_env(
                    size=config["size"], slippery=config["slippery"]
                )
                eval_env.reset(seed=seed)
                eval_env.action_space.seed(seed)

                log_dir = (
                    f"./logs_convergence_test/{algo['name'].lower()}/"
                    f"{config['name']}/seed_{seed}/"
                )
                os.makedirs(log_dir, exist_ok=True)

                eval_callback = SafetyEvalCallback(
                    eval_env=eval_env,
                    eval_freq=config["eval_freq"],
                    n_eval_episodes=n_eval_episodes,
                    log_path=log_dir,
                    deterministic=True,
                    verbose=0,
                )

                model = algo["klass"](
                    "MlpPolicy",
                    env,
                    verbose=0,
                    seed=seed,
                    **algo["kwargs"],
                )

                model.learn(
                    total_timesteps=config["total_timesteps"],
                    callback=eval_callback,
                )

                env.close()
                eval_env.close()

    print("\nTest de convergence terminé.")
    print("Lance ensuite : python experiments/plot_convergence.py")


if __name__ == "__main__":
    main()
