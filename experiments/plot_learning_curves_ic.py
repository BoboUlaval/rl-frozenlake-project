import os
import math
import numpy as np
import matplotlib.pyplot as plt

cases = [
    "4x4_deterministic",
    "4x4_stochastic",
    "8x8_stochastic",
]

seeds = [0, 1, 2]

os.makedirs("results/figures_ic", exist_ok=True)

def moving_average(x, window=3):
    return np.convolve(x, np.ones(window) / window, mode="same")

def load_curves(base_dir, case):
    all_means = []
    timesteps_ref = None

    for seed in seeds:
        path = f"{base_dir}/{case}/seed_{seed}/evaluations.npz"
        if not os.path.exists(path):
            print(f"Fichier manquant: {path}")
            return None, None, None

        data = np.load(path)
        timesteps = data["timesteps"]
        results = data["results"]
        mean_rewards = results.mean(axis=1)

        if timesteps_ref is None:
            timesteps_ref = timesteps

        all_means.append(mean_rewards)

    arr = np.array(all_means)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(mean)
    ic95 = 1.96 * std / math.sqrt(len(seeds)) if len(seeds) > 1 else np.zeros_like(mean)

    return timesteps_ref, moving_average(mean), ic95

for case in cases:
    dqn_t, dqn_mean, dqn_ic = load_curves("logs/dqn", case)
    ppo_t, ppo_mean, ppo_ic = load_curves("logs_ppo", case)

    if dqn_t is None or ppo_t is None:
        continue

    plt.figure(figsize=(8, 5))

    plt.plot(dqn_t, dqn_mean, label="DQN")
    plt.fill_between(dqn_t, dqn_mean - dqn_ic, dqn_mean + dqn_ic, alpha=0.2)

    plt.plot(ppo_t, ppo_mean, label="PPO")
    plt.fill_between(ppo_t, ppo_mean - ppo_ic, ppo_mean + ppo_ic, alpha=0.2)

    plt.xlabel("Nombre de pas d'entraînement")
    plt.ylabel("Reward moyen")
    plt.title(case.replace("_", " "))
    plt.suptitle("Évolution de la performance avec IC 95 %", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = f"results/figures_ic/{case}_learning_curve_ic95.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Figure sauvegardée : {output_path}")

print("Graphiques avec IC 95 % terminés.")