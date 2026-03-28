import os
import numpy as np
import matplotlib.pyplot as plt

cases = [
    "4x4_deterministic",
    "4x4_stochastic",
    "8x8_stochastic",
]

os.makedirs("results/figures", exist_ok=True)

# Lissage
def moving_average(x, window=3):
    return np.convolve(x, np.ones(window) / window, mode='same')

for case in cases:
    dqn_path = f"logs/{case}/evaluations.npz"
    ppo_path = f"logs_ppo/{case}/evaluations.npz"

    if not os.path.exists(dqn_path):
        print(f"Fichier manquant: {dqn_path}")
        continue
    if not os.path.exists(ppo_path):
        print(f"Fichier manquant: {ppo_path}")
        continue

    dqn_data = np.load(dqn_path)
    ppo_data = np.load(ppo_path)

    dqn_timesteps = dqn_data["timesteps"]
    dqn_results = dqn_data["results"]
    dqn_mean = moving_average(dqn_results.mean(axis=1))
    dqn_std = dqn_results.std(axis=1)

    ppo_timesteps = ppo_data["timesteps"]
    ppo_results = ppo_data["results"]
    ppo_mean = moving_average(ppo_results.mean(axis=1))
    ppo_std = ppo_results.std(axis=1)

    plt.figure(figsize=(8, 5))

    # DQN
    plt.plot(dqn_timesteps, dqn_mean, label="DQN")
    plt.fill_between(
        dqn_timesteps,
        dqn_mean - dqn_std,
        dqn_mean + dqn_std,
        alpha=0.2
    )

    # PPO
    plt.plot(ppo_timesteps, ppo_mean, label="PPO")
    plt.fill_between(
        ppo_timesteps,
        ppo_mean - ppo_std,
        ppo_mean + ppo_std,
        alpha=0.2
    )

    # Labels plus clean
    plt.xlabel("Nombre de pas d'entraînement")
    plt.ylabel("Reward moyen")

    # Titre plus clean ? Est ça marche?
    plt.title(case.replace("_", " "))
    plt.suptitle("Évolution de la performance durant l’apprentissage", fontsize=12)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = f"results/figures/{case}_learning_curve.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Figure sauvegardée : {output_path}")

print("Graphiques terminés.")