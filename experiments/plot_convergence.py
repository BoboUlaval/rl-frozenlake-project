"""
Visualisation des courbes de convergence pour identifier le budget
d'entraînement nécessaire à chaque cas d'étude.

Une figure par cas, avec récompense et taux de chute en sous-graphes.
Trace la moyenne sur les graines avec bande min/max (suffisant pour 2 seeds).

Usage (depuis la racine du projet) :
    python experiments/plot_convergence.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt


cases = [
    "4x4_deterministic",
    "4x4_stochastic",
    "8x8_stochastic",
]

algos = ["dqn", "ppo"]
seeds = [0, 1]

base_log_dir = "logs_convergence_test"
output_dir = "results/convergence"
os.makedirs(output_dir, exist_ok=True)


def moving_average(x, window=5):
    """Lissage léger pour rendre le plateau plus lisible."""
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="same")


def load_curves(algo, case, metric):
    """Retourne (timesteps, matrice [n_seeds, n_points])."""
    all_values = []
    timesteps_ref = None
    for seed in seeds:
        path = f"{base_log_dir}/{algo}/{case}/seed_{seed}/evaluations_safety.npz"
        if not os.path.exists(path):
            print(f"  Manquant : {path}")
            return None, None
        data = np.load(path)
        if timesteps_ref is None:
            timesteps_ref = data["timesteps"]
        all_values.append(data[metric])
    return timesteps_ref, np.array(all_values, dtype=float)


for case in cases:
    print(f"\nCas : {case}")
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)

    metrics = [
        ("mean_rewards", "Récompense moyenne", axes[0]),
        ("hole_rates", "Taux de chute", axes[1]),
    ]

    have_data = True
    for metric, ylabel, ax in metrics:
        for algo in algos:
            timesteps, values = load_curves(algo, case, metric)
            if values is None:
                have_data = False
                continue

            mean = moving_average(values.mean(axis=0))
            vmin = values.min(axis=0)
            vmax = values.max(axis=0)

            ax.plot(timesteps, mean, label=algo.upper(), linewidth=1.6)
            ax.fill_between(timesteps, vmin, vmax, alpha=0.2)

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    if not have_data:
        plt.close(fig)
        continue

    axes[0].legend(loc="best")
    axes[1].set_xlabel("Pas d'entraînement")
    axes[1].set_ylim(-0.02, 1.02)

    fig.suptitle(f"Test de convergence — {case.replace('_', ' ')}", fontsize=12)
    plt.tight_layout()

    out_path = f"{output_dir}/{case}_convergence.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure : {out_path}")

print("\nTerminé. Inspecte visuellement où les courbes plafonnent.")
print("Choisis ~1.3× le pas où le plateau commence comme budget final.")
