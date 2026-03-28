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
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="same")


def load_metric_curves(base_dir, case, metric_name):
    all_values = []
    timesteps_ref = None

    for seed in seeds:
        path = f"{base_dir}/{case}/seed_{seed}/evaluations_safety.npz"
        if not os.path.exists(path):
            print(f"Fichier manquant: {path}")
            return None, None, None

        data = np.load(path)
        timesteps = data["timesteps"]
        values = data[metric_name]

        if timesteps_ref is None:
            timesteps_ref = timesteps

        all_values.append(values)

    arr = np.array(all_values, dtype=float)
    mean = arr.mean(axis=0)

    if len(seeds) > 1:
        std = arr.std(axis=0, ddof=1)
        ic95 = 1.96 * std / math.sqrt(len(seeds))
    else:
        ic95 = np.zeros_like(mean)

    return timesteps_ref, moving_average(mean), ic95


for case in cases:
    dqn_t_reward, dqn_reward_mean, dqn_reward_ic = load_metric_curves(
        "logs/dqn", case, "mean_rewards"
    )
    ppo_t_reward, ppo_reward_mean, ppo_reward_ic = load_metric_curves(
        "logs_ppo", case, "mean_rewards"
    )

    dqn_t_hole, dqn_hole_mean, dqn_hole_ic = load_metric_curves(
        "logs/dqn", case, "hole_rates"
    )
    ppo_t_hole, ppo_hole_mean, ppo_hole_ic = load_metric_curves(
        "logs_ppo", case, "hole_rates"
    )

    if any(x is None for x in [
        dqn_t_reward, dqn_reward_mean, dqn_reward_ic,
        ppo_t_reward, ppo_reward_mean, ppo_reward_ic,
        dqn_t_hole, dqn_hole_mean, dqn_hole_ic,
        ppo_t_hole, ppo_hole_mean, ppo_hole_ic
    ]):
        continue

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)

    axes[0].plot(dqn_t_reward, dqn_reward_mean, label="DQN")
    axes[0].fill_between(
        dqn_t_reward,
        dqn_reward_mean - dqn_reward_ic,
        dqn_reward_mean + dqn_reward_ic,
        alpha=0.2
    )

    axes[0].plot(ppo_t_reward, ppo_reward_mean, label="PPO")
    axes[0].fill_between(
        ppo_t_reward,
        ppo_reward_mean - ppo_reward_ic,
        ppo_reward_mean + ppo_reward_ic,
        alpha=0.2
    )

    axes[0].set_ylabel("Reward moyen")
    axes[0].set_title("Récompense moyenne")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(dqn_t_hole, dqn_hole_mean, label="DQN")
    axes[1].fill_between(
        dqn_t_hole,
        dqn_hole_mean - dqn_hole_ic,
        dqn_hole_mean + dqn_hole_ic,
        alpha=0.2
    )

    axes[1].plot(ppo_t_hole, ppo_hole_mean, label="PPO")
    axes[1].fill_between(
        ppo_t_hole,
        ppo_hole_mean - ppo_hole_ic,
        ppo_hole_mean + ppo_hole_ic,
        alpha=0.2
    )

    axes[1].set_xlabel("Nombre de pas d'entraînement")
    axes[1].set_ylabel("Taux de chute")
    axes[1].set_title("Taux de chute")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True)

    case_title = case.replace("_", " ")
    fig.suptitle(f"Évolution des métriques avec IC 95 %\n{case_title}", fontsize=12)
    plt.tight_layout()

    output_path = f"results/figures_ic/{case}_combined_curve_ic95.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Figure sauvegardée : {output_path}")

print("Graphiques combinés avec IC 95 % terminés.")