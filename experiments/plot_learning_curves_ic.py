"""
Génère une figure compacte 3 panneaux côte-à-côte (un par cas d'étude),
chacun avec récompense moyenne et taux de chute (IC 95 %), pour les
4 variantes (DQN, PPO, DQN+Shield, PPO+Shield).

Code couleurs :
  - DQN  : bleu solide
  - PPO  : orange solide
  - DQN+Shield : bleu pointillé
  - PPO+Shield : orange pointillé
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt


cases = [
    ("4x4_deterministic", "4×4 déterministe"),
    ("4x4_stochastic",    "4×4 stochastique"),
    ("8x8_stochastic",    "8×8 stochastique"),
]

seeds = list(range(30))

# (label, dossier de logs, couleur, style)
variants = [
    ("DQN",        "logs/dqn",              "#1f77b4", "-"),
    ("PPO",        "logs_ppo",              "#ff7f0e", "-"),
    ("DQN+Shield", "logs/dqn_shielded",     "#1f77b4", "--"),
    ("PPO+Shield", "logs_ppo/ppo_shielded", "#ff7f0e", "--"),
]

os.makedirs("results/figures_ic", exist_ok=True)


def moving_average(x, window=3):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="same")


def load_metric_curves(base_dir, case, metric_name):
    all_values = []
    all_timesteps = []
    for seed in seeds:
        path = f"{base_dir}/{case}/seed_{seed}/evaluations_safety.npz"
        if not os.path.exists(path):
            return None, None, None
        d = np.load(path)
        all_timesteps.append(d["timesteps"])
        all_values.append(d[metric_name])

    min_len = min(len(v) for v in all_values)
    if min_len <= 1:
        return None, None, None
    trim = min_len - 1
    all_values = [v[:trim] for v in all_values]
    timesteps_ref = all_timesteps[0][:trim]

    arr = np.array(all_values, dtype=float)
    mean = arr.mean(axis=0)
    if len(seeds) > 1:
        std = arr.std(axis=0, ddof=1)
        ic95 = 1.96 * std / math.sqrt(len(seeds))
    else:
        ic95 = np.zeros_like(mean)
    return timesteps_ref, moving_average(mean), ic95


def plot_pair(ax_reward, ax_hole, base_dir, case, label, color, style):
    t, r_mean, r_ic = load_metric_curves(base_dir, case, "mean_rewards")
    _, h_mean, h_ic = load_metric_curves(base_dir, case, "hole_rates")
    if t is None:
        print(f"  Données manquantes : {label} / {case}")
        return
    ax_reward.plot(t, r_mean, label=label, color=color,
                   linestyle=style, linewidth=1.5)
    ax_reward.fill_between(t, r_mean - r_ic, r_mean + r_ic,
                           alpha=0.15, color=color)
    ax_hole.plot(t, h_mean, label=label, color=color,
                 linestyle=style, linewidth=1.5)
    ax_hole.fill_between(t, h_mean - h_ic, h_mean + h_ic,
                         alpha=0.15, color=color)


fig, axes = plt.subplots(2, 3, figsize=(13.5, 5.6), sharey="row")

for col, (case_key, case_title) in enumerate(cases):
    ax_r = axes[0, col]
    ax_h = axes[1, col]
    for label, base_dir, color, style in variants:
        plot_pair(ax_r, ax_h, base_dir, case_key, label, color, style)

    ax_r.set_title(case_title, fontsize=11)
    ax_r.grid(True, alpha=0.3)
    ax_h.grid(True, alpha=0.3)
    ax_h.set_xlabel("Pas d'entraînement", fontsize=9)
    ax_h.set_ylim(-0.02, 1.02)
    for ax in (ax_r, ax_h):
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

axes[0, 0].set_ylabel("Récompense moyenne", fontsize=10)
axes[1, 0].set_ylabel("Taux de chute", fontsize=10)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4,
           bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95])

out_pdf = "results/figures_ic/all_methods_combined.pdf"
out_png = "results/figures_ic/all_methods_combined.png"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, dpi=180, bbox_inches="tight")
plt.close()

print(f"Figures sauvegardées :\n  {out_pdf}\n  {out_png}")
