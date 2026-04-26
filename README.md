# RL FrozenLake — Étude de l'impact de la stochasticité sur l'évitement des états dangereux

Projet IFT-7201, Université Laval. Compare des baselines de RL généraliste
(DQN, PPO) à une approche de RL sûr (shielding, Alshiekh et al. AAAI 2018)
sur l'environnement FrozenLake de Gymnasium.

## Structure

```
rl-frozenlake-project/
├── envs/
│   └── frozenlake_envs.py            Wrappers (Monitor, OneHot)
├── shielding/
│   ├── __init__.py
│   ├── shield.py                     Calcul du shield (value iteration)
│   └── shielded_env.py               Wrappers pre/post-shield
├── callbacks/
│   ├── safety_eval_callback.py       Eval pour DQN/PPO/DQN-Shield
│   └── safety_eval_callback_maskable.py  Eval pour MaskablePPO
├── experiments/
│   ├── train_dqn.py                  Baselines DQN, 30 graines
│   ├── train_ppo.py                  Baselines PPO, 30 graines
│   ├── train_dqn_shielded.py         DQN + post-shield
│   ├── train_ppo_shielded.py         MaskablePPO + pre-shield
│   ├── evaluate_models_multi_seed.py Évaluation finale (4 algos)
│   └── plot_learning_curves_ic.py    Figures avec IC95
├── models/                           Modèles entraînés (.zip)
├── logs/, logs_ppo/                  Logs d'évaluation par seed
└── results/                          CSV et figures
```

## Installation

```bash
pip install stable-baselines3 sb3-contrib gymnasium matplotlib numpy
```

## Reproduction des expériences

L'ordre suivant entraîne tous les modèles, puis les évalue. Chaque script
utilise du multiprocessing (18 workers par défaut, ajustable en éditant
`n_workers`).

```bash
# 1. Baselines (entraînement)
python experiments/train_dqn.py
python experiments/train_ppo.py

# 2. Méthode adaptée : shielding
python experiments/train_dqn_shielded.py
python experiments/train_ppo_shielded.py

# 3. Évaluation finale et figures
python experiments/evaluate_models_multi_seed.py
python experiments/plot_learning_curves_ic.py
```

## Configuration

Les 3 cas d'étude (4×4 déterministe, 4×4 stochastique, 8×8 stochastique)
sont définis dans chaque script `train_*.py`. Les budgets ont été choisis
à partir d'un test de convergence préliminaire :

| Cas               | Budget   | Évaluations |
|-------------------|----------|-------------|
| 4×4 déterministe  | 15 000   | toutes les 600 pas |
| 4×4 stochastique  | 150 000  | toutes les 6 000 pas |
| 8×8 stochastique  | 500 000  | toutes les 20 000 pas |

Le shield est paramétré par :
- `horizon = 3` : nombre d'étapes d'anticipation
- `tolerance = 0.01` : marge autour du minimum pour qualifier une action de sûre

## Algorithmes

- **DQN**, **PPO** : implémentations Stable-Baselines3, hyperparamètres standards.
- **DQN + Shield (post-shield)** : l'agent choisit librement, mais
  `PostShieldWrapper` remplace l'action par une action sûre si elle ne l'est
  pas. Compatible DQN sans modification de la boucle d'apprentissage.
- **PPO + Shield (pre-shield)** : utilise `MaskablePPO` de sb3-contrib avec
  `ActionMasker`. Le masque d'actions sûres est consulté à chaque pas, et
  l'algorithme ne peut pas sampler une action interdite.

## Dépendances clés

- Python 3.10+
- stable-baselines3 ≥ 2.0
- sb3-contrib (pour MaskablePPO)
- gymnasium ≥ 0.28
