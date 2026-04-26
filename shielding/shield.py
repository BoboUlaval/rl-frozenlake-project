"""
Shield pour FrozenLake : calcule les actions sûres dans chaque état via
value iteration sur la fonction de transition connue

Stratégie : dans chaque état, autoriser les actions dont la probabilité
d'atteindre un trou en H étapes est au
plus min_a p_unsafe + tolerance. Cela priorise toujours les actions les
plus sûres en relâchant la contrainte dans les états où toutes les
actions sont équivalemment risquées.
Référence : Alshiekh et al. "Safe Reinforcement Learning via Shielding",
AAAI 2018.
"""
import numpy as np


class Shield:
    def __init__(self, base_env, horizon=3, tolerance=0.01):
        """
        base_env : environnement Gymnasium FrozenLake non-wrappé
                   (doit exposer .P, .desc, .observation_space.n,
                   .action_space.n)
        horizon  : nombre d'étapes à prévoir (H)
        tolerance: marge autour du minimum pour accepter une action
        """
        self.env = base_env
        self.horizon = horizon
        self.tolerance = tolerance
        self.n_states = base_env.observation_space.n
        self.n_actions = base_env.action_space.n

        self._compute_safe_actions()

    def _identify_terminals(self):
        desc = self.env.desc.flatten()
        self.holes = set()
        self.terminals = set()
        for s in range(self.n_states):
            cell = desc[s].decode("utf-8")
            if cell == "H":
                self.holes.add(s)
                self.terminals.add(s)
            elif cell == "G":
                self.terminals.add(s)

    def _compute_safe_actions(self):
        self._identify_terminals()

        # u[s] = probabilité minimale d'atteindre un trou (en suivant la
        # politique pessimiste), avec u^0[s] = 1 si s est un trou, 0 sinon
        u = np.zeros(self.n_states)
        for h in self.holes:
            u[h] = 1.0

        # Value iteration
        p_unsafe = None
        for _ in range(self.horizon):
            p_unsafe = np.zeros((self.n_states, self.n_actions))
            for s in range(self.n_states):
                if s in self.terminals:
                    continue
                for a in range(self.n_actions):
                    for prob, next_s, _reward, _done in self.env.P[s][a]:
                        p_unsafe[s, a] += prob * u[next_s]

            new_u = u.copy()
            for s in range(self.n_states):
                if s not in self.terminals:
                    new_u[s] = p_unsafe[s].min()
            u = new_u

        # p_unsafe est celui de la dernière itération (avec u^{H-1})
        self.p_unsafe = p_unsafe

        # safe_actions[s] : liste des actions autorisées dans s
        self.safe_actions = {}
        for s in range(self.n_states):
            if s in self.terminals:
                # Actions dans les états terminaux : peu importe
                self.safe_actions[s] = list(range(self.n_actions))
                continue
            min_p = self.p_unsafe[s].min()
            self.safe_actions[s] = [
                a for a in range(self.n_actions)
                if self.p_unsafe[s, a] <= min_p + self.tolerance
            ]

    def get_action_mask(self, state):
        """Masque booléen (n_actions,) : True = action autorisée."""
        mask = np.zeros(self.n_actions, dtype=bool)
        for a in self.safe_actions[state]:
            mask[a] = True
        return mask

    def is_safe(self, state, action):
        return action in self.safe_actions[state]

    def safe_action_set(self, state):
        return self.safe_actions[state]
