"""
Wrappers pour intégrer le Shield aux algorithmes de RL.

- PostShieldWrapper : pour DQN et autres algos sans masquage natif. Si l'agent
  choisit une action dangereuse, le wrapper la remplace par une action sûre
  choisie aléatoirement avant de la passer à env.step().

- make_shielded_env_pre : pour PPO via sb3-contrib MaskablePPO. Utilise un
  ActionMasker qui expose le masque d'actions sûres à chaque pas. L'algo
  ne peut pas sampler une action interdite.

Référence : Alshiekh et al. "Safe Reinforcement Learning via Shielding",
AAAI 2018, sections sur le pre-shield et le post-shield.
"""
import numpy as np
import gymnasium as gym
from envs.frozenlake_envs import make_env
from .shield import Shield



class PostShieldWrapper(gym.ActionWrapper):
    """
    Post-shield : intercepte l'action choisie par l'agent et la remplace
    par une action sûre si elle est dangereuse.

    Comptabilise le nombre d'interventions, exposé via wrapper.n_interventions.
    """

    def __init__(self, env, shield):
        super().__init__(env)
        self.shield = shield
        self.n_interventions = 0
        self.n_steps = 0
        # RNG dédié pour reproductibilité
        self._shield_rng = np.random.default_rng()

    def action(self, action):
        action = int(action)
        # État courant via l'env de base
        state = int(self.env.unwrapped.s)
        self.n_steps += 1
        if not self.shield.is_safe(state, action):
            self.n_interventions += 1
            safe = self.shield.safe_action_set(state)
            return int(self._shield_rng.choice(safe))
        return action

    def reset(self, **kwargs):
        seed = kwargs.get("seed", None)
        if seed is not None:
            self._shield_rng = np.random.default_rng(seed)
        return self.env.reset(**kwargs)


def make_shielded_env_post(size, slippery, horizon=3, tolerance=0.01):
    """Env avec post-shield pour DQN."""
    env = make_env(size=size, slippery=slippery)
    base = env.unwrapped
    shield = Shield(base, horizon=horizon, tolerance=tolerance)
    env = PostShieldWrapper(env, shield)
    env.shield = shield
    return env


def make_shielded_env_pre(size, slippery, horizon=3, tolerance=0.01):
    """Env avec pre-shield (action masking) pour MaskablePPO."""
    from sb3_contrib.common.wrappers import ActionMasker

    env = make_env(size=size, slippery=slippery)
    base = env.unwrapped
    shield = Shield(base, horizon=horizon, tolerance=tolerance)

    def mask_fn(_env):
        return shield.get_action_mask(int(_env.unwrapped.s))

    env = ActionMasker(env, mask_fn)
    env.shield = shield
    return env
