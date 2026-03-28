import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor


class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        n = env.observation_space.n
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n,),
            dtype=np.float32
        )
        self.n = n

    def observation(self, obs):
        vec = np.zeros(self.n, dtype=np.float32)
        vec[obs] = 1.0
        return vec


def make_env(size=4, slippery=True):
    env = gym.make(
        "FrozenLake-v1",
        map_name=f"{size}x{size}",
        is_slippery=slippery,
    )

    # Important: Monitor avant les autres wrappers
    env = Monitor(env)

    # Encodage one-hot pour SB3
    env = OneHotWrapper(env)

    return env