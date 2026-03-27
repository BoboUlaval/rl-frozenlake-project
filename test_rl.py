import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

# Wrapper one-hot
class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        n = env.observation_space.n
        self.observation_space = gym.spaces.Box(0, 1, shape=(n,), dtype=np.float32)
        self.n = n

    def observation(self, obs):
        vec = np.zeros(self.n, dtype=np.float32)
        vec[obs] = 1.0
        return vec

# Création env
env = gym.make("FrozenLake-v1", is_slippery=True)
env = OneHotWrapper(env)

# Modèle
model = DQN("MlpPolicy", env, verbose=1)

# Entraînement rapide
model.learn(total_timesteps=2000)

# Test ??
obs, _ = env.reset()
for _ in range(20):
    action, _ = model.predict(obs, deterministic=True) # ça marche mtn?
    action = int(action)
    obs, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        obs, _ = env.reset()

print("Test terminé")