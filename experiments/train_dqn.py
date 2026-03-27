import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import DQN
from envs.frozenlake_envs import make_env


# Cas d'étude
configs = [
    {"name": "4x4_deterministic", "size": 4, "slippery": False},
    {"name": "4x4_stochastic", "size": 4, "slippery": True},
    {"name": "8x8_stochastic", "size": 8, "slippery": True},
]

os.makedirs("models", exist_ok=True)

for config in configs:
    print(f"Training DQN on {config['name']}")

    env = make_env(size=config["size"], slippery=config["slippery"])

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=100,
        batch_size=32,
        gamma=0.99,
        verbose=1
    )

    model.learn(total_timesteps=5000)

    model.save(f"models/dqn_{config['name']}")

print("Training terminé")