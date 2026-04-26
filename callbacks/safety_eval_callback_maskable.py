"""
Variante de SafetyEvalCallback pour MaskablePPO : passe le masque
d'actions à model.predict() pour respecter le shield durant l'évaluation.

Utilisée uniquement par train_ppo_shielded.py. Pour DQN-shielded et les
baselines, le SafetyEvalCallback original convient.
"""
import os
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class SafetyEvalCallbackMaskable(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq=500,
        n_eval_episodes=100,
        log_path=None,
        deterministic=True,
        verbose=0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        self.deterministic = deterministic

        self.timesteps = []
        self.mean_rewards = []
        self.success_rates = []
        self.hole_rates = []
        self.mean_episode_lengths = []

        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            metrics = self.evaluate_policy()
            self.timesteps.append(self.num_timesteps)
            self.mean_rewards.append(metrics["mean_reward"])
            self.success_rates.append(metrics["success_rate"])
            self.hole_rates.append(metrics["hole_rate"])
            self.mean_episode_lengths.append(metrics["mean_episode_length"])
            self._save_results()
        return True

    def evaluate_policy(self):
        rewards, lengths = [], []
        successes = holes = 0

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = truncated = False
            ep_reward = 0.0
            ep_length = 0

            while not (done or truncated):
                # Récupère le masque depuis le wrapper ActionMasker
                # action_masks() est exposé par sb3-contrib
                from sb3_contrib.common.maskable.utils import get_action_masks
                action_masks = get_action_masks(self.eval_env)
                action, _ = self.model.predict(
                    obs,
                    deterministic=self.deterministic,
                    action_masks=action_masks,
                )
                action = int(action)
                obs, reward, done, truncated, _ = self.eval_env.step(action)
                ep_reward += reward
                ep_length += 1

            final_state = self.eval_env.unwrapped.s
            desc = self.eval_env.unwrapped.desc.flatten()
            cell = desc[final_state].decode("utf-8")

            rewards.append(ep_reward)
            lengths.append(ep_length)
            if ep_reward > 0:
                successes += 1
            elif cell == "H":
                holes += 1

        return {
            "mean_reward": float(np.mean(rewards)),
            "success_rate": successes / self.n_eval_episodes,
            "hole_rate": holes / self.n_eval_episodes,
            "mean_episode_length": float(np.mean(lengths)),
        }

    def _save_results(self):
        if self.log_path is None:
            return
        save_file = os.path.join(self.log_path, "evaluations_safety.npz")
        np.savez(
            save_file,
            timesteps=np.array(self.timesteps, dtype=np.int64),
            mean_rewards=np.array(self.mean_rewards, dtype=np.float64),
            success_rates=np.array(self.success_rates, dtype=np.float64),
            hole_rates=np.array(self.hole_rates, dtype=np.float64),
            mean_episode_lengths=np.array(
                self.mean_episode_lengths, dtype=np.float64
            ),
        )
