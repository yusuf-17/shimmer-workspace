"""
Custom callbacks for logging metrics to wandb.
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import wandb


class SimpleEpisodeStatsCallback(BaseCallback):
    """
    Callback that logs environment metrics per step to wandb.
    Extracts metrics like reach_rew, push_rew from the environment's info at each step.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.step_count = 0
        
    def _on_step(self) -> bool:
        """Called at each environment step - log metrics."""
        self.step_count += 1
        
        try:
            # For DummyVecEnv, access the buffer of infos from the last step
            if hasattr(self.model.env, 'buf_infos'):
                # buf_infos contains info from all parallel environments
                for env_idx, info in enumerate(self.model.env.buf_infos):
                    if isinstance(info, dict):
                        logs = {}
                        # Log all numerical metrics from info
                        for key, value in info.items():
                            if isinstance(value, (int, float, np.number)):
                                logs[f'{key}'] = float(value)
                        
                        if len(logs) > 0:
                            wandb.log(logs, commit=False)
        except Exception as e:
            if self.verbose > 0:
                print(f"Error in SimpleEpisodeStatsCallback: {e}")
        
        return True


        class DeterministicEvalCallback(BaseCallback):
            """Run a deterministic evaluation loop on a separate env and log success."""

            def __init__(
                self,
                eval_env,
                eval_freq: int = 10_000,
                n_eval_episodes: int = 5,
                deterministic: bool = True,
                log_prefix: str = "eval",
                verbose: int = 0,
                close_env: bool = True,
            ):
                super().__init__(verbose)
                self.eval_env = eval_env
                self.eval_freq = eval_freq
                self.n_eval_episodes = n_eval_episodes
                self.deterministic = deterministic
                self.log_prefix = log_prefix
                self.close_env = close_env
                self._last_eval_step = -1

            def _on_training_start(self) -> None:
                self._maybe_evaluate(force=True)

            def _on_step(self) -> bool:
                self._maybe_evaluate(force=False)
                return True

            def _on_training_end(self) -> None:
                if self.close_env and hasattr(self.eval_env, "close"):
                    self.eval_env.close()

            def _maybe_evaluate(self, force: bool) -> None:
                if self.eval_freq <= 0:
                    return

                if not force and self.num_timesteps - self._last_eval_step < self.eval_freq:
                    return

                metrics = self._evaluate()
                self._last_eval_step = self.num_timesteps

                wandb.log(metrics)

                if self.verbose > 0:
                    print(
                        f"[{self.log_prefix}] step={self.num_timesteps} "
                        f"success_rate={metrics[f'{self.log_prefix}/success_rate']:.4f} "
                        f"mean_return={metrics[f'{self.log_prefix}/mean_reward']:.4f}"
                    )

            def _evaluate(self):
                obs = self.eval_env.reset()
                episode_returns = []
                episode_lengths = []
                episode_successes = []

                current_return = 0.0
                current_length = 0

                while len(episode_returns) < self.n_eval_episodes:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, rewards, dones, infos = self.eval_env.step(action)

                    current_return += float(rewards[0])
                    current_length += 1

                    if bool(dones[0]):
                        info = infos[0] if isinstance(infos, (list, tuple)) and infos else {}
                        success_value = 0.0
                        if isinstance(info, dict):
                            if "is_success" in info:
                                success_value = float(info["is_success"])
                            elif "success" in info:
                                success_value = float(info["success"])

                        episode_returns.append(current_return)
                        episode_lengths.append(current_length)
                        episode_successes.append(success_value)

                        current_return = 0.0
                        current_length = 0

                return {
                    f"{self.log_prefix}/mean_reward": float(np.mean(episode_returns)),
                    f"{self.log_prefix}/std_reward": float(np.std(episode_returns)),
                    f"{self.log_prefix}/mean_ep_len": float(np.mean(episode_lengths)),
                    f"{self.log_prefix}/success_rate": float(np.mean(episode_successes)),
                    f"{self.log_prefix}/episodes": float(self.n_eval_episodes),
                }
