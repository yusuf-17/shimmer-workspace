import os
import sys
import random
from dataclasses import dataclass
from functools import cached_property
from typing import override
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import gymnasium as gym
import tyro
import wandb
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

from gymnasium.wrappers import TimeLimit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../metaworld_GW/Myworld/packages/metaworld")))
import metaworld
from metaworld.envs import SawyerTestTaskEnv
from metaworld.wrappers import RandomTaskSelectWrapper

from metaworld_algorithms.config.networks import ContinuousActionPolicyConfig
from metaworld_algorithms.config.nn import VanillaNetworkConfig
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import OnPolicyTrainingConfig
from metaworld_algorithms.config.envs import EnvConfig
from metaworld_algorithms.rl.algorithms import PPOConfig
from metaworld_algorithms.rl.algorithms import get_algorithm_for_config
from metaworld_algorithms.types import Agent, GymVectorEnv


ENV_NAME = "basketball-v3"
MAX_EPISODE_STEPS = 500

def make_env(rank, seed = 0, model=None, selection=None, monitor_dir=None, wrapper_class=None, monitor_kwargs=None, wrapper_kwargs=None):
    def _init():
        env = SawyerTestTaskEnv(render_mode='rgb_array',camera_name='corner4',width = 128, height=128)
        
        #test-gw-env basketball-v3
        env_name = "test-gw-env"
        ml1 = metaworld.MT50()
        task = [t for t in ml1.train_tasks if t.env_name == env_name][0]
        env.set_task(task)
        #env = FuseGwWrapper(env, model, selection) #VAEWrapper(env, domain_modules) # FuseGwWrapper(env, model, selection) #  ImgWrapper(env)
        #env = ObservationWrapper(env)
        env = TimeLimit(env, max_episode_steps=400)
        # env = FrameStack(env, 4)
        
        monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
        # Create the monitor folder if needed
        if monitor_path is not None and monitor_dir is not None:
            os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, filename=monitor_path)
        # Optionally, wrap the environment with the provided wrapper
        if wrapper_class is not None:
            env = wrapper_class(env, **wrapper_kwargs)

        obs, _ = env.reset()

        return env

    set_random_seed(seed)
    return _init

@dataclass(frozen=True)
class BasketballEnvConfig(EnvConfig):
    env_id: str = ENV_NAME

    @cached_property
    @override
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(
            np.array([-1, -1, -1, -1], dtype=np.float32),
            np.array([+1, +1, +1, +1], dtype=np.float32),
        )

    @cached_property
    @override
    def observation_space(self) -> gym.Space:
        env = SawyerBasketballEnvV3()
        space = env.observation_space
        env.close()
        return space

    @override
    def spawn(self, seed: int = 1) -> GymVectorEnv:
        return DummyVecEnv([make_env(i, seed=seed) for i in range(4)])


    @override
    def evaluate(
        self, envs: GymVectorEnv, agent: Agent
    ) -> tuple[float, float, dict[str, float]]:
        # Simple evaluation: run evaluation_num_episodes episodes
        successes = []
        returns = []
        obs, _ = envs.reset()
        ep_return = 0.0
        ep_successes = []

        for _ in range(self.evaluation_num_episodes * MAX_EPISODE_STEPS):
            action = agent.eval_action(obs)
            obs, reward, terminated, truncated, infos = envs.step(action)
            ep_return += float(reward[0])
            if terminated[0] or truncated[0]:
                success = float(infos.get("final_info", [{}])[0].get("success", 0.0))
                ep_successes.append(success)
                returns.append(ep_return)
                ep_return = 0.0
                if len(ep_successes) >= self.evaluation_num_episodes:
                    break

        mean_success = float(np.mean(ep_successes)) if ep_successes else 0.0
        mean_return = float(np.mean(returns)) if returns else 0.0
        return mean_success, mean_return, {ENV_NAME: mean_success}


@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None


def main() -> None:
    args = tyro.cli(Args)

    env_config = BasketballEnvConfig(
        env_id=ENV_NAME,
        terminate_on_success=False,
        use_one_hot=False,
        max_episode_steps=MAX_EPISODE_STEPS,
    )

    algorithm_config = PPOConfig(
        num_tasks=1,
        gamma=0.99,
        policy_config=ContinuousActionPolicyConfig(
            network_config=VanillaNetworkConfig(
                optimizer=OptimizerConfig(max_grad_norm=1.0),
            ),
            squash_tanh=False,
        ),
        vf_config=None,
        baseline_type="linear",
        num_epochs=16,
        num_gradient_steps=32,
        gae_lambda=0.97,
        target_kl=None,
        clip_vf_loss=False,
        normalize_advantages=False,
    )

    training_config = OnPolicyTrainingConfig(
        total_steps=int(2e7),
        rollout_steps=10_000,
        evaluation_frequency=1_000_000 // 500,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.track:
        assert args.wandb_project is not None
        wandb.init(
            project=args.wandb_project,
            config={
                "env_id": ENV_NAME,
                "seed": args.seed,
            },
        )

    envs = env_config.spawn(seed=args.seed)

    algorithm_cls = get_algorithm_for_config(algorithm_config)
    algorithm = algorithm_cls.initialize(algorithm_config, env_config, seed=args.seed)

    algorithm.train(
        config=training_config,
        envs=envs,
        env_config=env_config,
        seed=args.seed,
        track=args.track,
        checkpoint_manager=None,
        checkpoint_metadata=None,
        buffer_checkpoint=None,
    )

    envs.close()

    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()
