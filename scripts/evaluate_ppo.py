"""
Evaluate/Inference script for trained PPO model.
Loads a trained PPO model and uses it to interact with the environment.
"""
import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Myworld/packages/", "metaworld")))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
import torch
import numpy as np
from tqdm import tqdm

from shimmer_metaworld.modules.domains import load_pretrained_domains
from shimmer_metaworld.config import load_config
from shimmer_metaworld import DEBUG_MODE, PROJECT_DIR, LOGGER
from shimmer.modules.selection import FixedSharedSelection
from shimmer.modules.global_workspace import GlobalWorkspaceFusion
from metaworld.envs import SawyerTestTaskEnv
import metaworld


def make_env(rank, seed=0, model=None, selection=None, monitor_dir=None):
    """Create a single environment."""
    def _init():
        env = SawyerTestTaskEnv(render_mode='rgb_array', camera_name='corner4', width=128, height=128)
        
        env_name = "test-gw-env"
        ml1 = metaworld.MT50()
        task = [t for t in ml1.train_tasks if t.env_name == env_name][0]
        env.set_task(task)
        
        env = TimeLimit(env, max_episode_steps=400)
        
        monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
        if monitor_path is not None and monitor_dir is not None:
            os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, filename=monitor_path, info_keywords=('is_success',))
        
        obs, _ = env.reset()
        return env
    
    return _init


def evaluate_ppo(model_path, num_episodes=5, num_envs=1, render=False):
    """
    Load and evaluate a trained PPO model.
    
    Args:
        model_path: Path to the trained model.zip file
        num_episodes: Number of episodes to run
        num_envs: Number of parallel environments
        render: Whether to render the environment
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load global workspace if needed
    config = load_config(
        PROJECT_DIR / "shimmer_metaworld" / "config_template",
        load_files=["train_gw.yaml"],
        debug_mode=DEBUG_MODE,
    )
    
    domain_modules, gw_encoders, gw_decoders = load_pretrained_domains(
        config.domains,
        config.global_workspace.latent_dim,
        config.global_workspace.encoders.hidden_dim,
        config.global_workspace.encoders.n_layers,
        config.global_workspace.decoders.hidden_dim,
        config.global_workspace.decoders.n_layers,
        is_linear=config.global_workspace.linear_domains,
        bias=config.global_workspace.linear_domains_use_bias,
    )

    gw_ckpt = "fx9karna"
    ckpt_epoch = "250"
    ckpt_path = f'/mnt/datashare/yelhelw/checkpoints/shimmer-meta-{gw_ckpt}/epoch={ckpt_epoch}.ckpt'

    gw = GlobalWorkspaceFusion.load_from_checkpoint(
        ckpt_path,
        domain_mods=domain_modules,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders
    )
    gw.eval().freeze()
    
    selection = FixedSharedSelection()
    
    # Create environment
    print(f"Creating {num_envs} environment(s)...")
    env = DummyVecEnv([make_env(i, seed=0, model=gw, selection=selection) for i in range(num_envs)])
    
    # Load trained PPO model
    print(f"Loading PPO model from: {model_path}")
    model = PPO.load(model_path, env=env)
    model.policy.eval()
    
    print(f"Model loaded successfully!")
    print(f"Policy architecture: {model.policy}")
    
    # Run inference
    print(f"\nRunning inference for {num_episodes} episodes...")
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    
    obs = env.reset()
    episode_reward = np.zeros(num_envs)
    episode_length = np.zeros(num_envs)
    episode_count = 0
    
    with torch.no_grad():
        while episode_count < num_episodes:
            # Predict actions
            actions, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, rewards, dones, infos = env.step(actions)
            
            episode_reward += rewards
            episode_length += 1
            
            # Check if episodes are done
            for i in range(num_envs):
                if dones[i]:
                    success = infos[i].get('is_success', False) if isinstance(infos[i], dict) else False
                    episode_rewards.append(episode_reward[i])
                    episode_lengths.append(episode_length[i])
                    episode_successes.append(success)
                    episode_count += 1
                    
                    print(f"Episode {episode_count}/{num_episodes} - "
                          f"Reward: {episode_reward[i]:.2f}, "
                          f"Length: {int(episode_length[i])}, "
                          f"Success: {success}")
                    
                    episode_reward[i] = 0
                    episode_length[i] = 0
    
    env.close()
    
    # Print statistics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.0f}")
    print(f"Success Rate: {np.mean(episode_successes)*100:.1f}%")
    print("="*50)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a trained PPO model')
    parser.add_argument('--model', type=str, 
                       default='/home/yelhelw/metaworld_GW/shimmer-metaworld/models/2hbd488c/model.zip',
                       help='Path to the trained model.zip file')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--envs', type=int, default=1,
                       help='Number of parallel environments')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment')
    
    args = parser.parse_args()
    
    evaluate_ppo(
        model_path=args.model,
        num_episodes=args.episodes,
        num_envs=args.envs,
        render=args.render
    )
