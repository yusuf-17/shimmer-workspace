"""
Collect trajectories using a trained PPO model.
Loads the model and collects environment experiences for analysis/training.
"""
import sys
import os
from pathlib import Path
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Myworld/packages/", "metaworld")))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit
import torch
from tqdm import tqdm

from shimmer_metaworld.modules.domains import load_pretrained_domains
from shimmer_metaworld.config import load_config
from shimmer_metaworld import DEBUG_MODE, PROJECT_DIR
from shimmer.modules.selection import FixedSharedSelection
from shimmer.modules.global_workspace import GlobalWorkspaceFusion
from metaworld.envs import SawyerTestTaskEnv
import metaworld


def make_env_simple(rank, seed=0):
    """Create a single environment for inference."""
    def _init():
        env = SawyerTestTaskEnv(render_mode='rgb_array', camera_name='corner4', width=128, height=128)
        env_name = "test-gw-env"
        ml1 = metaworld.MT50()
        task = [t for t in ml1.train_tasks if t.env_name == env_name][0]
        env.set_task(task)
        env = TimeLimit(env, max_episode_steps=400)
        obs, _ = env.reset()
        return env
    
    return _init


def collect_trajectories(model_path, num_episodes=10, save_dir=None):
    """
    Collect trajectories from a trained PPO model.
    
    Args:
        model_path: Path to the trained model.zip file
        num_episodes: Number of episodes to collect
        save_dir: Directory to save the collected data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load global workspace
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
    
    # Create environment
    print(f"Creating environment...")
    num_envs = 1
    env = DummyVecEnv([make_env_simple(i) for i in range(num_envs)])
    
    # Load trained PPO model
    print(f"Loading PPO model from: {model_path}")
    model = PPO.load(model_path, env=env)
    model.policy.eval()
    
    # Prepare save directory
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        print(f"Will save trajectories to: {save_dir}")
    
    # Collect trajectories
    print(f"\nCollecting {num_episodes} trajectories...")
    
    trajectories = []
    obs = env.reset()
    episode_data = defaultdict(list)
    episode_count = 0
    
    with torch.no_grad():
        pbar = tqdm(total=num_episodes, desc="Episodes")
        while episode_count < num_episodes:
            # Predict actions
            actions, _states = model.predict(obs, deterministic=True)
            
            # Store observation and action
            episode_data['observations'].append(obs.copy())
            episode_data['actions'].append(actions.copy())
            
            # Step environment
            obs, rewards, dones, infos = env.step(actions)
            
            episode_data['rewards'].append(rewards)
            
            # Check if episode is done
            if dones[0]:
                trajectory = {
                    'observations': np.array(episode_data['observations']),
                    'actions': np.array(episode_data['actions']),
                    'rewards': np.array(episode_data['rewards']),
                    'success': infos[0].get('is_success', False) if isinstance(infos[0], dict) else False,
                }
                trajectories.append(trajectory)
                
                # Save individual trajectory if specified
                if save_dir is not None:
                    np.save(
                        f"{save_dir}/trajectory_{episode_count:04d}.npy",
                        trajectory,
                        allow_pickle=True
                    )
                
                episode_data = defaultdict(list)
                episode_count += 1
                pbar.update(1)
        
        pbar.close()
    
    env.close()
    
    # Print statistics
    print("\n" + "="*50)
    print("COLLECTION RESULTS")
    print("="*50)
    
    total_steps = sum(len(traj['observations']) for traj in trajectories)
    total_reward = sum(np.sum(traj['rewards']) for traj in trajectories)
    success_count = sum(1 for traj in trajectories if traj['success'])
    
    print(f"Episodes collected: {len(trajectories)}")
    print(f"Total steps: {total_steps}")
    print(f"Average steps per episode: {total_steps / len(trajectories):.1f}")
    print(f"Average reward: {total_reward / len(trajectories):.2f}")
    print(f"Success rate: {success_count / len(trajectories) * 100:.1f}%")
    
    if save_dir is not None:
        # Save summary
        np.save(
            f"{save_dir}/summary.npy",
            {
                'num_episodes': len(trajectories),
                'total_steps': total_steps,
                'avg_steps_per_episode': total_steps / len(trajectories),
                'avg_reward': total_reward / len(trajectories),
                'success_rate': success_count / len(trajectories),
            },
            allow_pickle=True
        )
        print(f"\nTrajectories saved to: {save_dir}")
    
    print("="*50)
    
    return trajectories


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect trajectories from a trained PPO model')
    parser.add_argument('--model', type=str,
                       default='/home/yelhelw/metaworld_GW/shimmer-metaworld/models/2hbd488c/model.zip',
                       help='Path to the trained model.zip file')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to collect')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save trajectories (optional)')
    
    args = parser.parse_args()
    
    collect_trajectories(
        model_path=args.model,
        num_episodes=args.episodes,
        save_dir=args.save_dir
    )
