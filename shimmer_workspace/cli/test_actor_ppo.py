from dataclasses import dataclass
import glob
import os
import sys
from typing import Any

from cfg_tools.utils import Path
import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
import torch
import tyro

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../", "")))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../Myworld/packages/", "metaworld")
    ),
)

import metaworld

from shimmer.modules.global_workspace import GlobalWorkspaceFusion
from shimmer.modules.selection import FixedSharedSelection
from shimmer_metaworld import DEBUG_MODE, PROJECT_DIR
from shimmer_metaworld.config import load_config
from shimmer_metaworld.modules.domains import load_pretrained_domains


class NormalizeAction(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        dataset = "/mnt/datashare/yelhelw/complex_dataset_V3/"
        actions = np.load(f"{dataset}/actions_train.npy", mmap_mode="r")
        self.actions_mean = np.mean(actions, axis=0)
        self.actions_std = np.std(actions, axis=0)

    def step(self, action: int):
        denorm_action = np.array(action, copy=True)
        for i in range(len(denorm_action)):
            denorm_action[i] = denorm_action[i] * self.actions_std[i] + self.actions_mean[i]
        return self.env.step(denorm_action)


class FuseGwWrapper(gym.wrappers.TransformObservation):
    def __init__(self, env: gym.Env, gw_model: Any, selection: Any) -> None:
        self.gw = gw_model
        self.visual_module = self.gw.domain_mods["v_latents"].visual_module
        self.gw_mod = self.gw.gw_mod
        self.encode_domain = self.gw.encode_domain
        self.fuse = self.gw_mod.fuse
        self.selection = selection
        self.device = next(gw_model.parameters()).device

        latent_dim = self.gw.workspace_dim
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(latent_dim,), dtype=np.float32
        )

        gym.utils.RecordConstructorArgs.__init__(self)
        super().__init__(env, lambda obs: self.transform(obs), observation_space=self.observation_space)

    def transform(self, observation: Any) -> np.ndarray:
        imgs = np.ascontiguousarray(np.flipud(observation))
        if imgs.ndim == 3:
            imgs = imgs[np.newaxis, ...]
        if imgs.ndim != 4:
            raise ValueError(f"Expected image observation with 4 dims, got shape {imgs.shape}.")

        imgs_tensor = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(
            self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        imgs_tensor.mul_(1.0 / 255.0)

        with torch.inference_mode():
            imgs_latent = self.visual_module.encode(imgs_tensor)
            gw_latents = self.gw_mod.encode(
                {"v_latents": self.encode_domain(imgs_latent, "v_latents")}
            )
            selection_scores = self.selection(gw_latents, gw_latents)
            gw_latent_v_fused = self.fuse(gw_latents, selection_scores)

        return gw_latent_v_fused.detach().cpu().numpy()[0].astype(np.float32, copy=False)


@dataclass(frozen=True)
class Args:
    model_path: str
    env_name: str = "soccer-v3"
    gw_ckpt: str = "r8stcltz"
    gw_epoch: int = 250
    gw_checkpoint_path: str | None = None
    seed: int = 0
    max_episode_steps: int = 400
    eval_steps: int = 10000
    deterministic: bool = True
    width: int = 128
    height: int = 128
    save_frames: bool = False
    frame_dir: str = "actor_eval_frames"
    frame_every: int = 1


def resolve_model_zip(path_or_dir: str) -> str:
    if os.path.isfile(path_or_dir):
        if path_or_dir.endswith(".zip"):
            return path_or_dir
        raise ValueError(f"Model path must point to a .zip file, got: {path_or_dir}")

    if not os.path.isdir(path_or_dir):
        raise FileNotFoundError(f"No such file or directory: {path_or_dir}")

    candidates = sorted(glob.glob(os.path.join(path_or_dir, "*.zip")), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(f"No .zip model found in directory: {path_or_dir}")
    return candidates[-1]


def resolve_gw_checkpoint(args: Args) -> str:
    if args.gw_checkpoint_path is not None:
        if not os.path.isfile(args.gw_checkpoint_path):
            raise FileNotFoundError(f"GW checkpoint not found: {args.gw_checkpoint_path}")
        return args.gw_checkpoint_path

    candidate = f"/mnt/datashare/yelhelw/checkpoints/shimmer-meta-{args.gw_ckpt}/epoch={args.gw_epoch}.ckpt"
    if os.path.isfile(candidate):
        return candidate

    all_ckpts = sorted(glob.glob(f"/mnt/datashare/yelhelw/checkpoints/shimmer-meta-{args.gw_ckpt}/*.ckpt"))
    if not all_ckpts:
        raise FileNotFoundError(f"No GW checkpoints found for run id: {args.gw_ckpt}")
    return all_ckpts[-1]


def make_env(args: Args, gw_model: Any, selection: Any) -> gym.Env:
    if gw_model is None:
        render_only = False
    else:
        render_only = True
    env = metaworld.make_mt_envs(
        args.env_name,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        terminate_on_success=False,
        use_one_hot=False,
        reward_function_version="v2",
        reward_normalization_method=None,
        normalize_observations=False,
        render_mode="rgb_array",
        camera_name="corner4",
        width=args.width,
        height=args.height,
        render_only=render_only,
    )
    if gw_model is None:
        pass
    else:
        env = FuseGwWrapper(env, gw_model, selection)
        env = NormalizeAction(env)
    return env


def main() -> None:
    args = tyro.cli(Args)

    model_zip = resolve_model_zip(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(
        PROJECT_DIR / "shimmer_metaworld" / "config_template",
        load_files=["train_gw.yaml"],
        debug_mode=DEBUG_MODE,
        argv=[]
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

    gw_checkpoint = resolve_gw_checkpoint(args)
    gw = GlobalWorkspaceFusion.load_from_checkpoint(
        gw_checkpoint,
        domain_mods=domain_modules,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
        weights_only=False,
    )
    gw.to(device)
    gw.eval().freeze()
    gw = None
    selection = FixedSharedSelection()
    env = make_env(args, gw, selection)

    model = PPO.load(model_zip, env=env, device=device)
    
    if args.save_frames:
        os.makedirs(args.frame_dir, exist_ok=True)

    obs, _ = env.reset(seed=args.seed)
    episode_reward = 0.0
    returns: list[float] = []
    successes: list[float] = []
    lengths: list[int] = []
    episode_len = 0

    for step in range(args.eval_steps):
        #print(step)
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += float(reward)
        episode_len += 1

        if args.save_frames and step % max(args.frame_every, 1) == 0:
            frame = env.render()
            frame = np.flipud(frame)
            Image.fromarray(frame).save(os.path.join(args.frame_dir, f"{step:06d}.png"))

        if terminated or truncated:
            print(episode_reward, info["success"])
            returns.append(episode_reward)
            successes.append(float(info.get("success", 0.0)))
            lengths.append(episode_len)
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_len = 0

    env.close()

    if returns:
        print(f"Episodes: {len(returns)}")
        print(f"Mean return: {np.mean(returns):.4f}")
        print(f"Mean success: {np.mean(successes):.4f}")
        print(f"Mean episode length: {np.mean(lengths):.2f}")
    else:
        print("No completed episodes within eval_steps.")


if __name__ == "__main__":
    main()
