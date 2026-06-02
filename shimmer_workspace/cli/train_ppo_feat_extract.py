from stable_baselines3 import PPO
import tyro
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from cfg_tools.utils import Path
import torch
import torch.nn as nn
import wandb
import os
import numpy as np
import sys
from typing import Any
from stable_baselines3.common.callbacks import CallbackList
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from torchvision import transforms
from gymnasium.wrappers import TimeLimit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../", "")))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../", "gw_dreamer")))
from shimmer_metaworld.modules.domains import load_pretrained_domains
from shimmer_metaworld.config import load_config
from shimmer_metaworld import DEBUG_MODE, PROJECT_DIR,LOGGER
from shimmer.modules.selection import FixedSharedSelection, RandomSelection

from shimmer.modules.selection import FixedSharedSelection, RandomSelection
from shimmer import ContrastiveLossType
from shimmer.modules.global_workspace import (
    GlobalWorkspaceFusion,
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,)
import gymnasium as gym

# Add the parent folder containing 'metaworld' and import env and policy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Myworld/packages/", "metaworld")))
import metaworld

from metaworld.envs import SawyerSoccerEnvV3,SawyerPushWallEnvV3, SawyerWallEnvV3

from dataclasses import dataclass

@dataclass(frozen=True)
class Args:
    init_gw : bool = False 
    vae_run : bool = False
    ckpt : str = "none" 
    num_envs : int | None = None

class features_extractor(BaseFeaturesExtractor):
    def __init__(self,observation_space,features_dim,model):
        super(features_extractor,self).__init__(observation_space, features_dim)
        self.model = model
        self._features_dim = features_dim
        self.gw_mod = self.model.gw_mod
        self.fuse = self.gw_mod.fuse
        self.device = next(model.parameters()).device
    def forward(self,observations):
        observations_t = torch.tensor(observations).to(self.device, dtype=torch.float32,).unsqueeze(0)
        
        gw_latents = self.gw_mod.encode({
                "attr": observations_t
            })

        gw_latent_fused = self.fuse(
            gw_latents,
            {"attr": torch.ones(gw_latents["attr"].size(0), device=self.device)}
            )
        decoded_latent_uni = self.gw_mod.decode(gw_latent_fused)

        decoded_act = decoded_latent_uni["act"].squeeze(0)

        return decoded_act

class TransformObs(gym.wrappers.TransformObservation):
    """Wraps a Metaworld environment to use Global Workspace latent representations.
    
    Encodes visual observations through the GW's visual VAE encoder and fuses them
    with the global workspace to produce latent observations.
    """
    
    def __init__(self, env: gym.Env,task,rank) -> None:
        dataset = Path("/mnt/datashare/yelhelw/complex_dataset_V3/")
        self.train = torch.from_numpy(np.load(dataset / f"attributes_train.npy")[:,:13])
        self.mean = torch.mean(self.train, dim=0)
        self.stdv = torch.std(self.train, dim=0)
        self.env_id = rank
        self.task = task
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(13,), dtype=np.float32
        )

        gym.utils.RecordConstructorArgs.__init__(self)
        super().__init__(env, lambda obs : self.transform(obs),observation_space=self.observation_space)  
        


    def transform(self, observation: Any) -> np.ndarray:
        data = env.get_attr("data")[self.env_id]
        wall = [0,0,-10]
        soccer_goal = [0,0,-10]
        if self.task == "soccer-v3":
            soccer_goal = data.body("goal_whole").xpos.tolist()
        elif self.task == "push-wall-v3":
            wall= data.body("wall").xpos.tolist()
        else:
            wall = data.body("wall").xpos.tolist()
            soccer_goal = data.body("goal_whole").xpos.tolist()
        proprio = observation[:4].tolist()
        ball = observation[4:7].tolist()
        

        att = proprio+ball+wall+soccer_goal
        
        for x in range(len(att)):
            att[x] = (att[x] - self.mean[x]) / self.stdv[x]
        
        return att


class NormalizeAction(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Sticky action.

    Paper: https://arxiv.org/abs/1709.06009
    Official implementation: https://github.com/mgbellemare/Arcade-Learning-Environment

    :param env: Environment to wrap
    :param action_repeat_probability: Probability of repeating the last action
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        dataset = Path("/mnt/datashare/yelhelw/complex_dataset_V3/")
        self.combin_index = np.load(dataset / f"combin_task_index.npy",allow_pickle=True)[0]
        
        self.train = np.clip(torch.from_numpy(
        np.load(dataset / f"actions_train.npy")), -10, 10)
        self.mean = torch.mean(self.train[:self.combin_index,:], dim=0)
        self.stdv = torch.std(self.train[:self.combin_index,:], dim=0)

    def step(self, action: int):
        for x in range(len(action)):
            action[x] = action[x] * self.stdv[x] + self.mean[x]
        return self.env.step(action)


class SuccessInfoWrapper(gym.Wrapper):
    """Ensure SB3 receives info['is_success'] so rollout/success_rate is logged."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(info, dict) and "is_success" not in info and "success" in info:
            info = dict(info)
            info["is_success"] = bool(info["success"])
        return obs, reward, terminated, truncated, info

class FuseGwWrapper(gym.wrappers.TransformObservation):
    """Wraps a Metaworld environment to use Global Workspace latent representations.
    
    Encodes visual observations through the GW's visual VAE encoder and fuses them
    with the global workspace to produce latent observations.
    """
    
    def __init__(self, env: gym.Env, gw_model: Any, selection: Any,vae_run = False,input_mod = "v",task="soccer-v3",rank=0) -> None:
        self.gw = gw_model
        self.input = input_mod
        self.task = task
        self.env_id = rank
        latent_dim = self.gw.workspace_dim
        self.visual_module = self.gw.domain_mods["v_latents"].visual_module
        self.gw_mod = self.gw.gw_mod
        self.encode_domain = self.gw.encode_domain
        self.fuse = self.gw_mod.fuse
        self.vae_run = vae_run
        
        if input_mod == "attr":
            dataset = "/mnt/datashare/yelhelw/complex_dataset_V3/"
            attributes = np.load(f"{dataset}/attributes_train.npy", mmap_mode="r")
            self.attributes_mean = np.mean(attributes, axis=0)
            self.attributes_std = np.std(attributes, axis=0)
        if self.vae_run:
            self.observation_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self.visual_module.latent_dim,), dtype=np.float32
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(latent_dim,), dtype=np.float32
            )

        gym.utils.RecordConstructorArgs.__init__(self)
        super().__init__(env, lambda obs : self.transform(obs),observation_space=self.observation_space)  
        
        self.device = next(gw_model.parameters()).device
        self.selection = selection

        print("Initalized successfully")
    def transform(self, observation: Any) -> np.ndarray:
        """Transform a single observation to one GW latent vector.

        VectorizeTransformObservation calls this once per environment item and
        then stacks the returned vectors. The return shape must be exactly
        ``(latent_dim,)``.
        """
        if self.input == "v":
            imgs = np.ascontiguousarray(np.flipud(observation))
            # Expect a single image (H, W, C); add a batch dimension for the model.
            if imgs.ndim == 3:
                imgs = imgs[np.newaxis, ...]

            # Expect a 4D input (B, H, W, C).
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
                if self.vae_run:
                    return imgs_latent.detach().cpu().numpy()[0].astype(np.float32, copy=False)
                gw_latents = self.gw_mod.encode({
                    "v_latents": self.encode_domain(imgs_latent, "v_latents")
                })

                gw_latent_fused = self.fuse(
                    gw_latents,
                    {"v_latents": torch.ones(gw_latents["v_latents"].size(0), device=self.device)}
                    )
        else:
            data = env.get_attr("data")[self.env_id]
            wall = [0,0,-10]
            soccer_goal = [0,0,-10]
            if self.task == "soccer-v3":
                soccer_goal = data.body("goal_whole").xpos.tolist()
            elif self.task == "push-wall-v3":
                wall= data.body("wall").xpos.tolist()
            else:
                wall = data.body("wall").xpos.tolist()
                soccer_goal = data.body("goal_whole").xpos.tolist()
            proprio = observation[:4].tolist()
            ball = observation[4:7].tolist()
            

            att = proprio+ball+wall+soccer_goal
           
            for x in range(len(att)):
                att[x] = (att[x] - self.attributes_mean[x]) / self.attributes_std[x]
            
            att = torch.tensor(att).to(self.device, dtype=torch.float32,).unsqueeze(0)
            gw_latents = self.gw_mod.encode({
                    "attr": att
                })

            gw_latent_fused = self.fuse(
                gw_latents,
                {"attr": torch.ones(gw_latents["attr"].size(0), device=self.device)}
                )
        # Return exactly one vector for one env item.
        return gw_latent_fused.detach().cpu().numpy()[0].astype(np.float32, copy=False)


def count_trainable_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def extract_gw_decoder_weights(gw, num_layers=4):
    state_dict = gw.state_dict()
    weights = []
    for layer in range(num_layers):
        weight_name = f"gw_mod.gw_decoders.act.{layer * 2}.weight"
        bias_name = f"gw_mod.gw_decoders.act.{layer * 2}.bias"
        if weight_name not in state_dict or bias_name not in state_dict:
            break
        weights.append((state_dict[weight_name], state_dict[bias_name]))
    return weights


def init_policy_from_gw(model, gw_weights):
    # SB3 MlpPolicy uses policy_net hidden linears plus action_net output linear.
    pi_linear_layers = [
        m for m in model.policy.mlp_extractor.policy_net if isinstance(m, nn.Linear)
    ]
    #+ [model.policy.action_net]
    target_layers = pi_linear_layers 
    #target_layers = []

    loaded = 0
    for idx, (src_w, src_b) in enumerate(gw_weights):
        if idx >= len(target_layers):
            break
        target = target_layers[idx]

        if target.weight.shape != src_w.shape or target.bias.shape != src_b.shape:
            print(
                f"Skipping pretrained layer {idx}: "
                f"GW {tuple(src_w.shape)}/{tuple(src_b.shape)} != "
                f"PPO {tuple(target.weight.shape)}/{tuple(target.bias.shape)}"
            )
            continue

        with torch.no_grad():
            target.weight.copy_(src_w.to(target.weight.device, dtype=target.weight.dtype))
            target.bias.copy_(src_b.to(target.bias.device, dtype=target.bias.dtype))
        loaded += 1

    return loaded, len(target_layers)


def evaluate_policy_deterministic(model, gw_model, selection, env_name, device, steps=100000, vae_run=False, input_mod="v", seed=0):
    """Run a deterministic rollout with `model.predict(..., deterministic=True)` for `steps` env steps.

    Returns (mean_episode_return, success_rate, n_episodes_seen).
    """
    print(model.policy,gw_model)
    eval_env = make_vec_env(gw_model, selection, seed=seed, num_goals=1, max_episode_steps=400, env_name=env_name, vae_run=vae_run, input_mod=input_mod)
    env = eval_env.envs[0]

    obs, _ = env.reset()
    episode_rewards = 0.0
    returns = []
    successes = []
    steps_done = 0

    while steps_done < steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_rewards += float(reward)
        steps_done += 1

        done = bool(terminated or truncated)
        if done:
            returns.append(episode_rewards)
            # prefer 'is_success' (Monitor / SB3 convention) then 'success'
            is_succ = False
            if isinstance(info, dict):
                is_succ = bool(info.get("is_success", info.get("success", False)))
            successes.append(1.0 if is_succ else 0.0)
            print(is_succ)
            obs, _ = env.reset()
            episode_rewards = 0.0

    # close the temporary env
    try:
        env.close()
    except Exception:
        pass

    if not returns:
        return 0.0, 0.0, 0

    return float(np.mean(returns)), float(np.mean(successes)), len(returns)


def resolve_model_zip(path_or_dir: str) -> str:
    if os.path.isfile(path_or_dir):
        if path_or_dir.endswith(".zip"):
            return path_or_dir
        raise ValueError(f"Model path must point to a .zip file, got: {path_or_dir}")

    if os.path.isdir(path_or_dir):
        candidates = sorted(
            [os.path.join(path_or_dir, p) for p in os.listdir(path_or_dir) if p.endswith(".zip")],
            key=os.path.getmtime,
        )
        if not candidates:
            raise FileNotFoundError(f"No .zip model found in directory: {path_or_dir}")
        return candidates[-1]

    # Convenience fallback: allow passing "<run_id>/model.zip" and resolve under models/.
    if not os.path.isabs(path_or_dir):
        candidate_under_models = os.path.join("models", path_or_dir)
        if os.path.isfile(candidate_under_models) and candidate_under_models.endswith(".zip"):
            return candidate_under_models
        if os.path.isdir(candidate_under_models):
            candidates = sorted(
                [
                    os.path.join(candidate_under_models, p)
                    for p in os.listdir(candidate_under_models)
                    if p.endswith(".zip")
                ],
                key=os.path.getmtime,
            )
            if candidates:
                return candidates[-1]

    raise FileNotFoundError(f"No such model file or directory: {path_or_dir}")


def make_vec_env(gw_model, selection, seed=0, num_goals=4, max_episode_steps=500, env_id="ML1-train", env_name=None,vae_run=False,input_mod="v"):
    if env_name is None:
        env_name = "soccer-v3"
    if gw_model is None:
        #if not using GW, we use standard metaworld observation without render only and wrappers
        print("Not using GW, creating standard Metaworld envs without render_only or wrappers.")
        def _make_env(rank):
            def _init():
                env = metaworld.make_mt_envs(
                    env_name,
                    seed=seed + rank,
                    max_episode_steps=max_episode_steps,
                    terminate_on_success=False,
                    use_one_hot=False,
                    reward_function_version="v2",
                    reward_normalization_method=None,
                    normalize_observations=False,
                    render_mode="rgb_array",
                    camera_name="corner4",
                    width=128,
                    height=128,
                )
                env = TransformObs(env,env_name,rank)
                env = NormalizeAction(env)
                env = SuccessInfoWrapper(env)
                return Monitor(env, info_keywords=("is_success",))

            return _init
    else:
        def _make_env(rank):
            def _init():
                if input_mod == "v":
                    render_only = True
                else:
                    render_only = False
                
                env = metaworld.make_mt_envs(
                    env_name,
                    seed=seed + rank,
                    max_episode_steps=max_episode_steps,
                    terminate_on_success=False,
                    use_one_hot=False,
                    reward_function_version="v2",
                    reward_normalization_method=None,
                    normalize_observations=False,
                    render_mode="rgb_array",
                    camera_name="corner4",
                    width=128,
                    height=128,
                    render_only=render_only,
                )
                '''
                env = SawyerSoccerEnvV3(render_mode='rgb_array',camera_name='corner4',width = 240, height=240)
                env.max_path_length = 400
                env._partially_observable = False
                env._freeze_rand_vec = False
                env._set_task_called = True
                '''
                env = FuseGwWrapper(env, gw_model, selection,vae_run=vae_run,input_mod=input_mod,task=env_name,rank=rank)
                if not vae_run:
                    env = NormalizeAction(env)
                env = SuccessInfoWrapper(env)
                return Monitor(env, info_keywords=("is_success",))

            return _init

    return DummyVecEnv([_make_env(rank) for rank in range(num_goals)])

if __name__ == '__main__':
    args = tyro.cli(Args)

    init_gw = args.init_gw
    vae_run = args.vae_run
    ckpt = args.ckpt
    n_envs = args.num_envs if args.num_envs is not None else (4 if vae_run else 10)
    n_envs = 4
    target_rollout_steps = 100_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(
        PROJECT_DIR / "shimmer_metaworld" / "config_template",
        load_files=["train_ppo.yaml"],
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

    #2 mod
    gw_ckpt = "8wfdyfub"
    ckpt_epoch = "243"

    #3 mod
    gw_ckpt = "r8stcltz"
    ckpt_epoch = "244"
    

    #act, attr
    #gw_ckpt = "uvqpvcof"
    #ckpt_epoch = "254"  
    
    ckpt_path = f'/mnt/datashare/yelhelw/checkpoints/shimmer-meta-{gw_ckpt}/epoch={ckpt_epoch}.ckpt'

    gw = GlobalWorkspaceFusion.load_from_checkpoint(ckpt_path, domain_mods=domain_modules,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,weights_only=False)

    gw.eval().freeze()
    
    selection = FixedSharedSelection()

    if target_rollout_steps % n_envs != 0:
        raise ValueError(
            f"num_envs={n_envs} must divide {target_rollout_steps} to keep the same rollout budget"
        )
    n_steps = target_rollout_steps // n_envs

    if vae_run:
        net_arch = [32,32,32,32,32,32,32]
    else:
        net_arch = []
    policy_kwargs = dict(
                net_arch=net_arch,
                features_extractor_class = features_extractor,
            features_extractor_kwargs = {
            "features_dim" : 4,
            "model" : gw
            },
            )
    #changed vf_coef from 0.5 to 0.001, mini_batch_size from 3125 to 1024
    run_config = {
        "model": "PPO",
        "policy_type": "MlpPolicy",
        "total_timesteps": 20_000_000,
        "learning_rate": 1e-3,
        "n_steps": n_steps,
        "policy_kwargs": policy_kwargs,
        "num_epochs": 16,
        "mini_batch_size": 3125,
        "gamma": 0.99,
        "gae_lambda": 0.97,
        "vf_coef": 0.001,
        "max_grad_norm": 1.0,
        "ent_coef": 0.005,
        "normalize_advantage": False,
        "target_kl": None,
        "clip_range_vf": None,
        "init_policy_from_gw": init_gw,
        "vae_run": vae_run,
        #"eval_episodes": 5,
        #"resume_model_path": "tto21w7h/model.zip",
    }

    if not run_config.get("init_policy_from_gw", False):
        print("Not initializing policy from GW, setting gw to None for env wrapper.")
        gw = None

    # Optional overrides for quick smoke testing without changing the default config.
    if "PPO_TOTAL_TIMESTEPS" in os.environ:
        run_config["total_timesteps"] = int(os.environ["PPO_TOTAL_TIMESTEPS"])
    if "PPO_N_STEPS" in os.environ:
        run_config["n_steps"] = int(os.environ["PPO_N_STEPS"])
    

    seed = 0

    

    env_name = ["soccer-v3", "test-gw-env", "push-wall-v3","soccer-aff"]
    env_name = env_name[0]
    env = make_vec_env(gw, selection, seed=seed, num_goals=n_envs, max_episode_steps=400, env_name=env_name,vae_run=vae_run,input_mod="attr")

    resume_model_path = run_config.get("resume_model_path")
    if resume_model_path:
        run = wandb.init(
        project="dreamerv3",
        config=run_config,
        sync_tensorboard=True,
       #monitor_gym=True,
        save_code=True,
         )
        model_zip = resolve_model_zip(resume_model_path)
        print(f"Loading PPO checkpoint for continued training: {model_zip}")
        model = PPO.load(model_zip, env=env, device=device)
        # Preserve the current run's tensorboard destination.
        model.tensorboard_log = f"runs/{run.id}"
        run.log({"resumed_from_checkpoint": 1, "resume_model_path": model_zip})
    else:
        run = wandb.init(
        project="dreamerv3",
        config=run_config,
        sync_tensorboard=True,
       #monitor_gym=True,
        save_code=True,
         )
        model = PPO(
            policy=run.config["policy_type"],
            env=env,
            learning_rate=run.config["learning_rate"],
            policy_kwargs=policy_kwargs,
            n_steps=run.config["n_steps"],
            n_epochs=run.config["num_epochs"],
            batch_size=run.config["mini_batch_size"],
            gamma=run.config["gamma"],
            gae_lambda=run.config["gae_lambda"],
            vf_coef=run.config["vf_coef"],
            ent_coef=run.config["ent_coef"],
            normalize_advantage=run.config["normalize_advantage"],
            target_kl=run.config["target_kl"],
            clip_range_vf=run.config["clip_range_vf"],
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            device = device,
        )
        model.policy.action_net=nn.Sequential()


    if run.config.get("init_policy_from_gw", False) and not resume_model_path:
        if vae_run:
            print("VAE RUN: skipping GW layer loading into PPO policy.")
        else:
            print("GO")
            gw_weights = extract_gw_decoder_weights(gw, num_layers=4)
            loaded_layers, total_target_layers = init_policy_from_gw(model, gw_weights)
            print(f"Loaded pretrained GW layers into PPO policy: {loaded_layers}/{total_target_layers}")
            run.log(
                {
                    "pretrained_init_enabled": 1,
                    "pretrained_layers_available": len(gw_weights),
                    "pretrained_layers_loaded": loaded_layers,
                    "pretrained_target_layers": total_target_layers,
                }
            )

    # Run a deterministic evaluation rollout before any optimization starts.
    try:
        print("Running deterministic evaluation rollout before training...")
        eval_return, eval_success, eval_eps = evaluate_policy_deterministic(
            model,
            gw,
            selection,
            env_name,
            device,
            steps=1000,
            vae_run=vae_run,
            input_mod="attr",
            seed=seed,
        )
        print(f"Deterministic eval — episodes: {eval_eps}, mean_return: {eval_return}, success_rate: {eval_success}")
        try:
            run.log({
                "det_eval/mean_return": float(eval_return),
                "det_eval/success_rate": float(eval_success),
                "det_eval/episodes": int(eval_eps),
            })
        except Exception:
            # If W&B run is not initialized or logging fails, keep going.
            pass
    except Exception as e:
        print("Deterministic evaluation failed:", e)

    trainable_params = count_trainable_parameters(model.policy)
    print(f"Trainable parameters: {trainable_params}")
    run.log({"trainable_parameters": trainable_params})

    # Create callback list to log metrics per step
    callbacks = CallbackList([
        WandbCallback(
            model_save_freq=10_000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
        #SimpleEpisodeStatsCallback(verbose=1),
    ])

    model.learn(
        total_timesteps=run.config["total_timesteps"],
        tb_log_name="ppo",
        progress_bar=True,
        callback=callbacks,
    )


    run.finish()
