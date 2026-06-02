from cProfile import label
from dataclasses import dataclass
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
import sys
import glob
import mujoco
from stable_baselines3.common.callbacks import CallbackList
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from gymnasium.wrappers import TimeLimit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../", "")))

import matplotlib.pyplot as plt
from shimmer_metaworld.modules.domains import load_pretrained_domains
from shimmer_metaworld.config import load_config
from shimmer_metaworld import DEBUG_MODE, PROJECT_DIR,LOGGER
from shimmer.modules.selection import FixedSharedSelection, RandomSelection

from shimmer.modules.selection import FixedSharedSelection, RandomSelection
from shimmer import ContrastiveLossType
from shimmer.modules.global_workspace import (
    GlobalWorkspaceFusion,
)
from shimmer_metaworld.callbacks import SimpleEpisodeStatsCallback
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from gymnasium.wrappers import StickyAction
# Add the parent folder containing 'metaworld' and import env and policy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Myworld/packages/", "metaworld")))
from metaworld.envs import SawyerSoccerEnvV3,SawyerPushWallEnvV3, SawyerWallEnvV3
from metaworld.envs.sawyer_push_wall_soccer import SawyerPushWallEnvV3 as SawyerTestTaskEnv
import metaworld

def count_trainable_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def make_env(rank, seed = 0, model=None, selection=None, monitor_dir=None, wrapper_class=None, monitor_kwargs=None, wrapper_kwargs=None):
    def _init():
        env = SawyerSoccer(render_mode='rgb_array',camera_name='corner4',width = 128, height=128)
        
        #test-gw-env basketball-v3
        env_name = "soccer-v3"
        ml1 = metaworld.MT50()
        task = [t for t in ml1.train_tasks if t.env_name == env_name][0]
        env.set_task(task)
        env = FuseGwWrapper(env, model, selection) #VAEWrapper(env, domain_modules) # FuseGwWrapper(env, model, selection) #  ImgWrapper(env)
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
class Args:
    task : str = "soccer-v3"

if __name__ == '__main__':
    args = tyro.cli(Args)
    task = args.task
    print(task)
    config = load_config(
        PROJECT_DIR / "shimmer_metaworld" / "config_template",
        load_files=["train_gw.yaml"],
        debug_mode=DEBUG_MODE,
    )
    seed = 0
    n_envs = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    three_mod_gw = {
        "gw_ckpt"   : "r8stcltz",
        "ckpt_epoch" : "34",
    }
    two_mod_gw = {
        "gw_ckpt"   : "8wfdyfub",
        "ckpt_epoch" : "255",
    }
    push_wall_gw = {
        "gw_ckpt"   : "ooodp88y",
        "ckpt_epoch" : "253",  }
    two_mod_gw = {
        "gw_ckpt"   : "klbnybq4",
        "ckpt_epoch" : "253",
    }
    #two_mod_gw = {
    #    "gw_ckpt"   : "5i7t9nbx",
    #    "ckpt_epoch" : "253",
    #}
    V1 = {
        "gw_ckpt"   : "60lfqbw3",
        "ckpt_epoch" : "249",
    }
    three_mod_gw = {
        "gw_ckpt"   : "n1pfpmrp",
        "ckpt_epoch" : "249",
    }
    #two_mod_gw = {
    #    "gw_ckpt"   : "2jbhhhjg",
     #   "ckpt_epoch" : "255",
    #}
    
    
    three_mod = True

    if three_mod:
        model = three_mod_gw
    else:
        model = two_mod_gw

    gw_ckpt = model["gw_ckpt"]
    ckpt_epoch = model["ckpt_epoch"]

    ckpt_path =  glob.glob(f'/mnt/datashare/yelhelw/checkpoints/shimmer-meta-{gw_ckpt}/*.ckpt')[0]
    print(ckpt_path)
    dataset = "/mnt/datashare/yelhelw/complex_dataset_V3/"
    gw = GlobalWorkspaceFusion.load_from_checkpoint(ckpt_path, domain_mods=domain_modules,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,weights_only=False)

    attributes = np.load(f"{dataset}/attributes_train.npy", mmap_mode="r")
    attributes_mean = np.mean(attributes, axis=0)
    attributes_std = np.std(attributes, axis=0)
    combin_index = np.load(f"{dataset}/combin_task_index.npy",allow_pickle=True)[0]
    #removed combin_index
    actions =  np.clip(np.load(f"{dataset}/actions_train.npy"), -10, 10)[:combin_index,:]
    actions_mean = np.mean(actions, axis=0)
    actions_std = np.std(actions, axis=0)

    '''
    plt.scatter(attributes[:len(actions),4], actions[:,2], c=actions[:,1], cmap='viridis')
    plt.colorbar(label='Action Value')
    plt.xlabel('Attribute 4')
    plt.ylabel('Actions 2')
    plt.title('Scatter Plot of Attributes Colored by Action Value')
    plt.savefig("attribute_action_scatter.png")
    '''
    

    gw.eval().freeze()
    
    selection = FixedSharedSelection()

    render = False
    if task == "soccer-v3":
        env = SawyerSoccerEnvV3(render_mode='rgb_array',camera_name='corner4',width = 240, height=240)
    elif task == "push-wall-v3":
        env = SawyerPushWallEnvV3(render_mode='rgb_array',camera_name='corner4',width = 240, height=240)
    elif task == "just-wall-v3":
        env = SawyerWallEnvV3(render_mode='rgb_array',camera_name='corner4',width = 240, height=240)
    else:
        env = SawyerTestTaskEnv(render_mode='rgb_array',camera_name='corner4',width = 240, height=240)
    env.max_path_length = 400
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    #env.model.geom("inner1").rgba = [1, 1, 1, 0] 
    #env.model.geom("outer1").rgba = [1, 1, 1, 0]
    #normal soccer ball
    #env.model.geom("outer").rgba = [0.85, 0.85, 0.85, 1]
    #env.model.geom("inner").rgba = [0.15, 0.15, 0.15, 1]
    #env.model.geom("wall").matid = 1
    #env = StickyAction(env, repeat_action_probability=0.9,repeat_action_duration=2)
    obs, _ = env.reset()
    
    episode_rewards = 0
    returns = []
    success = []
    step_count = []
    evaluation_steps = 100_000
    for step in range(evaluation_steps):
        if render == True:
            frame = env.render()
            frame = np.flipud(frame)  
            pil_frame = Image.fromarray(frame)
            image_path = f"soccer/{step}.png"
            pil_frame.save(image_path)
            #pil_frame = pil_frame.resize((128, 128), Image.BILINEAR) 
            #image = pil_frame.convert("RGB")
            #transform = transforms.ToTensor()
            #imgs_tensor = transform(image).to(device).unsqueeze(0)
        
        if task == "soccer-v3":
            wall = [0,0,-10]
            soccer_goal =  env.data.body("goal_whole").xpos.tolist()
        elif task == "push-wall-v3":
            wall = env.data.body("wall").xpos.tolist()
            soccer_goal =  [0,0,-10]
        else:
            wall = env.data.body("wall").xpos.tolist()
            soccer_goal =  env.data.body("goal_whole").xpos.tolist()
        
            
        #latent = {}
        #latent['v_latents'] = domain_modules['v_latents'].visual_module.encode(imgs_tensor)
        proprio = obs[:4].tolist()
        ball = obs[4:7].tolist()
        

        att = proprio+ball+wall+soccer_goal
        for x in range(len(att)):
            att[x] = (att[x] - attributes_mean[x]) / attributes_std[x]
        #latent['v_latents'] = domain_modules['v_latents'].visual_module.encode(imgs_tensor)
        gw_latent = gw.gw_mod.encode({"attr": torch.tensor(att).to(device, dtype=torch.float32).unsqueeze(0)})
        selection_scores = selection(gw_latent,gw_latent)
        gw_latent_v_fused = gw.gw_mod.fuse(
        gw_latent,
        selection_scores
        )
        decoded_latent_uni = gw.gw_mod.decode(gw_latent_v_fused)

        decoded_act = decoded_latent_uni["act"].squeeze(0).cpu().detach().numpy()
        for x in range(len(decoded_act)):
            decoded_act[x] = decoded_act[x] * actions_std[x] + actions_mean[x]
        obs, reward, done, truncate, info = env.step(decoded_act)
        episode_rewards += reward
    
        if info["success"] or done or truncate:
            success.append(info["success"])
            returns.append(episode_rewards)
            step_count.append(step)
            obs, _ = env.reset()
            episode_rewards = 0
            print("###########",step,"###########")
            print(np.mean(returns),np.mean(success))
    np.save(f"{task}_returns.npy", np.array(returns))
    np.save(f"{task}_success.npy", np.array(success))
    np.save(f"{task}_step_count.npy", np.array(step_count))

    plt.plot(step_count, np.mean(returns), label='Returns ' + task)
    plt.xlabel('Steps')
    plt.ylabel('Returns')  
    plt.title('Returns over Steps')
    plt.legend()
    plt.savefig(f"{task}_returns.png")

    plt.plot(step_count, np.mean(success), label='Success ' + task)
    plt.xlabel('Steps')
    plt.ylabel('Success')
    plt.title('Success over Steps')
    plt.legend()
    plt.savefig(f"{task}_success.png")

    print(f"Return: {np.mean(returns)}",np.mean(success))
    env.close()