from stable_baselines3 import PPO
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
from PIL import Image
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from gymnasium.wrappers import TimeLimit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../", "")))
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Myworld/packages/", "metaworld")))
from metaworld.envs import SawyerTestTaskEnv
import metaworld
from metaworld.policies.sawyer_push_wall_v3_policy import SawyerPushWallV3Policy


def PushWall(
        save_path : Path,
        n_observations : int = 0,
        camera_name : str = 'corner4',
        max_path_length : int = 0,
        random_selection : bool = True,
        save_images : bool = True,

):

    #TASK ENVIRONEMENT
    env = SawyerTestTaskEnv(render_mode='rgb_array',camera_name='corner4',width = 128, height=128)
    
    #define max steps before terminating
    if not max_path_length == 0:
        SawyerTestTaskEnv.max_path_length = max_path_length
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    #reset environment
    obs, info = env.reset()


    model_path = "/home/yelhelw/metaworld_GW/shimmer-metaworld/models/2hbd488c/model.zip"
    model = PPO.load(model_path, env=env)
    model.policy.eval()
    
    policy = SawyerPushWallV3Policy()
    print(f"Model loaded successfully!")
    print(f"Policy architecture: {model.policy}")
    #run variables
    ep = 0
    observations = 0
    
    #record goal coordinates: there seems to be a bug where the goal coordinates change


    att_total = []
    act_total = []
    while observations < n_observations:
        print(f"______{observations}___")
        
        obs, info = env.reset()

        goal = obs[-3:]
        
        done = False
        success = 0
        step = 0
        total_reward = 0

        visual = []
        actions = []
        att = []
        
        while not done:
            #intermediate variables for attributes, dimension 18:36 of obs are about past obs, -3: always goal, those don't interest us
            proprio = obs[:4].tolist()
            ball = obs[4:7].tolist()
            wall = env.data.body("wall").xpos.tolist()
            soccer_goal = [0,0,-10]   #placeholder for compatibility with other tasks
            task = 0 #push_wall task

            
            frame = env.render()
            # Predict actions
            action, _states = model.predict(obs, deterministic=True) 

            visual.append(frame)
            action = policy.get_action(obs)
            #apply chosen action
            obs, reward, terminate, truncate, info = env.step(action)
            obs[-3:] = goal
            
        
            #tracking variables
            step += 1
            total_reward += reward
            success = info.get("is_success")
            done = terminate or truncate or info.get("is_success")
            if step > 400:
                print(info.get("obj_to_target"))
            if step % 50 == 0:
                print(f"Episode {ep}, Step {step}, Reward so far: {total_reward:.2f}, Success :{info.get('is_success')}")

        if random_selection == True:
            #random selection of steps for dataset
            num_samples = int(0.25 * len(visual))  # 25% of steps
            selected_frames = np.random.choice(range(len(visual)), size=num_samples, replace=False)
        else:
            selected_frames = range(len(visual))
            num_samples = len(visual)
        
        if save_images == True:
            for index,i in enumerate(selected_frames):
                frame = np.flipud(visual[i])
                image_path = save_path/f"vision/wall_{observations+index:06d}.png"
                # Resize to 128x128
                pil_frame = Image.fromarray(frame)
                pil_frame = pil_frame.resize((128, 128), Image.BILINEAR)
                pil_frame.save(image_path)
        else :
            np.save(save_path/f"expert_vision_{ep}.npy", np.array(visual))
    
        observations += num_samples
        ep += 1
        
    env.close()
    return att_total, act_total, observations

   
if __name__ == "__main__":
    save_path = Path("./NewDataset/episodes")
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path/"vision").mkdir(parents=True, exist_ok=True)

    n_observations = 1000  # Total number of observations to collect
    camera_name = 'corner4'

    att, act, observations = PushWall(
        save_path=save_path,
        n_observations=n_observations,
        camera_name=camera_name,
        random_selection = False,
        save_images= True,
    )

    # Save attributes and actions as numpy arrays
    np.save(save_path/"attributes.npy", np.array(att))
    np.save(save_path/"actions.npy", np.array(act))

    print(f"Dataset collection complete. Total observations: {observations}")

