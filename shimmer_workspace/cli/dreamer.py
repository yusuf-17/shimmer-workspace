import os
import sys
import absl.logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import numpy as np
from collections import deque
import robodesk

from gymnasium.wrappers import TimeLimit

from shimmer_robodesk import DEBUG_MODE
from shimmer_robodesk.modules.contrastive_loss import VSEPPContrastiveLoss
from shimmer_robodesk.modules.domains import load_pretrained_domains

from gw_dreamer.config import load_config
from gw_dreamer.utils.wrappers.robodesk_wrapper import FuseGwWrapper, RawObservationWrapper, DictWrapper, TerminatedWrapper, RawObsGW
from gw_dreamer.utils.wrappers.from_gym import DreamerV3Wrapper
from gw_dreamer.utils.utils import normalize_to_3d
from gw_dreamer import PROJECT_DIR
from gw_dreamer.dreamer import Dreamer, Agent
from gw_dreamer.utils.buffer import ReplayBuffer
from gw_dreamer.utils.replay import PyTorchReplayBuffer
from gw_dreamer.utils.logger import WandbLogger, LoggersContainer

from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.optimizer import Optimizer
from shimmer.modules.selection import FixedSharedSelection
from shimmer import ContrastiveLossType
from shimmer.modules.global_workspace import (
    GlobalWorkspaceFusion,
)
from torch import set_float32_matmul_precision
from tqdm import tqdm

absl.logging.set_verbosity(absl.logging.ERROR)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':

    set_float32_matmul_precision("medium")
    
    config = load_config(
        PROJECT_DIR / "config",
        load_files=["train_gw_dreamer_gw_robodesk.yaml"],
        debug_mode=DEBUG_MODE,
    )

    config_dreamer = config.dreamer

    loggers = []
    if config.wandb.enabled:
        loggers.append(
            WandbLogger(
                project_name="gw_policy_metaworld",
                config= config
            )
        )
    
    logger = LoggersContainer(loggers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domain_modules, gw_encoders, gw_decoders = load_pretrained_domains(
        config.shimmer.domains,
        config.shimmer.global_workspace.latent_dim,
        config.shimmer.global_workspace.encoders.hidden_dim,
        config.shimmer.global_workspace.encoders.n_layers,
        config.shimmer.global_workspace.decoders.hidden_dim,
        config.shimmer.global_workspace.decoders.n_layers,
        is_linear=config.shimmer.global_workspace.linear_domains,
        bias=config.shimmer.global_workspace.linear_domains_use_bias,
    )

    contrastive_fn: ContrastiveLossType | None = None
    if config.shimmer.global_workspace.vsepp_contrastive_loss:
        contrastive_fn = VSEPPContrastiveLoss(
            config.shimmer.global_workspace.vsepp_margin,
            config.shimmer.global_workspace.vsepp_measure,
            config.shimmer.global_workspace.vsepp_max_violation,
            torch.tensor([1 / 0.07]).log(),
        )


    def get_scheduler(optimizer: Optimizer) -> OneCycleLR:
        return OneCycleLR(
            optimizer,
            config.shimmer.training.optim.max_lr,
            config.shimmer.training.max_steps,
            pct_start=config.shimmer.training.optim.pct_start,
            div_factor=config.shimmer.training.optim.max_lr / config.shimmer.training.optim.start_lr,
            final_div_factor=config.shimmer.training.optim.max_lr
            / config.shimmer.training.optim.end_lr,
        )
    
    #TODO add checkpoint_path
    gw =GlobalWorkspaceFusion.load_from_checkpoint(
        config.gw.checkpoint_path,
        {name: module.eval() for name, module in domain_modules.items()},
        gw_encoders,
        gw_decoders,
    )

    buffer = PyTorchReplayBuffer(
        config_dreamer.training.sequence_length,
        config_dreamer.training.buffer_size,
        '/mnt/datashare/yelhelw/metaworld_agent/',
        device=device
    )

    selection = FixedSharedSelection() #RandomSelection(0.2)

    env = SawyerTestTaskEnvV3(render_mode='rgb_array',camera_name='corner4')
    env = RawObservationWrapper(env)
    env = FuseGwWrapper(env,conf=config.shimmer)
    env = TimeLimit(env, max_episode_steps=500)
    env = DreamerV3Wrapper(env)

    eval_env = SawyerTestTaskEnvV3(render_mode='rgb_array',camera_name='corner4')
    eval_env = TerminatedWrapper(eval_env)
    eval_env = RawObservationWrapper(eval_env)
    eval_env = FuseGwWrapper(eval_env,conf=config.shimmer)
    eval_env = TimeLimit(eval_env, max_episode_steps=500)
    eval_env = DreamerV3Wrapper(eval_env)

    model = Agent(gw, env.observation_space, env.action_space, config.dreamer).to(device)

    print(f"Total trainable parameters: {count_trainable_parameters(model)}")
    print(f"Total parameters: {count_parameters(model)}")

    nb_steps = 0        

    act_losses = []
    crit_losses = []
    avg_return = []
    hidden_states = []
    images = []
    best_avg_return = -float('inf')
    nb_env_steps = 0
    learning_step = 0
    train_return = 0
    train_len = 0
    nb_done = 0
    tot_return = 0
    success = 0
    best_return = -float('inf')

    pbar = tqdm(total=config_dreamer.training.nb_total_env_steps, desc="Training Progress")
    pbar_rdm = tqdm(total=config_dreamer.training.rdm_data_size, desc="Random steps")

    act = {'reset': True}
    while nb_env_steps < config_dreamer.training.nb_total_env_steps:
        
        #################################################################################
        #                                 RANDOM POLICY                                 #     
        #################################################################################

        if buffer.empty:
            nb_rdm_env_steps = 0
            carry = None
            
            while nb_rdm_env_steps < config_dreamer.training.rdm_data_size:
                
                observation, _ = env.step(act)
                '''
                if observation['is_first']:
                    carry = None
                '''
                observation = normalize_to_3d(observation)
                tensor_observation = {key: torch.from_numpy(value).to(device) for key, value in observation.items()}
                tensor_observation['attr'] = [tensor_observation['attr_pos'], tensor_observation['attr_vel']]
                
                act, carry = model(tensor_observation, carry, random=True)
                act = {name: value[0,0] for name, value in act.items()}

                data = {
                    'v': tensor_observation['image'].flatten(0, 2),
                    'attr': tensor_observation['attr'].flatten(),
                    'action': act['action'],
                    'reward': tensor_observation['reward'].flatten(),
                    'is_first': tensor_observation['is_first'].flatten(),
                    'is_terminal': tensor_observation['is_terminal'].flatten(),
                    'is_last': tensor_observation['is_last'].flatten(),
                    'deter': carry[0]['deter'].flatten().detach(),
                    'gw': carry[0]['gw'].flatten().detach(),
                }
                buffer.add(data)

                nb_rdm_env_steps += 1
                pbar_rdm.update(1)

            pbar_rdm.close()
            act = {'reset': True}

        #################################################################################
        #                                     TRAIN                                     #     
        #################################################################################

        eval = False
        should_log_img = (nb_done % config_dreamer.training.log_img_every) == 0

        observation, info = env.step(act)
        if observation['is_first']:
            carry = None
            if nb_env_steps > 0:
                logger.log({"train/return_one_ep": train_return}, step = nb_env_steps)
                logger.log({"train/len_one_ep": train_len}, step = nb_env_steps)
                logger.log({"train/success_count": success}, step = nb_env_steps)

                if train_return.item() > best_avg_return:
                    best_avg_return = train_return
                    print('SAVE DIR: ', f'ckpt/best_dreamer_{logger.loggers[0].run.id}.pth')
                    torch.save(
                        {
                            'epoch': nb_env_steps,
                            'model_state_dict': model.state_dict(),
                            'dreamer_optimizer_state_dict': model.optimizer['dreamer'].state_dict(),
                            'gw_optimizer_state_dict': model.optimizer['gw'].state_dict(),
                            'loss': train_return,
                        }, 
                        f'ckpt/best_dreamer_{logger.loggers[0].run.id}.pth'
                    )

            train_return = 0
            train_len = 0
            success = 0
        else:
            train_len += 1

        observation = normalize_to_3d(observation)
        tensor_observation = {key: torch.from_numpy(value).to(device) for key, value in observation.items()}

        act, carry = model(tensor_observation, carry)
        act = {name: value[0,0] for name, value in act.items()}
        nb_env_steps += 1
        pbar.update(1)

        train_return += observation['reward']
        nb_done += observation['is_last']
        success += info['success']
        
        data = {
            'v': tensor_observation['image'].flatten(0, 2),
            'attr': tensor_observation['attr'].flatten(),
            'action': act['action'],
            'reward': tensor_observation['reward'].flatten(),
            'is_first': tensor_observation['is_first'].flatten(),
            'is_terminal': tensor_observation['is_terminal'].flatten(),
            'is_last': tensor_observation['is_last'].flatten(),
            'deter': carry[0]['deter'].flatten().detach(),
            'gw': carry[0]['gw'].flatten().detach(),
        }
        buffer.add(data)

        # Train Dreamer
        if nb_env_steps%config_dreamer.training.n_env_steps_btwn_train == 0:
            for i in range(config.dreamer.training.nb_agt_steps):
                for j in range(9): #9 + 1
                    data = buffer.sample(
                        batch_size=config_dreamer.training.batch_size,
                    )
                    data['attr'] = [data['attr_pos'], data['attr_vel']]
                    gw_loss, metrics = model.training_gw(data)
                    model.optimizer['gw'].zero_grad()
                    gw_loss.backward()
                    model.optimizer['gw'].step()
                    model.schedulers['gw'].step()


                # data shape (T,B,D)
                data = buffer.sample(
                    batch_size=config_dreamer.training.batch_size,
                )
                data['attr'] = [data['attr_pos'], data['attr_vel']]
                metrics = model.train_step(data)
                learning_step += 1

                for name, loss in metrics.items():
                    logger.log({name: loss}, step = nb_env_steps)
        
        logger.log({"nb_env_steps": nb_env_steps}, step = nb_env_steps)
        logger.log({"learning_step": learning_step}, step = nb_env_steps)

        if should_log_img:
            hidden_states.append(carry[0]['gw'])
            images.append(info['raw_obs'])

            if observation['is_last']:

                gw_lat = torch.stack(hidden_states)[:,0,0,:]
                uni_img = gw.gw_mod.gw_decoders.v_latents(gw_lat)
                imag_img = (domain_modules['v_latents'].visual_module.vae.decoder(uni_img) * 255.)
                imag_img = imag_img.detach().cpu().numpy().astype(np.uint8)

                images = np.array(images).transpose(0, 3, 1, 2)

                logger.log_videos("train/policy", images, step = nb_env_steps)
                logger.log_videos("train/imagination", imag_img, step = nb_env_steps)

                hidden_states = []
                images = []
        
        ##################################################################################
        #                                   EVALUATION                                   #     
        ##################################################################################

        if nb_env_steps%config_dreamer.training.eval_every==0:

            model.eval()
    
            with torch.no_grad():

                    eval_tot_return = 0
                    tot_len = 0
                    eval_images = []
                    eval_hidden_states = []

                    eval_done = False
                    eval_carry = None

                    act = {'reset': True}

                    while not eval_done:

                        eval_observation, info = eval_env.step(act)
                        if observation['is_last']:
                            carry = None
                        

                        eval_observation = normalize_to_3d(eval_observation)
                        tensor_eval_observation = {key: torch.from_numpy(value).to(device) for key, value in eval_observation.items()}
                        tensor_eval_observation['attr'] = [tensor_eval_observation['attr_pos'], tensor_eval_observation['attr_vel']]
                    
                        act, eval_carry = model(tensor_eval_observation, eval_carry)
                        act = {name: value[0,0] for name, value in act.items()}

                        eval_tot_return += eval_observation['reward'] 
                        
                        eval_done = eval_observation['is_last'] 

                        eval_hidden_states.append(eval_carry[0]['gw'])
                        eval_images.append(info['raw_obs'])

                        tot_len += 1
                
                    gw_lat = torch.stack(eval_hidden_states)[:,0,0,:]
                    uni_img = gw.gw_mod.gw_decoders.v_latents(gw_lat)
                    imag_img = (domain_modules['v_latents'].visual_module.vae.decoder(uni_img) * 255.)
                    imag_img = imag_img.cpu().numpy().astype(np.uint8)

                    eval_images = np.array(eval_images).transpose(0, 3, 1, 2)

                    diff = np.abs(imag_img - eval_images)

                    logger.log({"eval/return": eval_tot_return}, step = nb_env_steps)
                    logger.log({"eval/ep_len": tot_len}, step = nb_env_steps)
                    logger.log_videos("eval_policy", eval_images, step = nb_env_steps)
                    logger.log_videos("eval_imagination", imag_img, step = nb_env_steps)
                    logger.log_videos("eval_diff", diff, step = nb_env_steps)

            model.train()
            gw.eval()

    pbar.close()  
    logger.finish()