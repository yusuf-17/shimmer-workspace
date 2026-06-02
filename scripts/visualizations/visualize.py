from collections.abc import Sequence
from typing import cast
import umap
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
import torchvision
from PIL import Image
from torchvision.utils import make_grid

import sys,os
from pathlib import Path
sys.path.append(str(Path(os.getcwd()).resolve().parent.parent))

from shimmer.modules.global_workspace import (
    GlobalWorkspaceFusion,
)

from shimmer_metaworld import DEBUG_MODE, PROJECT_DIR,LOGGER
from shimmer_metaworld.config import load_config
from shimmer_metaworld.logging import get_pil_image, batch_to_device
from metaworld_dataset import (
    MetaworldDataModule,
    DomainDesc,
    get_default_domains,
)
from shimmer_metaworld.modules.domains import load_pretrained_domains
from collections.abc import Mapping
from shimmer.modules.selection import FixedSharedSelection
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import numpy as np

matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_grid_from_v_tensor(
    samples: Sequence[torch.Tensor],
    _: int,
    ncols: int,
) -> Image:
    image = make_grid(samples[0], nrow=ncols, pad_value=1).detach()
    return F.to_pil_image(image)


def to_device(data: torch.Tensor | Mapping[str, torch.Tensor] | list, device: str):
    """Put the data Tensor or list on the device (GPU or CPU)"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [value.to(device) for value in data]
    elif isinstance(data, Mapping):
        return {name: to_device(value, device) for name, value in data.items()}
    else:
        raise TypeError(f"Unsupported type: {type(data)}")

def get_latent_vectors(domain_modules,domain_module,data_module,):
    modalities = ["v","attr","act"]
    keys = frozenset({"v_latents","attr","act"})
    fuse = True

    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fusion_mech = FixedSharedSelection()

    data_module.prepare_data()
    data_module.setup()

    data_module.val_dataset = {keys :data_module.val_dataset[keys]}

    dataloaders = {
        "val" : data_module.val_dataloader(),
        #"train": data_module.train_dataloader(shuffle=False, drop_last=False),
    }

    for split, dataloader in dataloaders.items():
        latents: list[np.ndarray] = []
        latents_dict : dict = {m: [] for m in modalities}
        vae_latents_dict : dict = {m: [] for m in modalities}
        print(f"Saving {split}.")
        for batch, _, _ in tqdm(iter(dataloader), total=len(dataloader)):
            data = to_device(batch[keys], device)
    
            latent = {}
            for modality in modalities:
                if modality == 'v':
                    latent['v_latents'] = domain_modules['v_latents'].encode(data['v_latents'])
                    vae_latents_dict['v'].append(latent['v_latents'].detach().cpu().numpy().copy())
                else:
                    latent[modality] = domain_modules[modality].encode(data[modality])
                    vae_latents_dict[modality].append(latent[modality].detach().cpu().numpy().copy())
            else:
                latent = domain_module.gw_mod.encode(latent)
                if fuse:
                    selection_scores = fusion_mech(latent, latent)
                    latent_fuse = domain_module.gw_mod.fuse(latent, selection_scores)
                    latents.append(latent_fuse.detach().cpu().numpy().copy())
                
                    latent['v'] = latent.pop('v_latents')
                    #tanh is applied here to the modality vectors to make them correspond to the fused vector in latents
                    #that applies an activation function at the end
                    latents_dict = {m: latents_dict[m] + [torch.tanh(latent[m]).detach().cpu().numpy()] for m in latents_dict.keys() if m in latent}

                else:
                    latent['v'] = latent.pop('v_latents')
                    latents_dict = {m: latents_dict[m] + [latent[m].detach().cpu().numpy().copy()] for m in latents_dict.keys() if m in latent} 
        if len(latents) > 0:
            latent_vectors = np.concatenate(latents, axis=0)
            shuffle_latent_vectors = latent_vectors.copy()
            np.random.shuffle(shuffle_latent_vectors)
            latents_dict = {m: np.concatenate(v, axis=0) for m, v in latents_dict.items()}
            vae_latents_dict = {m: np.concatenate(v, axis=0) for m, v in vae_latents_dict.items()}
        else:
            latents_dict = {m: np.concatenate(v, axis=0) for m, v in latents_dict.items()}
            '''
            if using train dataset to limit to 40000 samples
            for m in latents_dict.keys():
                latents_dict[m] = latents_dict[m][:40000]
            '''
            vae_latents_dict = {m: np.concatenate(v, axis=0) for m, v in vae_latents_dict.items()}
            latent_vectors = np.concatenate([latents_dict[m] for m in latents_dict.keys()], axis=0)
            shuffle_latent_vectors = latent_vectors.copy()
            np.random.shuffle(shuffle_latent_vectors)
        print(shuffle_latent_vectors.shape)
        return latents_dict, latent_vectors,shuffle_latent_vectors, vae_latents_dict

def get_umap_plot(embedding_mod,embedding_all,labels,model_name):
    modality_cmap = ListedColormap(['red', 'green', 'orange'])
    all_embeddings = [embedding_all] + list(embedding_mod.values())
    x_min = min([emb[:, 0].min() for emb in all_embeddings])
    x_max = max([emb[:, 0].max() for emb in all_embeddings])
    y_min = min([emb[:, 1].min() for emb in all_embeddings])
    y_max = max([emb[:, 1].max() for emb in all_embeddings])

    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = 0.05
    x_limits = [x_min - padding * x_range, x_max + padding * x_range]
    y_limits = [y_min - padding * y_range, y_max + padding * y_range]

    actions = ['right-left', 'front-back', 'up-down', 'gripper']
    n_actions = len(actions)
    n_modalities = len(modalities)

    n_cols = n_modalities 
    n_rows = n_actions + 1

    fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(6 * n_cols + 2, 4 * n_rows))
    plt.subplots_adjust(hspace=0.3, wspace=0.3, right=0.92)

    ###################################
    # Top row: Modality plot centered #
    ###################################
    #axes[0, 0].remove()
    #axes[0, 2].remove()
    index = 0
    for modality in modalities:
        # GW embeddings
        modality_ax = axes[0, modalities.index(modality)]
        modality_ax.set_title(f'UMAP - {modality} in GW', fontsize=16, fontweight='bold')
        scatter = modality_ax.scatter(
            embedding_mod[modality][:, 0],
            embedding_mod[modality][:, 1],
            s=5,
        )
        modality_ax.set_xlim(x_limits)
        modality_ax.set_ylim(y_limits)

    modality_legend_elements = []
    for i, color in enumerate(modality_cmap.colors):
        modality_legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=8, 
                                        label=f'{modality_names[i]}'))

    axes[0, -1].axis('off')

    plt.suptitle('UMAP Visualization - Modalities and Attributes', fontsize=20, fontweight='bold', y=0.98)
    
    actions = ['right-left', 'front-back', 'up-down', 'gripper']
    x_disp = np.clip(labels[:, 0],-1,1)
    y_disp = np.clip(labels[:, 1],-1,1)
    z_disp = np.clip(labels[:, 2],-1,1)
    gripper = np.clip(labels[:, 3],-1,1)
    ###################################

    x_mouvement = ((x_disp<0.15)|(x_disp>0.30))
    x_stop = ((x_disp>0.15)|(x_disp<0.30))

    x_disp_binary = x_disp[x_mouvement]
    right_left_labels = np.where(x_disp_binary<0.15, 1,0)

    y_disp_binary = y_disp[(y_disp > -0.5)]
    front_back_labels = np.where((y_disp > -0.25), 1,0)
    binary_map = ListedColormap(['red', 'green'])

    for row, action in enumerate(actions):
        actual_row = row + 1
        
        for col, modality in enumerate(modalities):
            ax = axes[actual_row, col]
            embedding = embedding_mod[modality]
            if action == 'right-left':
                scatter = ax.scatter(
                    embedding[:, 0][x_mouvement],
                    embedding[:, 1][x_mouvement],
                    c=right_left_labels,
                    cmap=binary_map,
                    s=5,
                    alpha=0.9
                )
                ax.set_title(f'{modality.upper()} - Right-Left')
                
                if col == n_modalities - 1:
                    axes[actual_row, -1].axis('off')
                    pos = axes[actual_row, -1].get_position()
                    cbar_height = max(0.02, pos.height * 0.6) 
                    cbar_y = pos.y0 + (pos.height - cbar_height) / 2 
                    cbar_ax = fig.add_axes([pos.x0 + 0.02, cbar_y, 0.02, cbar_height])
                    cbar = plt.colorbar(scatter, cax=cbar_ax)
                
            elif action == 'front-back':
                scatter = ax.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    c=y_disp,
                    cmap='plasma',
                    s=5,
                    alpha=0.9
                )
                ax.set_title(f'{modality.upper()} -Front-Back')
                
                if col == n_modalities - 1:
                    axes[actual_row, -1].axis('off')
                    pos = axes[actual_row, -1].get_position()
                    cbar_height = max(0.02, pos.height * 0.6) 
                    cbar_y = pos.y0 + (pos.height - cbar_height) / 2
                    cbar_ax = fig.add_axes([pos.x0 + 0.02, cbar_y, 0.02, cbar_height])
                    cbar = plt.colorbar(scatter, cax=cbar_ax)
                
            elif action == 'up-down':
                scatter = ax.scatter(
                    embedding[:, 0][(gripper>0.2)&(x_disp<0)],
                    embedding[:, 1][(gripper>0.2)&(x_disp<0)],
                    c=gripper[(gripper>0.2)&(x_disp<0)],
                    cmap='plasma',
                    s=5,
                    alpha=0.9
                )
                ax.set_title(f'{modality.upper()} - Down & Grip')
                
                if col == n_modalities - 1:
                    axes[actual_row, -1].axis('off')
                    pos = axes[actual_row, -1].get_position()
                    cbar_height = max(0.02, pos.height * 0.6) 
                    cbar_y = pos.y0 + (pos.height - cbar_height) / 2
                    cbar_ax = fig.add_axes([pos.x0 + 0.02, cbar_y, 0.02, cbar_height])
                    cbar = plt.colorbar(scatter, cax=cbar_ax)
                
            elif action == 'gripper':
                scatter = ax.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    c=gripper,
                    cmap='plasma',
                    s=5,
                    alpha=0.7
                )
                ax.set_title(f'{modality.upper()} - Gripper')
                
                if col == n_modalities - 1:
                    axes[actual_row, -1].axis('off')
                    pos = axes[actual_row, -1].get_position()
                    cbar_height = max(0.02, pos.height * 0.6) 
                    cbar_y = pos.y0 + (pos.height - cbar_height) / 2
                    cbar_ax = fig.add_axes([pos.x0 + 0.02, cbar_y, 0.02, cbar_height])
                    cbar = plt.colorbar(scatter, cax=cbar_ax)
                
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
    plt.suptitle('UMAP Visualization - Modalities and Attributes', fontsize=20, fontweight='bold', y=0.98)
    
    plt.savefig(f"graphs/sweep/UMAP_{model_name}.png")
    plt.close()

models = [{
    "name": "z=12_fused",
    "gw_ckpt" : "jnvu796k",
    "ckpt_epoch" : "1016"
},
'''
{
    "name": "z_6_1.0",
    "gw_ckpt" : "ftn3bu69",
    "ckpt_epoch" : "193"
}
'''
]       

debug_mode = DEBUG_MODE
extra_config_files = ["train_gw.yaml"]
argv = []

LOGGER.debug(f"Debug mode: {debug_mode}")

config = load_config(
    PROJECT_DIR / "shimmer_metaworld"/ "config_template",
    load_files=extra_config_files,
    debug_mode=debug_mode,
    log_config=False,
    argv=argv,
)

#seed_everything(config.seed, workers=True)

domain_classes = get_default_domains(
    {domain.domain_type.kind.value for domain in config.domains}
)
print(config.domains)
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

#Load data
domain_classes = get_default_domains(["v_latents","attr","act"])
modalities = ["v","attr","act"]

data_module = MetaworldDataModule(
        '/mnt/datashare/yelhelw/dataset',
        domain_classes,
        config.domain_proportions,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
        ood_seed=config.ood_seed,
        domain_args=config.domain_data_args,
    )

for model_info in models:
    current_model = model_info["name"]
    gw_ckpt = model_info["gw_ckpt"]
    ckpt_epoch = model_info["ckpt_epoch"]

    ckpt_path = f'/mnt/datashare/yelhelw/checkpoints/shimmer-meta-{gw_ckpt}/epoch={ckpt_epoch}.ckpt'

    domain_module = GlobalWorkspaceFusion.load_from_checkpoint(ckpt_path, domain_mods=domain_modules,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders)
    domain_module.eval().freeze()
    domain_module.to(device)

    print(f"Processing model: {current_model}")
    latents_dict, latent_vectors, shuffle_latent_vectors, vae_latents_dict = get_latent_vectors(domain_modules,domain_module,data_module)
    
    print("Starting UMAP fitting...")
    reducer = umap.UMAP(n_neighbors=15,min_dist=0.1,metric="cosine",metric_kwds=None).fit(shuffle_latent_vectors)
    print("UMAP fitting done.")

    #get action labels
    labels = np.load('/mnt/datashare/yelhelw/dataset/actions_val.npy', mmap_mode="r")
    all_labels = np.repeat(labels, 3, axis=0)

    n_samples_per_modality = len(labels)
    modality_labels = []
    for i, modality in enumerate(modalities):
        modality_labels.extend([i] * n_samples_per_modality)
    modality_labels = np.array(modality_labels)

    modality_names = {0: 'Vision (v)', 1: 'Attributes (attr)', 2: 'Actions (act)'}
    x_disp = np.clip(labels[:, 0],-1,1)
    y_disp = np.clip(labels[:, 1],-1,1)
    z_disp = np.clip(labels[:, 2],-1,1)
    gripper = np.clip(labels[:, 3],-1,1)

    print("Starting UMAP transformation...")
    embedding_all = reducer.transform(latent_vectors)


    #Embeddings per modality
    embedding_mod_gw = {}
    embedding_mod_vae = {}

    for m in modalities:
        embedding = reducer.transform(latents_dict[m])
        embedding_mod_gw[m] = embedding

    get_umap_plot( embedding_mod_gw,embedding_all,labels,current_model)

    print(f"{current_model} GW UMAP done.")




