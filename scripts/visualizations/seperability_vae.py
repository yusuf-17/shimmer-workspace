

import numpy as np
from collections.abc import Mapping
from shimmer.modules.selection import FixedSharedSelection
from matplotlib.colors import ListedColormap
import torch
from torch import nn
from tqdm import tqdm
import glob
from scipy.stats import gaussian_kde
from collections.abc import Sequence
from typing import cast
from sklearn.utils import shuffle
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision.utils import make_grid

import sys,os
from pathlib import Path
sys.path.append(str(Path(os.getcwd()).resolve().parent.parent))

from shimmer.modules.global_workspace import (
    GlobalWorkspaceFusion,
)
import logging
from shimmer_metaworld import DEBUG_MODE, PROJECT_DIR,LOGGER
from shimmer_metaworld.config import DomainModuleVariant, load_config
from shimmer_metaworld.logging import get_pil_image, batch_to_device
from metaworld_dataset import (
    MetaworldDataModule,
    DomainDesc,
    get_default_domains,
)
from shimmer_metaworld.modules.domains.pretrained import load_pretrained_module
from shimmer_metaworld.modules.domains import load_pretrained_domains
from shimmer_metaworld.modules.domains.visual import VisualDomainModule

logger = logging.getLogger(__name__)
logging.basicConfig(filename='myapp.log', level=logging.INFO)
logger.info('Started')

matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_grid_from_v_tensor(
    samples: Sequence[torch.Tensor],
    _: int,
    ncols: int,
) -> Image:
    image = make_grid(samples[0], nrow=ncols, pad_value=1).detach()
    return F.to_pil_image(image)

import sys

subset = str(sys.argv[1]) if len(sys.argv) > 1 else "none"
print(subset)


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
    
class Probe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.flatten = nn.Flatten()
        # CrossEntropyLoss expects raw logits, so keep the probe linear.
        self.probe = nn.Linear(in_dim, num_classes)

    def fit(
        self,
        data_emb: torch.Tensor,
        labels: torch.Tensor,
        num_epochs: int,
        learning_rate: float = 0.001,
        batch_size: int = 32,
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        logger.info('Training the probe..')

        for epoch in range(num_epochs):
            self.train()
            permutation = torch.randperm(data_emb.size(0), device=data_emb.device)
            #logger.info(f'Epoch {epoch}')
            for i in range(0, data_emb.size(0), batch_size):
                idx = permutation[i:i + batch_size]
                batch_emb = data_emb[idx]
                batch_labels = labels[idx]

                logits = self.probe(batch_emb)
                loss = criterion(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        logger.info("Done")

    def evaluate(self, data_emb: torch.Tensor, labels: torch.Tensor, batch_size: int = 32) -> float:
        self.eval()
        correct = 0
        total = 0
        for i in range(0, data_emb.size(0), batch_size):
            logger.info(f"Evaluating batch {i}//{len(data_emb)/batch_size}")
            batch_emb = data_emb[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            with torch.no_grad():
                logits = self.probe(batch_emb)
            predicted = torch.argmax(logits, dim=-1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.numel()

        accuracy = correct / max(total, 1)
        logger.info(f"Probe accuracy: {accuracy:.4f}")
        return accuracy

import pickle

config = load_config(
        PROJECT_DIR / "shimmer_metaworld/config_template",
        load_files=["save_v_latents.yaml"],
        debug_mode=DEBUG_MODE,
        argv = []
    )
beta = 1
dataset_path = Path("/mnt/datashare/yelhelw/complex_dataset_V3")
labels = np.load(f"{dataset_path}/actions_val.npy", mmap_mode="r")
data_module = MetaworldDataModule(
    os.path.abspath(dataset_path),
    get_default_domains(["v"]),
    {frozenset(["v"]): 1.0},
    batch_size=config.training.batch_size,
    num_workers=config.training.num_workers,
    seed=config.seed,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
domain_checkpoint = None
for domain in config.domains:
    print(domain.domain_type)
    if domain.domain_type == DomainModuleVariant.v:
        domain_checkpoint = domain

assert (
    domain_checkpoint is not None
), "Please add domain_checkpoint entry in the configuration"
assert domain_checkpoint.domain_type == DomainModuleVariant.v
print(domain_checkpoint.checkpoint_path)
visual_domain = cast(
    VisualDomainModule,
    load_pretrained_module(domain_checkpoint),
)
visual_domain.to(device)
visual_domain.freeze()

data_module.prepare_data()
data_module.setup()

dataloaders = {
    "val": data_module.val_dataloader(),
}
for split, dataloader in dataloaders.items():
    latents: list[np.ndarray] = []

    print(f"Saving {split}.")
    for batch, _, _ in tqdm(iter(dataloader), total=len(dataloader)):
        if split == "train":
            images = batch[frozenset(["v"])]["v"].to(device)
        else:
            images = batch[frozenset(["v"])]["v"].to(device)
        latent = visual_domain.encode(images)
        latents.append(latent.detach().cpu().numpy())
    latent_vectors = np.concatenate(latents, axis=0)[:len(labels)]
    print(" concatenation done")
    shuffle_indices = np.random.permutation(latent_vectors.shape[0])
    shuffle_latent_vectors = latent_vectors[shuffle_indices]
    print("done")
for i in range(16):
    kde = gaussian_kde(latent_vectors[i])
    x = np.linspace(min(latent_vectors[i]),max(latent_vectors[i]),500)
    y = kde(x)
    plt.plot(x,y,linewidth=2)
plt.savefig("latents_0.05.png")
from collections.abc import Mapping
from shimmer.modules.selection import FixedSharedSelection
from matplotlib.colors import ListedColormap
from tqdm import tqdm


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

np.save("latent_vectors_val.npy", latent_vectors)
labels = np.load(f"{dataset_path}/actions_val.npy", mmap_mode="r")
labels_train = np.clip(np.load(f"{dataset_path}/actions_train.npy", mmap_mode="r"), -10, 10)
all_labels = np.repeat(labels, 3, axis=0)


attributes = np.load(f"{dataset_path}/attributes_val.npy", mmap_mode="r")[:len(labels)]
if subset == "random":
    probe_task_filter = (attributes[:, -1] == -1)
elif subset == "wall":
    probe_task_filter = np.isin(attributes[:, -1], [0, 1])
else:
    probe_task_filter = (labels[:, -1] > -100)
    probe_task_filter = np.isin(attributes[:, -1], [0, 1, 2])

np.save("attributes_val.npy", attributes)
latent_vectors = latent_vectors[probe_task_filter]
attributes = attributes[probe_task_filter]
labels = labels[probe_task_filter]

wall = attributes[:, 9]
ball = attributes[:, 6]
goal = attributes[:, 12]
modality_names = {0: 'Vision (v)', 1: 'Attributes (attr)', 2: 'Actions (act)'}


labels_norm = np.clip(labels.copy(), -1, 1)

#labels_norm = labels.copy()
#action_mean = np.mean(labels_train,axis=0)
#action_stdv = np.std(labels_train,axis=0)
#for x in range(4):
#    labels_norm[:,x] = (labels_norm[:,x]-action_mean[x])/ action_stdv[x]
#    labels_norm[:,x] = np.clip(labels_norm[:,x],-1,1)
x_disp = labels_norm[:, 0]
y_disp = labels_norm[:, 1]
z_disp = labels_norm[:, 2]
gripper = labels_norm[:, 3]
act_shuffle = np.array(labels_norm.copy())
np.random.shuffle(act_shuffle)
wall_binary = np.where((wall == -10), 0, 1)
x_disp_binary = np.where((x_disp<0),0,1)
goal_binary = np.where((goal == -10), 0, 1)
ball_binary = np.where((ball == -10), 0, 1)

# Create obj_code with different combinations
obj_code = np.zeros(len(wall_binary), dtype=object)
obj_code[:] = "none"


# wall_ball - wall AND ball (goal False)
mask_wall_ball = (wall_binary == 1) & (ball_binary == 1) & (goal_binary == 0)
obj_code[mask_wall_ball] = "wall_ball"

# wall - only wall
mask_wall_only = (wall_binary == 1) & (goal_binary == 0) & (ball_binary == 0)
obj_code[mask_wall_only] = "wall"
print(mask_wall_only)
# goal_ball - goal AND ball (wall False)
mask_goal_ball = (goal_binary == 1) & (ball_binary == 1) & (wall_binary == 0)
obj_code[mask_goal_ball] = "goal_ball"

color_map = {
    #"none": '#1f77b4',
    "all": '#ff7f0e',
    "wall_ball": '#2ca02c',
    "wall": '#d62728',
    "goal_ball": '#9467bd'
}


from sklearn.utils import shuffle
import logging 
logger = logging.getLogger(__name__)
logging.basicConfig(filename='myapp.log', level=logging.INFO)
logger.info('Started')

noise_std = 0  # adjust strength of noise
ball_color = attributes[:, -4].copy()
ball_present = ball_binary > -1

# Evaluate ball color only when a ball exists to avoid mixing "no ball" with a color class.
probe_latents = latent_vectors[ball_present]
probe_color_raw = ball_color[ball_present].astype(np.int64)
probe_color_raw = ball_binary
unique_colors, probe_color_labels = np.unique(probe_color_raw, return_inverse=True)
logger.info(f"Ball color classes in probe: {unique_colors.tolist()}")
logger.info(f"Ball color class counts: {np.bincount(probe_color_labels).tolist()}")

probe_labels_t = torch.from_numpy(probe_color_labels).long().to(device)
vae_probe = Probe(latent_vectors.shape[1], num_classes=len(unique_colors)).to(device)

vae_vectors_t = torch.from_numpy(probe_latents).float().to(device)
if noise_std > 0:
    vae_vectors_t = vae_vectors_t + torch.randn_like(vae_vectors_t) * noise_std

vae_probe.fit(data_emb=vae_vectors_t, labels=probe_labels_t, num_epochs=20)

accuracy_vae = vae_probe.evaluate(data_emb=vae_vectors_t, labels=probe_labels_t)


print(f"Accuracy VAE: {accuracy_vae}")

import umap
print("Fitting UMAP reducer...")
reducer = umap.UMAP(n_neighbors=70,min_dist=0.4,metric="cosine",metric_kwds=None).fit(shuffle_latent_vectors)
print("Transforming latent vectors...")
embedding_v = reducer.transform(latent_vectors)

embedding_v = embedding_v

modalities = ['v']
modality_cmap = ListedColormap(['red', 'green', 'orange'])
all_embeddings = [embedding_v] 
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
n_rows = n_actions + 3

fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(6 * n_cols + 2, 4 * n_rows))
plt.subplots_adjust(hspace=0.3, wspace=0.3, right=0.92)

###################################
# Top row: Modality plot centered #
###################################
#axes[0, 0].remove()
#axes[0, 2].remove()

axes[0, -1].axis('off')



actions = ['right-left', 'front-back',  'gripper','goal','ball','objects']
n_actions = len(actions)
n_modalities = len(modalities)
print(attributes[(attributes[:,-1]==0),6])
###################################
# Randomize plotting order to avoid points covering each other
###################################
np.random.seed(42)
shuffle_idx = np.random.permutation(len(x_disp))
wall_shuf = wall[shuffle_idx]
goal_shuf = goal[shuffle_idx]
ball_shuf = ball[shuffle_idx]
# Create shuffled versions of all data
x_disp_shuf = x_disp[shuffle_idx]
y_disp_shuf = y_disp[shuffle_idx]
z_disp_shuf = z_disp[shuffle_idx]
gripper_shuf = gripper[shuffle_idx]
modalities = ['v']

color_map = {
    #"none": '#1f77b4',
    #"all": '#ff7f0e',
    "wall_ball": '#2ca02c',
    "wall": '#d62728',
    "goal_ball": '#9467bd'
}

# Shuffle embeddings
#embedding_v_shuf = embedding_v[shuffle_idx]
#embedding_act_shuf = embedding_act[shuffle_idx]
###################################
# Define masks on shuffled data
###################################
x_mouvement = ((x_disp_shuf==1)|(x_disp_shuf==-1))
x_stop = (np.abs(x_disp_shuf)<0.1)

x_disp_binary = x_disp_shuf[x_mouvement]
right_left_labels = np.where(x_disp_binary<0, 1,0)

y_filter = (y_disp_shuf>=-0.1)
y_filtered= y_disp_shuf[y_filter]
front_back_labels = np.where((np.abs(y_filtered)<0.2), 0,1)

#wall_binary = np.where((wall_shuf==-10),0,1)
goal_binary = np.where((goal_shuf==-10),0,1)
#ball_binary = np.where((ball_shuf<0),0,1)
binary_map = ListedColormap(['red', 'green'])


for row, action in enumerate(actions):
    actual_row = row 

    ax = axes[actual_row, 0]

    embedding = embedding_v[shuffle_idx]
    if action == 'right-left':
            ax.scatter(
                embedding_v[:, 0],
                embedding_v[:, 1],
                c='grey',
                s=5,
                alpha=0.3
            )
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=x_disp_shuf,
                cmap='plasma',
                s=5,
                alpha=0.9
            )
            ax.set_title(f' Right-Left')

            axes[actual_row, -1].axis('off')
            pos = axes[actual_row, -1].get_position()
            cbar_height = max(0.02, pos.height * 0.6) 
            cbar_y = pos.y0 + (pos.height - cbar_height) / 2 
            cbar_ax = fig.add_axes([pos.x0 + 0.02, cbar_y, 0.02, cbar_height])
            cbar = plt.colorbar(scatter, cax=cbar_ax)
            
    elif action == 'front-back':
        ax.scatter(
            embedding_v[:, 0],
            embedding_v[:, 1],
            c='grey',
            s=5,
            alpha=0.3
        )
        scatter = ax.scatter(
            embedding[y_filter, 0],
            embedding[y_filter, 1],
            c=y_disp_shuf[y_filter],
            cmap='plasma',
            s=5,
            alpha=0.9
        )
        ax.set_title(f'Front-Stop')

        axes[actual_row, -1].axis('off')
        pos = axes[actual_row, -1].get_position()
        cbar_height = max(0.02, pos.height * 0.6) 
        cbar_y = pos.y0 + (pos.height - cbar_height) / 2
        cbar_ax = fig.add_axes([pos.x0 + 0.02, cbar_y, 0.02, cbar_height])
        cbar = plt.colorbar(scatter, cax=cbar_ax)
    elif action == 'gripper':
        ax.scatter(
            embedding_v[:, 0],
            embedding_v[:, 1],
            c='grey',
            s=5,
            alpha=0.3
        )
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=gripper_shuf,
            cmap='plasma',
            s=5,
            alpha=0.7
        )
        ax.set_title(f'Gripper')
        

        axes[actual_row, -1].axis('off')
        pos = axes[actual_row, -1].get_position()
        cbar_height = max(0.02, pos.height * 0.6) 
        cbar_y = pos.y0 + (pos.height - cbar_height) / 2
        cbar_ax = fig.add_axes([pos.x0 + 0.02, cbar_y, 0.02, cbar_height])
        cbar = plt.colorbar(scatter, cax=cbar_ax)
    elif action == 'goal':
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=goal_binary[:],
            cmap=binary_map,
            s=5,
            alpha=0.9
        )
        ax.set_title(f'Goal')

        axes[actual_row, -1].axis('off')
        pos = axes[actual_row, -1].get_position()
        cbar_height = max(0.02, pos.height * 0.6) 
        cbar_y = pos.y0 + (pos.height - cbar_height) / 2
        cbar_ax = fig.add_axes([pos.x0 + 0.02, cbar_y, 0.02, cbar_height])
        cbar = plt.colorbar(scatter, cax=cbar_ax)
    elif action == 'ball':
        scatter = ax.scatter(
            embedding_v[:, 0],
            embedding_v[:, 1],
            c=ball_color,
            cmap='plasma',
            s=5,
            alpha=0.7
        )
        ax.set_title(f'Ball color')
        

        axes[actual_row, -1].axis('off')
        pos = axes[actual_row, -1].get_position()
        cbar_height = max(0.02, pos.height * 0.6) 
        cbar_y = pos.y0 + (pos.height - cbar_height) / 2
        cbar_ax = fig.add_axes([pos.x0 + 0.02, cbar_y, 0.02, cbar_height])
        cbar = plt.colorbar(scatter, cax=cbar_ax)
    
    elif action == 'objects':
        idx = np.random.permutation(len(embedding_v))

        ax.scatter(
            embedding_v[idx, 0],
            embedding_v[idx, 1],
            c=[color_map[label] for label in obj_code[idx]],
            alpha=0.8,
            s=5,
        )
        '''
        for obj_label, color in color_map.items():
            mask = obj_code == obj_label
            ax.scatter(embedding_v[mask, 0],
            embedding_v[mask, 1],
            c=color,
            alpha=0.8,
            s=5,
            label=obj_label,
            )
        '''
        ax.set_title(f'Objects')

        axes[actual_row, -1].axis('off')
        pos = axes[actual_row, -1].get_position()
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=obj_label)
                    for obj_label, color in color_map.items()]
        axes[actual_row, -1].legend(handles=handles, loc='center left', fontsize=9)
        
        
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

plt.suptitle('UMAP Visualization - Modalities and Attributes', fontsize=20, fontweight='bold', y=0.98)
plt.savefig(f"graphs/vae/complex_V3/UMAP_V3_shuffled.png")
plt.close()