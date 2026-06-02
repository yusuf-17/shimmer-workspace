from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import torch
import sys
sys.path.append(str(Path(os.getcwd()).resolve().parent.parent))
from metaworld_dataset import (
    MetaworldDataModule,
    color_blind_visual_domain,
    get_default_domains,
)
from tqdm import tqdm

from shimmer_metaworld import DEBUG_MODE, PROJECT_DIR
from shimmer_metaworld.config import DomainModuleVariant, load_config
from shimmer_metaworld.modules.domains.pretrained import load_pretrained_module
from shimmer_metaworld.modules.domains.visual import VisualDomainModule
from collections.abc import Mapping
from shimmer.modules.selection import FixedSharedSelection
from matplotlib.colors import ListedColormap
import umap


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


import pickle

config = load_config(
        PROJECT_DIR / "shimmer_metaworld/config_template",
        load_files=["save_v_latents.yaml"],
        debug_mode=DEBUG_MODE,
    )
beta = 1
additional_transforms: dict[str, list[Callable[[Any], Any]]] = {}
if config.domain_modules.visual.color_blind:
    additional_transforms["v"] = [color_blind_visual_domain]

data_module = MetaworldDataModule(
    os.path.abspath('/mnt/datashare/yelhelw/complex_dataset/'),
    get_default_domains(["v"]),
    {frozenset(["v"]): 1.0},
    batch_size=config.training.batch_size,
    num_workers=config.training.num_workers,
    seed=config.seed,
    additional_transforms=additional_transforms,
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

    latent_vectors = np.concatenate(latents, axis=0)
    shuffle_latent_vectors = latent_vectors.copy()
    np.random.shuffle(shuffle_latent_vectors)

print(len(latent_vectors))

from umap.umap_ import nearest_neighbors
import pickle



knn = nearest_neighbors(shuffle_latent_vectors,
                              n_neighbors=100,
                              metric="cosine",
                              metric_kwds=None,
                              angular=True,
                              random_state=None,
                             )

n_neighbors = [5, 25,50, 100]
min_dists = [0, 0.2, 0.5, 0.9]
embeddings = np.zeros((4, 4, 40065, 2))
for i, k in enumerate(n_neighbors):
    for j, dist in enumerate(min_dists):
        print(k,dist)
        reducer = umap.UMAP(n_neighbors=k,
                                                      min_dist=dist,
                                                      precomputed_knn=knn
                                                      ).fit(shuffle_latent_vectors)
                                                      
        print("reducer done")
        embeddings[i, j] = reducer.transform(latent_vectors)


import matplotlib.pyplot as plt
labels = np.load('/mnt/datashare/yelhelw/complex_dataset/actions_val.npy', mmap_mode="r")
labels_train = np.clip(np.load('/mnt/datashare/yelhelw/complex_dataset/actions_train.npy', mmap_mode="r"), -10, 10)
all_labels = np.repeat(labels, 3, axis=0)
attributes = np.load('/mnt/datashare/yelhelw/complex_dataset/attributes_val.npy', mmap_mode="r")
wall = attributes[:,9]
ball = attributes[:,6]
goal = attributes[:,12]

print(ball)
modality_names = {0: 'Vision (v)', 1: 'Attributes (attr)', 2: 'Actions (act)'}


labels_norm = labels.copy()

#labels_norm = labels.copy()
action_mean = np.mean(labels_train,axis=0)
action_stdv = np.std(labels_train,axis=0)
for x in range(4):
    labels_norm[:,x] = (labels_norm[:,x]-action_mean[x])/ action_stdv[x]
#    labels_norm[:,x] = np.clip(labels_norm[:,x],-1,1)
x_disp = labels_norm[:, 0]
y_disp = labels_norm[:, 1]
z_disp = labels_norm[:, 2]
gripper = labels_norm[:, 3]

wall_binary = np.where((wall==-10),0,1)
goal_binary = np.where((goal==-10),0,1)
ball_binary = np.where((ball<0),0,1)

# Create obj_code with different combinations
obj_code = np.zeros(len(wall_binary), dtype=object)
obj_code[:] = "none"

# all - wall AND goal AND ball
mask_all = (wall_binary == 1) & (goal_binary == 1) & (ball_binary == 1)
obj_code[mask_all] = "all"

# wall_ball - wall AND ball (goal False)
mask_wall_ball = (wall_binary == 1) & (ball_binary == 1) & (goal_binary == 0)
obj_code[mask_wall_ball] = "wall_ball"

# wall - only wall
mask_wall_only = (wall_binary == 1) & (goal_binary == 0) & (ball_binary == 0)
obj_code[mask_wall_only] = "wall"

# goal_ball - goal AND ball (wall False)
mask_goal_ball = (goal_binary == 1) & (ball_binary == 1) & (wall_binary == 0)
obj_code[mask_goal_ball] = "goal_ball"

# Create color mapping for obj_codes
color_map = {
    "none": '#1f77b4',
    "all": '#ff7f0e',
    "wall_ball": '#2ca02c',
    "wall": '#d62728',
    "goal_ball": '#9467bd'
}

fig, axs = plt.subplots(4, 4, figsize=(20, 20))
for i, ax_row in enumerate(axs):
    for j, ax in enumerate(ax_row):
        # Plot each category separately to enable legend
        for obj_label, color in color_map.items():
            mask = obj_code == obj_label
            ax.scatter(embeddings[i, j, mask, 0],
                       embeddings[i, j, mask, 1],
                       c=color,
                       alpha=0.8,
                       s=1,
                       label=obj_label,
                       )
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_title("min_dist = {}".format(min_dists[j]), size=15)
        if j == 0:
            ax.set_ylabel("n_neighbors = {}".format(n_neighbors[i]), size=15)

# Add legend to the figure
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=12)

fig.suptitle("UMAP embedding of vae latents with grid of parameters", y=0.92, size=20)
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig("graphs/vae/UMAP_sweep.png")
plt.close()