

from dataclasses import dataclass

import numpy as np
from collections.abc import Mapping
from shimmer.modules.selection import FixedSharedSelection
from matplotlib.colors import ListedColormap
import torch
from torch import nn
from tqdm import tqdm
import glob
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

import tyro
sys.path.append(str(Path(os.getcwd()).resolve().parent.parent))

from shimmer.modules.global_workspace import (
    GlobalWorkspaceFusion,
)
import logging
from shimmer_metaworld import DEBUG_MODE, PROJECT_DIR,LOGGER
from shimmer_metaworld.config import load_config
from shimmer_metaworld.logging import get_pil_image, batch_to_device
from metaworld_dataset import (
    MetaworldDataModule,
    DomainDesc,
    get_default_domains,
)
from shimmer_metaworld.modules.domains import load_pretrained_domains
logger = logging.getLogger(__name__)
logging.basicConfig(filename='myapp.log', level=logging.INFO)
logger.info('Started')

matplotlib.use("Agg")

@dataclass(frozen=True)
class Args:
    subset : str = "none"
    test : str = "ball"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_grid_from_v_tensor(
    samples: Sequence[torch.Tensor],
    _: int,
    ncols: int,
) -> Image:
    image = make_grid(samples[0], nrow=ncols, pad_value=1).detach()
    return F.to_pil_image(image)

import sys


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
    
import torch
from torch import nn
    
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


complex_first_chosen = {
    "gw_ckpt" : "d5izsqr0",
    "ckpt_epoch" : "255"
}
complex_new_b2 = {
    "gw_ckpt" : "60lfqbw3",
    "ckpt_epoch" : "255"
}
complex_no_cont = {
    "gw_ckpt" : "umoy6xcx",
    "ckpt_epoch" : "255"
}
complex_clip = {
    "gw_ckpt" : "aiduy2li",
    "ckpt_epoch" : "1252"
}
complex_low_cont = {
    "gw_ckpt" : "6qaa4b7l",
    "ckpt_epoch" : "1252"
}
complex_high_cont = {
    "gw_ckpt" : "a361gio5",
    "ckpt_epoch" : "1252"
}

complex_V3 = {
    "gw_ckpt" : "8wfdyfub",
    "ckpt_epoch" : "255"
}
complex_V3_CLIP = {
    "gw_ckpt" : "o5xymig4",
    "ckpt_epoch" : "255"
}
complex_V3_no_cont = {
    "gw_ckpt" : "9o76tnap",
    "ckpt_epoch" : "250"
}
complex_V3_no_cont_full_cycle = {
    "gw_ckpt" : "dx395548",
    "ckpt_epoch" : "250"
}
complex_V3_05_cont = {
    "gw_ckpt" : "6kfz3nyp",
    "ckpt_epoch" : "250"
}
complex_V3_01_cont = {
    "gw_ckpt" : "0g6qx9t1",
    "ckpt_epoch" : "250"
}
complex_V3_08_cont = {
    "gw_ckpt" : "6jxxuroa",
    "ckpt_epoch" : "250"
}
complex_V3_06_cont = {
    "gw_ckpt" : "5617nld8",
    "ckpt_epoch" : "250"
}
complex_V3_055_cont = {
    "gw_ckpt" : "mae3igrw",
    "ckpt_epoch" : "250"
}
complex_V3_09_cont = {
    "gw_ckpt" : "bc05btrg",
    "ckpt_epoch" : "250"
}

complex_V3_b_05 = {
    "gw_ckpt" : "hduzszl0",
    "ckpt_epoch" : "250"
}
current_model = "act"

complex_V3__CLIP_b_05 = {
    "gw_ckpt" : "k49kadci",
    "ckpt_epoch" : "250"
}
complex_V3__05_b_05 = {
    "gw_ckpt" : "lgftgymm",
    "ckpt_epoch" : "250"
}
complex_V3__01_b_05 = {
    "gw_ckpt" : "9rm4sgdn",
    "ckpt_epoch" : "250"
}
complex_V3__0_b_05 = {
    "gw_ckpt" : "ijcpsgvw",
    "ckpt_epoch" : "250"
}
complex_V3__01_b_05 = {
    "gw_ckpt" : "tn6u8sk1",
    "ckpt_epoch" : "250"
}
complex_V3__16_b_05 = {
    "gw_ckpt" : "axdz8rfo",
    "ckpt_epoch" : "250"
}

complex_V3_b_0 = {
    "gw_ckpt" : "nloje1e5",
    "ckpt_epoch" : "250"
}
complex_V3_CLIP_b_0 = {
    "gw_ckpt" : "vbifoy5g",
    "ckpt_epoch" : "250"
}
complex_V3_no_cont_b_0 = {
    "gw_ckpt" : "nxbrkewn",
    "ckpt_epoch" : "250"
}

complex_V3_no_cont_b_01 = {
    "gw_ckpt" : "if3g0rv1",
    "ckpt_epoch" : "250"
}

complex_V3_b_01 = {
    "gw_ckpt" : "juwynyqw",
    "ckpt_epoch" : "250"
}

complex_V3_CLIP_b_01 = {
    "gw_ckpt" : "c8izir6m",
    "ckpt_epoch" : "250"
}


complex_V3_b15 = {
    "gw_ckpt" : "uvqpvcof",
    "ckpt_epoch" : "250"
}

complex_V3_CLIP_b15 = {
    "gw_ckpt" : "fjbhgrnw",
    "ckpt_epoch" : "250"
}

if current_model == "act":
    gw_ckpt = complex_V3_CLIP_b15["gw_ckpt"]
    ckpt_epoch = complex_V3_09_cont["ckpt_epoch"]

args = tyro.cli(Args)
subset = args.subset
test = args.test

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
print(gw_ckpt)
#seed_everything(config.seed, workers=True)

domain_classes = get_default_domains(
    {domain.domain_type.kind.value for domain in config.domains}
)

domain_modules, gw_encoders, gw_decoders = load_pretrained_domains(
    config.domains,
    12,
    config.global_workspace.encoders.hidden_dim,
    config.global_workspace.encoders.n_layers,
    config.global_workspace.decoders.hidden_dim,
    config.global_workspace.decoders.n_layers,
    is_linear=config.global_workspace.linear_domains,
    bias=config.global_workspace.linear_domains_use_bias,
)

print(config.domains)

ckpt_path = glob.glob(f'/mnt/datashare/yelhelw/checkpoints/shimmer-meta-{gw_ckpt}/*.ckpt')[0]

domain_module = GlobalWorkspaceFusion.load_from_checkpoint(ckpt_path, domain_mods=domain_modules,
    gw_encoders=gw_encoders,
    gw_decoders=gw_decoders,weights_only=False)
domain_module.eval().freeze()

domain_module.to(device)





#Load data
domain_classes = get_default_domains(["v"])
dataset_path = Path("/mnt/datashare/yelhelw/complex_dataset_V3")
domain_proportions = {frozenset(["v"]): 1.0}
print(domain_classes)

data_module = MetaworldDataModule(
        dataset_path,
        domain_classes,
        domain_proportions,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
        ood_seed=config.ood_seed,
        domain_args=config.domain_data_args,
    )



modality_names = {0: 'Vision (v)', 1: 'Attributes (attr)', 2: 'Actions (act)'}


#saved_latents = np.load("/mnt/datashare/yelhelw/complex_dataset_V3/saved_latents/val/domain_v.npy")
#print(len(saved_latents))

modalities = ["v"]
keys = frozenset({"v"})
fuse = False
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fusion_mech = FixedSharedSelection()

data_module.prepare_data()
data_module.setup()
print(data_module.val_dataset)
data_module.val_dataset = {keys :data_module.val_dataset[keys]}
dataloaders = {
    #"train": data_module.train_dataloader(shuffle=False, drop_last=False),
    "val" : data_module.val_dataloader(),
}
vision_only = False
for split, dataloader in dataloaders.items():
    latents: list[np.ndarray] = []
    latents_dict : dict = {m: [] for m in modalities}
    vae_latents_dict : dict = {m: [] for m in modalities}
    print(f"Saving {split}.")
    for batch, _, _ in tqdm(iter(dataloader), total=len(dataloader)):
        data = to_device(batch[keys], device)
   
        latent = {}
        vae_latent = {}
        for modality in modalities:
            if modality == 'v':
                latent['v_latents'] = domain_modules['v_latents'].visual_module.encode(data['v'])
                #latent['v_latents'] = data['v_latents']
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
        vae_latents_dict = {m: np.concatenate(v, axis=0) for m, v in vae_latents_dict.items()}
        #train only on vision latents to resemble Kuske : 
        if vision_only:
            latent_vectors = latents_dict['v']
        else:
            latent_vectors = np.concatenate([latents_dict[m] for m in latents_dict.keys()], axis=0)

        vae_vectors = vae_latents_dict['v']
        shuffle_latent_vectors = latent_vectors.copy()
        np.random.shuffle(shuffle_latent_vectors)
    
print(latent_vectors.shape,vae_vectors.shape)


labels = np.load(f"{dataset_path}/actions_val.npy", mmap_mode="r")
labels_train = np.clip(np.load(f"{dataset_path}/actions_train.npy", mmap_mode="r"), -10, 10)
all_labels = np.repeat(labels, 3, axis=0)

attributes = np.load(f"{dataset_path}/attributes_val.npy", mmap_mode="r")



#subset = "none" 
if subset == "random":
    probe_task_filter = (attributes[:, -1] == -1)
    ball = attributes[:,5]
elif subset == "wall":
    #probe_task_filter = (wall!=-10)
    probe_task_filter = np.isin(attributes[:, -1], [0, 1])
elif subset == "none":
    probe_task_filter = (attributes[:, -1] != -100)
    #probe_task_filter = np.isin(attributes[:, -1], [0, 1, 2])
else:
    probe_task_filter = (attributes[:, -1])
vae_vectors = vae_vectors[probe_task_filter]
latent_vectors = latent_vectors[probe_task_filter]
labels = labels[probe_task_filter]
attributes = attributes[probe_task_filter]

wall = attributes[:,9]
ball = attributes[:,6]
goal = attributes[:,12]

modality_names = {0: 'Vision (v)', 1: 'Attributes (attr)', 2: 'Actions (act)'}

labels_norm = np.clip(labels.copy(), -1, 1)

x_disp = labels_norm[:, 0]
y_disp = labels_norm[:, 1]
z_disp = labels_norm[:, 2]
gripper = np.clip(labels_norm[:, 3], 0, 1)

print("Transforming latent vectors...")
#act_shuffle = np.array(labels_norm.copy())
#np.random.shuffle(act_shuffle)

wall_binary = np.where((wall==-10),0,1)
print(np.where(wall==-10)[0])
goal_binary = np.where((goal==-10),0,1)
ball_binary = np.where((ball==-10),0,1)



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
(
print(f"Object code distribution: {np.unique(obj_code, return_counts=True)}"))
color_map = {
    "none": '#1f77b4',
    "all": '#ff7f0e',
    "wall_ball": '#2ca02c',
    "wall": '#d62728',
    "goal_ball": '#9467bd'
}


ball_color = attributes[:, -4].copy()
ball_present = ball_binary > -1

# Evaluate ball color only when a ball exists to avoid mixing "no ball" with a color class.
probe_latents = latent_vectors[ball_present]
probe_color_raw = ball_color[ball_present].astype(np.int64)
if test == "ball":
    probe_color_raw = ball_binary
elif test == "task":
    probe_color_raw = attributes[:, -1].astype(np.int64)
unique_colors, probe_color_labels = np.unique(probe_color_raw, return_inverse=True)
logger.info(f"Ball color classes in probe: {unique_colors.tolist()}")
logger.info(f"Ball color class counts: {np.bincount(probe_color_labels).tolist()}")

probe_labels_t = torch.from_numpy(probe_color_labels).long().to(device)
vae_probe = Probe(vae_vectors.shape[1], num_classes=len(unique_colors)).to(device)

vae_vectors_t = torch.from_numpy(vae_vectors).float().to(device)
vae_probe.fit(data_emb=vae_vectors_t, labels=probe_labels_t, num_epochs=20)
gw_probe = Probe(latent_vectors.shape[1], num_classes=len(unique_colors)).to(device)

gw_probe.fit(data_emb=torch.from_numpy(latent_vectors).to(device), labels=probe_labels_t,
                num_epochs=20)

accuracy_vae = vae_probe.evaluate(data_emb=vae_vectors_t, labels=probe_labels_t)
accuracy_gw = gw_probe.evaluate(data_emb=torch.from_numpy(latent_vectors).to(device), labels=probe_labels_t)
print(f"Accuracy VAE: {accuracy_vae}, Accuracy GW: {accuracy_gw}")

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
n_rows = n_actions + 2

fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(6 * n_cols + 2, 4 * n_rows))
plt.subplots_adjust(hspace=0.3, wspace=0.3, right=0.92)

###################################
# Top row: Modality plot centered #
###################################
#axes[0, 0].remove()
#axes[0, 2].remove()

axes[0, -1].axis('off')



actions = ['right-left', 'front-back',  'gripper','balls','objects']
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
    elif action == 'balls':
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
        ax.set_title(f'Objects')
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
        ax.set_title(f'Objects')
        '''

        axes[actual_row, -1].axis('off')
        pos = axes[actual_row, -1].get_position()
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=obj_label)
                    for obj_label, color in color_map.items()]
        axes[actual_row, -1].legend(handles=handles, loc='center left', fontsize=9)
        
        
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

plt.suptitle('UMAP Visualization - Modalities and Attributes', fontsize=20, fontweight='bold', y=0.98)
plt.savefig(f"graphs/UMAP_complex_V3_b15_CLIP_all.png")
plt.close()