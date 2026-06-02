

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_grid_from_v_tensor(
    samples: Sequence[torch.Tensor],
    _: int,
    ncols: int,
) -> Image:
    image = make_grid(samples[0], nrow=ncols, pad_value=1).detach()
    return F.to_pil_image(image)

import sys

obj = str(sys.argv[1])

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
    
#Load data
domain_classes = get_default_domains(["v_latents","attr"])
dataset_path = Path('/mnt/datashare/yelhelw/complex_dataset_V3/')
print(domain_classes)
print(config.domain_proportions)
data_module = MetaworldDataModule(
        dataset_path,
        domain_classes,
        config.domain_proportions,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
        ood_seed=config.ood_seed,
        domain_args=config.domain_data_args,
    )



modality_names = {0: 'Vision (v)', 1: 'Attributes (attr)', 2: 'Actions (act)'}


saved_latents = np.load("/mnt/datashare/yelhelw/complex_dataset_V3/saved_latents/val/domain_v.npy")
print(len(saved_latents))

modalities = ["v"]
keys = frozenset({"v_latents","attr"})
fuse = False
print()
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fusion_mech = FixedSharedSelection()

data_module.prepare_data()
data_module.setup()

data_module.val_dataset = {keys :data_module.val_dataset[keys]}
dataloaders = {
    #"train": data_module.train_dataloader(shuffle=False, drop_last=False),
    "val" : data_module.val_dataloader(),
}

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
        vae_latents_dict = {m: np.concatenate(v, axis=0) for m, v in vae_latents_dict.items()}
        #latent_vectors = np.concatenate([latents_dict[m] for m in latents_dict.keys()], axis=0)
        #train only on vision latents to resemble Kuske : 
        latent_vectors = latents_dict['v']
        vae_vectors = vae_latents_dict['v']
        shuffle_latent_vectors = latent_vectors.copy()
        np.random.shuffle(shuffle_latent_vectors)
    
print(latent_vectors.shape,vae_vectors.shape)


act_labels = np.load(dataset_path/'actions_val.npy', mmap_mode="r")

labels_norm = np.clip(act_labels.copy(), -1, 1)


attributes = np.load(dataset_path/'attributes_val.npy', mmap_mode="r").astype(int)
wall = attributes[:len(latent_vectors),9]
ball = attributes[:len(latent_vectors),5]
goal = attributes[:len(latent_vectors),12]

floor_color = attributes[:,-4]
wall_color = attributes[:,-3]
ball_color = attributes[:,-2]
wall_binary = np.where((wall==-10),0,1)
goal_binary = np.where((goal==-10),0,1)
ball_binary = np.where((ball==0),0,1)

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

obj_binary = ball_binary if obj == "ball" else wall_binary if obj == "wall" else goal_binary
labels = torch.from_numpy(obj_binary).long()

#task_filter = attributes[:len(latent_vectors),-1]==-1
#task_filter = (attributes[:len(latent_vectors),-1]<3) & (attributes[:len(latent_vectors),-1]>-1)
#task_filter = attributes[:len(latent_vectors),-1]>-5
task_filter = attributes[:len(latent_vectors),-1]==-1
print(len(task_filter))

latent_vectors, vae_vectors, labels = shuffle(latent_vectors[task_filter], vae_vectors[task_filter], labels[task_filter], random_state=42)

vae_probe = Probe(vae_vectors.shape[1]).to(device)
gw_probe = Probe(latent_vectors.shape[1]).to(device)

vae_probe.train(data_emb=torch.from_numpy(vae_vectors).to(device), labels=labels.to(device),
                num_epochs=5)
gw_probe.train(data_emb=torch.from_numpy(latent_vectors).to(device), labels=labels.to(device),
                num_epochs=5)

accuracy_vae = vae_probe.evaluate(data_emb=torch.from_numpy(vae_vectors).to(device), labels=labels.to(device))
accuracy_gw = gw_probe.evaluate(data_emb=torch.from_numpy(latent_vectors).to(device), labels=labels.to(device))
print(f"Accuracy VAE: {accuracy_vae}, Accuracy GW: {accuracy_gw}")
'''
from sklearn import svm
model = svm.SVC(kernel='linear')
import numpy as np

obj_binary = ball_binary
print('ball_binary')
points = np.concatenate([latent_vectors[(obj_binary == 1)],latent_vectors[(obj_binary == 0)]],axis=0)
#for act use labels_norm instead of vae_vectors
points_vae = np.concatenate([vae_vectors[(obj_binary == 1)],vae_vectors[(obj_binary == 0)]],axis=0)
labels = np.concatenate([np.ones(len(latent_vectors[(obj_binary == 1)])),np.zeros(len(latent_vectors[(obj_binary == 0)]))])
points, points_vae, labels = shuffle(points, points_vae, labels, random_state=42)
print("fitting model GW")
model.fit(points,labels)
print("fitting model VAE")
model_vae = svm.SVC(kernel='linear') 
model_vae.fit(points_vae,labels)

#embedding_points = np.concatenate([embedding_wall,embedding_no_wall])

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

y_pred = model.predict(points)
y_pred_vae = model_vae.predict(points_vae)
print(accuracy_score(labels, y_pred),accuracy_score(labels, y_pred_vae))
print(precision_score(labels, y_pred),precision_score(labels, y_pred_vae))
print(recall_score(labels, y_pred),recall_score(labels, y_pred_vae))
print(f1_score(labels, y_pred),f1_score(labels, y_pred_vae))
print(confusion_matrix(labels, y_pred),confusion_matrix(labels, y_pred_vae))


'''