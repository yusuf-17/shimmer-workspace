from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import torch
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
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


def main():
    dataset_path = Path("/mnt/datashare/yelhelw/complex_dataset_V3/")
    (dataset_path / "domain_splits").mkdir(parents=True, exist_ok=True)
    actions_train = np.load(dataset_path / "actions_train.npy",allow_pickle=True)
    actions_val = np.load(dataset_path / "actions_val.npy",allow_pickle=True)

    attributes_train = np.load(dataset_path / "attributes_train.npy",allow_pickle=True)    
    attributes_val = np.load(dataset_path / "attributes_val.npy",allow_pickle=True)
    
    combin_index = np.load(dataset_path / "combin_task_index.npy",allow_pickle=True)
    combin_index[0] = np.where(attributes_train[:,-1]==2)[0][-1]+1
    combin_index[1] = np.where(attributes_val[:,-1]==2)[0][-1]+1

    #no soccer
    #combin_index[0] = np.where(attributes_train[:,-1]==0)[0][-1]+1
    #combin_index[1] = np.where(attributes_val[:,-1]==0)[0][-1]+1

    
    print(combin_index)
    full_dataset = [0,0]
    full_dataset[0] = len(attributes_train)
    full_dataset[1] = len(attributes_val)

    #just push wall
    #full_dataset[0] = np.where(attributes_train[:,-1]==0)[0][-1]+1
    #full_dataset[1] = np.where(attributes_val[:,-1]==0)[0][-1]+1

    print(full_dataset)
    domain_split_train = {
        frozenset({"v_latents","attr","act"}): np.array([i for i in range(combin_index[0])]),
        frozenset({"v_latents","attr"}): np.array([i for i in range(full_dataset[0])]),
    }
    domain_split_val = {
        frozenset({"v_latents","attr","act"}): np.array([i for i in range(combin_index[1])]),
        frozenset({"v_latents","attr"}): np.array([i for i in range(full_dataset[1])]),
    }

    np.save(dataset_path / "domain_splits/domain_train.npy", domain_split_train)
    np.save(dataset_path / "domain_splits/domain_val.npy", domain_split_val)
    print(domain_split_train,domain_split_val)
if __name__ == "__main__":
    main()
