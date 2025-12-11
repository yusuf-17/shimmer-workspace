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
from shimmer_metaworld.modules.domains.attribute import AttributeDomainModule


def main():
    config = load_config(
        PROJECT_DIR / "shimmer_metaworld/config_template",
        load_files=["save_v_latents.yaml"],
        debug_mode=DEBUG_MODE,
    )
    beta = 0.15
    additional_transforms: dict[str, list[Callable[[Any], Any]]] = {}
    if config.domain_modules.visual.color_blind:
        additional_transforms["v"] = [color_blind_visual_domain]

    data_module = MetaworldDataModule(
        os.path.abspath('/mnt/datashare/yelhelw/'),
        get_default_domains(["attr"]),
        {frozenset(["attr"]): 1.0},
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
        additional_transforms=additional_transforms,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain_checkpoint = None
    for domain in config.domains:
        print(domain.domain_type)
        if domain.domain_type == DomainModuleVariant.attr:
            domain_checkpoint = domain

    assert (
        domain_checkpoint is not None
    ), "Please add domain_checkpoint entry in the configuration"
    assert domain_checkpoint.domain_type == DomainModuleVariant.attr

    attr_domain = cast(
        AttributeDomainModule,
        load_pretrained_module(domain_checkpoint),
    )
    attr_domain.to(device)
    attr_domain.freeze()

    data_module.prepare_data()
    data_module.setup()

    dataloaders = {
        "train": data_module.train_dataloader(shuffle=False, drop_last=False),
        "val": data_module.val_dataloader(),
    }

    for split, dataloader in dataloaders.items():
        latents: list[np.ndarray] = []

        print(f"Saving {split}.")
        for batch, _, _ in tqdm(iter(dataloader), total=len(dataloader)):
            if split == "train":
                images = batch[frozenset(["attr"])]["attr"].to(device)
            else:
                images = batch[frozenset(["attr"])]["attr"].to(device)
            latent = attr_domain.encode(images.float())
            latents.append(latent.detach().cpu().numpy())

        latent_vectors = np.concatenate(latents, axis=0)

        presaved_path = config.domain_data_args["attr_latents"]["presaved_path"]
        Path(f"{config.dataset.path}/saved_latents/{split}/").mkdir(
            parents=True, exist_ok=True
        )
        path = config.dataset.path / f"saved_latents/{split}/{presaved_path}"
        print(f"Saving in {path}.")
        np.save(path, latent_vectors)

        if split == 'train':
            figure = plt.figure()
            for i in range(16):
                kde = gaussian_kde(latent_vectors[i])
                x = np.linspace(min(latent_vectors[i]),max(latent_vectors[i]),500)
                y = kde(x)
                plt.plot(x,y,linewidth=2)
        plt.savefig(f"latents_{beta}.png")


if __name__ == "__main__":
    main()
