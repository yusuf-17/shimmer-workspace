from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers.wandb import WandbLogger
from migrate_ckpt.migrate import get_folder_migrations
import wandb
import sys, os, shutil
# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import numpy as np

from shimmer_metaworld.modules.domains.visual import VisualDomainModule
from metaworld_dataset import (
    MetaworldDataModule,
    color_blind_visual_domain,
    get_default_domains,
)

from PIL import Image
import torchvision.transforms as transforms

from shimmer_metaworld import DEBUG_MODE, LOGGER, PROJECT_DIR
from shimmer_metaworld.ckpt_migrations import SaveMigrations
from shimmer_metaworld.config import load_config
from shimmer_metaworld.logging import LogVisualCallback



def train_visual_domain(
    config_path: Path,
    debug_mode: bool | None = None,
    log_config: bool = False,
    extra_config_files: list[str] | None = None,
    argv: list[str] | None = None,
):
    if debug_mode is None:
        debug_mode = DEBUG_MODE
    if extra_config_files is None:
        extra_config_files = ["train_v.yaml"]
    if argv is None:
        argv = []

    LOGGER.debug(f"Debug mode: {debug_mode}")

    config = load_config(
        config_path,
        load_files=extra_config_files,
        debug_mode=debug_mode,
        log_config=log_config,
        argv=argv,
    )


    pl.seed_everything(config.seed, workers=True)

    #additional_transforms: dict[str, list[Callable[[Any], Any]]] = {}
    #if config.domain_modules.visual.color_blind:
    #    LOGGER.info("v domain will be color blind.")
    #    additional_transforms["v"] = [color_blind_visual_domain]

    data_module = MetaworldDataModule(
        os.path.abspath('/mnt/datashare/yelhelw/expert_frames_final/'),
        get_default_domains(["v"]),
        {frozenset(["v"]): 1.0},
        batch_size=128,
        num_workers=8,
    )


    #val_samples = data_module.get_samples("val", 32)[frozenset(["v"])]["v"]
    train_samples = []
    #train_samples = data_module.get_samples("train", 32)[frozenset(["v"])]["v"]
    for r in range(320000,320032):
        img = Image.open(f"/mnt/datashare/yelhelw/expert_frames_final/train/rand_env_{r}.png")
        transform = transforms.ToTensor()
        train_samples.append(transform(img))
    train_samples = torch.stack(train_samples,dim=0)

    val_samples = []
    for r in range(80241,80241+32):
        img = Image.open(f"/mnt/datashare/yelhelw/expert_frames_final/val/rand_env_0{r}.png")
        transform = transforms.ToTensor()
        val_samples.append(transform(img))
    val_samples = torch.stack(val_samples,dim=0)

    for b in [0.05, 0.5, 1]:
        print("########",b)
        v_domain_module = VisualDomainModule(
            num_channels=config.domain_modules.visual.num_channels,
            optim_lr=config.training.optim.lr,
            beta=b,
            optim_weight_decay=config.training.optim.weight_decay,
            scheduler_args={
                "max_lr": config.training.optim.max_lr,
                "total_steps": config.training.max_steps,
            },
        )
        LOGGER.debug(
            f"log_val_medias_every_n_epochs {config.logging.log_val_medias_every_n_epochs}"
        )

        callbacks: list[pl.Callback] = [
            LearningRateMonitor(logging_interval="step"),
            LogVisualCallback(
                val_samples,
                log_key="images/val_attr",
                mode="val",
                every_n_epochs=config.logging.log_val_medias_every_n_epochs,
                ncols=8,
            ),
            LogVisualCallback(
                train_samples,
                log_key="images/train_attr",
                mode="train",
                every_n_epochs=config.logging.log_train_medias_every_n_epochs,
                ncols=8,
            ),
        # ProfilerCallback()
        ]

        if config.training.enable_progress_bar:
            callbacks.append(RichProgressBar())

        wandb_logger = None
        if config.wandb.enabled:
            if config.title is not None:
                run_name = config.title
            else:
                run_name = f"v_vae_z=16,b={b}"
            wandb_kwargs: dict[str, Any] = {}
            if config.desc is not None:
                wandb_kwargs["notes"] = config.desc
            wandb_logger = WandbLogger(
                save_dir=config.wandb.save_dir,
                project=config.wandb.project,
                entity=config.wandb.entity,
                tags=["train_gw"],
                name=run_name,
                **wandb_kwargs,
            )
            wandb_logger.experiment.config.update(config.model_dump())

            checkpoint_dir = (
                config.default_root_dir / f"{wandb_logger.name}-{wandb_logger.version}"
            )
            callbacks.extend(
                [
                    SaveMigrations(
                        get_folder_migrations(
                            PROJECT_DIR / "shimmer_metaworld" / "migrations" / "visual_mod"
                        )
                    ),
                    ModelCheckpoint(
                        dirpath=checkpoint_dir,
                        filename="{epoch}",
                        monitor="val/loss",
                        mode="min",
                        save_top_k=1,
                    ),
                ]
            )
        LOGGER.debug(f"wandb logger: {wandb_logger}")

        torch.set_float32_matmul_precision(config.training.float32_matmul_precision)


        trainer = pl.Trainer(
            logger=wandb_logger,
            fast_dev_run=config.training.fast_dev_run,
            max_steps=config.training.max_steps,
            enable_progress_bar=True,
            default_root_dir=config.default_root_dir,
            callbacks=callbacks,
            precision=config.training.precision,
            accelerator=config.training.accelerator,
            devices=config.training.devices
        )

        trainer.fit(v_domain_module, data_module)
        trainer.validate(v_domain_module, data_module, "best")

        wandb.finish()

if __name__ == "__main__":
    train_visual_domain(PROJECT_DIR / "shimmer_metaworld"/ "config_template")