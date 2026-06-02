import sys, os, shutil
# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


from pathlib import Path
from typing import Any
import numpy as np
import click
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers.wandb import WandbLogger
import wandb
from migrate_ckpt.migrate import get_folder_migrations
from metaworld_dataset import (
    MetaworldDataModule,
    get_default_domains,)

from shimmer_metaworld import DEBUG_MODE, LOGGER, PROJECT_DIR
from shimmer_metaworld.ckpt_migrations import (
    SaveMigrations,
)
from shimmer_metaworld.config import load_config
from shimmer_metaworld.logging import LogAttributesCallback, LogVisualCallback, LogActionsCallback
from shimmer_metaworld.modules.domains.visual import VisualDomainModule
from shimmer_metaworld.modules.domains.attribute import AttributeDomainModule, ActionDomainModule

from PIL import Image
import torchvision.transforms as transforms

DEFAULT_CALLBACKS: dict[str, type[LogAttributesCallback]] = {
    "v": LogVisualCallback,
    "attr": LogAttributesCallback,
    "act" : LogActionsCallback,
}

DEFAULT_DOMAINS: dict[str, type[VisualDomainModule]] = {
    "v": VisualDomainModule,
    "attr": AttributeDomainModule,
    "act" : ActionDomainModule,
}

DOMAIN_CONF : dict[str, str] = {
    "v" : "train_v.yaml",
    "attr" : "train_attr.yaml",
    "act" : "train_act.yaml",
}

def GW_sweep(
        config_path: Path,
        data_path : Path,
        domain,
        val_samples,
        train_samples,
        sweep,
        debug_mode=None,
        extra_config_files=None,
        argv=None,
        ):
    LogCallback =  DEFAULT_CALLBACKS[domain]
    

    callbacks: list[pl.Callback] = [
            LearningRateMonitor(logging_interval="step"),
            LogCallback(
                val_samples,
                log_key="images/val_attr",
                mode="val",
                every_n_epochs=config.logging.log_val_medias_every_n_epochs,
                ncols=8,
            ),
            LogCallback(
                train_samples,
                log_key="images/train_attr",
                mode="train",
                every_n_epochs=config.logging.log_train_medias_every_n_epochs,
                #image_size=32,
                ncols=8,
            ),
            ]

def unimodal_sweep(
        config_path: Path,
        data_path : Path,
        domain,
        val_samples,
        train_samples,
        sweep,
        ):
    LogCallback =  DEFAULT_CALLBACKS[domain]
    

    callbacks: list[pl.Callback] = [
            LearningRateMonitor(logging_interval="step"),
            LogCallback(
                val_samples,
                log_key="images/val_attr",
                mode="val",
                every_n_epochs=config.logging.log_val_medias_every_n_epochs,
                ncols=8,
            ),
            LogCallback(
                train_samples,
                log_key="images/train_attr",
                mode="train",
                every_n_epochs=config.logging.log_train_medias_every_n_epochs,
                #image_size=32,
                ncols=8,
            ),
            ]   

    if debug_mode is None:
        debug_mode = DEBUG_MODE
    if extra_config_files is None:
        extra_config_files = DOMAIN_CONF[domain]
    if argv is None:
        argv = []

    config = load_config(
        config_path,
        load_files = extra_config_files,
        debug_mode=debug_mode,
        log_config=False,
        argv=argv,
    )


    pl.seed_everything(config.seed, workers=True)

    data_module = MetaworldDataModule(
        data_path,
        get_default_domains([domain]),
        {frozenset([domain]): 1.0},
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model_parameters = {
        "hidden_dim" : config.domain_modules.attribute.hidden_dim,
        "beta" : config.domains_modules.attribute.beta,
        "optim_lr" : config.training.optim.lr,
        "optim_weight_decay" : config.training.optim.weight_decay,
        "scheduler_args" : {
            "max_lr": config.training.optim.max_lr,
            "total_steps": config.training.max_steps,
        }
    }
    for p in sweep:
        for i in p[1]:
            print("########",p[0],i)
            model_parameters[p[0]] = i
            DomainModule = DEFAULT_DOMAINS[domain](**model_parameters)
            
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

            trainer.fit(DomainModule, data_module)
            trainer.validate(DomainModule, data_module, "best")

            wandb.finish()

def main(domain):
    if domain == "visual":
        config_path = Path()
        data_path = Path('/mnt/datashare/yelhelw/expert_frames_final/')

        train_samples = []
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

        unimodal_sweep( 
            config_path,
            data_path,
            val_samples,
            train_samples,

        )

    
