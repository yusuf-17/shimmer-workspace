from pathlib import Path
from typing import Any
import sys, os, shutil
import numpy as np
# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
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
from shimmer_metaworld.logging import LogActionsCallback
from shimmer_metaworld.modules.domains.attribute import ActionDomainModule


def train_act_domain(
    config_path: Path,
    debug_mode: bool | None = None,
    log_config: bool = False,
    extra_config_files: list[str] | None = None,
    argv: list[str] | None = None,
):
    if debug_mode is None:
        debug_mode = DEBUG_MODE
    if extra_config_files is None:
        extra_config_files = ["train_attr.yaml"]
    if argv is None:
        argv = []
    config = load_config

    LOGGER.debug(f"Debug mode: {debug_mode}")

    config = load_config(
        config_path,
        load_files=extra_config_files,
        debug_mode=debug_mode,
        log_config=log_config,
        argv=argv,
    )

    pl.seed_everything(config.seed, workers=True)

    data_module = MetaworldDataModule(
        os.path.abspath('/mnt/datashare/yelhelw/dataset'),
        get_default_domains(["act"]),
        {frozenset(["act"]): 1.0},
        batch_size=128,
        num_workers=8,
    )

    train_samples = data_module.get_samples("train", 32,offset=100)[frozenset(["act"])]["act"]
    val_samples = data_module.get_samples("val", 32,offset=100)[frozenset(["act"])]["act"]
   
    print(train_samples[:].min(),val_samples[:].max())
    
    for b in np.arange(0.05,0.25,0.05,dtype=np.float32):
        print("########",b)
        attr_domain_module = ActionDomainModule(
        latent_dim=1,
        hidden_dim=32,
        beta=0.005,
        optim_lr=config.training.optim.lr,
        optim_weight_decay=config.training.optim.weight_decay,
        scheduler_args={
            "max_lr": config.training.optim.max_lr,
            "total_steps": config.training.max_steps,
        },
        )

        callbacks: list[pl.Callback] = [
            LearningRateMonitor(logging_interval="step"),
            LogActionsCallback(
                train_samples,
                log_key="images/train_act",
                mode="train",
                every_n_epochs=config.logging.log_train_medias_every_n_epochs,
                #image_size=32,
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
                run_name = f"act_vae_z=1_b={b:.2f}"
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
                            PROJECT_DIR / "shimmer_metaworld" / "migrations" / "act_mod"
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
            devices=config.training.devices,
        )

        trainer.fit(attr_domain_module, data_module)
        wandb.finish()


if __name__ == "__main__":
    train_act_domain(PROJECT_DIR / "shimmer_metaworld"/ "config_template")