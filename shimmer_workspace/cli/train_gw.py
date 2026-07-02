import logging
from collections.abc import Callable
from typing import Any
from cfg_tools.utils import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import click
import torch
from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
import pytorch_lightning as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from shimmer import (
    ContrastiveLossType,
    GlobalWorkspaceBase,
    SaveMigrations,
)
from shimmer.modules.global_workspace import (
    GlobalWorkspace2Domains,
    GlobalWorkspaceFusion,
)
from metaworld_dataset import (
    MetaworldDataModule,
    color_blind_visual_domain,
    get_default_domains,
)
import torch
from torch import set_float32_matmul_precision
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.optimizer import Optimizer

from shimmer_metaworld import DEBUG_MODE, LOGGER,PROJECT_DIR
from shimmer_metaworld.config import load_config
from shimmer_metaworld.dataset.pre_process import TokenizeCaptions
from shimmer_metaworld.logging import LogGWImagesCallback
from shimmer_metaworld.modules.contrastive_loss import VSEPPContrastiveLoss
from shimmer_metaworld.modules.domains import load_pretrained_domains
import numpy as np
torch.set_default_dtype(torch.float32)

class ProfilerCallback(Callback):
        def __init__(self, every_n_steps: int = 100):
            super().__init__()
            self.every_n_steps = every_n_steps

        def on_train_batch_end(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule, 
            outputs, batch, batch_idx: int
        ):
            global_step = trainer.global_step
            if global_step % self.every_n_steps == 0 and global_step > 0:
                print(f"\n=== Profiler summary at step {global_step} ===")
                print(trainer.profiler.summary())

def train_gw(
    config_path: Path,
    debug_mode: bool | None = None,
    log_config: bool = False,
    extra_config_files: list[str] | None = None,
    argv: list[str] | None = None,
):
    if debug_mode is None:
        debug_mode = DEBUG_MODE
    if extra_config_files is None:
        extra_config_files = ["train_gw.yaml"]
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

    seed_everything(config.seed, workers=True)

    domain_classes = get_default_domains(
        {domain.domain_type.kind.value for domain in config.domains}
    )
    print(config.domains)
    data_module = MetaworldDataModule(
        '/mnt/datashare/yelhelw/',
        domain_classes,
        config.domain_proportions,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
        ood_seed=config.ood_seed,
        domain_args=config.domain_data_args,
    )
    
    

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
    #print(domain_modules, gw_encoders, gw_decoders)

    contrastive_fn: ContrastiveLossType | None = None
    if config.global_workspace.vsepp_contrastive_loss:
        contrastive_fn = VSEPPContrastiveLoss(
            config.global_workspace.vsepp_margin,
            config.global_workspace.vsepp_measure,
            config.global_workspace.vsepp_max_violation,
            torch.tensor([1 / 0.07]).log(),
        )

    def get_scheduler(optimizer: Optimizer) -> OneCycleLR:
        return OneCycleLR(
            optimizer,
            config.training.optim.max_lr,
            config.training.max_steps,
            pct_start=config.training.optim.pct_start,
            div_factor=config.training.optim.max_lr / config.training.optim.start_lr,
            final_div_factor=config.training.optim.max_lr
            / config.training.optim.end_lr,
        )

    module: GlobalWorkspaceBase
    gw_type: str
    if config.global_workspace.use_fusion_model:
        gw_type = "gw_fusion"
        module = GlobalWorkspaceFusion(
            domain_modules,
            gw_encoders,
            gw_decoders,
            config.global_workspace.latent_dim,
            config.global_workspace.loss_coefficients,
            config.global_workspace.selection_temperature,
            config.training.optim.lr,
            config.training.optim.weight_decay,
            learn_logit_scale=config.global_workspace.learn_logit_scale,
            contrastive_loss=contrastive_fn,
            scheduler=get_scheduler,
        )
        module.float()
    else:
        gw_type = "gw"

        module = GlobalWorkspace2Domains(
            domain_modules,
            gw_encoders,
            gw_decoders,
            config.global_workspace.latent_dim,
            config.global_workspace.loss_coefficients,
            config.training.optim.lr,
            config.training.optim.weight_decay,
            learn_logit_scale=config.global_workspace.learn_logit_scale,
            contrastive_loss=contrastive_fn,
            scheduler=get_scheduler,
        )

    train_samples = data_module.get_samples("train", 32,offset=1)
    val_samples = data_module.get_samples("val", 32,offset=1)
    #test_samples = data_module.get_samples("test", 32)

    for domains in val_samples:
        for domain in domains:
            val_samples[frozenset([domain])] = {domain: val_samples[domains][domain]}
            #test_samples[frozenset([domain])] = {domain: test_samples[domains][domain]}
        break
    
    # helper
    '''
    def to_float32(obj):
        if torch.is_tensor(obj):
            if obj.dtype.is_floating_point:
                return obj.to(torch.float32)
            return obj
        if isinstance(obj, dict):
            return {k: to_float32(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(to_float32(x) for x in obj)
        return obj
    
    train_samples = to_float32(train_samples)
    '''
    def find_double_tensors(obj, prefix="root"):
        if torch.is_tensor(obj):
            if obj.dtype == torch.float64:
                print(f"{prefix}: dtype={obj.dtype} device={obj.device} shape={tuple(obj.shape)}")
        elif isinstance(obj, dict):
            for k,v in obj.items():
                find_double_tensors(v, f"{prefix}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i,v in enumerate(obj):
                find_double_tensors(v, f"{prefix}[{i}]")

    # call after model creation and in training_step (on batch, outputs, loss, optimizer state)
    #print(find_double_tensors(train_samples))

    
    
    callbacks = []
    callbacks = [
        LearningRateMonitor(logging_interval="step"),

    ]

    if config.ood_seed is not None:
        train_samples_ood = data_module.get_samples("train", 32, ood=True)
        val_samples_ood = data_module.get_samples("val", 32, ood=True)
        #test_samples_ood = data_module.get_samples("test", 32, ood=True)

        for domains in val_samples_ood:
            for domain in domains:
                val_samples_ood[frozenset([domain])] = {
                    domain: val_samples_ood[domains][domain]
                }
                '''
                test_samples_ood[frozenset([domain])] = {
                    domain: test_samples_ood[domains][domain]
                }
                '''
            break
        callbacks.extend(
            [
                '''
                LogGWImagesCallback(
                    val_samples_ood,
                    log_key="images/val/ood",
                    mode="val",
                    every_n_epochs=config.logging.log_val_medias_every_n_epochs,
                    filter=config.logging.filter_images,
                ),
                LogGWImagesCallback(
                    train_samples_ood,
                    log_key="images/train/ood",
                    mode="train",
                    every_n_epochs=config.logging.log_train_medias_every_n_epochs,
                    filter=config.logging.filter_images,
                ),
                '''
            ]
        )

    if config.training.enable_progress_bar:
        callbacks.append(RichProgressBar())
        #callbacks.append(ProfilerCallback(every_n_steps=200))
    wandb_logger = None
    if config.wandb.enabled:
        if config.title is not None:
            run_name = config.title
        else:
            run_name = f"{gw_type}_z={config.global_workspace.latent_dim}"
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
                SaveMigrations(),
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename="{epoch}",
                    monitor="val/loss",
                    mode="min",
                    save_top_k=1,
                ),
            ]
        )
    print(callbacks)
    set_float32_matmul_precision(config.training.float32_matmul_precision)

    trainer = Trainer(
        logger=wandb_logger,
        fast_dev_run=config.training.fast_dev_run,
        max_steps=config.training.max_steps,
        enable_progress_bar=config.training.enable_progress_bar,
        default_root_dir=config.default_root_dir,
        callbacks=callbacks,
        precision=config.training.precision,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
    )

    trainer.fit(module, data_module)
    trainer.validate(module, data_module, "best")
    #trainer.test(module, data_module, "best")
if __name__ == "__main__":
    train_gw(PROJECT_DIR / "shimmer_metaworld"/ "config_template")
'''
@click.command(
    "gw",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    },
    help="Train the Global Workspace",
)
@click.option(
    "--config_path",
    "-c",
    default="./config",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),  # type: ignore
)
@click.option("--debug", "-d", is_flag=True, default=None)
@click.option("--log_config", is_flag=True, default=False)
@click.option(
    "--extra_config_files",
    "-e",
    multiple=True,
    type=str,
    help=(
        "Additional files to `local.yaml` to load in the config path. "
        "By default `train_gw.yaml`"
    ),
)
@click.pass_context
def train_gw_command(
    ctx: click.Context,
    config_path: Path,
    debug: bool | None,
    log_config: bool,
    extra_config_files: list[str],
):
    return train_gw(
        config_path,
        debug,
        log_config,
        extra_config_files if len(extra_config_files) else None,
        ctx.args,
    )
'''