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
import wandb
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
#For sweeps, contrastive_coef and fused_coef can be passed, others are set at 1
def train_gw(
    config_path: Path,
    contrast_coef,
    demi_coef,
    trans_coef,
    cycle_coef,
    z,
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
        "/mnt/datashare/yelhelw/complex_dataset_V3",
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
        z,
        config.global_workspace.encoders.hidden_dim,
        config.global_workspace.encoders.n_layers,
        config.global_workspace.decoders.hidden_dim,
        config.global_workspace.decoders.n_layers,
        is_linear=config.global_workspace.linear_domains,
        bias=config.global_workspace.linear_domains_use_bias,
    )

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

    loss_coefficients = config.global_workspace.loss_coefficients
    loss_coefficients['contrastives'] = contrast_coef
    loss_coefficients['demi_cycles'] = demi_coef
    loss_coefficients['translations'] = trans_coef
    loss_coefficients['cycles'] = cycle_coef
    loss_coefficients.pop('fused', None)
    module: GlobalWorkspaceBase
    gw_type: str
    if config.global_workspace.use_fusion_model:
        gw_type = "gw_fusion"
        module = GlobalWorkspaceFusion(
            domain_modules,
            gw_encoders,
            gw_decoders,
            z,
            loss_coefficients,
            selection_temperature=config.global_workspace.selection_temperature,
            optim_lr=config.training.optim.lr,
            optim_weight_decay=config.training.optim.weight_decay,
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
    
    print(domains for domains in train_samples)
    for domains in val_samples:
        for domain in domains:
            train_samples[frozenset([domain])] = {domain: train_samples[domains][domain]} 
            val_samples[frozenset([domain])] = {domain: val_samples[domains][domain]}
            #test_samples[frozenset([domain])] = {domain: test_samples[domains][domain]}
        break
    
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
    ]

    '''
    LogGWImagesCallback(
    val_samples,
    log_key="images/val",
    mode="val",
    every_n_epochs=config.logging.log_val_medias_every_n_epochs,
    filter=config.logging.filter_images,
    ),
    LogGWImagesCallback(
        train_samples,
        log_key="images/train",
        mode="train",
        every_n_epochs=config.logging.log_train_medias_every_n_epochs,
        filter=config.logging.filter_images,
    ),
    '''
    if config.training.enable_progress_bar:
        callbacks.append(RichProgressBar())
        #callbacks.append(ProfilerCallback(every_n_steps=200))
    wandb_logger = None
    if config.wandb.enabled:
        if config.title is not None:
            run_name = config.title
        else:
            run_name = f"{gw_type}_2_mod_b=1.5,z={z}_con={contrast_coef}_demi={demi_coef}_trans={trans_coef}_cycle={cycle_coef}"
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
    #trainer.validate(module, data_module, "best")
    #trainer.test(module, data_module, "best")
    wandb.finish()

if __name__ == "__main__":
    CLIP = (12,1.0,0,0,0)
    Kuske = (12,1.0,1.0,2.0,1.0)
    parameters = [
        #(16,0.1,1.0,1.0,0.1),
        #(16,0,1.0,1.0,0.1),
        #(12,0,1.0,1.0,0.1),
        (12,0.1,1.0,1.0,0.1),
        #CLIP,
    ]
    for param in parameters:
        z,c,d,t,cyc = param
        print(f"Training with contrastive_coef={c}")
        train_gw(
            PROJECT_DIR / "shimmer_metaworld"/ "config_template",
            contrast_coef=c,
            demi_coef=d,
            trans_coef=t,
            cycle_coef=cyc,
            z = z,
        )
