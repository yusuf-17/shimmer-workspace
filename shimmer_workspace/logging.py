import io
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Generic, Literal, TypeVar, cast

import lightning.pytorch as pl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import Logger, TensorBoardLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import gridspec
from matplotlib.figure import Figure
import matplotlib.colors as colors
from PIL import Image
from shimmer.modules.global_workspace import GlobalWorkspaceBase, GWPredictionsBase
from simple_shapes_dataset import (
    UnnormalizeAttributes,
    tensor_to_attribute,
)
from simple_shapes_dataset.cli import generate_image
from tokenizers.implementations import ByteLevelBPETokenizer
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid
from torch import nn
from shimmer_metaworld import LOGGER
from shimmer_metaworld.modules.domains.text import GRUTextDomainModule, Text2Attr
from shimmer_metaworld.modules.domains.visual import VisualLatentDomainModule

matplotlib.use("Agg")

_T = TypeVar("_T")


def log_image(
    logger: Logger,
    key: str,
    image: torch.Tensor | Image.Image,
    tensorboard_step: int | None = None,
):
    if isinstance(logger, WandbLogger):
        logger.log_image(key, [image])
    elif isinstance(logger, TensorBoardLogger):
        torch_image = to_tensor(image) if isinstance(image, Image.Image) else image
        logger.experiment.add_image(key, torch_image, tensorboard_step)
    else:
        LOGGER.warning(
            "[Sample Logger] Only logging to tensorboard or wandb is supported"
        )
        return


def log_text(
    logger: Logger,
    key: str,
    columns: list[str],
    data: list[list[str]],
    tensorboard_step: int | None = None,
):
    if isinstance(logger, WandbLogger):
        logger.log_text(key, columns, data)
    elif isinstance(logger, TensorBoardLogger):
        text = ", ".join(columns) + "\n"
        text += "\n".join([", ".join(d) for d in data])
        logger.experiment.add_text(key, text, tensorboard_step)
    else:
        LOGGER.warning(
            "[Sample Logger] Only logging to tensorboard or wandb is supported"
        )
        return


class LogSamplesCallback(Generic[_T], ABC, pl.Callback):
    def __init__(
        self,
        reference_samples: _T,
        log_key: str,
        mode: Literal["train", "val", "test"],
        every_n_epochs: int | None = 1,
    ) -> None:
        super().__init__()
        self.reference_samples = reference_samples
        self.every_n_epochs = every_n_epochs
        self.log_key = log_key
        self.mode = mode
        self._global_step = 0

    def get_step(self) -> int:
        self._global_step += 1
        return self._global_step - 1

    def to(self, samples: _T, device: torch.device) -> _T:
        raise NotImplementedError

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        if stage != "fit":
            return
        device = trainer.strategy.root_device
        self.reference_samples = self.to(self.reference_samples, device)
        for logger in trainer.loggers:
            self.log_samples(logger, self.reference_samples, "reference")

    def on_callback(
        self,
        loggers: Sequence[Logger],
        pl_module: pl.LightningModule,
    ) -> None:
        if not len(loggers):
            LOGGER.debug("[LOGGER] No logger found.")
            return

        samples = self.to(self.reference_samples, pl_module.device)

        with torch.no_grad():
            pl_module.eval()
            generated_samples = pl_module(samples)
            pl_module.train()

        for logger in loggers:
            self.log_samples(logger, generated_samples, "prediction")

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.mode != "train":
            return

        if (
            self.every_n_epochs is None
            or trainer.current_epoch % self.every_n_epochs != 0
        ):
            LOGGER.debug("[LOGGER] on_train_epoch_end")
            return

        LOGGER.debug("[LOGGER] on_train_epoch_end called")
        return self.on_callback(trainer.loggers, pl_module)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.mode == "test":
            return

        return self.on_callback(trainer.loggers, pl_module)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.mode != "val":
            return

        if (
            self.every_n_epochs is None
            or trainer.current_epoch % self.every_n_epochs != 0
        ):
            return

        return self.on_callback(trainer.loggers, pl_module)

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.mode != "test":
            return

        return self.on_callback(trainer.loggers, pl_module)

    @abstractmethod
    def log_samples(self, logger: Logger, samples: _T, mode: str) -> None: ...


def get_pil_image(figure: Figure) -> Image.Image:
    buf = io.BytesIO()
    figure.savefig(buf)
    buf.seek(0)
    return Image.open(buf)

def attr_to_image(samples: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 17-dim vectors into images with each object as a patch.

    Args:
        samples: (batch_size, 17) tensor with values in [-1, 1]

    Returns:
        images: (batch_size, 3, H, W) RGB images
    """
    B = samples.shape[0]
    device = samples.device
    samples_np = samples.detach().cpu().numpy()

    imgs= []
    for i in range(B):
        s = samples_np[i]

        eff_xy = s[0:2]
        eff_z = s[2]
        gripper = s[3]
        ball_xy = s[4:6]
        wall_xy = s[11:13]
        goal_xy = s[14:16]


        fig = plt.figure()
        ax = fig.subplots()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal", adjustable="box")

        # plot: effector
        eff_marker_size = eff_z * (0.5 + (np.tanh(eff_z) + 1)/2)  # scale by z safely
        # map effector color scalar to colormap
        cmap_obj = plt.get_cmap('cool')
        norm = colors.Normalize(vmin=-1,vmax=1)
        eff_color = cmap_obj(norm(eff_z))

        ax.scatter(eff_xy[0], eff_xy[1], s=(gripper*25)+10, c=[eff_color], marker="X", edgecolors="k", linewidths=0.6, label="effector")

        # ball
        ax.scatter(ball_xy[0], ball_xy[1], c='blue', marker="o", label="ball")

        # wall
        ax.scatter(wall_xy[0], wall_xy[1], c='green', marker="s", label="wall")

        # goal
        ax.scatter(goal_xy[0], goal_xy[1], c='red', marker="D", label="goal")
        image = get_pil_image(fig)
        img_t = to_tensor(image)
        imgs.append(img_t)
        plt.close(fig)
        
    return torch.stack(imgs,dim=0)



class LogAttributesCallback(LogSamplesCallback[torch.Tensor]):
    def __init__(
        self,
        reference_samples: torch.Tensor,
        log_key: str,
        mode: Literal["train", "val", "test"],
        every_n_epochs: int | None = 1,
        ncols: int = 8,
    ) -> None:
        super().__init__(reference_samples, log_key, mode, every_n_epochs)
        self.ncols = ncols

    def to(self, samples: torch.Tensor, device: torch.device) -> torch.Tensor:
        return samples.to(device)
    import torch.nn.functional as F

    def safe_divide(self,x):
        max_val = x.abs().max(dim=1, keepdim=True)[0]  # per sample
        max_val[max_val == 0] = 1.0  # avoid division by zero
        return x / max_val
    def log_samples(self, logger: Logger, samples: torch.Tensor, mode: str) -> None:
        images = make_grid(attr_to_image(samples.float()), nrow=4, pad_value=1)
        log_image(logger, f"{self.log_key}_{mode}", images)

def act_to_image(samples: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 17-dim vectors into images with each object as a patch.

    Args:
        samples: (batch_size, 17) tensor with values in [-1, 1]

    Returns:
        images: (batch_size, 3, H, W) RGB images
    """
    B = samples.shape[0]
    device = samples.device
    samples_np = samples.detach().cpu().numpy()

    imgs= []
    for i in range(B):
        s = samples_np[i]

        disp_x = s[0]
        disp_y = s[1]
        disp_z = s[2]
        gripper = s[3]

        fig = plt.figure()
        ax = fig.subplots()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal", adjustable="box")

            # --- XY displacement arrow ---
        ax.arrow(0, 0, disp_x, disp_y,
                head_width=0.05, length_includes_head=True, color='blue')

        # --- Z displacement arrow ---
        # Represent Z as an arrow from (0,0) pointing upward in a different color
        ax.arrow(0, 0, 0, disp_z,
                head_width=0.05, length_includes_head=True, color='red')

        # --- Gripper as a point ---
        # Map gripper to something like color or size
        ax.scatter(gripper, 0, s=50, c="red", marker='o')

        image = get_pil_image(fig)
        img_t = to_tensor(image)
        imgs.append(img_t)
        plt.close(fig)
        
    return torch.stack(imgs,dim=0)


#TODO: write actions callback logging images with displacement vectors
class LogActionsCallback(LogSamplesCallback[torch.Tensor]):
    def __init__(
        self,
        reference_samples: torch.Tensor,
        log_key: str,
        mode: Literal["train", "val", "test"],
        every_n_epochs: int | None = 1,
        ncols: int = 8,
    ) -> None:
        super().__init__(reference_samples, log_key, mode, every_n_epochs)
        self.ncols = ncols

    def to(self, samples: torch.Tensor, device: torch.device) -> torch.Tensor:
        return samples.to(device)
    import torch.nn.functional as F

    def safe_divide(self,x):
        max_val = x.abs().max(dim=1, keepdim=True)[0]  # per sample
        max_val[max_val == 0] = 1.0  # avoid division by zero
        return x / max_val
    def log_samples(self, logger: Logger, samples: torch.Tensor, mode: str) -> None:

        images = make_grid(vector_to_image(samples.float()), nrow=4, pad_value=1)
        log_image(logger, f"{self.log_key}_{mode}", images)


class LogTextCallback(LogSamplesCallback[Mapping[str, torch.Tensor]]):
    def __init__(
        self,
        reference_samples: Mapping[str, torch.Tensor],
        log_key: str,
        mode: Literal["train", "val", "test"],
        image_size: int,
        vocab: str,
        merges: str,
        every_n_epochs: int | None = 1,
        ncols: int = 8,
    ) -> None:
        super().__init__(reference_samples, log_key, mode, every_n_epochs)
        self.image_size = image_size
        self.ncols = ncols
        self.tokenizer = ByteLevelBPETokenizer(vocab, merges)

    def to(
        self, samples: Mapping[str, torch.Tensor], device: torch.device
    ) -> dict[str, torch.Tensor]:
        return {x: samples[x].to(device) for x in samples}

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        if stage != "fit":
            return
        assert isinstance(pl_module, GRUTextDomainModule)
        device = trainer.strategy.root_device
        self.reference_samples = self.to(self.reference_samples, device)
        for logger in trainer.loggers:
            self.log_samples(logger, self.reference_samples, "reference")

    def on_callback(
        self,
        loggers: Sequence[Logger],
        pl_module: pl.LightningModule,
    ) -> None:
        assert isinstance(pl_module, GRUTextDomainModule)

        samples = self.to(self.reference_samples, pl_module.device)

        if not len(loggers):
            LOGGER.debug("[LOGGER] No logger found.")
            return

        with torch.no_grad():
            pl_module.eval()
            generated_samples = pl_module(samples)
            pl_module.train()

        for logger in loggers:
            self.log_samples(logger, generated_samples, "prediction")

    def log_samples(
        self, logger: Logger, samples: Mapping[str, torch.Tensor], mode: str
    ) -> None:
        if not isinstance(logger, WandbLogger):
            LOGGER.warning("Only logging to wandb is supported")
            return

        assert self.tokenizer is not None
        text = self.tokenizer.decode_batch(
            samples["tokens"].detach().cpu().tolist(), skip_special_tokens=True
        )
        text = [[t.replace("<pad>", "")] for t in text]
        log_text(logger, f"{self.log_key}_{mode}_str", ["text"], text, self.get_step())


class LogVisualCallback(LogSamplesCallback[torch.Tensor]):
    def __init__(
        self,
        reference_samples: torch.Tensor,
        log_key: str,
        mode: Literal["train", "val", "test"],
        every_n_epochs: int | None = 1,
        ncols: int = 8,
    ) -> None:
        super().__init__(reference_samples, log_key, mode, every_n_epochs)
        self.ncols = ncols

    def to(self, samples: torch.Tensor, device: torch.device) -> torch.Tensor:
        return samples.to(device)

    def log_samples(self, logger: Logger, samples: torch.Tensor, mode: str) -> None:
        images = make_grid(samples, nrow=self.ncols, pad_value=1)
        log_image(logger, f"{self.log_key}_{mode}", images)


class LogText2AttrCallback(
    LogSamplesCallback[
        Mapping[str, Mapping[str, torch.Tensor] | Sequence[torch.Tensor]]
    ]
):
    def __init__(
        self,
        reference_samples: Mapping[
            str, Mapping[str, torch.Tensor] | Sequence[torch.Tensor]
        ],
        log_key: str,
        mode: Literal["train", "val", "test"],
        every_n_epochs: int | None = 1,
        image_size: int = 32,
        ncols: int = 8,
        vocab: str | None = None,
        merges: str | None = None,
    ) -> None:
        super().__init__(reference_samples, log_key, mode, every_n_epochs)
        self.image_size = image_size
        self.ncols = ncols
        self.tokenizer = ByteLevelBPETokenizer(vocab, merges)
        self.reference_samples = reference_samples

    def to(
        self,
        samples: Mapping[str, Mapping[str, torch.Tensor] | Sequence[torch.Tensor]],
        device: torch.device,
    ) -> dict[str, dict[str, torch.Tensor] | list[torch.Tensor]]:
        latents: dict[str, dict[str, torch.Tensor] | list[torch.Tensor]] = {}
        for domain_name, domain in samples.items():
            if isinstance(domain, dict):
                latents[domain_name] = {k: x.to(device) for k, x in domain.items()}
            elif isinstance(domain, list):
                latents[domain_name] = [x.to(device) for x in domain]
        return latents

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        if stage != "fit":
            return
        assert isinstance(pl_module, Text2Attr)
        device = trainer.strategy.root_device
        self.reference_samples = self.to(self.reference_samples, device)
        for logger in trainer.loggers:
            self.log_samples(logger, self.reference_samples, "reference")

    def on_callback(
        self,
        loggers: Sequence[Logger],
        pl_module: pl.LightningModule,
    ) -> None:
        assert isinstance(pl_module, Text2Attr)

        samples = self.to(self.reference_samples, pl_module.device)

        if not len(loggers):
            LOGGER.debug("[LOGGER] No logger found.")
            return

        with torch.no_grad():
            pl_module.eval()
            generated_samples = pl_module(samples["t"])
            pl_module.train()

        for logger in loggers:
            self.log_samples(logger, generated_samples, "prediction")

    def log_samples(
        self,
        logger: Logger,
        samples: Mapping[str, Mapping[str, torch.Tensor] | Sequence[torch.Tensor]],
        mode: str,
    ) -> None:
        for domain_name, domain in samples.items():
            if domain_name == "t":
                assert self.tokenizer is not None
                assert isinstance(domain, dict)
                text = self.tokenizer.decode_batch(
                    domain["tokens"].detach().cpu().tolist(), skip_special_tokens=True
                )
                text = [[t.replace("<pad>", "")] for t in text]
                log_text(
                    logger,
                    f"{self.log_key}_{mode}_str",
                    ["text"],
                    text,
                    self.get_step(),
                )
            elif domain_name == "attr":
                assert isinstance(domain, list)
                image = attribute_image_grid(
                    domain,
                    image_size=self.image_size,
                    ncols=self.ncols,
                )
                log_image(logger, f"{self.log_key}_{mode}", image, self.get_step())


def batch_to_device(
    samples: Mapping[
        frozenset[str],
        Mapping[str, Any],
    ],
    device: torch.device,
) -> dict[frozenset[str], dict[str, Any]]:
    out: dict[frozenset[str], dict[str, Any]] = {}
    for domain_names, domains in samples.items():
        latents: dict[str, Any] = {}
        for domain_name, domain in domains.items():
            if isinstance(domain, torch.Tensor):
                latents[domain_name] = domain.to(device)
            elif (
                isinstance(domain, Mapping)
                and len(domain)
                and isinstance(next(iter(domain.values())), torch.Tensor)
            ):
                latents[domain_name] = {k: x.to(device) for k, x in domain.items()}
            elif (
                isinstance(domain, Sequence)
                and len(domain)
                and isinstance(domain[0], torch.Tensor)
            ):
                latents[domain_name] = [x.to(device) for x in domain]
            else:
                latents[domain_name] = domain
        out[domain_names] = latents
    return out


class LogGWImagesCallback(pl.Callback):
    def __init__(
        self,
        reference_samples: Mapping[frozenset[str], Mapping[str, Any]],
        log_key: str,
        mode: Literal["train", "val", "test"],
        every_n_epochs: int | None = 1,
        image_size: int = 32,
        ncols: int = 8,
        filter: Sequence[str] | None = None,
        vocab: str | None = None,
        merges: str | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.reference_samples = reference_samples
        self.every_n_epochs = every_n_epochs
        self.log_key = log_key
        self.image_size = image_size
        self.ncols = ncols
        self.filter = filter
        self.tokenizer = None
        if vocab is not None and merges is not None:
            self.tokenizer = ByteLevelBPETokenizer(vocab, merges)
        self._global_step = 0

    def get_step(self):
        self._global_step += 1
        return self._global_step - 1

    def to(
        self,
        samples: Mapping[
            frozenset[str],
            Mapping[
                str, torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor]
            ],
        ],
        device: torch.device,
    ) -> dict[
        frozenset[str],
        dict[str, torch.Tensor | list[torch.Tensor] | dict[Any, torch.Tensor]],
    ]:
        return batch_to_device(samples, device)

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        if stage != "fit":
            return
        assert isinstance(pl_module, GlobalWorkspaceBase)
        device = trainer.strategy.root_device
        self.reference_samples = self.to(self.reference_samples, device)

        for domain_names, domains in self.reference_samples.items():
            for domain_name, domain_tensor in domains.items():
                for logger in trainer.loggers:
                    self.log_samples(
                        logger,
                        pl_module,
                        domain_tensor,
                        domain_name,
                        f"ref_{'-'.join(domain_names)}_{domain_name}",
                    )

    def on_callback(
        self,
        loggers: Sequence[Logger],
        pl_module: GlobalWorkspaceBase,
    ) -> None:
        if not (len(loggers)):
            return

        with torch.no_grad():
            latent_groups = pl_module.encode_domains(self.reference_samples)
            predictions = cast(GWPredictionsBase, pl_module(latent_groups))

            for logger in loggers:
                for domains, preds in predictions["broadcasts"].items():
                    domain_from = ",".join(domains)
                    for domain, pred in preds.items():
                        log_name = f"pred_trans_{domain_from}_to_{domain}"
                        if self.filter is not None and log_name not in self.filter:
                            continue
                        self.log_samples(
                            logger,
                            pl_module,
                            pl_module.decode_domain(pred, domain),
                            domain,
                            log_name,
                        )
                for domains, preds in predictions["cycles"].items():
                    domain_from = ",".join(domains)
                    for domain, pred in preds.items():
                        log_name = f"pred_cycle_{domain_from}_to_{domain}"
                        if self.filter is not None and log_name not in self.filter:
                            continue
                        self.log_samples(
                            logger,
                            pl_module,
                            pl_module.decode_domain(pred, domain),
                            domain,
                            log_name,
                        )

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.mode != "train":
            return

        if not isinstance(pl_module, GlobalWorkspaceBase):
            return

        if (
            self.every_n_epochs is None
            or trainer.current_epoch % self.every_n_epochs != 0
        ):
            return

        return self.on_callback(trainer.loggers, pl_module)

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.mode != "val":
            return

        if not isinstance(pl_module, GlobalWorkspaceBase):
            return

        if (
            self.every_n_epochs is None
            or trainer.current_epoch % self.every_n_epochs != 0
        ):
            return

        return self.on_callback(trainer.loggers, pl_module)

    def on_test_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.mode != "test":
            return

        if not isinstance(pl_module, GlobalWorkspaceBase):
            return

        return self.on_callback(trainer.loggers, pl_module)

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.mode == "test":
            return

        if not isinstance(pl_module, GlobalWorkspaceBase):
            return

        return self.on_callback(trainer.loggers, pl_module)

    def log_samples(
        self,
        logger: Logger,
        pl_module: GlobalWorkspaceBase,
        samples: Any,
        domain: str,
        mode: str,
    ) -> None:
        match domain:
            case "v":
                self.log_visual_samples(logger, samples, mode)
            case "v_latents":
                assert "v_latents" in pl_module.domain_mods

                module = cast(
                    VisualLatentDomainModule,
                    pl_module.domain_mods["v_latents"],
                )
                self.log_visual_samples(logger, module.decode_images(samples), mode)
            case "attr":
                self.log_attribute_samples(logger, samples, mode)
            case "act":
                self.log_act_samples(logger, samples, mode)

    def log_visual_samples(
        self,
        logger: Logger,
        samples: Any,
        mode: str,
    ) -> None:
        images = make_grid(samples, nrow=self.ncols, pad_value=1)
        log_image(logger, f"{self.log_key}/{mode}", images, self.get_step())

    def log_attribute_samples(
        self,
        logger: Logger,
        samples: Any,
        mode: str,
    ) -> None:
        images = make_grid(attr_to_image(samples.float()), nrow=4, pad_value=1)
        log_image(logger, f"{self.log_key}_{mode}", images)

    def log_act_samples(
        self,
        logger: Logger,
        samples: Any,
        mode: str,
    ) -> None:
        images = make_grid(act_to_image(samples.float()), nrow=4, pad_value=1)
        log_image(logger, f"{self.log_key}_{mode}", images)
