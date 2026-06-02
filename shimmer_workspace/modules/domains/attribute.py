from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from shimmer import LossOutput
from shimmer.modules.domain import DomainModule
from shimmer.modules.global_workspace import SchedulerArgs
from shimmer.modules.vae import (
    VAE,
    VAEDecoder,
    VAEEncoder,
    gaussian_nll,
    kl_divergence_loss,
)
from torch.nn.functional import mse_loss
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR


class Encoder(VAEEncoder):
    def __init__(
        self,
        in_dim : int,
        hidden_dim: int,
        out_dim: int,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

        self.q_mean = nn.Linear(self.out_dim, self.out_dim)
        self.q_logvar = nn.Linear(self.out_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(x.float()).view(x.size(0), -1)
        out = out.view(out.size(0), -1)

        return self.q_mean(out), self.q_logvar(out)


class Decoder(VAEDecoder):
    def __init__(
        self,
        in_dim : int,
        latent_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.in_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.decoder(x.float())


class AttributeDomainModule(DomainModule):
    in_dim = 13
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        beta: float = 1,
        coef_categories: float = 1,
        coef_attributes: float = 1,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0,
        scheduler_args: SchedulerArgs | None = None,
    ):
        """
        Defines the Attribute domain module.

        Args:
            latent_dim (`int`): the latent dimension of the module
            hidden_dim (`int`): hidden dimension of the VAE encoders and decoders
            beta (`float`): for beta-VAE
            coef_categories (`float`): loss coefficient attributed to the category
                (Defaults to 1.0)
            coef_attributes (`float`): loss coefficient attributed to the rest of the
                attributes (Defaults to 1.0)
            optim_lr (`float`): learning rate for the optimizer
            optim_weight_decay (`float`): weight decay for the optimizer
            scheduler_args (`SchedulerArgs | None`): Scheduler arguments
        """
        super().__init__(latent_dim)
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.coef_categories = coef_categories
        self.coef_attributes = coef_attributes

        vae_encoder = Encoder(self.in_dim,self.hidden_dim, self.latent_dim)
        vae_decoder = Decoder(self.in_dim,self.latent_dim, self.hidden_dim)
        self.vae = VAE(vae_encoder, vae_decoder, beta)

        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay

        self.scheduler_args = SchedulerArgs(
            max_lr=optim_lr,
            total_steps=1,
        )
        self.scheduler_args.update(scheduler_args or {})
    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any
    ) -> LossOutput:
        return LossOutput(F.mse_loss(pred, target, reduction="mean"))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x.float())

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.vae.decode(z.float())
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.decode(self.encode(x.float()))

    def generic_step(
        self,
        x: torch.Tensor,
        mode: str = "train",
    ) -> torch.Tensor:
        # this calles model.__call__(x)  since it inherets from nn.Module
        # which internally runs: vae.forward(x)
        (mean, logvar), reconstruction = self.vae(x.float())

        reconstruction_loss = mse_loss(reconstruction, x, reduction="mean")

        kl_loss = kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + self.vae.beta * kl_loss

        self.log(f"{mode}/reconstruction_loss", reconstruction_loss)
        self.log(f"{mode}/kl_loss", kl_loss)
        self.log(f"{mode}/loss", total_loss)
        return total_loss

    def validation_step(  # type: ignore
        self, batch: Mapping[frozenset[str], Mapping[str, Sequence[torch.Tensor]]], _
    ) -> torch.Tensor:
        x = batch[frozenset(["attr"])]["attr"].float()
        return self.generic_step(x, "val")

    def training_step(  # type: ignore
        self,
        batch: Mapping[frozenset[str], Mapping[str, Sequence[torch.Tensor]]],
        _,
    ) -> torch.Tensor:
        x = batch[frozenset(["attr"])]["attr"].float()
        return self.generic_step(x, "train")

    def configure_optimizers(  # type: ignore
        self,
    ) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optim_lr,
            weight_decay=self.optim_weight_decay,
        )
        lr_scheduler = OneCycleLR(optimizer, **self.scheduler_args)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

class ActionDomainModule(DomainModule):
    in_dim = 1
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        beta: float = 1,
        coef_categories: float = 1,
        coef_attributes: float = 1,
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0,
        scheduler_args: SchedulerArgs | None = None,
    ):
        """
        Defines the Actions domain module.

        Args:
            latent_dim (`int`): the latent dimension of the module
            hidden_dim (`int`): hidden dimension of the VAE encoders and decoders
            beta (`float`): for beta-VAE
            coef_categories (`float`): loss coefficient attributed to the category
                (Defaults to 1.0)
            coef_attributes (`float`): loss coefficient attributed to the rest of the
                attributes (Defaults to 1.0)
            optim_lr (`float`): learning rate for the optimizer
            optim_weight_decay (`float`): weight decay for the optimizer
            scheduler_args (`SchedulerArgs | None`): Scheduler arguments
        """
        super().__init__(latent_dim)
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.coef_categories = coef_categories
        self.coef_attributes = coef_attributes

        vae_encoder = Encoder(self.in_dim,self.hidden_dim, self.latent_dim)
        vae_decoder = Decoder(self.in_dim,self.latent_dim, self.hidden_dim)
        self.vae = VAE(vae_encoder, vae_decoder, beta)

        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay

        self.scheduler_args = SchedulerArgs(
            max_lr=optim_lr,
            total_steps=1,
        )
        self.scheduler_args.update(scheduler_args or {})
    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any
    ) -> LossOutput:
        return LossOutput(F.mse_loss(pred, target, reduction="sum"))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x.float())

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.vae.decode(z.float())
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.decode(self.encode(x.float()))

    def generic_step(
        self,
        x: torch.Tensor,
        mode: str = "train",
    ) -> torch.Tensor:
        # this calles model.__call__(x)  since it inherets from nn.Module
        # which internally runs: vae.forward(x)
        (mean, logvar), reconstruction = self.vae(x)

        reconstruction_loss = mse_loss(reconstruction, x, reduction = "sum")

        kl_loss = kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + self.vae.beta * kl_loss

        self.log(f"{mode}/reconstruction_loss", reconstruction_loss)
        self.log(f"{mode}/kl_loss", kl_loss)
        self.log(f"{mode}/loss", total_loss)
        return total_loss

    def validation_step(  # type: ignore
        self, batch: Mapping[frozenset[str], Mapping[str, Sequence[torch.Tensor]]], _
    ) -> torch.Tensor:
        x = batch[frozenset(["act"])]["act"].float()
        return self.generic_step(x, "val")

    def training_step(  # type: ignore
        self,
        batch: Mapping[frozenset[str], Mapping[str, Sequence[torch.Tensor]]],
        _,
    ) -> torch.Tensor:
        x = batch[frozenset(["act"])]["act"].float()
        return self.generic_step(x, "train")

    def configure_optimizers(  # type: ignore
        self,
    ) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optim_lr,
            weight_decay=self.optim_weight_decay,
        )
        lr_scheduler = OneCycleLR(optimizer, **self.scheduler_args)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }
    
class AttributeLegacyDomainModule(DomainModule):
    latent_dim = 13

    def __init__(self):
        super().__init__(self.latent_dim)
        self.save_hyperparameters()

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any
    ) -> LossOutput:
        pred  = self.decode(pred)
        target  = self.decode(target)

        loss = F.mse_loss(pred, target, reduction="mean")

        return LossOutput(loss)

    def encode(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> list:
        return z

    def forward(self, x: Sequence[torch.Tensor]) -> list[torch.Tensor]:  # type: ignore
        return self.decode(self.encode(x))


class ActionLegacyDomainModule(DomainModule):
    latent_dim = 4

    def __init__(self):
        super().__init__(self.latent_dim)
        self.save_hyperparameters()

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any
    ) -> LossOutput:
        pred  = self.decode(pred)
        target  = self.decode(target)

        loss = F.mse_loss(pred, target, reduction="mean")

        return LossOutput(loss)

    def encode(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> list:
        return z

    def forward(self, x: Sequence[torch.Tensor]) -> list[torch.Tensor]:  # type: ignore
        return self.decode(self.encode(x))
