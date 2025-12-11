from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PIL.Image import Image
from shimmer.modules.vae import VAE, VAEDecoder, VAEEncoder
from torch import nn
from PIL.Image import Image
from torchvision.utils import make_grid
import torchvision.transforms.functional as F


class RAEEncoder(VAEEncoder):
    def __init__(
        self,
        num_channels: int,
        ae_dim: int = 1028,
        z_dim: int = 16,
        kernel_dim: int = 4,
        padding: int = 1,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.dims = [
            ae_dim // (2**i) for i in reversed(range(5))
        ]  # 1 2 4 8 # 32 64 128 256

        self.kernel_dim = kernel_dim
        self.padding = padding
        self.dims[-1] = ae_dim
        self.use_batchnorm = use_batchnorm

        self.out_dim = self.dims[4] * 2 * 2
        self.z_dim = z_dim

        self.layers = nn.Sequential(
            nn.Conv2d(
                num_channels,
                self.dims[0],
                kernel_size=self.kernel_dim,
                stride=4,
                padding=0,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[0]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.dims[0],
                self.dims[1],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[1]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.dims[1],
                self.dims[2],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[2]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.dims[2],
                self.dims[3],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[3]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.dims[3],
                self.dims[4],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[4]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
        )

        self.q_mean = nn.Linear(self.out_dim, self.z_dim)
        self.q_logvar = nn.Linear(self.out_dim, self.z_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.layers(x).view(x.size(0), -1)
        out = out.view(out.size(0), -1)

        return self.q_mean(out), self.q_logvar(out)


class RAEDecoder(VAEDecoder):
    def __init__(
        self,
        num_channels: int,
        z_dim: int = 16,
        ae_dim: int = 1028,
        kernel_dim: int = 4,
        padding: int = 1,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.dims = [ae_dim // (2**i) for i in reversed(range(5))]
        self.dims[-1] = ae_dim

        self.kernel_dim = kernel_dim
        self.padding = padding
        self.use_batchnorm = use_batchnorm

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                z_dim,
                self.dims[4],
                kernel_size=4,
                stride=4,
                padding = 0,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[4]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.dims[4],
                self.dims[3],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[3]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.dims[3],
                self.dims[2],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[2]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.dims[2],
                self.dims[1],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[1]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.dims[1],
                self.dims[0],
                kernel_size=self.kernel_dim,
                stride=4,
                padding=0,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[0]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
        )

        self.out_layer = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(
                self.dims[0],
                self.num_channels,
                kernel_size=self.kernel_dim,
                stride=1,
                padding=self.padding,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.out_layer(self.layers(z[:, :, None, None]))


def dim_exploration_figure(
    vae: VAE,
    z_size: int,
    device: torch.device,
    ax_from_tensors: Callable[[Any, int, int], Image],
    num_samples: int = 5,
    range_start: int = -6,
    range_end: int = 6,
    image_size: int = 32,
    plot_dims: Sequence[int] | None = None,
    fig_dim: int = 5,
) -> Figure:
    possible_dims = plot_dims or np.arange(z_size)

    #fig_size = (len(possible_dims) - 1) * fig_dim

    fig = plt.figure(constrained_layout=True, figsize=(num_samples*2, len(possible_dims)*2))

    gs = GridSpec(len(possible_dims), 1, figure=fig)
    for i, dim_i in enumerate(possible_dims):
        ax = fig.add_subplot(gs[i])

        z = torch.zeros(num_samples,z_size).to(device)

        for p in range(num_samples):
            step = range_start + (range_end - range_start) * float(p) / float(
                num_samples - 1
            )
            z[p, dim_i] = step

        decoded_x = vae.decoder(z)

        #img_grid = ax_from_tensors(decoded_x, image_size, num_samples)
        image = make_grid(decoded_x, nrow=num_samples, pad_value=1).detach()
        
        ax.imshow(F.to_pil_image(image))
        ax.set_xlabel(f"range")
        ax.set_ylabel(f"dim {dim_i}")

    return fig
