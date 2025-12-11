from collections.abc import Sequence
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
import wandb
from PIL import Image
from torchvision.utils import make_grid

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from shimmer_metaworld import DEBUG_MODE, PROJECT_DIR
from shimmer_metaworld.ckpt_migrations import migrate_model
from shimmer_metaworld.config import load_config
from shimmer_metaworld.logging import get_pil_image
from shimmer_metaworld.modules.domains.visual import VisualDomainModule
from shimmer_metaworld.modules.vae import RAEEncoder, dim_exploration_figure
import torchvision.transforms as transforms
matplotlib.use("Agg")


def image_grid_from_v_tensor(
    samples: Sequence[torch.Tensor],
    _: int,
    ncols: int,
) -> Image:
    image = make_grid(samples[0], nrow=ncols, pad_value=1).detach()
    return F.to_pil_image(image)


def main() -> None:
    config = load_config(
        PROJECT_DIR / "shimmer_metaworld"/"config_template",
        load_files=["viz_vae_v.yaml"],
        debug_mode=DEBUG_MODE,
    )

    if config.visualization is None:
        raise ValueError("Visualization config should be defined for this script.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = config.default_root_dir / config.visualization.explore_vae.checkpoint
    migrate_model(ckpt_path, PROJECT_DIR / "shimmer_metaworld" / "migrations" / "visual_mod")
    domain_module = VisualDomainModule.load_from_checkpoint(ckpt_path)
    domain_module.eval().freeze()

    num_samples = 15
    range_start = -2
    range_end = 2
    val_samples = []
    '''
    img = Image.open("/mnt/datashare/yelhelw/expert_frames_final/val/rand_env_082400.png")
    transform = transforms.ToTensor()
    val_samples.append(transform(img))
    with torch.no_grad():
        z = domain_module.vae.encoder(img)
    z_mean = z.mean(0)
    z_std = z.std(0)
    print(z_mean,z_std)
    '''
    fig = dim_exploration_figure(
        domain_module.vae,
        cast(RAEEncoder, domain_module.vae.encoder).z_dim,
        device,
        image_grid_from_v_tensor,
        num_samples,
        range_start,
        range_end,
    )
    caption = f"VAE_Exploration_from_{range_start}_to_{range_end}"
    image = get_pil_image(fig)

    if config.visualization.explore_vae.wandb_name is not None:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            id=config.visualization.explore_vae.wandb_name,
            resume=True,
        )
        wandb.log({"vae_exploration": [wandb.Image(image, caption=caption)]})

    else:
        plt.savefig(PROJECT_DIR / f"data/{caption}.pdf")
        plt.show()


if __name__ == "__main__":
    main()
