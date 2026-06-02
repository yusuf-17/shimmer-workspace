import argparse
import glob
import re

from shimmer.modules.global_workspace import GlobalWorkspaceFusion
from shimmer_metaworld import DEBUG_MODE, PROJECT_DIR
from shimmer_metaworld.config import load_config
from shimmer_metaworld.modules.domains import load_pretrained_domains


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GW action decoder layers in a checkpoint")
    parser.add_argument("--gw-ckpt", default="8wfdyfub", help="Wandb/run checkpoint id suffix")
    parser.add_argument("--epoch", default="250", help="Checkpoint epoch number")
    args = parser.parse_args()

    config = load_config(
        PROJECT_DIR / "shimmer_metaworld" / "config_template",
        load_files=["train_gw.yaml"],
        debug_mode=DEBUG_MODE,
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

    ckpt_path = glob.glob(f'/mnt/datashare/yelhelw/checkpoints/shimmer-meta-{args.gw_ckpt}/*.ckpt')[0]
    gw = GlobalWorkspaceFusion.load_from_checkpoint(
        ckpt_path,
        domain_mods=domain_modules,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
        weights_only=False,
    )

    state_dict = gw.state_dict()
    key_pattern = re.compile(r"^gw_mod\.gw_decoders\.act\.(\d+)\.(weight|bias)$")

    matched = []
    for key in state_dict:
        m = key_pattern.match(key)
        if m:
            matched.append((int(m.group(1)), m.group(2), key, tuple(state_dict[key].shape)))

    matched.sort(key=lambda x: (x[0], x[1]))

    print(f"Checkpoint: {ckpt_path}")
    print(f"Matched act decoder parameter keys: {len(matched)}")

    if not matched:
        print("No gw_mod.gw_decoders.act.* keys found.")
        return

    layer_indices = sorted({idx for idx, _, _, _ in matched})
    linear_layer_count = len(layer_indices)
    print(f"Decoder linear layer indices: {layer_indices}")
    print(f"Decoder linear layer count: {linear_layer_count}")

    print("\nPer-layer shapes:")
    for idx in layer_indices:
        weight_item = next((item for item in matched if item[0] == idx and item[1] == "weight"), None)
        bias_item = next((item for item in matched if item[0] == idx and item[1] == "bias"), None)
        w_shape = weight_item[3] if weight_item else None
        b_shape = bias_item[3] if bias_item else None
        print(f"  act.{idx}: weight={w_shape}, bias={b_shape}")


if __name__ == "__main__":
    main()
