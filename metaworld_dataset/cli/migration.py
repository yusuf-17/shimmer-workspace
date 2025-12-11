from collections.abc import Callable
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from metaworld_dataset.cli.alignments import update_alignment_split_format
from metaworld_dataset.cli.create_dataset import create_unpaired_attributes
from metaworld_dataset.version import __version__


def update_0p1_to_1p0(dataset_path: Path, dry_run: bool):
    click.echo("Migrating from 0.1.0 to 1.0.0")
    click.echo("Updating split format...")
    if (dataset_path / "domain_splits").exists() and not (
        dataset_path / "domain_splits_v2"
    ).exists():
        update_alignment_split_format(dataset_path, dry_run)
    # Add unpaired attributes
    if not (dataset_path / "train_unpaired.npy").exists():
        click.echo("Adding unpaired attributes using seed [0]...")

        if not dry_run:
            create_unpaired_attributes(0, dataset_path)
    if (dataset_path / "train_bert-base-uncased.npy").exists():
        (dataset_path / "train_bert-base-uncased.npy").rename(
            dataset_path / "train_latent.npy"
        )
        (dataset_path / "val_bert-base-uncased.npy").rename(
            dataset_path / "val_latent.npy"
        )
        (dataset_path / "test_bert-base-uncased.npy").rename(
            dataset_path / "test_latent.npy"
        )
    if (dataset_path / "bert-base-uncased_mean.npy").exists():
        (dataset_path / "bert-base-uncased_mean.npy").rename(
            dataset_path / "latent_mean.npy"
        )
        (dataset_path / "bert-base-uncased_std.npy").rename(
            dataset_path / "latent_std.npy"
        )
    if (dataset_path / "mean_bert-base-uncased.npy").exists():
        (dataset_path / "mean_bert-base-uncased.npy").rename(
            dataset_path / "latent_mean.npy"
        )
        (dataset_path / "std_bert-base-uncased.npy").rename(
            dataset_path / "latent_std.npy"
        )

    # Only keep the first 500,00 samples
    old_length = np.load(dataset_path / "train_labels.npy").shape[0]
    if old_length == 1_000_000 and not dry_run:
        click.echo(
            "Detected oversized dataset. Keeping only the first 500,000 samples."
        )
        train_labels = np.load(dataset_path / "train_labels.npy")
        train_labels = train_labels[:500_000]
        np.save(dataset_path / "train_labels.npy", train_labels)

        train_unpaired_attr = np.load(dataset_path / "train_unpaired.npy")[:500_000]
        np.save(dataset_path / "train_unpaired.npy", train_unpaired_attr)

        train_captions = np.load(
            dataset_path / "train_captions.npy", allow_pickle=True
        )[:500_000]
        np.save(dataset_path / "train_captions.npy", train_captions)

        train_captions_choices = np.load(
            dataset_path / "train_caption_choices.npy", allow_pickle=True
        )[:500_000]
        np.save(dataset_path / "train_caption_choices.npy", train_captions_choices)

        train_bert_latents = np.load(dataset_path / "train_latent.npy")[:500_000]
        np.save(dataset_path / "train_latent.npy", train_bert_latents)

        mean = train_bert_latents.mean(axis=0)
        std = train_bert_latents.std(axis=0)
        np.save(dataset_path / "latent_mean.npy", mean)
        np.save(dataset_path / "latent_std.npy", std)

        click.echo("Removing extra images...")
        for k in tqdm(range(500_000, old_length)):
            (dataset_path / "train" / f"{k}.png").unlink()


migrations: list[Callable[[Path, bool], None]] = [update_0p1_to_1p0]

version_to_migration_start_index = {"0.1.0": 0}


def migrate_dataset(dataset_path: Path, dry_run: bool) -> None:
    current_version = "0.1.0"
    if (dataset_path / "version.txt").exists():
        with open(dataset_path / "version.txt") as version:
            current_version = version.read()
    click.echo(f"Detected database version: {current_version}")

    click.echo("Migrating")
    if current_version in version_to_migration_start_index:
        for migration in migrations[
            version_to_migration_start_index[current_version] :
        ]:
            migration(dataset_path, dry_run)

    if not dry_run:
        with open(dataset_path / "version.txt", "w") as version_file:
            version_file.write(__version__)
    click.echo(f"Dataset now on version {__version__}.")


@click.command("migrate", help="Update the dataset to the latest version")
@click.option(
    "--dataset_path",
    "-p",
    default=".",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the dataset",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Whether to do a dry-run pass",
)
def migrate_dataset_command(dataset_path: Path, dry_run: bool) -> None:
    migrate_dataset(dataset_path, dry_run)
