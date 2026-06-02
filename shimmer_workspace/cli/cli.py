import click

from shimmer_metaworld.cli.config import config_group
from shimmer_metaworld.cli.download import download_group
from shimmer_metaworld.cli.extract import save_v_latents_command
from shimmer_metaworld.cli.migrate import migrate_domains_command
from shimmer_metaworld.cli.train_attr import train_attr_command
from shimmer_metaworld.cli.train_gw import train_gw_command
from shimmer_metaworld.cli.train_t import train_t_command
from shimmer_metaworld.cli.train_v import train_v_command


@click.group()
def cli():
    pass


cli.add_command(migrate_domains_command)
cli.add_command(download_group)
cli.add_command(config_group)


@cli.group("train")
def train_group():
    pass


train_group.add_command(train_v_command)
train_group.add_command(train_attr_command)
train_group.add_command(train_t_command)
train_group.add_command(train_gw_command)


@cli.group("extract")
def extract_group():
    pass


extract_group.add_command(save_v_latents_command)
