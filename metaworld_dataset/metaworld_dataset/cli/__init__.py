import click

#from simple_shapes_dataset.cli.download import download_dataset
#from simple_shapes_dataset.cli.migration import migrate_dataset_command

from .alignments import alignment_group, create_domain_split
from .create_dataset import (
    create_dataset,
    create_ood_split,
    create_unpaired_attributes,
    unpaired_attributes_command,
)
from .odd_one_out import create_odd_one_out_dataset
from .ood_splits import (
    BinsBoundary,
    BoundaryBase,
    BoundaryInfo,
    BoundaryKind,
    ChoiceBoundary,
    MultiBinsBoundary,
    attr_boundaries,
    filter_dataset,
    ood_split,
)
from .utils import (
    Dataset,
    define_domain_split,
    generate_class,
    generate_color,
    generate_dataset,
    generate_image,
    generate_location,
    generate_rotation,
    generate_scale,
    generate_unpaired_attr,
    get_deterministic_name,
    get_diamond_patch,
    get_domain_alignment,
    get_egg_patch,
    get_transformed_coordinates,
    get_triangle_patch,
    load_labels,
    load_labels_old,
    save_bert_latents,
    save_dataset,
    save_labels,
)


@click.group()
def cli():
    pass


cli.add_command(create_dataset)
cli.add_command(alignment_group)
cli.add_command(unpaired_attributes_command)
cli.add_command(create_ood_split)
cli.add_command(create_odd_one_out_dataset)
#cli.add_command(download_dataset)
#cli.add_command(migrate_dataset_command)


__all__ = [
    "alignment_group",
    "create_domain_split",
    "create_dataset",
    "create_ood_split",
    "create_unpaired_attributes",
    "unpaired_attributes_command",
    "create_odd_one_out_dataset",
    "BinsBoundary",
    "BoundaryBase",
    "BoundaryInfo",
    "BoundaryKind",
    "ChoiceBoundary",
    "MultiBinsBoundary",
    "attr_boundaries",
    "filter_dataset",
    "ood_split",
    "Dataset",
    "define_domain_split",
    "generate_class",
    "generate_color",
    "generate_dataset",
    "generate_image",
    "generate_location",
    "generate_rotation",
    "generate_scale",
    "generate_unpaired_attr",
    "get_deterministic_name",
    "get_diamond_patch",
    "get_domain_alignment",
    "get_egg_patch",
    "get_transformed_coordinates",
    "get_triangle_patch",
    "load_labels",
    "load_labels_old",
    "save_bert_latents",
    "save_dataset",
    "save_labels",
    "cli",
    "migrate_dataset_command",
]
