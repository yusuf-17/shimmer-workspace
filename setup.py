from pathlib import Path
from setuptools import setup

metaworld_dataset_path: str = (Path(__file__).parent / "metaworld_dataset").resolve().as_uri()


setup(
    name = "shimmer_workspace",
)