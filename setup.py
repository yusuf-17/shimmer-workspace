from pathlib import Path
from setuptools import setup

metaworld_dataset_path: str = (Path(__file__).parent / "packages/metaworld-dataset").resolve().as_uri()


setup(
    name = "meta_world",
    install_requires=[
        f"metaworld-dataset @{metaworld_dataset_path}",
    ],
)