from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Generic, NamedTuple, TypedDict, TypeVar

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class DomainDesc:
    base: str
    kind: str


class DomainType(Enum):
    """
    Different domain types available. Each model type is
    described as with a `DomainDesc` which represents a base type and a kind.

    For example "v_latents" has a base of "v" as it a special representation of
    the visual domain.
    """

    v = DomainDesc("v", "v")  # Uses images and visual VAE to encode images
    # Uses pre-saved latent representations extracted from the visual VAE
    v_latents = DomainDesc("v", "v_latents")
    attr = DomainDesc("attr", "attr")
    act = DomainDesc("act", "act")
    raw_text = DomainDesc("t", "raw_text")  # raw text representations (str)
    t = DomainDesc("t", "t")  # loads BERT representations of the raw text
    tact = DomainDesc("tact", "tact")  

# TODO: Consider handling CPU usage
# with a workaround in:
# https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662


_T = TypeVar("_T")


class DataDomain(ABC, Generic[_T]):
    """
    Base class for a domain of the SimpleShapesDataset.
    All domains extend this base class and implement the
    __getitem__ and __len__ methods.
    """

    @abstractmethod
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        transform: Callable[[Any], _T] | None = None,
        additional_args: dict[str, Any] | None = None,
    ) -> None:
        """
        Params:
            dataset_path (str | pathlib.Path): Path to the dataset.
            split (str): The split of the dataset to use. One of "train", "val", "test".
            transform (Any -> Any): Optional transform to apply to the data.
            additional_args (dict[str, Any]): Optional additional arguments to pass
                to the domain.
        """
        ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> _T: ...


class MetaworldImages(DataDomain):
    """
    Domain for the images of the SimpleShapesDataset.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        transform: Callable[[Image.Image], Any] | None = None,
        additional_args: dict[str, Any] | None = None,
    ) -> None:
        assert split in ("train", "val", "test"), "Invalid split"

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_path = (self.dataset_path / self.split).resolve()
        #TODO reformat the folder so it has png indices so we can go back to the normal index
        self.image_files = sorted(
        self.image_path.glob("*.png"),
        key=lambda f: int(f.stem.split("_")[-1])  # sort by the zero-padded number
        )
        self.transform = transform
        self.additional_args = additional_args
        self._dataset_size: int | None = None

    @property
    def dataset_size(self) -> int:
        if self._dataset_size is None:
            self._dataset_size = self._get_dataset_length()
        return self._dataset_size

    def _get_dataset_length(self) -> int:
        size = 0
        for file in self.image_path.iterdir():
            if file.suffix == ".png":
                size += 1
        return size

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int):
        """
        Params:
            index: The index of the image to retrieve.
        Returns:
            A PIL image at the given index.
        """
        path = self.image_files[index]
        #path = self.image_path / f"{index}.png"
        with Image.open(path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                return self.transform(image)
            return image


class PretrainedVisualAdditionalArgs(TypedDict):
    presaved_path: str
    use_unpaired: bool


class MetaworldPretrainedImages(DataDomain):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        transform: Callable[[torch.Tensor], Any] | None = None,
        additional_args: PretrainedVisualAdditionalArgs | None = None,
    ) -> None:
        assert split in ("train", "val", "test"), "Invalid split"

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transform = transform
        default_args = PretrainedVisualAdditionalArgs(
            presaved_path=".", use_unpaired=False
        )
        self.additional_args = default_args
        if additional_args is not None:
            self.additional_args.update(additional_args)

        self.presaved_path = (
            self.dataset_path
            / f"saved_latents/{split}/{self.additional_args['presaved_path']}"
        )
        self.latents = torch.from_numpy(np.load(self.presaved_path.resolve()))
        self.dataset_size = self.latents.size(0)

        if self.additional_args["use_unpaired"]:
            assert (self.dataset_path / f"{split}_unpaired.npy").exists()
            unpaired = np.load(self.dataset_path / f"{split}_unpaired.npy")
            self.unpaired = torch.from_numpy(unpaired[:, 1]).float()
        else:
            self.unpaired = torch.zeros((self.latents.size(0),)).float()

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int):
        if self.additional_args["use_unpaired"]:
            x = torch.cat(
                [self.latents[index], self.unpaired[index].unsqueeze(0)], dim=0
            )
        else:
            x = self.latents[index]
        return x.float()


class Attribute(NamedTuple):
    """
    NamedTuple for the attributes of the MetaworldDataset.
    NamedTuples are used as they are correcly handled by pytorch's collate function.
    """

    proprio_x: torch.Tensor
    proprio_y : torch.Tensor
    proprio_z : torch.Tensor
    proprio_gripper : torch.Tensor
    ball: torch.Tensor
    wall: torch.Tensor
    soccer_goal: torch.Tensor
    unpaired: torch.Tensor | None


class Action(NamedTuple):
    """
    NamedTuple for the actions of the MetaworldDataset.
    NamedTuples are used as they are correcly handled by pytorch's collate function.
    """
    dis_x: torch.Tensor
    dis_y: torch.Tensor
    dis_z: torch.Tensor
    gripper: torch.Tensor
    unpaired: torch.Tensor | None

class AttributesAdditionalArgs(TypedDict):
    n_unpaired: int



class MetaworldAttributes(DataDomain):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        transform: Callable[[Attribute], Any] | None = None,
        additional_args: AttributesAdditionalArgs | None = None,
    ) -> None:
        assert split in ("train", "val", "test"), "Invalid split"

        self.dataset_path = Path(dataset_path).resolve()
        self.split = split
        self.labels: torch.Tensor = torch.from_numpy(
            np.load(self.dataset_path / f"attributes_{split}.npy")
        )
        self.transform = transform

        default_args = AttributesAdditionalArgs(n_unpaired=0)
        self.additional_args = additional_args or default_args
        self.dataset_size = self.labels.size(0)

        self.unpaired = None
        if self.additional_args["n_unpaired"] >= 1:
            if not (self.dataset_path / f"{split}_unpaired.npy").exists():
                raise ValueError(
                    "Asking for an unpaired attribute, "
                    "but there is no unpaired label file."
                )
            self.unpaired = torch.from_numpy(
                np.load(self.dataset_path / f"{split}_unpaired.npy")[
                    :, 2 : 2 + self.additional_args["n_unpaired"]
                ]
            ).float()

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int):
        """
        Returns:
            An Attribute named tuple at the given index.
        """
        label = self.labels[index]
        unpaired = self.unpaired[index] if self.unpaired is not None else None
        #we define an Attributes named tuple to be able to preprocess each part of the Attributes
        #seperately then they are combined with transform attribute_to_tensor 
        item = Attribute(
            proprio_x=label[0],
            proprio_y=label[1],
            proprio_z=label[2],
            proprio_gripper=label[3],
            ball = label[4:11],
            wall = label[11:14],
            soccer_goal=label[14:17],
            unpaired=self.unpaired,
            )

        if self.transform is not None:
            return self.transform(item)
        return item.float()


class MetaworldActions(DataDomain):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        transform: Callable[[Attribute], Any] | None = None,
        additional_args: AttributesAdditionalArgs | None = None,
    ) -> None:
        assert split in ("train", "val", "test"), "Invalid split"

        self.dataset_path = Path(dataset_path).resolve()
        self.split = split
        self.labels: torch.Tensor = torch.from_numpy(
            np.load(self.dataset_path / f"actions_{split}.npy")
        )
        self.transform = transform

        default_args = AttributesAdditionalArgs(n_unpaired=0)
        self.additional_args = additional_args or default_args
        self.dataset_size = self.labels.size(0)

        self.unpaired = None
        if self.additional_args["n_unpaired"] >= 1:
            if not (self.dataset_path / f"{split}_unpaired.npy").exists():
                raise ValueError(
                    "Asking for an unpaired attribute, "
                    "but there is no unpaired label file."
                )
            self.unpaired = torch.from_numpy(
                np.load(self.dataset_path / f"{split}_unpaired.npy")[
                    :, 2 : 2 + self.additional_args["n_unpaired"]
                ]
            ).float()

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int):
        """
        Returns:
            An Attribute named tuple at the given index.
        """
        label = self.labels[index]
        unpaired = self.unpaired[index] if self.unpaired is not None else None

        item = Action(
            dis_x=label[0],
            dis_y = label[1],
            dis_z=label[2],
            gripper = label[3],
            unpaired= unpaired
        )

        if self.transform is not None:
            return self.transform(item)
        return item.float()

class Choice(NamedTuple):
    structure: int
    groups: list[int]
    writers: dict[str, dict[str, int]]
    variants: dict[str, int]


class RawText(NamedTuple):
    caption: str
    choice: Choice


class Text(NamedTuple):
    caption: str
    bert: torch.Tensor
    choice: Choice
    attr: Attribute


class SimpleShapesRawText(DataDomain):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        transform: Callable[[RawText], Any] | None = None,
        additional_args: dict[str, Any] | None = None,
    ) -> None:
        assert split in ("train", "val", "test"), "Invalid split"

        self.dataset_path = Path(dataset_path).resolve()
        self.split = split

        self.captions = np.load(self.dataset_path / f"{split}_captions.npy")
        self.choices = np.load(
            self.dataset_path / f"{split}_caption_choices.npy",
            allow_pickle=True,
        )
        self.transform = transform
        self.additional_args = additional_args or {}
        self.dataset_size = len(self.captions)

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int):
        item = RawText(
            caption=self.captions[index], choice=Choice(**self.choices[index])
        )

        if self.transform is not None:
            return self.transform(item)
        return item


class SimpleShapesText(DataDomain):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        transform: Callable[[Text], Any] | None = None,
        additional_args: dict[str, Any] | None = None,
    ) -> None:
        """
        Possible additional args:
            latent_filename: The name of the model used to get the latent file.
                It will load files of the form {split}_{latent_filename}.npy.
        """
        assert split in ("train", "val", "test"), "Invalid split"

        self.dataset_path = Path(dataset_path).resolve()
        self.split = split

        self.additional_args = additional_args or {}
        self.latent_filename = self.additional_args.get("latent_filename", "latent")

        self.raw_text = SimpleShapesRawText(self.dataset_path, self.split)
        self.attributes = MetaworldAttributes(self.dataset_path, self.split)

        self.bert_mean = torch.from_numpy(
            np.load(self.dataset_path / f"{self.latent_filename}_mean.npy")
        )
        self.bert_std = torch.from_numpy(
            np.load(self.dataset_path / f"{self.latent_filename}_std.npy")
        )

        bert_data = torch.from_numpy(
            np.load(self.dataset_path / f"{self.split}_{self.latent_filename}.npy")
        )
        assert bert_data.ndim == 2
        self.bert_data = (bert_data - self.bert_mean) / self.bert_std
        self.transform = transform
        self.dataset_size = self.bert_data.size(0)

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int):
        item = Text(
            caption=self.raw_text[index].caption,
            bert=self.bert_data[index],
            choice=self.raw_text[index].choice,
            attr=self.attributes[index],
        )

        if self.transform is not None:
            return self.transform(item)
        return item


DEFAULT_DOMAINS: dict[str, type[DataDomain]] = {
    "v": MetaworldImages,
    "v_latents": MetaworldPretrainedImages,
    "attr": MetaworldAttributes,
    "raw_text": SimpleShapesRawText,
    "t": SimpleShapesText,
    "act" : MetaworldActions,
}


#gets required domains and gives appropriate data domains
def get_default_domains(
    domains: Iterable[DomainDesc | str],
) -> dict[DomainDesc, type[DataDomain]]:
    domain_classes = {}
    for domain in domains:
        if isinstance(domain, str):
            domain = DomainType[domain].value
        domain_classes[domain] = DEFAULT_DOMAINS[domain.kind]
    return domain_classes


def get_default_domains_dataset(
    domains: Iterable[DomainDesc | str],
    dataset_path: str | Path,
    split: str,
    transforms: Mapping[str, Callable[[Any], Any]],
    domain_args: Mapping[str, Any] = {},
) -> dict[DomainDesc, DataDomain]:
    domain_classes = {}
    for domain in domains:
        if isinstance(domain, str):
            domain = DomainType[domain].value
        domain_classes[domain] = DEFAULT_DOMAINS[domain.kind](
            dataset_path,
            split,
            transforms[domain.kind],
            domain_args.get(domain.kind, None),
        )
    return domain_classes
