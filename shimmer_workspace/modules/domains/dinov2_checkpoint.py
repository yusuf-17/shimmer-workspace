import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from typing import Union
from PIL.Image import Image
import numpy as np
from shimmer.modules import DomainModule, LossOutput
import torch.nn.functional as F
from typing import Any

class DINOv2FeatureExtractor(DomainModule):
    """
    A checkpoint-compatible module that extracts features from images using DINOv2.
    Wraps AutoImageProcessor and AutoModel for seamless preprocessing and inference.
    """

    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = None):
        """
        Initialize the DINOv2 feature extractor.

        Args:
            model_name: Name of the pretrained model (default: "facebook/dinov2-base")
            device: Device to use ("cuda", "cpu", or None for automatic selection)
        """
        super().__init__(latent_dim=768)  # DINOv2-base has a hidden size of 768
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self.model_name = model_name
       
        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        
        # Freeze model parameters (features are extracted, not fine-tuned)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images: Union[Image, np.ndarray, torch.Tensor, list]) -> torch.Tensor:
        """
        Extract features from input images.

        Args:
            images: Input image(s). Can be:
                - PIL Image
                - numpy array (H, W, C)
                - torch tensor (C, H, W) or (B, C, H, W)
                - list of images

        Returns:
            Feature tensor of shape (B, num_patches, hidden_dim) or (num_patches, hidden_dim)
        """
        # Handle different input types
        if isinstance(images, torch.Tensor):
            # If already a tensor, assume it's in the right format but may need preprocessing
            if images.ndim == 4:
                # Batch of images (B, C, H, W)
                pil_images = [
                    TensorToPIL(images[i]) for i in range(images.shape[0])
                ]
            elif images.ndim == 3:
                # Single image (C, H, W)
                pil_images = TensorToPIL(images)
            elif images.ndim == 2:
                # Single grayscale image (H, W)
                pil_images = TensorToPIL(images)
            else:
                raise ValueError(f"Unsupported tensor shape: {images.shape}")
        elif isinstance(images, np.ndarray):
            # Numpy array
            if images.ndim == 2:
                # Single grayscale image (H, W)
                pil_images = NumpyToPIL(images)
            elif images.ndim == 3 and images.shape[2] in [1, 3, 4]:
                # Single image (H, W, C)
                pil_images = NumpyToPIL(images)
            elif images.ndim == 4 and images.shape[-1] in [1, 3, 4]:
                # Batch of images
                pil_images = [NumpyToPIL(images[i]) for i in range(images.shape[0])]
            else:
                raise ValueError(f"Unsupported array shape: {images.shape}")
        elif isinstance(images, Image):
            pil_images = images
        elif isinstance(images, list):
            pil_images = images
        else:
            raise ValueError(f"Unsupported image type: {type(images)}")

        # Process images through processor
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)

        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding so output is [B, D] and compatible with GW losses.
            features = outputs.last_hidden_state[:, 0, :]

        return features

    def get_config(self) -> dict:
        """Get configuration for saving/loading."""
        return {
            "model_name": self.model_name,
            "device": self.device,
        }

    @classmethod
    def from_config(cls, config: dict) -> "DINOv2FeatureExtractor":
        """Load from configuration."""
        return cls(**config)

    def encode(self, images: Union[Image, np.ndarray, torch.Tensor, list]) -> torch.Tensor:
        """Alias for forward to match expected interface."""
        return self.forward(images)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Alias for forward to match expected interface."""
        return z

    @property
    def device(self) -> str:
        return self._device
    
    
    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any
    ) -> LossOutput:
        return LossOutput(F.mse_loss(pred, target, reduction="mean"))

def TensorToPIL(tensor: torch.Tensor) -> Image:
    """Convert torch tensor image to RGB PIL Image."""
    if tensor.dim() == 3:
        # Handle CHW tensors by moving channels to the last axis.
        if tensor.shape[0] in [1, 3, 4]:
            tensor = tensor.permute(1, 2, 0)
    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported tensor shape for image conversion: {tensor.shape}")
    
    # Ensure values are in [0, 255]
    if tensor.max() <= 1.0:
        tensor = (tensor * 255).byte()
    else:
        tensor = tensor.byte()
    
    array = tensor.cpu().numpy()

    # Promote grayscale to RGB for DINOv2.
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)

    from PIL import Image as PILImage
    return PILImage.fromarray(array).convert("RGB")


def NumpyToPIL(array: np.ndarray) -> Image:
    """Convert numpy image array to RGB PIL Image."""
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = (array * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)

    # Promote grayscale or single-channel inputs to RGB for DINOv2.
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    
    from PIL import Image as PILImage
    return PILImage.fromarray(array).convert("RGB")
