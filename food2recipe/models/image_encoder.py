# File: food2recipe/models/image_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from food2recipe.core.settings import load_settings
from food2recipe.core.logging_utils import setup_logger

logger = setup_logger("image_encoder")

class ImageEncoder:
    def __init__(self, settings=None):
        self.settings = settings or load_settings()
        self.device = torch.device(self.settings.DEVICE if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        self._load_model()

    def _load_model(self):
        backend = self.settings.MODEL_BACKEND.lower()
        model_name = self.settings.MODEL_NAME
        pretrained = self.settings.PRETRAINED_DATASET
        
        logger.info(f"Loading model: {backend} - {model_name} (on {self.device})")
        
        if backend == "open_clip":
            try:
                import open_clip
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name, 
                    pretrained=pretrained,
                    device=self.device
                )
                self.model = model
                self.preprocess = preprocess # Note: We use our own deterministic transform usually, but keep this ref
                logger.info("OpenCLIP model loaded.")
            except ImportError:
                logger.warning("open_clip not found. Install it or switch backend. Fallback to timm?")
                raise
        elif backend == "timm":
            try:
                import timm
                # Load a model that outputs embeddings (remove classifier)
                self.model = timm.create_model(model_name, pretrained=True, num_classes=0).to(self.device)
                self.model.eval()
                logger.info("Timm model loaded.")
            except ImportError:
                logger.error("timm not installed.")
                raise
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def encode(self, images_tensor):
        """
        Encodes a batch of images.
        images_tensor: Tensor of shape (B, C, H, W)
        Returns: Normalized embeddings (B, D)
        """
        # Ensure input is on device
        images_tensor = images_tensor.to(self.device)
        
        with torch.no_grad():
            if self.settings.MODEL_BACKEND == "open_clip":
                features = self.model.encode_image(images_tensor)
            else:
                features = self.model(images_tensor)
                
            # L2 Normalize
            features = F.normalize(features, p=2, dim=1)
            
        return features.cpu()

if __name__ == "__main__":
    # Test smoke
    enc = ImageEncoder()
    print("Encoder initialized.")
