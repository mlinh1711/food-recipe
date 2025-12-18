# File: food2recipe/preprocessing/image_preprocess.py
from PIL import Image
from torchvision import transforms
from food2recipe.core.settings import load_settings

# Standard ImageNet / CLIP normalization stats
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

def get_transforms(mode="inference", image_size=224):
    """
    Returns image transforms.
    Use OpenAI CLIP mean/std for best results if using CLIP.
    """
    if mode == "train":
        # For building index/training, we might want simple resize or some augmentation
        # But for retrieval index, usually we use deterministic resize too unless doing data aug for robustness
        # Requirement: "Transform for training/build index: resize, center crop, normalize"
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:
        # Inference: Deterministic
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

def load_and_transform_image(image_path_or_file, transform):
    """
    Loads an image (path or file-like) and applies transform.
    """
    try:
        image = Image.open(image_path_or_file).convert("RGB")
        return transform(image)
    except Exception as e:
        # Handle partially corrupted images or read errors
        raise ValueError(f"Error processing image: {e}")
