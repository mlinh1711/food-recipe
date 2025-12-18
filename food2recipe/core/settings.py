# File: food2recipe/core/settings.py
import os
from pathlib import Path
from typing import List, Optional, Dict
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """
    Project configuration settings.
    Reads from environment variables or .env file.
    Default paths assume the script logic runs relative to the project root.
    """
    
    # --- Paths ---
    # Default: assumes project root is where the script is run or 2 levels up from this file
    # food2recipe/core/settings.py -> food2recipe/core -> food2recipe -> ROOT
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    # Data Directories
    DATA_DIR: Path = Field(default_factory=lambda: Path("data")) # Default to data/ folder relative to execution
    IMAGES_DIR: Optional[Path] = Field(default=None) # Will be computed if None
    URLS_DIR: Optional[Path] = Field(default=None)   # Will be computed if None
    RECIPES_CSV: Optional[Path] = Field(default=None) # Will be computed if None
    

    ARTIFACTS_DIR: Optional[Path] = Field(default=None)
    REPORTS_DIR: Optional[Path] = Field(default=None)

    # --- Model Config ---
    MODEL_BACKEND: str = Field(default="open_clip", description="open_clip or timm")
    MODEL_NAME: str = Field(default="ViT-B-32", description="Model architecture name")
    PRETRAINED_DATASET: str = Field(default="laion2b_s34b_b79k", description="Pretrained weights tag")
    DEVICE: str = Field(default="cpu")
    
    # --- Processing ---
    IMAGE_SIZE: int = 224
    
    # --- Retrieval ---
    USE_FAISS: bool = True
    TOP_K: int = 5
    CONFIDENCE_THRESHOLD: float = 0.6  # If similarity < threshold -> Uncertain
    
    # --- CSV Mapping ---
    # Allow mapping CSV columns via config
    FOOD_COL: str = "class_name"
    INGREDIENTS_COL: str = "ingredients"
    INSTRUCTIONS_COL: str = "instructions"
    TITLE_COL: str = "vietnamese_name" # Optional display title
    
    class Config:
        env_file = ".env"
        extra = "ignore" # Ignore extra env vars

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set defaults if not provided, allowing for flexible 'Data root'
        # If user explicitly sets DATA_DIR in env, we use it.
        # Otherwise it defaults to 'data' in CWD.
        
        # NOTE: For the specific user context where 'Images' is in root:
        # We can implement a smart check or just stick to the requested "Data root: data/" structure.
        # But to be robust to the user's current directory state (images in root),
        # let's defaulting to looking in data/ first, then fall back to root if not found?
        # For 'Production-Lite' we strictly follow the config.
        
        if self.IMAGES_DIR is None:
            self.IMAGES_DIR = self.DATA_DIR / "Images"
        if self.URLS_DIR is None:
            self.URLS_DIR = self.DATA_DIR / "Urls"
        if self.RECIPES_CSV is None:
            self.RECIPES_CSV = self.DATA_DIR / "vnfood30_recipes.csv"
            
        if self.ARTIFACTS_DIR is None:
            self.ARTIFACTS_DIR = self.BASE_DIR / "food2recipe" / "artifacts"
        if self.REPORTS_DIR is None:
            self.REPORTS_DIR = self.BASE_DIR / "food2recipe" / "reports"

        # --- Sanity check paths ---
        if not self.IMAGES_DIR.exists():
            raise FileNotFoundError(f"IMAGES_DIR not found: {self.IMAGES_DIR.resolve()}")

        if not self.URLS_DIR.exists():
            raise FileNotFoundError(f"URLS_DIR not found: {self.URLS_DIR.resolve()}")

        if not self.RECIPES_CSV.exists():
            raise FileNotFoundError(f"RECIPES_CSV not found: {self.RECIPES_CSV.resolve()}")
    
        # Ensure output dirs exist
        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(self.REPORTS_DIR, exist_ok=True)

def load_settings() -> Settings:
    return Settings()
