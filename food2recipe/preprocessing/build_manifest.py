# File: food2recipe/preprocessing/build_manifest.py
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from food2recipe.core.settings import load_settings
from food2recipe.core.logging_utils import setup_logger

# Import the SAME normalizer used by recipe processing
# Why: image folder labels and recipe CSV labels may differ in case/spacing/diacritics.
# Using one shared normalizer ensures consistent join keys across modules.
from food2recipe.preprocessing.text_preprocess import normalize_food_name

logger = setup_logger("build_manifest")


def _normalize_split_name(split_name: str) -> str:
    """
    Normalize split folder names to a consistent set: train / val / test.

    Why we need this:
    - Your dataset folders are named like: Train, Test, Validate (capitalized)
    - But downstream code filters by 'train' and 'val'
    - Without normalization, filtering returns 0 rows, so no embeddings are generated.
    """
    s = (split_name or "").strip().lower()

    if s in ["train", "training"]:
        return "train"
    if s in ["val", "valid", "validation", "validate"]:
        return "val"
    if s in ["test", "testing"]:
        return "test"

    # Unknown split name: keep normalized lowercase version so it is still visible in manifest/logs
    return s


def build_manifest(settings=None) -> Path:
    """
    Scans the Images directory and creates a manifest.csv.

    Expected structure:
      Images/{split}/{food_name}/*.(jpg|jpeg|png)

    Output columns:
      - image_path: absolute path to image
      - split: normalized split name (train/val/test)
      - split_raw: original split folder name
      - food_name: NORMALIZED key (the one used for joining with recipe CSV)
      - food_name_raw: original class folder name (for debugging)
    """
    settings = settings or load_settings()
    images_dir = Path(settings.IMAGES_DIR)

    if not images_dir.exists():
        logger.error(f"Images directory not found at: {images_dir}")
        raise FileNotFoundError(f"Images directory not found at: {images_dir}")

    # Find split folders
    split_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower())
    splits_raw = [d.name for d in split_dirs]
    logger.info(f"Scanning images in {images_dir}, found splits: {splits_raw}")

    records = []

    for split_path in split_dirs:
        split_raw = split_path.name
        split_norm = _normalize_split_name(split_raw)

        # Each folder inside split is a class name
        class_dirs = [d for d in split_path.iterdir() if d.is_dir()]
        class_dirs = sorted(class_dirs, key=lambda p: p.name.lower())

        if not class_dirs:
            logger.warning(f"No class folders found under split: {split_path}")
            continue

        for class_path in tqdm(class_dirs, desc=f"Scanning {split_raw}"):
            food_name_raw = class_path.name

            # Normalize folder label to match recipe keys
            # Example:
            #   folder: "BanhBeo" / "Banh beo" / "Bánh Bèo"
            #   key:    "banh_beo"
            food_key = normalize_food_name(food_name_raw)

            if not food_key:
                # If normalization produced empty string, skip to avoid bad keys breaking retrieval/eval
                logger.warning(f"Skipping class with empty normalized key. Raw folder name: {food_name_raw}")
                continue

            # Gather all images
            images = []
            images += list(class_path.glob("*.jpg"))
            images += list(class_path.glob("*.jpeg"))
            images += list(class_path.glob("*.png"))

            if not images:
                continue

            for img_path in images:
                records.append(
                    {
                        "image_path": str(img_path.resolve()),
                        "split": split_norm,          # normalized for downstream filtering
                        "split_raw": split_raw,       # keep raw split folder
                        "food_name": food_key,        # normalized join key (IMPORTANT)
                        "food_name_raw": food_name_raw,  # raw folder name for debugging
                    }
                )

    if not records:
        logger.warning(f"No images found in {images_dir}. Check structure: Images/<split>/<class>/*.jpg")
        return None

    df = pd.DataFrame(records)

    # Quick sanity summary so you can see if normalization worked
    num_images = len(df)
    num_classes_norm = df["food_name"].nunique()
    num_classes_raw = df["food_name_raw"].nunique()
    split_counts = df["split"].value_counts().to_dict()

    logger.info(f"Found {num_images} images.")
    logger.info(f"Classes (normalized): {num_classes_norm}, Classes (raw folders): {num_classes_raw}")
    logger.info(f"Split counts after normalization: {split_counts}")

    # Helpful debug: show a few label mappings (raw -> normalized)
    sample_map = (
        df[["food_name_raw", "food_name"]]
        .drop_duplicates()
        .head(10)
        .to_dict(orient="records")
    )
    logger.info(f"Sample class mapping (raw -> key): {sample_map}")

    # Save manifest to artifacts
    output_path = Path(settings.ARTIFACTS_DIR) / "manifest.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Manifest saved to {output_path}")

    return output_path


if __name__ == "__main__":
    try:
        build_manifest()
    except Exception as e:
        logger.exception(f"Failed to build manifest: {e}")
