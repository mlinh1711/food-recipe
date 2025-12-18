# File: food2recipe/preprocessing/text_preprocess.py
import re
import unicodedata
import pandas as pd
from typing import Dict, Optional, Any

from food2recipe.core.settings import load_settings
from food2recipe.core.logging_utils import setup_logger

logger = setup_logger("text_preprocess")


def normalize_food_name(name: str) -> str:
    """
    Normalize food names so that:
    - CSV class_name like "Banh beo"
    - image folder name like "BanhBeo" / "banh_beo" / "Bánh Bèo"
    can match the same key.

    Strategy:
    1) lowercase
    2) remove Vietnamese accents
    3) convert anything non-alphanumeric to underscore
    4) collapse multiple underscores, strip underscores

    Example:
    - "Banh beo"  -> "banh_beo"
    - "Bánh Bèo"  -> "banh_beo"
    - "BanhBeo"   -> "banhbeo"  (NOTE: camelcase won't insert underscore automatically)
      If your folders are camelcase, it still matches if you normalize folders the same way.
    """
    if not isinstance(name, str):
        return ""

    s = name.strip().lower()
    if not s:
        return ""

    # Remove accents (Vietnamese diacritics)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

    # Replace non-alphanumeric with underscore
    s = re.sub(r"[^a-z0-9]+", "_", s)

    # Collapse multiple underscores and strip
    s = re.sub(r"_+", "_", s).strip("_")
    return s


class RecipeProcessor:
    """
    Loads recipes CSV, validates schema, builds lookup map keyed by normalized food_name.

    Why normalized key:
    - Your image labels come from folder names
    - Your recipes labels come from CSV class_name
    - They may differ by spacing/diacritics/case
    - Normalize both sides to match reliably
    """

    def __init__(self, settings=None):
        self.settings = settings or load_settings()
        # Map normalized_key -> recipe_obj
        self.recipes_data: Dict[str, Any] = {}

    def load_and_process(self):
        """
        Loads CSV, validates columns, cleans text, and builds lookup map.
        """
        csv_path = self.settings.RECIPES_CSV

        if not csv_path.exists():
            logger.error(f"Recipes CSV not found at {csv_path}")
            raise FileNotFoundError(f"Recipes CSV not found at {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            raise

        logger.info(f"Loaded CSV with shape {df.shape}. Columns: {list(df.columns)}")

        # Column Mapping (configured from .env / settings)
        col_food = self.settings.FOOD_COL
        col_ing = self.settings.INGREDIENTS_COL
        col_instr = self.settings.INSTRUCTIONS_COL
        col_title = self.settings.TITLE_COL  # optional display

        # Validation
        required = [col_food, col_ing, col_instr]
        missing = [c for c in required if c not in df.columns]
        if missing:
            msg = (
                f"Missing required columns in CSV: {missing}.\n"
                f"Available columns: {list(df.columns)}\n"
                "Action required: Update .env mappings (FOOD_COL, etc.) or fix CSV header."
            )
            logger.error(msg)
            raise ValueError(msg)

        # Basic cleaning: fill NaN, strip
        df = df.fillna("")
        df[col_food] = df[col_food].astype(str).str.strip()
        df[col_ing] = df[col_ing].astype(str)
        df[col_instr] = df[col_instr].astype(str)

        # Create normalized key for stable matching
        # This is the key that downstream should use.
        df["food_key"] = df[col_food].apply(normalize_food_name)

        # If some rows normalize to empty string, drop them to avoid bad keys
        bad_rows = df["food_key"].eq("")
        if bad_rows.any():
            logger.warning(f"Dropping {int(bad_rows.sum())} rows with empty normalized food_key.")
            df = df.loc[~bad_rows].copy()

        # Group by normalized key, pick best recipe if duplicates exist
        grouped = df.groupby("food_key", dropna=False)

        self.recipes_data = {}

        for food_key, group in grouped:
            # Heuristic: pick recipe with longest instructions
            # Reason: often a longer instruction text indicates a more complete recipe.
            instr_lens = group[col_instr].str.len()
            best_idx = instr_lens.idxmax()
            best_row = group.loc[best_idx]

            recipe_obj = {
                # Use normalized key as primary key
                "food_key": food_key,

                # Keep original names for display/debug
                "food_name_raw": str(best_row[col_food]).strip(),

                "ingredients": self._clean_text(best_row[col_ing]),
                "instructions": self._clean_text(best_row[col_instr]),
                "raw_ingredients": best_row[col_ing],
                "raw_instructions": best_row[col_instr],
            }

            # Optional: add a nicer title for UI if available
            if col_title in df.columns:
                recipe_obj["title"] = str(best_row[col_title]).strip()

            self.recipes_data[food_key] = recipe_obj

        logger.info(f"Processed recipes for {len(self.recipes_data)} unique food items (normalized keys).")

        # Useful debug: show a few keys so you can compare with image folder keys
        sample_keys = list(self.recipes_data.keys())[:5]
        logger.info(f"Sample recipe keys: {sample_keys}")

    def _clean_text(self, text: str) -> str:
        """
        Basic text cleaning.
        Keep it simple to avoid breaking Vietnamese punctuation.
        """
        if not isinstance(text, str):
            return ""
        # Remove multiple spaces, preserve line meaning minimally
        text = text.replace("\r", "\n")
        text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
        return text

    def get_recipe(self, food_name_or_key: str) -> Optional[Dict[str, Any]]:
        """
        Accept either raw food name (e.g., 'Banh beo') or already-normalized key (e.g., 'banh_beo').
        This makes integration easier: caller can just pass folder name or predicted label.
        """
        if not isinstance(food_name_or_key, str):
            return None
        key = normalize_food_name(food_name_or_key)
        return self.recipes_data.get(key)


if __name__ == "__main__":
    processor = RecipeProcessor()
    processor.load_and_process()
    # Quick manual test
    print(processor.get_recipe("Banh beo"))
    print(processor.get_recipe("Bánh Bèo"))
