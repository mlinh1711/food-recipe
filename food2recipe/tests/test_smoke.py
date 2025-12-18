# File: food2recipe/tests/test_smoke.py
import unittest
from pathlib import Path
from food2recipe.core.settings import load_settings
from food2recipe.preprocessing.text_preprocess import RecipeProcessor
from food2recipe.models.image_encoder import ImageEncoder

class SmokeTest(unittest.TestCase):
    def test_settings(self):
        settings = load_settings()
        self.assertIsNotNone(settings.DATA_DIR)
        
    def test_text_proc(self):
        # Only testing if valid init, not actual loading if CSV missing
        proc = RecipeProcessor()
        self.assertIsNotNone(proc)
        
    def test_encoder_init(self):
        # This requires torch/clip, might be slow or fail on some envs
        # Only run if open_clip/timm installed
        try:
            settings = load_settings()
            enc = ImageEncoder(settings)
            self.assertIsNotNone(enc.model)
        except ImportError:
            pass

if __name__ == "__main__":
    unittest.main()
