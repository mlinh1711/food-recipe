# File: food2recipe/scripts/build_index.py
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from food2recipe.core.settings import load_settings
from food2recipe.core.logging_utils import setup_logger
from food2recipe.preprocessing.build_manifest import build_manifest
from food2recipe.models.image_encoder import ImageEncoder
from food2recipe.retrieval.index_faiss import RetrievalIndex
from food2recipe.preprocessing.image_preprocess import get_transforms, load_and_transform_image
from PIL import Image

logger = setup_logger("build_index")

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            # We use a helper from image_preprocess to ensure consistency
            # But standard Dataset needs to return tensors directly
            # Re-implementing simplified load here for speed
            image = Image.open(path).convert("RGB")
            return self.transform(image), str(path)
        except Exception as e:
            # Return dummy or handle error? For index build, skipping is safer but complicated In batch
            # We'll just return zeros and filter later? 
            # Better: The manifest building step should have vetted files, but let's be safe.
            print(f"Error loading {path}: {e}")
            return torch.zeros((3, 224, 224)), ""

def main():
    settings = load_settings()
    
    # 1. Manifest
    # Always rebuild? Or check existence? Let's rebuild to be safe
    logger.info("Building manifest...")
    manifest_path = build_manifest(settings)
    if not manifest_path:
        logger.error("No manifest created. Exiting.")
        sys.exit(1)
        
    df = pd.read_csv(manifest_path)
    
    # Filter: Use 'train' and 'val' for index. 'test' is for eval.
    # Configurable?
    index_splits = ['train', 'val']
    df_index = df[df['split'].isin(index_splits)].copy()
    
    logger.info(f"Using {len(df_index)} images from splits {index_splits} for indexing.")
    
    # 2. Encoder
    encoder = ImageEncoder(settings)
    transform = get_transforms(mode="train", image_size=settings.IMAGE_SIZE)
    
    # 3. Batch Process
    dataset = ImageDataset(df_index['image_path'].tolist(), transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0) # workers=0 for compatibility
    
    all_embeddings = []
    valid_paths = []
    
    logger.info("Encoding images...")
    for imgs, paths in tqdm(dataloader):
        # Filter out failed loads (empty paths)
        # Note: parallel loading with 0 workers is safe, but with >0 might need robust collate
        # Simplified:
        valid_mask = [p != "" for p in paths]
        if not any(valid_mask):
            continue
            
        imgs = imgs[valid_mask]
        current_valid_paths = [p for p, m in zip(paths, valid_mask) if m]
        
        embeddings = encoder.encode(imgs)
        all_embeddings.append(embeddings.numpy())
        valid_paths.extend(current_valid_paths)
        
    if not all_embeddings:
        logger.error("No embeddings generated.")
        sys.exit(1)
        
    final_embeddings = np.vstack(all_embeddings)
    
    # 4. Prepare Metadata
    # Need to map back path -> food_name
    # Efficient look up
    path_to_row = df_index.set_index("image_path").to_dict("index")
    
    metadata = []
    for p in valid_paths:
        row = path_to_row[p]
        metadata.append({
            "image_path": p,
            "food_name": row["food_name"],
            "split": row["split"]
        })
        
    # 5. Build & Save Index
    index = RetrievalIndex(settings)
    index.build(final_embeddings, metadata)
    
    save_dir = settings.ARTIFACTS_DIR / "index"
    index.save(save_dir)
    logger.info("Index build complete!")

if __name__ == "__main__":
    main()
