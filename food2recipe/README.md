# food2recipe

A "production-lite" AI project to recognize food from images and retrieve recipes.

## Structure
- `app/`: Streamlit UI
- `core/`: Config & Logging
- `models/`: Image Encoders (CLIP/Timm)
- `retrieval/`: FAISS Indexing & Logic
- `scripts/`: Build & Eval scripts
- `data/`: Dataset storage (Assumed local)

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Data**
   Ensure your data is in `data/` or configure `.env`.
   Required structure:
   ```
   data/
       Images/
           train/
           val/
           test/
       vnfood30_recipes.csv
   ```

3. **Build Index**
   Pre-processes images and builds the vector search index.
   ```bash
   python -m food2recipe.scripts.build_index
   ```

4. **Run Evaluation (Optional)**
   Checks accuracy on test split.
   ```bash
   python -m food2recipe.scripts.run_eval
   ```

5. **Run App**
   Start the web interface.
   ```bash
   streamlit run food2recipe/app/streamlit_app.py
   ```

## Design Notes
- Uses **OpenCLIP** for embedding generation by default.
- Uses **FAISS** for fast similarity search (~L2 normalized cosine).
- Caches models and data in Streamlit for performance.
- Configurable via `core/settings.py` and `.env`.
