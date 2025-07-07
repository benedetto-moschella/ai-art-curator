# File: create_embeddings_pylint_approved.py
"""
Scans a specified image directory, generates CLIP embeddings for each image,
and saves both the embeddings and their file paths into chunked block files.

This script is intended to be run locally to preprocess an image dataset,
following Python best practices and conventions.
"""

import os
import torch
import clip
from PIL import Image
from tqdm import tqdm

# --- 1. CONFIGURATION CONSTANTS ---
# Constants are defined at the top for easy configuration.
IMAGE_ROOT = "./art_dataset"
SAVE_DIR = "./clip_blocks_output"
BLOCK_SIZE = 1000
MODEL_NAME = "ViT-B/32"


def scan_image_files(root_path: str) -> list:
    """
    Scans a root directory recursively and returns a sorted list of image file paths.

    Args:
        root_path (str): The path to the root directory to scan.

    Returns:
        list: A sorted list of full paths to the found image files.
    """
    print("\nScanning for images in the local folder...")
    all_paths = []
    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                all_paths.append(os.path.join(dirpath, file))

    image_paths = sorted(all_paths)
    if not image_paths:
        # The line is broken into two for readability, respecting the character limit.
        print(
            f"🔴 ERROR: No images found at path '{root_path}'. "
            "Check if the folder exists and contains images."
        )
        return []

    print(f"✅ Found {len(image_paths)} images.")
    return image_paths


def main():
    """
    Main function to run the entire embedding creation and saving process.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Images will be read from: '{os.path.abspath(IMAGE_ROOT)}'")
    print(f"Embeddings will be saved to: '{os.path.abspath(SAVE_DIR)}'")

    # --- 2. LOADING THE CLIP MODEL ---
    print("\nLoading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, preprocess = clip.load(MODEL_NAME, device=device)
    print("✅ Model loaded successfully.")

    # --- 3. FINDING THE IMAGES ---
    # Call the new helper function to get image paths.
    image_paths = scan_image_files(IMAGE_ROOT)
    if not image_paths:
        return # Exit if no images were found

    # --- 4. CREATING AND SAVING EMBEDDINGS ---
    print("\nStarting the embedding creation process...")
    block_idx = 0
    buffer_embeddings = []
    buffer_paths = []

    for path in tqdm(image_paths, desc="Processing images"):
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(image)
                emb /= emb.norm(dim=-1, keepdim=True)

                buffer_embeddings.append(emb.cpu())
                buffer_paths.append(path)
        except (IOError, Image.UnidentifiedImageError) as e:
            print(f"⚠️ Error on image (will be skipped): {path} | Detail: {e}")

        # Save a block every BLOCK_SIZE images or at the end of the loop
        if len(buffer_embeddings) >= BLOCK_SIZE or path == image_paths[-1]:
            # This 'if' statement is now on two lines.
            if not buffer_embeddings:
                continue

            print(f"\n💾 Saving block {block_idx} with {len(buffer_paths)} images...")
            torch.save(
                torch.cat(buffer_embeddings),
                os.path.join(SAVE_DIR, f"embeddings_block_{block_idx}.pt")
            )

            with open(os.path.join(SAVE_DIR, f"paths_block_{block_idx}.txt"),
                      "w", encoding="utf-8") as f:
                f.write("\n".join(buffer_paths) + "\n")

            print(f"✅ Block {block_idx} saved.")

            block_idx += 1
            buffer_embeddings = []
            buffer_paths = []

    print("\n🎉 Indexing process completed!")


if __name__ == "__main__":
    main()
