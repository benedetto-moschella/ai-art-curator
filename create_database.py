# File: create_database.py
"""
This script loads pre-computed CLIP embeddings and their corresponding file paths
from block files, parses the paths to extract rich metadata (author, title,
year, movement), and populates a persistent ChromaDB vector database.
"""

import os
import re
import shutil
import torch
import chromadb
from tqdm import tqdm

# --- 1. CONFIGURATION CONSTANTS ---
EMBEDDINGS_DIR = "./clip_blocks_output"
DB_PATH = "./data/chroma_db"
COLLECTION_NAME = "art_collection_final"


def parse_metadata_from_path(path: str) -> dict:
    """
    Extracts metadata from a file path assuming a structure like
    `.../Movement/author_title-YYYY.jpg`.
    """
    try:
        parts = path.split(os.sep)
        movement = parts[-2].replace('_', ' ').title()
        filename = parts[-1]

        # Regex to parse "author_title-YYYY.ext" or "author_title.ext"
        pattern = re.compile(r'^(.*?)_(.*?)(?:-(\d{4}))?\..*$')
        match = pattern.match(filename)

        if not match:
            # Fallback if the filename doesn't match the author_title pattern
            return {
                "path": path, "author": "Unknown",
                "title": os.path.splitext(filename)[0],
                "year": "", "movement": movement
            }

        author = match.group(1).replace('-', ' ').title()
        title_part = match.group(2)

        # Cleans the title by removing the year at the end
        title = re.sub(r'[\s-]?\d{4}$', '', title_part).replace('-', ' ').capitalize().strip()

        # Extracts the year if present
        year_match = re.search(r'(\d{4})$', title_part)
        year = year_match.group(1) if year_match else ""

        return {
            "path": path, "author": author, "title": title,
            "year": year, "movement": movement
        }

    except IndexError:
        # Fallback if the path structure is unexpected
        return {
            "path": path, "author": "Unknown", "title": "Untitled",
            "year": "", "movement": "N/A"
        }


def main():
    """
    Main function to orchestrate the database creation process.
    It initializes the DB, finds and parses embedding blocks,
    and populates the database with embeddings and rich metadata.
    """
    # --- 2. DATABASE SETUP ---
    print(f"Initializing ChromaDB in '{DB_PATH}'...")
    if os.path.exists(DB_PATH):
        print("Found an old database, removing it to start fresh...")
        shutil.rmtree(DB_PATH)
    os.makedirs(DB_PATH, exist_ok=True)

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # --- 3. LOADING CYCLE ---
    print(f"Reading embedding blocks from '{EMBEDDINGS_DIR}'...")
    if not os.path.exists(EMBEDDINGS_DIR):
        raise FileNotFoundError(
            f"Embeddings folder '{EMBEDDINGS_DIR}' not found."
        )

    embedding_files = sorted([f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".pt")])
    all_path_files_in_dir = os.listdir(EMBEDDINGS_DIR)

    for emb_file in tqdm(embedding_files, desc="Loading Blocks into DB"):
        try:
            block_index_str = emb_file.replace('embeddings_block_', '').replace('.pt', '')
            expected_path_file = f"paths_block_{block_index_str}.txt"
        except ValueError:
            continue
        if expected_path_file not in all_path_files_in_dir:
            continue

        path_to_open = os.path.join(EMBEDDINGS_DIR, expected_path_file)
        embeddings_tensor = torch.load(os.path.join(EMBEDDINGS_DIR, emb_file))

        with open(path_to_open, "r", encoding="utf-8") as file_handle:
            full_path_list = [line.strip() for line in file_handle.readlines()]

        if not full_path_list or collection.get(ids=full_path_list[0:1])['ids']:
            continue

        metadatas_list = [parse_metadata_from_path(path) for path in full_path_list]

        collection.add(
            ids=full_path_list,
            embeddings=embeddings_tensor.cpu().tolist(),
            metadatas=metadatas_list
        )

    print(
        f"✅ Process completed. The DB has been created and now contains "
        f"{collection.count()} artworks."
    )


if __name__ == "__main__":
    main()
