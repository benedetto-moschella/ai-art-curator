# File: art_engine_final.py
"""
Art Therapy Engine.

This module provides a class `ArtEngine` that encapsulates the logic for
recommending artworks based on a user's mood. It uses a hybrid approach
with CLIP for semantic search and Google Gemini for creative reasoning.
"""

import os
import logging
import chromadb
import torch
import clip
import google.generativeai as genai

# Configure the logger for the module
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# pylint: disable=R0903
class ArtEngine:
    """An engine that suggests artworks to soothe a user's mood."""

    def __init__(self, db_path: str, collection_name: str):
        """
        Initializes the ArtEngine by loading all required models and the database.
        """
        print("AI Engine: Initializing models and DB...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Configure and load models
        try:
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            self.llm = genai.GenerativeModel('gemini-1.5-flash-latest')
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", self.device)
        except Exception as e:
            # Re-raising the exception with the original traceback for better debugging.
            raise RuntimeError(
                f"Failed to load AI models. Check API key and network. Error: {e}"
            ) from e

        # Load the vector database
        try:
            client = chromadb.PersistentClient(path=db_path)
            self.collection = client.get_collection(name=collection_name)
            print(
                f"Database '{collection_name}' loaded with "
                f"{self.collection.count()} artworks."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ChromaDB collection. Ensure the DB was created. Error: {e}"
            ) from e

        print("✅ AI Engine initialized successfully.")

    def _get_recipe_from_mood(self, mood: str) -> str:
        """Generates a keyword-based visual recipe from a mood using an LLM."""
        prompt = (f"You are an art therapist. A user is feeling: '{mood}'. "
                  f"Respond with a list of up to 10 comma-separated keywords "
                  f"that describe the visual antidote.")
        response = self.llm.generate_content(prompt)
        recipe = response.text.strip()
        safe_recipe = ", ".join([k.strip() for k in recipe.split(',')][:7])
        # Using lazy formatting for logging.
        logger.info("Recipe for CLIP: '%s'", safe_recipe)
        return safe_recipe

    def _find_best_match(self, recipe: str, exclude_ids: list = None) -> dict | None:
        """Finds the best matching artwork in ChromaDB using CLIP."""
        with torch.no_grad():
            text_tokens = clip.tokenize([recipe]).to(self.device)
            query_embedding = self.clip_model.encode_text(text_tokens)
            query_embedding /= query_embedding.norm(dim=-1, keepdim=True)

        results = self.collection.query(
            query_embeddings=query_embedding.cpu().tolist(),
            n_results=5
        )
        if not results or not results['ids'][0]:
            return None

        # Find the first valid result not in the exclusion list
        for i, result_id in enumerate(results['ids'][0]):
            if exclude_ids is None or result_id not in exclude_ids:
                return results['metadatas'][0][i]
        return None

    def _get_explanation(self, mood: str, metadata: dict) -> str:
        """Generates the final empathetic explanation for the chosen artwork."""
        author = metadata.get("author", "Unknown")
        title = metadata.get("title", "Untitled")
        year = metadata.get("year", "")
        movement = metadata.get("movement", "N/A")

        prompt = (f"You are an empathetic and concise art critic. "
                  f"A user is feeling '{mood}'. The chosen artwork for them is "
                  f"'{title}' ({year}) by {author}, from the {movement} movement. "
                  f"Write a personal and touching explanation (2-3 sentences max), "
                  f"without repeating the info you already know.")
        response = self.llm.generate_content(prompt)
        return response.text.strip()

    def get_art_for_mood(self, mood: str, exclude_ids: list = None) -> dict | None:
        """
        The main public method. Takes a mood and returns a dictionary with artwork data.
        """
        # 1. Get visual recipe
        recipe = self._get_recipe_from_mood(mood)
        if not recipe:
            return None

        # 2. Find best artwork match
        chosen_metadata = self._find_best_match(recipe, exclude_ids)
        if not chosen_metadata:
            return None

        # 3. Get empathetic explanation
        explanation = self._get_explanation(mood, chosen_metadata)

        # 4. Return all data combined
        chosen_metadata['explanation'] = explanation
        return chosen_metadata
