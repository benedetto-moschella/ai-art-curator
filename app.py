# File: app_final.py
"""
A command-line interface for the Art Therapy Engine.

This application prompts the user for their mood, passes it to the ArtEngine,
and displays the recommended artwork and its details in the console.
"""

import logging
from dotenv import load_dotenv
from PIL import Image

# Import the ArtEngine class from our refactored engine file
# Make sure the filename is correct (e.g., art_engine_refactored.py)
from art_engine import ArtEngine

# --- CONFIGURATION CONSTANTS ---
DB_PATH = "./data/chroma_db"
COLLECTION_NAME = "art_collection_final"

# Configure a logger for this application module
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def display_artwork(artwork_data: dict):
    """Formats and prints the artwork data to the console."""
    print("\n--- Recommended Artwork ---")
    print(f'"{artwork_data.get("title", "Untitled")}" ({artwork_data.get("year", "")})')
    print(f'by {artwork_data.get("author", "Unknown")}')
    print(f'Movement: {artwork_data.get("movement", "N/A")}')
    # Corrected: No f-string needed if there are no variables.
    print(f"\n{artwork_data.get('explanation', '')}")
    print("-" * 25)

    # Attempt to show the image
    try:
        Image.open(artwork_data['path']).show()
    except FileNotFoundError:
        print(
            f"\nWARNING: Could not display image. Make sure the path "
            f"'{artwork_data['path']}' is correct."
        )
    except OSError as e:
        # Catching a more specific OSError for file/display related issues
        print(f"\nWARNING: Could not display image. Error: {e}")


def main():
    """
    Main loop to run the AI Art Curator application.
    """
    # Load environment variables from a .env file
    load_dotenv()

    # --- Initialize the ArtEngine ---
    try:
        engine = ArtEngine(db_path=DB_PATH, collection_name=COLLECTION_NAME)
    except RuntimeError as e:
        print(f"🔴 Failed to initialize ArtEngine: {e}")
        return

    # --- Main Application Loop ---
    print("\n--- AI Art Curator ---")
    print("Describe how you are feeling to receive an artwork. Type 'exit' to quit.")

    shown_in_session = []
    while True:
        try:
            user_mood = input("\n> How are you feeling today? ")
            if user_mood.lower() == 'exit':
                break

            print("\nProcessing... 🤔")
            artwork_data = engine.get_art_for_mood(user_mood, exclude_ids=shown_in_session)

            if artwork_data:
                shown_in_session.append(artwork_data['path'])
                display_artwork(artwork_data)
            else:
                print(
                    "Sorry, I couldn't find a suitable artwork. "
                    "Try describing your mood in a different way."
                )

        except (KeyboardInterrupt, EOFError):
            # Allow clean exit with Ctrl+C or Ctrl+D
            break
        # pylint: disable=W0718
        # We are deliberately catching a broad exception here to ensure the UI
        # never crashes on an unexpected error, logging it instead.
        except Exception as e:
            logger.error("A critical error occurred in the main loop: %s", e, exc_info=True)
            print("🔴 A critical error occurred. Please check the logs.")

    print("\nThank you for using the AI Art Curator. See you soon!")


if __name__ == "__main__":
    main()
