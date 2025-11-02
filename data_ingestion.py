"""
This script populates a Qdrant collection with basic astrology content.
Run it ONCE (locally or in a controlled environment) to create/fill the collection.
"""

import os
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings


# Get credentials locally 
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "astrology-zodiac"


ASTRO_DOCS = [
    "Aries is a fire sign ruled by Mars. It represents initiative, courage, and impulse. Approximate dates: March 21 to April 19.",
    "Taurus is an earth sign ruled by Venus. It represents stability, pleasure, and material security. Dates: April 20 to May 20.",
    "Gemini is an air sign ruled by Mercury. It represents communication, versatility, and curiosity. Dates: May 21 to June 20.",
    "Cancer is a water sign ruled by the Moon. It represents emotions, protection, memories, and family. Dates: June 21 to July 22.",
    "Fire signs (Aries, Leo, Sagittarius) usually match well with other fire signs and with air signs.",
    "Earth signs (Taurus, Virgo, Capricorn) usually match well with other earth signs and with water signs.",
    "For a complete birth chart analysis you need: birth date, birth time, and birth place.",
    "Compatibility is not only about the Sun sign, but we can give a basic view by element.",
    "Leo is a fire sign ruled by the Sun. It represents creativity, leadership, and self-expression.",
    "Virgo is an earth sign ruled by Mercury. It represents organization, service, and attention to detail.",
]

# create embeddings using huggingface model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def main() -> None:
    """
    Connects to Qdrant and uploads astrology documents as vectors.
    """
    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    print("Uploading documents to Qdrant...")
    result= Qdrant.from_texts(
        texts=ASTRO_DOCS,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
    )

    print(f"Done! Collection '{COLLECTION_NAME}' is populated.")

if __name__ == "__main__":
    main()

