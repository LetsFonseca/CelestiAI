"""
CLI tool to ingest parsed PDF text into Qdrant, matching data_ingestion.py style.

Usage examples:

  - From a text file produced elsewhere:
      python ingest_qdrant_from_text.py --text-file path/to/parsed.txt \
        --collection astrology-zodiac

  - Parse a PDF on the fly using Docling and ingest:
      python ingest_qdrant_from_text.py --pdf-path dataset/astrology_signs.pdf \
        --collection astrology-zodiac

  - Read text from STDIN:
      cat parsed.txt | python ingest_qdrant_from_text.py --stdin

Requires env vars:
  QDRANT_URL, QDRANT_API_KEY

Embeddings:
  Uses the same model as data_ingestion.py: sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

# Optional import: only when using --pdf-path
try:
    from pdf_docling_parser import DoclingPdfParser
except Exception:
    DoclingPdfParser = None  # type: ignore

try:
    # Prefer modern splitters package present in requirements
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    # Fallback if package alias differs in some environments
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore


DEFAULT_COLLECTION = "astrology-zodiac"


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Missing required env var {name}. Set it before running ingestion."
        )
    return val


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    return splitter.split_text(text)


def load_text(
    *,
    text_file: Optional[Path] = None,
    pdf_path: Optional[Path] = None,
    read_stdin: bool = False,
) -> str:
    # Highest priority: explicit --text-file
    if text_file is not None:
        data = text_file.read_text(encoding="utf-8", errors="ignore")
        return data.strip()

    # Next: --pdf-path via Docling
    if pdf_path is not None:
        if DoclingPdfParser is None:
            raise RuntimeError(
                "Docling parser import failed. Ensure pdf_docling_parser.py is available and dependencies installed."
            )
        parser = DoclingPdfParser()
        return parser.parse_to_text(pdf_path)

    # Finally: --stdin
    if read_stdin:
        data = sys.stdin.read()
        return data.strip()

    raise ValueError("No input provided. Use --text-file, --pdf-path, or --stdin.")


def ingest_text(text: str, collection_name: str = DEFAULT_COLLECTION) -> None:
    qdrant_url = _require_env("QDRANT_URL")
    qdrant_api_key = _require_env("QDRANT_API_KEY")

    print("Preparing embeddings and chunking text...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    chunks = chunk_text(text)
    if not chunks:
        raise RuntimeError("No chunks produced from input text; nothing to ingest.")

    print(f"Connecting to Qdrant and uploading {len(chunks)} chunks...")
    _ = Qdrant.from_texts(
        texts=chunks,
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
    )

    print(f"Done! Collection '{collection_name}' now contains the ingested chunks.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest parsed PDF text into Qdrant.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text-file", type=Path, help="Path to a UTF-8 text file to ingest.")
    g.add_argument("--pdf-path", type=Path, help="Path to a PDF to parse via Docling, then ingest.")
    g.add_argument("--stdin", action="store_true", help="Read text to ingest from STDIN.")

    p.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    text = load_text(text_file=args.text_file, pdf_path=args.pdf_path, read_stdin=args.stdin)
    ingest_text(text, collection_name=args.collection)


if __name__ == "__main__":
    main()

