from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

from docling.document_converter import DocumentConverter


class DoclingPdfParser:
    """
    Thin wrapper around Docling's DocumentConverter to parse PDF files.

    Usage:
        parser = DoclingPdfParser()

        md = parser.parse_to_markdown("file.pdf")
        text = parser.parse_to_text("file.pdf")

        # from bytes (e.g. uploaded file in a web app)
        md2 = parser.parse_bytes_to_markdown(pdf_bytes)
    """

    def __init__(self, converter: Optional[DocumentConverter] = None) -> None:
        """
        Initialize the parser.

        :param converter: Optional custom DocumentConverter instance.
                          If not provided, a default one will be created.
        """
        self.converter = converter or DocumentConverter()

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _ensure_path(path: Union[str, Path]) -> Path:
        """Normalize path argument and validate that it exists."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"PDF file not found: {p}")
        return p

    # -----------------------------
    # Public API – from file path
    # -----------------------------
    def parse_to_markdown(self, pdf_path: Union[str, Path]) -> str:
        """
        Convert a PDF file to Markdown using Docling.

        :param pdf_path: Path to the PDF file.
        :return: Markdown string.
        """
        pdf_path = self._ensure_path(pdf_path)

        result = self.converter.convert(str(pdf_path))
        # Docling returns a Document object with export helpers
        markdown = result.document.export_to_markdown()
        return markdown

    def parse_to_text(self, pdf_path: Union[str, Path]) -> str:
        """
        Convert a PDF file to plain text (using Markdown export as base).

        :param pdf_path: Path to the PDF file.
        :return: Plain text string (Markdown stripped of basic markers).
        """
        markdown = self.parse_to_markdown(pdf_path)
        text = markdown.replace("#", "").replace("*", "").replace("<!-- image -->","")
        
        return text.strip()

    # -----------------------------
    # Public API – from bytes
    # -----------------------------
    def parse_bytes_to_markdown(self, pdf_bytes: bytes) -> str:
        """
        Convert PDF bytes to Markdown. Useful for uploads in web apps.

        :param pdf_bytes: Raw PDF bytes.
        :return: Markdown string.
        """
        result = self.converter.convert_bytes(
            pdf_bytes,
            mime_type="application/pdf",
        )
        markdown = result.document.export_to_markdown()
        return markdown

    def parse_bytes_to_text(self, pdf_bytes: bytes) -> str:
        """
        Convert PDF bytes to plain text.

        :param pdf_bytes: Raw PDF bytes.
        :return: Plain text string.
        """
        markdown = self.parse_bytes_to_markdown(pdf_bytes)
        text = markdown.replace("#", "").replace("*", "").replace("<!-- image -->","")
        return text.strip()
    
# if __name__ == "__main__":
#     parser = DoclingPdfParser()
#     # Demo using the known dataset PDF. Prints a short preview.
#     try:
#         md = parser.parse_to_markdown("dataset/astrology_signs.pdf")
#         print("Markdown preview:\n", md[:500])
#         txt = parser.parse_to_text("dataset/astrology_signs.pdf")
#         print("\nText preview:\n", txt[:500])
#     except Exception as e:
#         print(f"Docling parse failed: {e}")
