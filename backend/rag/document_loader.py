from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import fitz as fitz_t
else:
    fitz_t = None


def _get_fitz():
    """Lazily import fitz (PyMuPDF) so the backend can start without it."""
    import fitz

    return fitz


@dataclass(frozen=True)
class Document:
    source: str
    content: str
    page: int
    chunk_index: int


class PDFLoader:
    """Parse PDF files into text chunks suitable for embedding."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self, pdf_path: Path) -> list[Document]:
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = _get_fitz().open(str(pdf_path))
        documents: list[Document] = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text(sort=True).strip()
            if not text:
                continue
            chunks = self._chunk_text(text)
            for ci, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        source=pdf_path.name,
                        content=chunk,
                        page=page_num,
                        chunk_index=ci,
                    )
                )

        doc.close()
        return documents

    def load_directory(self, dir_path: Path) -> list[Document]:
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        pdf_files = sorted(dir_path.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {dir_path}")

        documents: list[Document] = []
        for pdf_path in pdf_files:
            documents.extend(self.load(pdf_path))
        return documents

    def _chunk_text(self, text: str) -> list[str]:
        """Split text at sentence boundaries, merging into fixed-size chunks with overlap."""
        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= self.chunk_size:
            return [text]

        # Split into sentences while keeping delimiters
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) <= self.chunk_size:
                current.append(sent)
                current_len += len(sent) + 1  # +1 for space
            else:
                if current:
                    chunks.append(" ".join(current))
                # If a single sentence exceeds chunk_size, split it further
                if len(sent) > self.chunk_size:
                    for i in range(0, len(sent), self.chunk_size - self.chunk_overlap):
                        chunks.append(sent[i : i + self.chunk_size])
                    current = []
                    current_len = 0
                else:
                    current = [sent]
                    current_len = len(sent)

        if current:
            chunks.append(" ".join(current))

        return chunks
