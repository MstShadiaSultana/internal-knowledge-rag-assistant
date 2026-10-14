"""Extract text from local files while retaining document and page provenance."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class DocumentText:
    source: str
    page: int | None
    text: str


def iter_document_texts(data_dir: Path) -> Iterator[DocumentText]:
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(path)
                for page_number, page in enumerate(reader.pages, start=1):
                    text = (page.extract_text() or "").strip()
                    if text:
                        yield DocumentText(path.name, page_number, text)
            except Exception as exc:
                raise ValueError(f"Could not read PDF {path.name}") from exc
        else:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                yield DocumentText(path.name, None, text)
