import io
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_pdf(file_bytes: bytes, filename: str) -> List[Document]:
    """Extract text from each page of a PDF, one Document per page."""
    reader = PdfReader(io.BytesIO(file_bytes))
    docs: List[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": filename, "page": page_num},
            ))
    return docs


def _load_text(file_bytes: bytes, filename: str) -> List[Document]:
    """Load a plain-text or Markdown file as a single Document."""
    text = file_bytes.decode("utf-8", errors="replace")
    return [Document(
        page_content=text,
        metadata={"source": filename, "page": 1},
    )]


# ── Public API ────────────────────────────────────────────────────────────────

def process_uploaded_file(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Load a file and split it into chunks ready for embedding.

    Raises ValueError for unsupported file types.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    raw_docs = _load_pdf(file_bytes, filename) if ext == ".pdf" else _load_text(file_bytes, filename)
    return _split_documents(raw_docs)


def _split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
