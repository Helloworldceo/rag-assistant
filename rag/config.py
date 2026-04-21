from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")

# ── Text splitting ─────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Retrieval ─────────────────────────────────────────────────────────────────
NUM_RETRIEVED_DOCS = 4

# ── DeepSeek (OpenAI-compatible) ──────────────────────────────────────────────
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_LLM_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.0

# ── Embeddings (free local model, no API key required) ────────────────────────
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Supported file extensions ─────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
