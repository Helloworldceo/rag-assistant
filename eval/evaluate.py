#!/usr/bin/env python3
"""
RAGAS Evaluation Harness for RAG Assistant
==========================================

Measures retrieval and generation quality across four RAGAS metrics:
  • Faithfulness        — is every claim in the answer grounded in the context?
  • Response Relevancy  — does the answer directly address the question?
  • Context Precision   — did the retriever find useful chunks?
  • Context Recall      — did the retriever miss any critical information?

Usage:
  # Run against the built-in sample document (fully self-contained):
  python eval/evaluate.py

  # Use a custom dataset JSON (must follow the sample_dataset.json schema):
  python eval/evaluate.py --dataset path/to/my_dataset.json

  # Save numeric results to JSON:
  python eval/evaluate.py --output eval/results.json

  # Keep the sample document in the vector store after evaluation:
  python eval/evaluate.py --keep-sample

Requirements:
  pip install ragas>=0.2.0 datasets>=2.14.0
  (already included in requirements.txt)
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# ── Bootstrap: add project root to sys.path ──────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Validate API key early ────────────────────────────────────────────────────
if not os.getenv("DEEPSEEK_API_KEY"):
    print("[ERROR] DEEPSEEK_API_KEY is not set. Add it to .env and retry.")
    sys.exit(1)

# ── Project imports ───────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from rag.chain import RAGChain
from rag.config import (
    DEEPSEEK_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# ── RAGAS imports (0.2.x API) ─────────────────────────────────────────────────
try:
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecision,
        LLMContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # SingleTurnSample location changed between minor versions
    try:
        from ragas import SingleTurnSample
    except ImportError:
        from ragas.dataset_schema import SingleTurnSample

except ImportError as exc:
    print(f"[ERROR] RAGAS import failed: {exc}")
    print("Install with: pip install ragas>=0.2.0")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
EVAL_COLLECTION = "rag_eval_temp"
SAMPLE_DOC = Path(__file__).parent / "sample_doc.txt"
SAMPLE_DATASET = Path(__file__).parent / "sample_dataset.json"

# ANSI colours
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED   = "\033[91m"
_CYAN  = "\033[96m"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _colour_score(score: float) -> str:
    """Colour-code a 0-1 score: green ≥ 0.8, yellow ≥ 0.6, red < 0.6."""
    if score >= 0.8:
        colour = _GREEN
    elif score >= 0.6:
        colour = _YELLOW
    else:
        colour = _RED
    bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    return f"{colour}{bar}  {score:.3f}{_RESET}"


def _ingest_sample_doc(chroma_dir: str) -> Chroma:
    """Load sample_doc.txt into a dedicated temporary Chroma collection."""
    text = SAMPLE_DOC.read_text(encoding="utf-8")
    raw_doc = Document(page_content=text, metadata={"source": "sample_doc.txt"})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents([raw_doc])

    embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
    store = Chroma(
        collection_name=EVAL_COLLECTION,
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )
    store.add_documents(chunks)
    print(f"  Ingested {len(chunks)} chunks from sample_doc.txt")
    return store


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the RAG Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        default=str(SAMPLE_DATASET),
        help=f"Path to evaluation dataset JSON (default: {SAMPLE_DATASET})",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Save metric scores to this JSON file (e.g. eval/results.json)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of chunks to retrieve per question (default: 4)",
    )
    parser.add_argument(
        "--keep-sample",
        action="store_true",
        help="Do not delete the temporary eval Chroma collection after evaluation",
    )
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        sys.exit(1)

    with open(dataset_path, encoding="utf-8") as f:
        raw = json.load(f)
    samples_data = raw.get("samples", [])
    if not samples_data:
        print("[ERROR] Dataset has no samples. Check the JSON structure.")
        sys.exit(1)

    print(f"\n{_BOLD}RAG Assistant · RAGAS Evaluation{_RESET}")
    print("─" * 50)
    print(f"Dataset : {dataset_path} ({len(samples_data)} samples)")

    # ── Build temporary vector store with sample doc ──────────────────────────
    use_temp_dir = not args.keep_sample
    chroma_dir = str(ROOT / "chroma_db_eval") if args.keep_sample else tempfile.mkdtemp(prefix="rag_eval_")

    print(f"\nStep 1 — Ingesting reference document…")
    store = _ingest_sample_doc(chroma_dir)
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": args.k})

    # ── Build RAG chain against eval collection ───────────────────────────────
    chain = RAGChain(retriever=retriever)

    # ── Run questions through the chain ──────────────────────────────────────
    print(f"\nStep 2 — Running {len(samples_data)} questions through the RAG chain…")
    ragas_samples = []

    for i, item in enumerate(samples_data, start=1):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        print(f"  [{i}/{len(samples_data)}] {question[:72]}…")

        result = chain.ask(question=question, chat_history=[])

        ragas_samples.append(
            SingleTurnSample(
                user_input=question,
                response=result["answer"],
                retrieved_contexts=[doc.page_content for doc in result["source_documents"]],
                reference=ground_truth,
            )
        )

    # ── Configure RAGAS evaluator ─────────────────────────────────────────────
    print(f"\nStep 3 — Configuring RAGAS with DeepSeek as the judge LLM…")
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=DEFAULT_LLM_MODEL,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.0,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
    )

    eval_dataset = EvaluationDataset(samples=ragas_samples)

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecision(),
        LLMContextRecall(),
    ]

    # ── Run evaluation ────────────────────────────────────────────────────────
    print(f"\nStep 4 — Running RAGAS evaluation (may take 1–3 minutes)…\n")
    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    df = results.to_pandas()

    # ── Print results table ───────────────────────────────────────────────────
    metric_cols = {
        "faithfulness":        "Faithfulness       (hallucination resistance)",
        "response_relevancy":  "Response Relevancy (answer addresses question)",
        "llm_context_precision": "Context Precision  (retriever signal:noise)",
        "llm_context_recall":  "Context Recall     (retriever coverage)",
    }

    print("\n" + "═" * 65)
    print(f"  {_BOLD}RAGAS Evaluation Results{_RESET}  —  {len(samples_data)} samples")
    print("═" * 65)

    scores: dict[str, float] = {}
    for col, label in metric_cols.items():
        if col in df.columns:
            score = float(df[col].mean())
            scores[col] = score
            print(f"  {label:<44}  {_colour_score(score)}")

    print("═" * 65)
    print("\nScoring guide:  ≥ 0.80  good  │  0.60–0.79  needs work  │  < 0.60  poor\n")

    # ── Save results ──────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": str(dataset_path),
            "num_samples": len(samples_data),
            "k": args.k,
            "metrics": scores,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {out_path}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if not args.keep_sample:
        try:
            store.delete_collection()
        except Exception:
            pass


if __name__ == "__main__":
    main()
