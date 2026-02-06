from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.config import RESULT_DIR, SA_DIR

# Optional: load .env so OPENAI_* vars are available when running standalone
try:
    from dotenv import load_dotenv

    for _candidate in (
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().parents[1] / "Experiment" / ".env",
    ):
        if _candidate.exists():
            load_dotenv(_candidate)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
LINEART_ROOT = ROOT / "lineart-board"
if str(LINEART_ROOT) not in sys.path:
    sys.path.insert(0, str(LINEART_ROOT))

from semantic_graph import similarity  # type: ignore
from app import embedding_client  # type: ignore


def _iter_samples(split: str) -> Iterable[Tuple[Path, Dict]]:
    split_dir = SA_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(f"split directory not found: {split_dir}")
    for path in sorted(split_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            yield path, json.load(f)


def _rank_candidates(query_embedding: List[float], block_embeddings: Dict[str, List[float]], candidates: List[str]) -> List[str]:
    scored: List[Tuple[float, str]] = []
    for bid in candidates:
        emb = block_embeddings.get(bid)
        if emb is None:
            continue
        dist = similarity.cosine_distance(query_embedding, emb)
        scored.append((dist, bid))
    scored.sort(key=lambda item: item[0])
    return [bid for _, bid in scored]


def _metrics_from_ranks(ranks: List[int]) -> Dict[str, float]:
    if not ranks:
        return {"MRR": 0.0, "R1": 0.0, "R3": 0.0}
    mrr = sum(1.0 / r for r in ranks) / len(ranks)
    r1 = sum(1 for r in ranks if r == 1) / len(ranks)
    r3 = sum(1 for r in ranks if r <= 3) / len(ranks)
    return {"MRR": mrr, "R1": r1, "R3": r3}


def eval_text_only(split: str = "test") -> Dict[str, float]:
    ranks: List[int] = []
    n_queries = 0
    for _, sample in _iter_samples(split):
        blocks = {b["block_id"]: b for b in sample.get("blocks", [])}
        block_embeddings = {}
        for block_id, block in blocks.items():
            text = block.get("text") or ""
            block_embeddings[block_id] = embedding_client.embed_text(text)
        for query in sample.get("retrieval_queries", []):
            candidate_ids = query.get("candidate_block_ids") or []
            answer_id = query.get("answer_block_id")
            if answer_id not in candidate_ids:
                continue
            query_text = query.get("query_text") or ""
            query_emb = embedding_client.embed_text(query_text)
            ranking = _rank_candidates(query_emb, block_embeddings, candidate_ids)
            if answer_id not in ranking:
                continue
            rank_position = ranking.index(answer_id) + 1  # 1-based rank
            ranks.append(rank_position)
            n_queries += 1
    metrics = _metrics_from_ranks(ranks)
    metrics["n_queries"] = n_queries
    return metrics


def save_results(split: str, metrics: Dict[str, Dict[str, float]]) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / "sa_retrieval.json"
    payload = {"split": split, "metrics": metrics}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="SA-Canvas retrieval evaluation")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"], help="Dataset split.")
    parser.add_argument("--save", action="store_true", help="Write results to results/sa_retrieval.json")
    args = parser.parse_args()

    metrics = {"text_only": eval_text_only(split=args.split)}
    if args.save:
        out_path = save_results(args.split, metrics)
        print(f"Saved metrics to {out_path}")
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
