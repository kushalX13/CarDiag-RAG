"""CLI: Train a bi-encoder dense retriever on train_triples.jsonl."""

import os
# Avoid OpenMP duplicate lib error on macOS (libomp.dylib already initialized)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import random

import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import DATA_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
BIENCODER_MODEL_DIR = os.path.join(DATA_DIR, "models", "biencoder")
TRAIN_TRIPLES_PATH = os.path.join(PROCESSED_DIR, "train_triples.jsonl")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_triples(path: str) -> list[dict]:
    """Load JSONL triples file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class TriplesDataset(Dataset):
    """Dataset of (query, [pos + negs]) for bi-encoder training."""

    def __init__(self, triples: list[dict]):
        self.samples = []
        for t in triples:
            query = t.get("query_text", "")
            pos = t.get("pos", {})
            negs = t.get("negs", [])[:8]  # up to 8 negs
            pos_text = pos.get("text", "") if isinstance(pos, dict) else ""
            passages = [pos_text]
            for n in negs:
                passages.append(n.get("text", "") if isinstance(n, dict) else "")
            if query and passages:
                self.samples.append((query, passages))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[str, list[str]]:
        return self.samples[idx]


def collate_triples(batch: list[tuple[str, list[str]]]) -> tuple[list[str], list[list[str]]]:
    """Collate batch of (query, passages)."""
    queries = [b[0] for b in batch]
    passages_list = [b[1] for b in batch]
    return queries, passages_list


def train_epoch(
    model: SentenceTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
) -> float:
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    for queries, passages_list in tqdm(dataloader, desc="Training"):
        # Per-query: (query, [pos, neg1, neg2, ...]), label 0 = pos at index 0
        batch_queries = []
        batch_passages_flat = []
        batch_sizes = []

        for q, passages in zip(queries, passages_list):
            batch_queries.append(q)
            batch_passages_flat.extend(passages)
            batch_sizes.append(len(passages))

        # Encode via forward (encode() uses no_grad and breaks gradients)
        features_q = model.tokenize(batch_queries)
        features_p = model.tokenize(batch_passages_flat)
        features_q = {k: v.to(device) for k, v in features_q.items()}
        features_p = {k: v.to(device) for k, v in features_p.items()}
        with torch.amp.autocast("cuda", enabled=use_amp):
            q_out = model.forward(features_q)
            p_out = model.forward(features_p)
        q_emb = q_out["sentence_embedding"]
        p_emb = p_out["sentence_embedding"]

        # Compute scores per (query, passage) and reshape for cross-entropy
        offset = 0
        logits_list = []
        for i, size in enumerate(batch_sizes):
            q = q_emb[i : i + 1]  # (1, dim)
            p = p_emb[offset : offset + size]  # (size, dim)
            scores = torch.matmul(q, p.T).squeeze(0)  # (size,)
            logits_list.append(scores)
            offset += size

        # Pad to max size in batch for stacking
        max_size = max(batch_sizes)
        padded_logits = []
        for logits in logits_list:
            if len(logits) < max_size:
                pad = torch.full(
                    (max_size - len(logits),),
                    float("-inf"),
                    device=logits.device,
                    dtype=logits.dtype,
                )
                logits = torch.cat([logits, pad])
            padded_logits.append(logits)
        logits = torch.stack(padded_logits)  # (batch, max_size)

        labels_t = torch.zeros(len(queries), dtype=torch.long, device=device)
        loss = torch.nn.functional.cross_entropy(logits, labels_t)

        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bi-encoder dense retriever")
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base SentenceTransformer model name",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--train-path",
        type=str,
        default=TRAIN_TRIPLES_PATH,
        help="Path to train_triples.jsonl",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument(
        "--max-triples",
        type=int,
        default=None,
        help="If set, only load/process the first N training triples",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    use_amp = torch.cuda.is_available()  # AMP only on CUDA; MPS doesn't support it

    triples = load_triples(args.train_path)
    if args.max_triples is not None:
        triples = triples[: args.max_triples]
        logger.info("Limited to first %d training triples", len(triples))
    else:
        logger.info("Loaded %d training triples", len(triples))
    dataset = TriplesDataset(triples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_triples,
        pin_memory=device.type == "cuda",
    )

    model = SentenceTransformer(args.model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(BIENCODER_MODEL_DIR, exist_ok=True)

    for epoch in range(args.epochs):
        loss = train_epoch(model, dataloader, optimizer, device, use_amp)
        logger.info("Epoch %d: loss = %.4f", epoch + 1, loss)

    model.save(BIENCODER_MODEL_DIR)
    config = {
        "base_model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
    }
    config_path = os.path.join(BIENCODER_MODEL_DIR, "train_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info("Saved model to %s", BIENCODER_MODEL_DIR)


if __name__ == "__main__":
    main()
