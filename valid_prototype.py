# valid_prototype.py
"""
Classify contextual word embeddings from wiki_zh against Cilin prototype vectors.

Pipeline:
  1. Read txt files from raw_data/wiki_zh (one sentence per line)
  2. Jieba-tokenize each sentence → List[str]
  3. For each word in each sentence, run SimCSEModel to get contextual word embedding
  4. Compute cosine similarity against all prototype vectors (one forward pass per sentence)
  5. Assign word to the closest prototype group (if similarity >= threshold)
  6. Save group_id -> [words] mapping as JSON

Output JSON format:
  {
    "Aa01A01=": ["词A", "词B", ...],
    "Aa01A02=": ["词C", ...],
    ...
    "_meta": { "total_groups_assigned": ..., ... }
  }
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import jieba
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset_preprocess import parse_cilin
from model import SimCSEConfig, SimCSEModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, encoder_name: str, device: str) -> SimCSEModel:
    cfg = SimCSEConfig(encoder_name=encoder_name)
    model = SimCSEModel(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Support both raw state_dict and full checkpoint dicts
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def load_prototypes(proto_path: str, device: str) -> Tuple[List[str], torch.Tensor]:
    """
    Returns:
        group_ids : list of group id strings (e.g. "Aa01A01=")
        proto_matrix : [G, H] normalized float tensor on device
    """
    data = torch.load(proto_path, map_location="cpu", weights_only=False)
    proto_dict: Dict[str, torch.Tensor] = data["prototypes"]
    group_ids = list(proto_dict.keys())
    proto_matrix = torch.stack(
        [F.normalize(proto_dict[g].float(), p=2, dim=-1) for g in group_ids], dim=0
    ).to(device)  # [G, H]
    return group_ids, proto_matrix


def parse_group_to_words(cilin_path: str) -> Dict[str, List[str]]:
    """Build group_id ('Aa01A01=') -> list of member words from cilin."""
    group_to_words: Dict[str, List[str]] = {}
    with open(cilin_path, "r", encoding="utf-8-sig") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            head = parts[0]
            if head.endswith("="):
                group_to_words[head] = parts[1:]
    return group_to_words


def read_sentences(
    input_dir: Path,
    max_sentences: int,
    seed: int,
) -> List[str]:
    """
    Read raw text from *.txt files under input_dir.
    Files are shuffled; sentences are collected until max_sentences reached.
    max_sentences=0 means no limit.
    """
    files = sorted(input_dir.glob("**/*.txt"))
    rng = random.Random(seed)
    rng.shuffle(files)

    sentences: List[str] = []
    for fp in files:
        if max_sentences > 0 and len(sentences) >= max_sentences:
            break
        try:
            for line in fp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    sentences.append(line)
                if max_sentences > 0 and len(sentences) >= max_sentences:
                    break
        except Exception:
            continue
    return sentences


@torch.no_grad()
def get_all_word_embeddings(
    model: SimCSEModel,
    tokenizer,
    tokens: List[str],
    device: str,
    max_len: int,
) -> Dict[int, torch.Tensor]:
    """
    Single forward pass for one sentence.
    Returns word_idx -> L2-normalized contextual embedding [H] for each word.
    Words truncated beyond max_len will be absent from the dict.
    """
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    try:
        word_ids = enc.word_ids(batch_index=0)
    except Exception:
        return {}

    enc = {k: v.to(device) for k, v in enc.items()}
    out = model.bert(**enc, return_dict=True)
    hidden = out.last_hidden_state[0]  # [L, H]

    result: Dict[int, torch.Tensor] = {}
    for widx in range(len(tokens)):
        tok_indices = [pos for pos, wid in enumerate(word_ids) if wid == widx]
        if not tok_indices:
            continue
        idx_tensor = torch.tensor(tok_indices, device=device)
        emb = hidden[idx_tensor].mean(dim=0)
        emb = F.normalize(emb, p=2, dim=-1)
        result[widx] = emb
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate prototype classification on wiki_zh word embeddings."
    )
    parser.add_argument("--input", default="raw_data/wiki_zh",
                        help="Directory of raw txt files (one sentence per line).")
    parser.add_argument("--model", default="checkpoints/20260120/simcse_cilin_best.pt",
                        help="Path to SimCSEModel checkpoint.")
    parser.add_argument("--prototype", default="prototypes/20260121.pt",
                        help="Path to prototype .pt file.")
    parser.add_argument("--encoder", default="hfl/chinese-roberta-wwm-ext",
                        help="HuggingFace encoder name used during training.")
    parser.add_argument("--cilin", default="same.txt",
                        help="Path to Cilin same.txt.")
    parser.add_argument("--output", default="valid_prototype_results.json",
                        help="Output JSON path.")
    parser.add_argument("--max-sentences", type=int, default=10000,
                        help="Max sentences to process (0 = no limit; be careful with large corpora).")
    parser.add_argument("--min-sim", type=float, default=0.5,
                        help="Minimum cosine similarity to assign a word to a group.")
    parser.add_argument("--max-len", type=int, default=64,
                        help="Max subword token length passed to BERT.")
    parser.add_argument("--device", default="auto",
                        help="'auto', 'cuda', or 'cpu'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cilin-only", action="store_true",
                        help="Only classify words that appear in the Cilin vocabulary "
                             "(much faster; skips out-of-vocabulary words).")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # ---- Load model ----
    print(f"Loading model from {args.model} ...")
    model = load_model(args.model, args.encoder, device)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    # ---- Load prototypes ----
    print(f"Loading prototypes from {args.prototype} ...")
    group_ids, proto_matrix = load_prototypes(args.prototype, device)
    n_groups, hidden_dim = proto_matrix.shape
    print(f"  {n_groups} groups, hidden dim = {hidden_dim}")

    # ---- Load Cilin ----
    print(f"Parsing Cilin from {args.cilin} ...")
    syn_dict, related_dict = parse_cilin(args.cilin)
    cilin_vocab: Set[str] = set(syn_dict.keys()) | set(related_dict.keys())
    group_members: Dict[str, List[str]] = parse_group_to_words(args.cilin)
    print(f"  Cilin vocab: {len(cilin_vocab)} words, {len(group_members)} groups (= only)")

    # ---- Read sentences ----
    print(f"Reading sentences from {args.input} ...")
    sentences = read_sentences(
        Path(args.input),
        max_sentences=args.max_sentences,
        seed=args.seed,
    )
    print(f"  {len(sentences)} sentences loaded")

    # ---- Classify ----
    # group_id -> set of unique words classified into that group
    group_to_predicted_words: Dict[str, Set[str]] = defaultdict(set)
    # word -> (best_group_id, best_sim)  for deduplication across contexts
    word_best: Dict[str, Tuple[str, float]] = {}
    n_unmatched = 0

    print("Classifying word embeddings ...")
    for sent in tqdm(sentences, desc="sentences"):
        tokens: List[str] = [t for t in jieba.cut(sent, cut_all=False) if t.strip()]
        if len(tokens) < 2:
            continue

        # One forward pass: get all word embeddings for this sentence
        word_embs = get_all_word_embeddings(model, tokenizer, tokens, device, args.max_len)

        for widx, emb in word_embs.items():
            word = tokens[widx]
            if not word.strip():
                continue
            if args.cilin_only and word not in cilin_vocab:
                continue

            # Cosine similarity against all prototype vectors: [G]
            sims = proto_matrix @ emb  # both are L2-normalized

            best_sim_val, best_idx = sims.max(dim=0)
            best_sim_val = best_sim_val.item()
            best_group = group_ids[best_idx.item()]

            if best_sim_val >= args.min_sim:
                group_to_predicted_words[best_group].add(word)
                # Keep track of the best (highest-sim) assignment per word string
                prev = word_best.get(word)
                if prev is None or best_sim_val > prev[1]:
                    word_best[word] = (best_group, round(best_sim_val, 4))
            else:
                n_unmatched += 1

    # ---- Build output ----
    print("Building output JSON ...")
    output: Dict = {}

    # Group entries: only include groups that received at least one word
    for gid in group_ids:
        predicted = group_to_predicted_words.get(gid)
        if not predicted:
            continue
        output[gid] = {
            "predicted_words": sorted(predicted),
            "cilin_words": group_members.get(gid, []),
        }

    n_assigned_groups = len(output)
    n_assigned_words = len(word_best)

    output["_meta"] = {
        "model": args.model,
        "prototype": args.prototype,
        "sentences_processed": len(sentences),
        "min_sim_threshold": args.min_sim,
        "cilin_only": args.cilin_only,
        "total_groups_with_predictions": n_assigned_groups,
        "total_unique_words_assigned": n_assigned_words,
        "total_word_occurrences_unmatched": n_unmatched,
    }

    # ---- Save ----
    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    print(f"\nSaved → {out_path}")
    print(f"  Groups with predictions : {n_assigned_groups}")
    print(f"  Unique words assigned   : {n_assigned_words}")
    print(f"  Unmatched word occ.     : {n_unmatched}")


if __name__ == "__main__":
    main()
