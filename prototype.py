# prototype.py
"""
Build Cilin sense prototypes from a trained SimCSEModel.

Pipeline (keeps your Step0/1, and uses margin-only rejection for all-polysemy groups):
  Step0: collect occurrence-level contextual word vectors for each '=' group, with per-word cap
  Step1: robust core selection (Keep) via trimmed-mean center and top-q cosine
  Step2: select SafeGroups by cohesion quantile
  Step3: build SafeProto for SafeGroups
  Step4: build prototypes:
        - non-all-polysemy groups: prefer anchor-only (monosemous words) if available
        - all-polysemy groups: margin-only rejection against SafeProto, then robust center

Output: torch.save(dict) containing:
  - "prototypes": {group_id: tensor[H]}
  - "meta": {group_id: {...stats...}}
  - "params": {...hyperparams...}

Assumptions:
  - dataset_json is a JSON list of tokenized sentences: List[List[str]]
  - cilin_txt is your same.txt / cilin file; we use only '=' lines as groups
  - model checkpoint is either:
      * a torch.save() dict with key "model_state_dict" or "state_dict", or
      * a raw state_dict
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Your SimCSEModel / SimCSEConfig
from model import SimCSEConfig, SimCSEModel


def parse_cilin_equals(path: str | Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Parse cilin file and keep only '=' lines as synonym groups."""
    group2words: Dict[str, List[str]] = {}
    word2groups: Dict[str, List[str]] = defaultdict(list)

    with Path(path).open("r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip().lstrip("\ufeff")
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            code = parts[0]
            if not code.endswith("="):
                continue
            words = parts[1:]
            seen = set()
            uniq = []
            for w in words:
                if w not in seen:
                    uniq.append(w)
                    seen.add(w)
            if len(uniq) < 2:
                continue
            group2words[code] = uniq
            for w in uniq:
                word2groups[w].append(code)

    return group2words, dict(word2groups)


@torch.no_grad()
def trimmed_mean_center(vecs: torch.Tensor, trim: float = 0.1) -> torch.Tensor:
    """
    vecs: [N, H], expected roughly L2-normalized. Returns unit vector center.
    2-pass trimmed mean based on cosine similarity.
    """
    if vecs.ndim != 2 or vecs.size(0) == 0:
        raise ValueError("trimmed_mean_center expects vecs [N,H] with N>0")
    mu = F.normalize(vecs.mean(dim=0), p=2, dim=0)
    if trim <= 0:
        return mu
    scores = (vecs @ mu)
    k = max(1, int(math.ceil(vecs.size(0) * (1 - trim))))
    topk = torch.topk(scores, k=k, largest=True).indices
    mu2 = F.normalize(vecs[topk].mean(dim=0), p=2, dim=0)
    return mu2


@torch.no_grad()
def top_q_items(items: List[Any], scores: torch.Tensor, q: float) -> List[Any]:
    """Keep top-q proportion by score."""
    assert 0 < q <= 1.0
    n = len(items)
    if n == 0:
        return []
    k = max(1, int(math.ceil(n * q)))
    idx = torch.topk(scores, k=k, largest=True).indices.tolist()
    return [items[i] for i in idx]


def load_tokenized_sentences(
    path: str | Path,
    min_len: int = 3,
    *,
    max_sentences: Optional[int] = None,
) -> List[List[str]]:
    """
    Load tokenized sentences from JSON.

    Supported formats:
    - `List[List[str]]` (each inner list is a tokenized sentence)
    - `List[Dict]` or `Dict` rows containing a `tokens: List[str]` field
      (e.g., NLI-style pair files where each file stores 2 dicts)
    """

    def _extract_tokens(row: Any) -> Optional[List[str]]:
        if isinstance(row, list):
            return [str(x) for x in row]
        if isinstance(row, dict):
            toks = row.get("tokens")
            if isinstance(toks, list):
                return [str(x) for x in toks]
        return None

    p = Path(path)
    files = [p] if p.is_file() else p.glob("*.json")
    out: List[List[str]] = []
    for f in files:
        try:
            with f.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception as e:
            print(f"[WARN] failed reading {f}: {e}")
            continue

        rows: Iterable[Any]
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = [data]
        else:
            continue

        for row in rows:
            toks = _extract_tokens(row)
            if toks is None or len(toks) < min_len:
                continue
            out.append(toks)
            if max_sentences is not None and len(out) >= max_sentences:
                return out

    if not out:
        raise ValueError(
            f"No valid tokenized sentences found under {path}. "
            "Expected JSON like List[List[str]] or rows with a 'tokens' field."
        )
    return out


def build_word_index(sentences: List[List[str]], vocab: Optional[set[str]] = None) -> Dict[str, List[Tuple[int, int]]]:
    idx: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    use_vocab = vocab is not None
    for sid, toks in enumerate(sentences):
        for pos, w in enumerate(toks):
            if use_vocab and w not in vocab:
                continue
            idx[w].append((sid, pos))
    return idx


def sample_occurrences(word_occ: List[Tuple[int, int]], max_per_word: int, rng: random.Random) -> List[Tuple[int, int]]:
    if len(word_occ) <= max_per_word:
        return word_occ
    return rng.sample(word_occ, k=max_per_word)


class SentenceEncodingCache:
    """Small LRU cache for sentence-level word embeddings."""
    def __init__(self, max_size: int = 2048):
        self.max_size = max_size
        self._cache: Dict[int, Any] = {}
        self._order = deque()

    def get(self, key: int):
        return self._cache.get(key)

    def put(self, key: int, value: Any):
        if key in self._cache:
            return
        self._cache[key] = value
        self._order.append(key)
        while len(self._order) > self.max_size:
            old = self._order.popleft()
            if old in self._cache:
                del self._cache[old]


def build_word_mask(word_ids_list: List[List[Optional[int]]], sentences_tokens: List[List[str]]) -> torch.BoolTensor:
    if not word_ids_list:
        return torch.BoolTensor([])

    max_len = max(len(row) for row in word_ids_list)
    max_words = max(len(ws) for ws in sentences_tokens) if sentences_tokens else 0

    ids = []
    for row in word_ids_list:
        cleaned = [-1 if wid is None else int(wid) for wid in row]
        if len(cleaned) < max_len:
            cleaned += [-1] * (max_len - len(cleaned))
        ids.append(cleaned)

    word_ids_tensor = torch.tensor(ids)  # [B, L]
    word_idx = torch.arange(max_words).view(1, max_words, 1)
    return word_ids_tensor.unsqueeze(1) == word_idx


def mean_pooling_by_word(token_embeddings: torch.Tensor, word_mask: torch.Tensor) -> torch.Tensor:
    """
    token_embeddings: [B, L, H]
    word_mask: [B, W, L]
    return: [B, W, H]
    """
    mask_expanded = word_mask.unsqueeze(-1).float()  # [B, W, L, 1]
    token_expanded = token_embeddings.unsqueeze(1)   # [B, 1, L, H]
    sum_embeddings = (token_expanded * mask_expanded).sum(dim=2)
    sum_mask = word_mask.sum(dim=2).clamp(min=1e-9).unsqueeze(-1)
    return sum_embeddings / sum_mask


@torch.no_grad()
def encode_batch_last_hidden(
    model: SimCSEModel,
    tokenizer,
    sentences_tokens: List[List[str]],
    device: str,
    max_len: int,
) -> Tuple[torch.Tensor, List[List[Optional[int]]]]:
    enc = tokenizer(
        sentences_tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    _, last_hidden = model.encode(input_ids, attention_mask, output_hidden=True)  # [B,L,H]

    word_ids_list: List[List[Optional[int]]] = []
    for i in range(len(sentences_tokens)):
        word_ids_list.append(enc.word_ids(batch_index=i))

    word_mask = build_word_mask(word_ids_list, sentences_tokens).to(last_hidden.device)
    word_embeddings = mean_pooling_by_word(last_hidden, word_mask)
    word_embeddings = word_embeddings.detach().cpu()

    return word_embeddings, word_ids_list


@torch.no_grad()
def extract_word_vec(word_embeddings_1: torch.Tensor, target_word_index: int) -> Optional[torch.Tensor]:
    if target_word_index < 0 or target_word_index >= word_embeddings_1.size(0):
        return None
    v = word_embeddings_1[target_word_index]
    v = F.normalize(v, p=2, dim=0)
    return v


def load_model(cfg_path_or_name: str, ckpt_path: str | Path, device: str) -> SimCSEModel:
    if cfg_path_or_name and Path(cfg_path_or_name).exists():
        cfg_obj = json.loads(Path(cfg_path_or_name).read_text(encoding="utf-8"))
        cfg = SimCSEConfig(**cfg_obj)
    elif cfg_path_or_name:
        cfg = SimCSEConfig(encoder_name=cfg_path_or_name)
    else:
        cfg = SimCSEConfig()

    model = SimCSEModel(cfg).to(device)
    model.eval()

    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        sd = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")
    return model


def build_prototypes(
    sentences: List[List[str]],
    group2words: Dict[str, List[str]],
    word2groups: Dict[str, List[str]],
    model: SimCSEModel,
    tokenizer,
    device: str,
    seed: int,
    max_len: int,
    max_per_word: int,
    keep_q: float,
    trim: float,
    safe_coh_q: float,
    min_keep: int,
    min_anchor_words: int,
    min_keep_anchor: int,
    margin: float,
    min_residual: int,
    cache_size: int,
    *,
    show_progress: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    def _get_tqdm():
        try:
            from tqdm.auto import tqdm as _tqdm  # type: ignore
        except Exception:
            return None
        return _tqdm

    _tqdm = _get_tqdm() if show_progress else None

    def _progress(iterable: Iterable[Any], *, total: Optional[int] = None, desc: str = "") -> Iterable[Any]:
        if _tqdm is not None:
            return _tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
        if total is None or total <= 0:
            return iterable

        bar_len = 30
        update_every = max(1, total // 200)

        def _gen():
            for i, x in enumerate(iterable, start=1):
                if i == 1 or i == total or (i % update_every) == 0:
                    filled = int(bar_len * i / total)
                    bar = "#" * filled + "-" * (bar_len - filled)
                    print(f"\r{desc:20s} [{bar}] {i}/{total}", end="", flush=True)
                yield x
            print()

        return _gen()

    rng = random.Random(seed)

    print("[STAGE] build_word_index")
    vocab = set(word2groups.keys())
    inv_index = build_word_index(sentences, vocab=vocab)

    cache = SentenceEncodingCache(max_size=cache_size)

    def get_sentence_encoding(sid: int, toks: List[str]):
        cached = cache.get(sid)
        if cached is not None:
            return cached
        word_embeddings_b, _ = encode_batch_last_hidden(model, tokenizer, [toks], device=device, max_len=max_len)
        word_embeddings = word_embeddings_b[0]  # [W, H] CPU
        cache.put(sid, word_embeddings)
        return word_embeddings

    # Step0: Occ
    print("[STAGE] Step0: collect occurrences (encode sentences on-demand)")
    Occ: Dict[str, List[Dict[str, Any]]] = {g: [] for g in group2words.keys()}
    group_items = group2words.items()
    if isinstance(group2words, dict):
        group_items = list(group_items)
    for g, words in _progress(group_items, total=len(group2words), desc="Occ (groups)"):
        for w in words:
            occs = inv_index.get(w, [])
            if not occs:
                continue
            sampled = sample_occurrences(occs, max_per_word=max_per_word, rng=rng)
            for sid, pos in sampled:
                toks = sentences[sid]
                word_embeddings = get_sentence_encoding(sid, toks)
                v = extract_word_vec(word_embeddings, target_word_index=pos)
                if v is None:
                    continue
                Occ[g].append({"word": w, "vec": v, "sid": sid, "pos": pos})

    # Step1: Keep + mu0 + cohesion
    print("[STAGE] Step1: compute Keep/mu0/cohesion")
    Keep: Dict[str, List[Dict[str, Any]]] = {}
    mu0: Dict[str, torch.Tensor] = {}
    cohesion: Dict[str, float] = {}
    too_small = set()

    occ_items = Occ.items()
    if isinstance(Occ, dict):
        occ_items = list(occ_items)
    for g, items in _progress(occ_items, total=len(Occ), desc="Keep (groups)"):
        if len(items) < min_keep:
            too_small.add(g)
            continue
        V = torch.stack([it["vec"] for it in items], dim=0)
        c = trimmed_mean_center(V, trim=trim)
        mu0[g] = c
        scores = (V @ c)
        Keep[g] = top_q_items(items, scores, q=keep_q)
        Vk = torch.stack([it["vec"] for it in Keep[g]], dim=0)
        cohesion[g] = float((Vk @ c).mean().item())

    coh_vals = [cohesion[g] for g in cohesion.keys() if g not in too_small]
    SafeGroups: List[str] = []
    if coh_vals:
        coh_t = float(torch.tensor(coh_vals).quantile(safe_coh_q).item())
        for g in cohesion.keys():
            if g in too_small:
                continue
            if cohesion[g] >= coh_t and len(Keep.get(g, [])) >= min_keep:
                SafeGroups.append(g)
    else:
        coh_t = float("nan")

    # Step3: SafeProto
    print("[STAGE] Step3: build SafeProto")
    SafeProto: Dict[str, torch.Tensor] = {}
    for s in _progress(SafeGroups, total=len(SafeGroups), desc="SafeProto"):
        V = torch.stack([it["vec"] for it in Keep[s]], dim=0)
        SafeProto[s] = trimmed_mean_center(V, trim=trim)

    def mono_words(g: str) -> List[str]:
        return [w for w in group2words[g] if len(word2groups.get(w, [])) == 1]

    def best_safe_sim(e: torch.Tensor) -> float:
        best = -1.0
        for p in SafeProto.values():
            sim = float((e @ p).item())
            if sim > best:
                best = sim
        return best

    prototypes: Dict[str, torch.Tensor] = {}
    meta: Dict[str, Dict[str, Any]] = {}

    print("[STAGE] Step4: build final prototypes")
    for g in _progress(list(group2words.keys()), total=len(group2words), desc="Proto (groups)"):
        if g in too_small or g not in Keep:
            meta[g] = {"status": "too_small_or_empty", "n_occ": len(Occ.get(g, [])), "n_keep": len(Keep.get(g, [])) if g in Keep else 0}
            continue

        anchors = mono_words(g)
        all_poly = (len(anchors) == 0)

        # Anchor proto if possible
        if not all_poly and len(anchors) >= min_anchor_words:
            anchor_set = set(anchors)
            anchor_vecs = [it["vec"] for it in Keep[g] if it["word"] in anchor_set]
            if len(anchor_vecs) >= min_keep_anchor:
                V = torch.stack(anchor_vecs, dim=0)
                p = trimmed_mean_center(V, trim=trim)
                prototypes[g] = p
                Vk = torch.stack([it["vec"] for it in Keep[g]], dim=0)
                meta[g] = {
                    "status": "anchor_proto",
                    "all_polysemy": False,
                    "n_occ": len(Occ[g]),
                    "n_keep": len(Keep[g]),
                    "n_anchor_words": len(anchors),
                    "n_anchor_occ": len(anchor_vecs),
                    "cohesion": cohesion.get(g, None),
                    "quality_keep": float((Vk @ p).mean().item()),
                }
                continue

        # All-polysemy: margin-only rejection against SafeProto
        if all_poly and SafeProto:
            resid = []
            for it in Keep[g]:
                e = it["vec"]
                sg = float((e @ mu0[g]).item())
                s1 = best_safe_sim(e)
                if (s1 - sg) >= margin:
                    continue
                resid.append(e)

            if len(resid) >= min_residual:
                V = torch.stack(resid, dim=0)
                p = trimmed_mean_center(V, trim=trim)
                prototypes[g] = p

                Vk = torch.stack([it["vec"] for it in Keep[g]], dim=0)
                meta[g] = {
                    "status": "allpoly_residual_proto",
                    "all_polysemy": True,
                    "n_occ": len(Occ[g]),
                    "n_keep": len(Keep[g]),
                    "n_residual": len(resid),
                    "cohesion": cohesion.get(g, None),
                    "quality_keep": float((Vk @ p).mean().item()),
                    "quality_residual": float((V @ p).mean().item()),
                    "margin": margin,
                    "n_safe_groups": len(SafeGroups),
                    "coh_threshold": coh_t,
                }
                continue

            # fallback to Keep
            Vk = torch.stack([it["vec"] for it in Keep[g]], dim=0)
            p = trimmed_mean_center(Vk, trim=trim)
            prototypes[g] = p
            meta[g] = {
                "status": "allpoly_fallback_keep_proto",
                "all_polysemy": True,
                "n_occ": len(Occ[g]),
                "n_keep": len(Keep[g]),
                "n_residual": len(resid),
                "cohesion": cohesion.get(g, None),
                "quality_keep": float((Vk @ p).mean().item()),
                "margin": margin,
                "n_safe_groups": len(SafeGroups),
                "coh_threshold": coh_t,
            }
            continue

        # Fallback for others
        Vk = torch.stack([it["vec"] for it in Keep[g]], dim=0)
        p = trimmed_mean_center(Vk, trim=trim)
        prototypes[g] = p
        meta[g] = {
            "status": "keep_proto",
            "all_polysemy": all_poly,
            "n_occ": len(Occ[g]),
            "n_keep": len(Keep[g]),
            "cohesion": cohesion.get(g, None),
            "quality_keep": float((Vk @ p).mean().item()),
            "n_safe_groups": len(SafeGroups),
            "coh_threshold": coh_t,
        }

    params = {
        "seed": seed,
        "max_len": max_len,
        "max_per_word": max_per_word,
        "keep_q": keep_q,
        "trim": trim,
        "safe_coh_q": safe_coh_q,
        "min_keep": min_keep,
        "min_anchor_words": min_anchor_words,
        "min_keep_anchor": min_keep_anchor,
        "margin": margin,
        "min_residual": min_residual,
        "cache_size": cache_size,
        "safe_coh_threshold": coh_t,
        "n_groups": len(group2words),
        "n_safe_groups": len(SafeGroups),
    }
    return prototypes, meta, params


def main():
    ap = argparse.ArgumentParser(description="Build Cilin prototypes and torch.save(dict).")
    ap.add_argument("--cilin", type=str, default="same.txt", help="Path to cilin same.txt")
    ap.add_argument("--data", type=str, required=True, help="Tokenized sentences JSON file or directory (e.g., data/nli_zh)")
    ap.add_argument("--ckpt", type=str, required=True, help="Trained SimCSEModel checkpoint path")
    ap.add_argument("--cfg", type=str, default="", help="SimCSEConfig JSON path OR encoder_name string (optional)")
    ap.add_argument("--out", type=str, default="prototypes.pt", help="Output path (torch.save)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-sentences", type=int, default=0, help="Max sentences to load (0 = no limit)")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--max-per-word", type=int, default=30)
    ap.add_argument("--keep-q", type=float, default=0.6)
    ap.add_argument("--trim", type=float, default=0.1)
    ap.add_argument("--safe-coh-q", type=float, default=0.7)
    ap.add_argument("--min-keep", type=int, default=20)
    ap.add_argument("--min-anchor-words", type=int, default=2)
    ap.add_argument("--min-keep-anchor", type=int, default=20)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--min-residual", type=int, default=20)
    ap.add_argument("--cache-size", type=int, default=2048)

    args = ap.parse_args()

    group2words, word2groups = parse_cilin_equals(args.cilin)
    print(f"[INFO] '=' groups: {len(group2words):,} | unique words in '=': {len(word2groups):,}")

    max_sentences = args.max_sentences if args.max_sentences and args.max_sentences > 0 else None
    sentences = load_tokenized_sentences(args.data, max_sentences=max_sentences)
    print(f"[INFO] sentences: {len(sentences):,}")

    model = load_model(args.cfg, args.ckpt, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(model.cfg.encoder_name, use_fast=True)

    prototypes, meta, params = build_prototypes(
        sentences=sentences,
        group2words=group2words,
        word2groups=word2groups,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        seed=args.seed,
        max_len=args.max_len,
        max_per_word=args.max_per_word,
        keep_q=args.keep_q,
        trim=args.trim,
        safe_coh_q=args.safe_coh_q,
        min_keep=args.min_keep,
        min_anchor_words=args.min_anchor_words,
        min_keep_anchor=args.min_keep_anchor,
        margin=args.margin,
        min_residual=args.min_residual,
        cache_size=args.cache_size,
    )

    out_obj = {"prototypes": prototypes, "meta": meta, "params": params}
    torch.save(out_obj, args.out)
    print(f"[INFO] saved prototypes: {len(prototypes):,} -> {args.out}")

    status_count = defaultdict(int)
    for m in meta.values():
        status_count[m.get("status", "unknown")] += 1
    print("[INFO] status counts:")
    for k, v in sorted(status_count.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
