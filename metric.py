# metric.py
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from model import SimCSEModel


TEMPLATES = [
    "我覺得{w}很重要。",
    "最近大家都在討論{w}。",
    "很多人關心{w}這件事。",
]


@torch.no_grad()
def local_uniformity(z: torch.Tensor, k: int = 5, max_samples: int = 2048) -> float:
    """
    Local Uniformity U_K = (1/N) Σ_i log( (1/K) Σ_{j∈NN(i,K)} exp(-2||z_i - z_j||^2) )
    If less than 2 samples are provided, returns 0.0.
    max_samples can be used to subsample embeddings to avoid OOM on very large batches.
    """
    if z is None or z.numel() == 0 or z.dim() != 2:
        return 0.0

    n = z.size(0)
    if n < 2:
        return 0.0

    # Subsample to control memory footprint when computing pairwise distances.
    if n > max_samples:
        idx = torch.randperm(n, device=z.device)[:max_samples]
        z = z[idx]
        n = z.size(0)

    k_eff = min(max(k, 1), n - 1)

    # Normalize to ensure pairwise distance is on the unit hypersphere.
    z = F.normalize(z, p=2, dim=-1)

    # Pairwise squared Euclidean distance on normalized vectors.
    dist = torch.cdist(z, z, p=2.0)  # [n, n]
    dist_sq = dist.pow(2)
    dist_sq.fill_diagonal_(float("inf"))  # exclude self from kNN

    knn_dist, _ = torch.topk(dist_sq, k=k_eff, dim=1, largest=False)
    exp_terms = torch.exp(-2.0 * knn_dist)
    local_vals = torch.log(exp_terms.mean(dim=1))
    return float(local_vals.mean().item())


@torch.no_grad()
def retrieval_metrics(
    z_query: torch.Tensor,
    z_cand: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """
    Pair retrieval metrics for aligned pairs (positive is diagonal).
    Returns Recall@K and MRR for query->candidate retrieval.
    """
    if (
        z_query is None
        or z_cand is None
        or z_query.dim() != 2
        or z_cand.dim() != 2
        or z_query.size(0) == 0
        or z_cand.size(0) == 0
    ):
        out = {"mrr": 0.0, "n": 0.0}
        for k in topk:
            out[f"r@{int(k)}"] = 0.0
        return out

    n = min(z_query.size(0), z_cand.size(0))
    z_query = F.normalize(z_query[:n], p=2, dim=-1)
    z_cand = F.normalize(z_cand[:n], p=2, dim=-1)

    sim = z_query @ z_cand.T  # [N, N], positive index = diagonal
    pos = sim.diag().unsqueeze(1)  # [N, 1]
    # rank = 1 + number of candidates with strictly higher similarity than positive
    ranks = 1 + (sim > pos).sum(dim=1)

    metrics: Dict[str, float] = {"mrr": float((1.0 / ranks.float()).mean().item()), "n": float(n)}
    for k in topk:
        k_int = int(k)
        if k_int <= 0:
            continue
        metrics[f"r@{k_int}"] = float((ranks <= k_int).float().mean().item())
    return metrics


@torch.no_grad()
def paired_retrieval_metrics(
    z1: torch.Tensor,
    z2: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """
    Symmetric retrieval on aligned pairs:
      - z1 -> z2
      - z2 -> z1
    Returns per-direction metrics plus bidirectional averages.
    """
    m12 = retrieval_metrics(z1, z2, topk=topk)
    m21 = retrieval_metrics(z2, z1, topk=topk)

    out: Dict[str, float] = {
        "n": min(m12.get("n", 0.0), m21.get("n", 0.0)),
        "mrr_12": m12.get("mrr", 0.0),
        "mrr_21": m21.get("mrr", 0.0),
        "mrr": 0.5 * (m12.get("mrr", 0.0) + m21.get("mrr", 0.0)),
    }
    for k in topk:
        k_int = int(k)
        key = f"r@{k_int}"
        v12 = m12.get(key, 0.0)
        v21 = m21.get(key, 0.0)
        out[f"{key}_12"] = v12
        out[f"{key}_21"] = v21
        out[key] = 0.5 * (v12 + v21)
    return out


@torch.no_grad()
def word_level_retrieval_metrics_from_entries(
    model: SimCSEModel,
    tokenizer: PreTrainedTokenizerBase,
    entries: List[Dict[str, Any]],
    device: str,
    num_pairs: int = 512,
    max_len: int = 64,
    topk: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """
    Build aligned retrieval pairs from dataset entries using positive replacements.
    Query/candidate embeddings are contextual *word* embeddings at the replaced index.
    """
    candidates: List[Tuple[List[str], int, List[str]]] = []
    for item in entries:
        tokens = item.get("tokens", [])
        pos_map = item.get("pos", {}) or {}
        if not isinstance(tokens, list) or not isinstance(pos_map, dict):
            continue
        for k, variants in pos_map.items():
            if not variants:
                continue
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            # One variant per position to avoid over-weighting a single sentence/index.
            picked = random.choice(variants)
            if not isinstance(picked, list):
                continue
            if idx < 0 or idx >= len(tokens) or idx >= len(picked):
                continue
            if str(tokens[idx]) == str(picked[idx]):
                continue
            candidates.append(([str(t) for t in tokens], idx, [str(t) for t in picked]))

    if not candidates:
        out = {"mrr": 0.0, "n": 0.0}
        for k in topk:
            out[f"r@{int(k)}"] = 0.0
        return out

    if len(candidates) > num_pairs:
        candidates = random.sample(candidates, num_pairs)

    model_was_training = model.training
    model.eval()
    emb_cache: Dict[Tuple[Tuple[str, ...], int], torch.Tensor] = {}
    z_anchor_list: List[torch.Tensor] = []
    z_pos_list: List[torch.Tensor] = []

    for tokens, idx, pos_tokens in candidates:
        z_anchor_list.append(
            _get_contextual_embedding_from_tokens(
                model, tokenizer, tokens, idx, device, max_len=max_len, cache=emb_cache
            ).detach().cpu()
        )
        z_pos_list.append(
            _get_contextual_embedding_from_tokens(
                model, tokenizer, pos_tokens, idx, device, max_len=max_len, cache=emb_cache
            ).detach().cpu()
        )

    if model_was_training:
        model.train()

    if not z_anchor_list or not z_pos_list:
        out = {"mrr": 0.0, "n": 0.0}
        for k in topk:
            out[f"r@{int(k)}"] = 0.0
        return out

    z_anchor = torch.stack(z_anchor_list, dim=0)
    z_pos = torch.stack(z_pos_list, dim=0)
    return paired_retrieval_metrics(z_anchor, z_pos, topk=topk)


def _span_to_token_idx(offsets: torch.Tensor, char_start: int, char_end: int) -> int:
    indices = []
    for i, (s, e) in enumerate(offsets.tolist()):
        if s == e == 0:
            continue
        if not (e <= char_start or s >= char_end):
            indices.append(i)
    if not indices:
        return 0
    return indices[0]


@torch.no_grad()
def _get_contextual_embedding_from_tokens(
    model: SimCSEModel,
    tokenizer: PreTrainedTokenizerBase,
    tokens: List[str],
    target_word_idx: int,
    device: str,
    max_len: int = 64,
    cache: Optional[Dict[Tuple[Tuple[str, ...], int], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    從 tokenized sentence 直接取某個 word index 的 contextual embedding。
    會對應該詞所有 subword 做平均；若對不到則 fallback CLS。
    """
    key: Optional[Tuple[Tuple[str, ...], int]] = None
    if cache is not None:
        key = (tuple(tokens), int(target_word_idx))
        cached = cache.get(key)
        if cached is not None:
            return cached

    if target_word_idx < 0 or target_word_idx >= len(tokens):
        # Fallback to CLS via string path when index is invalid.
        text = "".join(tokens)
        target_word = tokens[0] if tokens else ""
        vec = _get_contextual_embedding(model, tokenizer, text, target_word, device, max_len)
        if cache is not None and key is not None:
            cache[key] = vec
        return vec

    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    # word_ids is available for fast tokenizers; fallback to string-based helper otherwise.
    try:
        word_ids = enc.word_ids(batch_index=0)
    except Exception:
        text = "".join(tokens)
        target_word = tokens[target_word_idx]
        vec = _get_contextual_embedding(model, tokenizer, text, target_word, device, max_len)
        if cache is not None and key is not None:
            cache[key] = vec
        return vec

    enc = {k: v.to(device) for k, v in enc.items()}
    out = model.bert(**enc, return_dict=True)
    hidden = out.last_hidden_state[0]  # [L, H]

    mask = torch.tensor([wid == target_word_idx for wid in word_ids], device=hidden.device, dtype=torch.bool)
    if mask.any():
        vec = hidden[mask].mean(dim=0)
    else:
        vec = hidden[0]  # CLS fallback (e.g., truncated away)
    vec = F.normalize(vec, p=2, dim=-1)

    if cache is not None and key is not None:
        cache[key] = vec
    return vec


@torch.no_grad()
def _get_contextual_embedding(
    model: SimCSEModel,
    tokenizer: PreTrainedTokenizerBase,
    sentence: str,
    target_word: str,
    device: str,
    max_len: int = 64,
) -> torch.Tensor:
    """
    在一個句子中，取得 target_word 的 contextual embedding。
    若找不到該詞，則回傳 CLS 向量作為 fallback。
    """
    idx = sentence.find(target_word)
    if idx == -1:
        enc = tokenizer(
            sentence,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        ).to(device)
        out = model.bert(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            return_dict=True,
        )
        cls_emb = out.last_hidden_state[0, 0]
        return F.normalize(cls_emb, p=2, dim=-1)

    char_start = idx
    char_end = idx + len(target_word)

    enc = tokenizer(
        sentence,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_offsets_mapping=True,
    ).to(device)
    offsets = enc["offset_mapping"][0]
    token_idx = _span_to_token_idx(offsets, char_start, char_end)

    out = model.bert(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        return_dict=True,
    )
    token_emb = out.last_hidden_state[0, token_idx]
    return F.normalize(token_emb, p=2, dim=-1)


@torch.no_grad()
def contextual_synonym_consistency(
    model: SimCSEModel,
    tokenizer: PreTrainedTokenizerBase,
    syn_dict: Dict[str, List[str]],
    device: str,
    num_pairs: int = 200,
    max_len: int = 64,
) -> float:
    """
    Contextual Synonym Consistency (CSC):
    對於 (w, s) ∈ Cilin '='，在多個 template context 中替換 w → s，
    比較 E(w|context) 與 E(s|context) 的 cosine similarity，取平均。
    """
    pairs: List[Tuple[str, str]] = []
    for w, syns in syn_dict.items():
        for s in syns:
            if w != s:
                pairs.append((w, s))

    if not pairs:
        return 0.0
    if len(pairs) > num_pairs:
        pairs = random.sample(pairs, num_pairs)

    sims: List[float] = []
    model_was_training = model.training
    model.eval()

    for w, s in pairs:
        emb_ws = []
        emb_ss = []
        for tmpl in TEMPLATES:
            sent_w = tmpl.format(w=w)
            sent_s = tmpl.format(w=s)
            ew = _get_contextual_embedding(model, tokenizer, sent_w, w, device, max_len)
            es = _get_contextual_embedding(model, tokenizer, sent_s, s, device, max_len)
            emb_ws.append(ew)
            emb_ss.append(es)

        ew_mean = torch.stack(emb_ws, dim=0).mean(dim=0)
        es_mean = torch.stack(emb_ss, dim=0).mean(dim=0)
        ew_mean = F.normalize(ew_mean, p=2, dim=-1)
        es_mean = F.normalize(es_mean, p=2, dim=-1)
        cos = F.cosine_similarity(ew_mean.unsqueeze(0), es_mean.unsqueeze(0)).item()
        sims.append(cos)

    if model_was_training:
        model.train()

    if not sims:
        return 0.0
    return float(sum(sims) / len(sims))


@torch.no_grad()
def contextual_synonym_margin(
    model: SimCSEModel,
    tokenizer: PreTrainedTokenizerBase,
    syn_dict: Dict[str, List[str]],
    related_dict: Dict[str, List[str]],
    device: str,
    num_pairs: int = 200,
    max_len: int = 64,
) -> dict:
    """
    Contextual Synonym Margin (CSM):
    對於同時有 '=' 與 '#' 的詞 w：
      - μ_same = mean cos(E(w|C), E(s|C)) over s ∈ '='
      - μ_related = mean cos(E(w|C), E(r|C)) over r ∈ '#'
      - margin = μ_same - μ_related

    回傳 dict：
      {
        "mu_same": float,
        "mu_related": float,
        "margin": float
      }
    """
    candidate_ws = [w for w in syn_dict.keys() if w in related_dict and syn_dict[w] and related_dict[w]]
    if not candidate_ws:
        return {"mu_same": 0.0, "mu_related": 0.0, "margin": 0.0}

    if len(candidate_ws) > num_pairs:
        candidate_ws = random.sample(candidate_ws, num_pairs)

    same_sims: List[float] = []
    rel_sims: List[float] = []

    model_was_training = model.training
    model.eval()

    for w in candidate_ws:
        # w 的 embedding 用多模板平均
        emb_w_list = []
        for tmpl in TEMPLATES:
            sent_w = tmpl.format(w=w)
            ew = _get_contextual_embedding(model, tokenizer, sent_w, w, device, max_len)
            emb_w_list.append(ew)
        emb_w = F.normalize(torch.stack(emb_w_list, dim=0).mean(dim=0), p=2, dim=-1)

        # 同義詞
        for s in syn_dict[w]:
            emb_s_list = []
            for tmpl in TEMPLATES:
                sent_s = tmpl.format(w=s)
                es = _get_contextual_embedding(model, tokenizer, sent_s, s, device, max_len)
                emb_s_list.append(es)
            emb_s = F.normalize(torch.stack(emb_s_list, dim=0).mean(dim=0), p=2, dim=-1)
            same_sims.append(F.cosine_similarity(emb_w.unsqueeze(0), emb_s.unsqueeze(0)).item())

        # 相關詞
        for r in related_dict[w]:
            emb_r_list = []
            for tmpl in TEMPLATES:
                sent_r = tmpl.format(w=r)
                er = _get_contextual_embedding(model, tokenizer, sent_r, r, device, max_len)
                emb_r_list.append(er)
            emb_r = F.normalize(torch.stack(emb_r_list, dim=0).mean(dim=0), p=2, dim=-1)
            rel_sims.append(F.cosine_similarity(emb_w.unsqueeze(0), emb_r.unsqueeze(0)).item())

    if model_was_training:
        model.train()

    if not same_sims or not rel_sims:
        return {"mu_same": 0.0, "mu_related": 0.0, "margin": 0.0}

    mu_same = float(sum(same_sims) / len(same_sims))
    mu_rel = float(sum(rel_sims) / len(rel_sims))
    margin = mu_same - mu_rel

    return {"mu_same": mu_same, "mu_related": mu_rel, "margin": margin}


@torch.no_grad()
def contextual_synonym_consistency_from_entries(
    model: SimCSEModel,
    tokenizer: PreTrainedTokenizerBase,
    entries: List[Dict[str, Any]],
    device: str,
    num_pairs: int = 200,
    max_len: int = 64,
) -> float:
    """
    Dataset-based CSC:
    使用 preprocessed entry 中的 pos 替換句，直接在真實語料上下文比較
    E(orig_word | sentence) 與 E(replaced_syn | sentence_pos) 的 cosine。
    """
    candidates: List[Tuple[List[str], int, List[str]]] = []
    for item in entries:
        tokens = item.get("tokens", [])
        pos_map = item.get("pos", {}) or {}
        if not isinstance(tokens, list) or not isinstance(pos_map, dict):
            continue
        for k, variants in pos_map.items():
            if not variants:
                continue
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            # Limit candidate explosion: one sampled variant per index.
            var = random.choice(variants)
            if not isinstance(var, list):
                continue
            candidates.append(([str(t) for t in tokens], idx, [str(t) for t in var]))

    if not candidates:
        return 0.0
    if len(candidates) > num_pairs:
        candidates = random.sample(candidates, num_pairs)

    sims: List[float] = []
    model_was_training = model.training
    model.eval()
    emb_cache: Dict[Tuple[Tuple[str, ...], int], torch.Tensor] = {}

    for tokens, idx, pos_tokens in candidates:
        e_anchor = _get_contextual_embedding_from_tokens(
            model, tokenizer, tokens, idx, device, max_len=max_len, cache=emb_cache
        )
        e_pos = _get_contextual_embedding_from_tokens(
            model, tokenizer, pos_tokens, idx, device, max_len=max_len, cache=emb_cache
        )
        sims.append(F.cosine_similarity(e_anchor.unsqueeze(0), e_pos.unsqueeze(0)).item())

    if model_was_training:
        model.train()
    return float(sum(sims) / len(sims)) if sims else 0.0


@torch.no_grad()
def contextual_synonym_margin_from_entries(
    model: SimCSEModel,
    tokenizer: PreTrainedTokenizerBase,
    entries: List[Dict[str, Any]],
    device: str,
    num_pairs: int = 200,
    max_len: int = 64,
) -> dict:
    """
    Dataset-based CSM:
    使用同一 entry、同一 token index 上的 pos 與 hard_neg 替換句，計算
      mu_same = mean cos(E(anchor), E(pos))
      mu_related = mean cos(E(anchor), E(neg))
      margin = mu_same - mu_related
    """
    candidates: List[Tuple[List[str], int, List[str], List[str]]] = []
    for item in entries:
        tokens = item.get("tokens", [])
        pos_map = item.get("pos", {}) or {}
        hard_map = item.get("hard_neg", {}) or {}
        if not isinstance(tokens, list) or not isinstance(pos_map, dict) or not isinstance(hard_map, dict):
            continue
        common_keys = set(pos_map.keys()) & set(hard_map.keys())
        for k in common_keys:
            pos_vars = pos_map.get(k) or []
            neg_vars = hard_map.get(k) or []
            if not pos_vars or not neg_vars:
                continue
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            pos_toks = random.choice(pos_vars)
            neg_toks = random.choice(neg_vars)
            if not isinstance(pos_toks, list) or not isinstance(neg_toks, list):
                continue
            candidates.append(
                ([str(t) for t in tokens], idx, [str(t) for t in pos_toks], [str(t) for t in neg_toks])
            )

    if not candidates:
        return {"mu_same": 0.0, "mu_related": 0.0, "margin": 0.0}
    if len(candidates) > num_pairs:
        candidates = random.sample(candidates, num_pairs)

    same_sims: List[float] = []
    rel_sims: List[float] = []
    model_was_training = model.training
    model.eval()
    emb_cache: Dict[Tuple[Tuple[str, ...], int], torch.Tensor] = {}

    for tokens, idx, pos_tokens, neg_tokens in candidates:
        e_anchor = _get_contextual_embedding_from_tokens(
            model, tokenizer, tokens, idx, device, max_len=max_len, cache=emb_cache
        )
        e_pos = _get_contextual_embedding_from_tokens(
            model, tokenizer, pos_tokens, idx, device, max_len=max_len, cache=emb_cache
        )
        e_neg = _get_contextual_embedding_from_tokens(
            model, tokenizer, neg_tokens, idx, device, max_len=max_len, cache=emb_cache
        )
        same_sims.append(F.cosine_similarity(e_anchor.unsqueeze(0), e_pos.unsqueeze(0)).item())
        rel_sims.append(F.cosine_similarity(e_anchor.unsqueeze(0), e_neg.unsqueeze(0)).item())

    if model_was_training:
        model.train()

    if not same_sims or not rel_sims:
        return {"mu_same": 0.0, "mu_related": 0.0, "margin": 0.0}
    mu_same = float(sum(same_sims) / len(same_sims))
    mu_rel = float(sum(rel_sims) / len(rel_sims))
    return {"mu_same": mu_same, "mu_related": mu_rel, "margin": mu_same - mu_rel}
