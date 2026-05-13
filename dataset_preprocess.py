# dataset_preprocess.py
import argparse
import json
import os
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertForMaskedLM


def parse_cilin(path: str):
    """
    解析 same.txt / 詞林擴展版：
        Aa01A02= 人類 生人 全人類
        Aa01B03# 良民 順民
        Aa01A05@ 二人 三人 兩人 雙人

    回傳:
        syn_dict:     word -> [同義詞列表]   (對應 '=')   => 用來產生 positive
        related_dict: word -> [相關詞列表]   (對應 '#')   => 用來產生 hard negative
    """
    syn_dict: Dict[str, List[str]] = {}
    related_dict: Dict[str, List[str]] = {}

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            code = parts[0]
            words = parts[1:]
            tag = code[-1]  # 最後一位: '=', '#', '@'

            if tag == "=":
                for w in words:
                    syn_dict.setdefault(w, [])
                    syn_dict[w].extend([x for x in words if x != w])
            elif tag == "#":
                for w in words:
                    related_dict.setdefault(w, [])
                    related_dict[w].extend([x for x in words if x != w])
            else:
                # '@' 或其他：當作獨立詞，不用於構建 pair
                continue

    for d in (syn_dict, related_dict):
        for k, v in d.items():
            d[k] = sorted(set(v))

    return syn_dict, related_dict


# ======== Embedding-based filter ========

class EmbeddingFilter:
    def __init__(
        self,
        model_name: str = "hfl/chinese-bert-wwm-ext",
        device: str = "cuda",
        word_threshold: float = 0.4,
        ctx_threshold: float = 0.25,
    ):
        self.device = device
        self.word_threshold = word_threshold
        self.ctx_threshold = ctx_threshold

        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        base_model = BertModel.from_pretrained(model_name)
        emb_matrix = base_model.get_input_embeddings().weight.detach()  # [V, H]
        self.embeddings = F.normalize(emb_matrix.to(device), p=2, dim=-1)
        del base_model
        # cache for word embeddings to avoid re-tokenizing the same word repeatedly
        self.word_cache: dict[str, Optional[torch.Tensor]] = {}

    def _get_word_embedding(self, word: str) -> Optional[torch.Tensor]:
        if word in self.word_cache:
            return self.word_cache[word]
        tokens = self.tokenizer.tokenize(word)
        if not tokens:
            self.word_cache[word] = None
            return None
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids_tensor = torch.tensor(ids, device=self.device, dtype=torch.long)
        vecs = self.embeddings[ids_tensor]  # [k, H]
        emb = vecs.mean(dim=0)
        emb = F.normalize(emb, p=2, dim=-1)
        self.word_cache[word] = emb
        return emb

    def _get_context_embedding(self, tokens: List[str], skip_idx: int) -> Optional[torch.Tensor]:
        embs = []
        for i, w in enumerate(tokens):
            if i == skip_idx:
                continue
            e = self._get_word_embedding(w)
            if e is not None:
                embs.append(e)
        if not embs:
            return None
        ctx = torch.stack(embs, dim=0).mean(dim=0)
        ctx = F.normalize(ctx, p=2, dim=-1)
        return ctx

    def is_good_syn(
        self,
        tokens: List[str],
        idx: int,
        target_word: str,
        candidate: str,
        ctx_emb: Optional[torch.Tensor] = None,
    ) -> bool:
        tgt_emb = self._get_word_embedding(target_word)
        if tgt_emb is None:
            return False

        syn_emb = self._get_word_embedding(candidate)
        if syn_emb is None:
            return False

        cos_word = torch.dot(tgt_emb, syn_emb).item()
        if cos_word < self.word_threshold:
            return False

        if ctx_emb is None:
            ctx_emb = self._get_context_embedding(tokens, idx)
        if ctx_emb is None:
            # 沒有 context，就只看 cos_word
            return cos_word >= self.word_threshold

        cos_ctx = torch.dot(ctx_emb, syn_emb).item()
        if cos_ctx < self.ctx_threshold:
            return False

        return True


# ======== MLM-based filter（比較慢，可選） ========

class MLMFilter:
    def __init__(
        self,
        model_name: str = "hfl/chinese-bert-wwm-ext",
        device: str = "cuda",
        prob_threshold: float = 0.01,
        topk: int = 50,
    ):
        self.device = device
        self.prob_threshold = prob_threshold
        self.topk = topk

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.mlm = BertForMaskedLM.from_pretrained(model_name).to(device)
        self.mlm.eval()
        # cache tokenization for candidate words to avoid re-tokenizing
        self._cand_token_cache: dict[str, Optional[int]] = {}

    def _get_candidate_id(self, candidate: str) -> Optional[int]:
        """
        Convert candidate to a single token id; return None if it splits into multiple pieces.
        """
        if candidate in self._cand_token_cache:
            return self._cand_token_cache[candidate]
        tks = self.tokenizer.tokenize(candidate)
        if len(tks) != 1:
            self._cand_token_cache[candidate] = None
            return None
        cid = self.tokenizer.convert_tokens_to_ids(tks[0])
        self._cand_token_cache[candidate] = cid
        return cid

    def get_mask_probs_batch(
        self,
        tokens_list: List[List[str]],
        idx_list: List[int],
    ) -> List[Optional[torch.Tensor]]:
        """
        Batch encode multiple sentences (or multiple positions of the same sentence)
        with [MASK] inserted, returning a list of probability vectors.
        """
        assert len(tokens_list) == len(idx_list), "tokens_list and idx_list must align"
        if not tokens_list:
            return []

        masked_sentences: List[str] = []
        for toks, idx in zip(tokens_list, idx_list):
            if idx < 0 or idx >= len(toks):
                masked_sentences.append("")  # placeholder, will return None
                continue
            words = list(toks)
            words[idx] = self.tokenizer.mask_token
            masked_sentences.append("".join(words))

        enc = self.tokenizer(
            masked_sentences,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding=True,
        ).to(self.device)

        input_ids = enc["input_ids"]
        outputs = self.mlm(**enc)
        logits = outputs.logits  # [B, L, V]

        probs_list: List[Optional[torch.Tensor]] = []
        for row, masked in enumerate(masked_sentences):
            if not masked:
                probs_list.append(None)
                continue
            mask_positions = (input_ids[row] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_positions) == 0:
                probs_list.append(None)
                continue
            mask_idx = mask_positions[0].item()
            probs = torch.softmax(logits[row, mask_idx], dim=-1)  # [V]
            probs_list.append(probs)
        return probs_list

    def get_mask_probs(self, tokens: List[str], idx: int) -> Optional[torch.Tensor]:
        """
        Encode sentence with a single [MASK] at idx and return probability vector at that position.
        """
        res = self.get_mask_probs_batch([tokens], [idx])
        return res[0] if res else None

    @torch.no_grad()
    def filter_candidates(
        self,
        tokens: List[str],
        idx: int,
        candidates: List[str],
        probs: Optional[torch.Tensor] = None,
        max_keep: Optional[int] = None,
    ) -> List[str]:
        """
        Filter a batch of candidates for one masked position using a single forward pass.
        """
        if probs is None:
            probs = self.get_mask_probs(tokens, idx)
        if probs is None:
            return []

        topk_ids: Optional[set[int]] = None
        if self.topk is not None and self.topk > 0:
            k = min(self.topk, probs.numel())
            topk_ids = set(torch.topk(probs, k=k).indices.tolist())

        accepted: List[str] = []
        for cand in candidates:
            cid = self._get_candidate_id(cand)
            if cid is None:
                continue
            p = probs[cid].item()
            if p < self.prob_threshold:
                continue
            if topk_ids is not None and cid not in topk_ids:
                continue
            accepted.append(cand)
            if max_keep is not None and len(accepted) >= max_keep:
                break
        return accepted

    @torch.no_grad()
    def is_good_syn(
        self,
        tokens: List[str],
        idx: int,
        target_word: str,
        candidate: str,
    ) -> bool:
        # 向後相容：重用 batched 道路，只傳單一 candidate
        keep = self.filter_candidates(tokens, idx, [candidate], max_keep=1)
        return bool(keep)


# ======== 載入 tokenized sentences ========

def load_tokenized_sentences(path: str, min_len: int = 3) -> List[List[str]]:
    """
    path 可以是單一 JSON 檔 (List[List[str]]) 或一個目錄（裡面多個 JSON 檔）。
    """
    base = Path(path)
    files = sorted(base.glob("*.json")) if base.is_dir() else [base]

    sentences: List[List[str]] = []
    for fp in files:
        if not fp.is_file():
            continue
        with fp.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        for tokens in data:
            tok_len = len(tokens)
            if tok_len >= min_len:
                sentences.append([str(t) for t in tokens])
    return sentences


# ======== 主預處理邏輯 ========

def build_entry_for_sentence(
    tokens: List[str],
    syn_dict: Dict[str, List[str]],
    related_dict: Dict[str, List[str]],
    pos_filter,
    max_pos_per_idx: int = 4,
    max_neg_per_idx: int = 4,
) -> Dict[str, Any]:
    """
    對單一句子：
      - 為每個 index 產生 pos / hard_neg 的替換句
      - pos 使用 syn_dict('=')
      - hard_neg 使用 related_dict('#')
    """
    n = len(tokens)
    pos: Dict[str, List[List[str]]] = {}
    hard_neg: Dict[str, List[List[str]]] = {}

    ctx_cache: Dict[int, Optional[torch.Tensor]] = {}
    mlm_prob_cache: Dict[int, Optional[torch.Tensor]] = {}
    # pre-compute MLM probs for all candidate positions in one batch
    if isinstance(pos_filter, MLMFilter):
        mlm_indices: List[int] = []
        for i, w in enumerate(tokens):
            if syn_dict.get(w) or related_dict.get(w):
                mlm_indices.append(i)
        if mlm_indices:
            batch_probs = pos_filter.get_mask_probs_batch([tokens for _ in mlm_indices], mlm_indices)
            for idx_val, probs in zip(mlm_indices, batch_probs):
                mlm_prob_cache[idx_val] = probs

    def get_ctx(idx: int) -> Optional[torch.Tensor]:
        if not isinstance(pos_filter, EmbeddingFilter):
            return None
        if idx not in ctx_cache:
            ctx_cache[idx] = pos_filter._get_context_embedding(tokens, idx)
        return ctx_cache[idx]

    def get_mask_probs(idx: int) -> Optional[torch.Tensor]:
        if not isinstance(pos_filter, MLMFilter):
            return None
        if idx not in mlm_prob_cache:
            mlm_prob_cache[idx] = pos_filter.get_mask_probs(tokens, idx)
        return mlm_prob_cache[idx]

    def select_candidates_batch(cands: List[str], tgt_emb: torch.Tensor, ctx_emb: Optional[torch.Tensor]):
        cand_embs: list[torch.Tensor] = []
        cand_texts: list[str] = []
        for c in cands:
            emb = pos_filter._get_word_embedding(c)
            if emb is None:
                continue
            cand_texts.append(c)
            cand_embs.append(emb)
        if not cand_embs:
            return []
        mat = torch.stack(cand_embs, dim=0)  # [M, H]
        cos_word = torch.mv(mat, tgt_emb)
        mask = cos_word >= pos_filter.word_threshold
        if ctx_emb is not None:
            cos_ctx = torch.mv(mat, ctx_emb)
            mask = mask & (cos_ctx >= pos_filter.ctx_threshold)
        keep = [cand_texts[i] for i, m in enumerate(mask.tolist()) if m]
        return keep

    use_batch = isinstance(pos_filter, EmbeddingFilter)

    # positive：同義詞
    for i in range(n):
        w = tokens[i]
        candidates = syn_dict.get(w, [])
        if not candidates:
            continue

        accepted: List[List[str]] = []
        if use_batch:
            tgt_emb = pos_filter._get_word_embedding(w)
            if tgt_emb is None:
                continue
            batch_keep = select_candidates_batch(candidates, tgt_emb, get_ctx(i))
            for cand in batch_keep[:max_pos_per_idx]:
                new_tokens = list(tokens)
                new_tokens[i] = cand
                accepted.append(new_tokens)
        elif isinstance(pos_filter, MLMFilter):
            probs = get_mask_probs(i)
            if probs is None:
                continue
            batch_keep = pos_filter.filter_candidates(
                tokens,
                i,
                candidates,
                probs=probs,
                max_keep=max_pos_per_idx,
            )
            for cand in batch_keep:
                new_tokens = list(tokens)
                new_tokens[i] = cand
                accepted.append(new_tokens)
        else:
            for cand in candidates:
                if len(accepted) >= max_pos_per_idx:
                    break
                if use_batch:
                    ok = pos_filter.is_good_syn(tokens, i, w, cand, ctx_emb=get_ctx(i))
                else:
                    ok = pos_filter.is_good_syn(tokens, i, w, cand)
                if not ok:
                    continue
                new_tokens = list(tokens)
                new_tokens[i] = cand
                accepted.append(new_tokens)

        if accepted:
            pos[str(i)] = accepted

    # hard negative：相關詞（#）
    for i in range(n):
        w = tokens[i]
        candidates = related_dict.get(w, [])
        if not candidates:
            continue

        accepted: List[List[str]] = []
        if use_batch:
            tgt_emb = pos_filter._get_word_embedding(w)
            if tgt_emb is None:
                continue
            batch_keep = select_candidates_batch(candidates, tgt_emb, get_ctx(i))
            for cand in batch_keep[:max_neg_per_idx]:
                new_tokens = list(tokens)
                new_tokens[i] = cand
                accepted.append(new_tokens)
        elif isinstance(pos_filter, MLMFilter):
            probs = get_mask_probs(i)
            if probs is None:
                continue
            batch_keep = pos_filter.filter_candidates(
                tokens,
                i,
                candidates,
                probs=probs,
                max_keep=max_neg_per_idx,
            )
            for cand in batch_keep:
                new_tokens = list(tokens)
                new_tokens[i] = cand
                accepted.append(new_tokens)
        else:
            for cand in candidates:
                if len(accepted) >= max_neg_per_idx:
                    break
                # 可以選擇要不要過濾；這裡也跑一下 is_good_syn，
                # 但你如果覺得太嚴格，可以改成「不過濾，只要有就替換」。
                if use_batch:
                    ok = pos_filter.is_good_syn(tokens, i, w, cand, ctx_emb=get_ctx(i))
                else:
                    ok = pos_filter.is_good_syn(tokens, i, w, cand)
                if not ok:
                    continue
                new_tokens = list(tokens)
                new_tokens[i] = cand
                accepted.append(new_tokens)

        if accepted:
            hard_neg[str(i)] = accepted

    return {
        "tokens": tokens,
        "pos": pos,
        "hard_neg": hard_neg,
    }


# 全域 filter 供多進程使用
_GLOBAL_FILTER = None
_GLOBAL_SYN = None
_GLOBAL_REL = None
_GLOBAL_POS_MAX = None
_GLOBAL_NEG_MAX = None
_GLOBAL_MODE = None


def _init_worker(
    filter_mode,
    model_name,
    device,
    word_threshold,
    ctx_threshold,
    prob_threshold,
    topk,
    syn_dict,
    related_dict,
    max_pos,
    max_neg,
):
    global _GLOBAL_FILTER, _GLOBAL_SYN, _GLOBAL_REL, _GLOBAL_POS_MAX, _GLOBAL_NEG_MAX, _GLOBAL_MODE
    _GLOBAL_SYN = syn_dict
    _GLOBAL_REL = related_dict
    _GLOBAL_POS_MAX = max_pos
    _GLOBAL_NEG_MAX = max_neg
    _GLOBAL_MODE = filter_mode
    worker_device = device
    if isinstance(device, str) and device.startswith("cuda"):
        # 避免 CUDA 在 fork 後初始化失敗，worker 一律用 CPU。
        worker_device = "cpu"
    if filter_mode == "embed":
        _GLOBAL_FILTER = EmbeddingFilter(
            model_name=model_name,
            device=worker_device,
            word_threshold=word_threshold,
            ctx_threshold=ctx_threshold,
        )
    else:
        _GLOBAL_FILTER = MLMFilter(
            model_name=model_name,
            device=worker_device,
            prob_threshold=prob_threshold,
            topk=topk,
        )


def _process_one_global(tokens: List[str]) -> Dict[str, Any]:
    return build_entry_for_sentence(
        tokens,
        _GLOBAL_SYN,
        _GLOBAL_REL,
        pos_filter=_GLOBAL_FILTER,
        max_pos_per_idx=_GLOBAL_POS_MAX,
        max_neg_per_idx=_GLOBAL_NEG_MAX,
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for SimCSE + Cilin (offline JSON).")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to tokenized sentences (JSON or directory of JSONs).")
    parser.add_argument("--cilin", type=str, required=True,
                        help="Path to same.txt / Cilin-style thesaurus.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory; each input JSON will write a same-named JSON here.")
    parser.add_argument("--model-name", type=str, default="hfl/chinese-bert-wwm-ext",
                        help="Base BERT model name.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for embeddings/MLM.")
    parser.add_argument("--filter-mode", choices=("embed", "mlm"), default="embed",
                        help="Use embedding similarity or MLM prob for filtering.")
    parser.add_argument("--max-pos-per-idx", type=int, default=4,
                        help="Maximum positive variants per token index.")
    parser.add_argument("--max-neg-per-idx", type=int, default=4,
                        help="Maximum hard negative variants per token index.")
    parser.add_argument("--min-len", type=int, default=3,
                        help="Minimum tokenized sentence length to keep.")
    parser.add_argument("--max-len", type=int, default=15,
                        help="Maximum tokenized sentence length to keep (skip if exceeded).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes for preprocessing.")
    parser.add_argument("--chunksize", type=int, default=64,
                        help="Batch size per worker map to balance throughput.")
    parser.add_argument("--word-threshold", type=float, default=0.4,
                        help="Embedding cosine threshold for target vs candidate.")
    parser.add_argument("--ctx-threshold", type=float, default=0.25,
                        help="Embedding cosine threshold for context vs candidate.")
    parser.add_argument("--prob-threshold", type=float, default=0.01,
                        help="MLM probability threshold for candidate (mlm mode).")
    parser.add_argument("--topk", type=int, default=50,
                        help="MLM top-k inclusion check (mlm mode).")
    args = parser.parse_args()

    print("Loading Cilin...")
    syn_dict, related_dict = parse_cilin(args.cilin)

    input_path = Path(args.input)
    input_files = sorted(input_path.glob("*.json")) if input_path.is_dir() else [input_path]

    out_dir = Path(args.output)
    if out_dir.exists() and out_dir.is_file():
        raise ValueError("--output should be a directory, not an existing file.")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing filter (mode={args.filter_mode})...")
    if args.filter_mode == "embed":
        pos_filter = EmbeddingFilter(
            model_name=args.model_name,
            device=args.device,
            word_threshold=args.word_threshold,
            ctx_threshold=args.ctx_threshold,
        )
    else:
        pos_filter = MLMFilter(
            model_name=args.model_name,
            device=args.device,
            prob_threshold=args.prob_threshold,
            topk=args.topk,
        )

    workers = max(1, args.workers)

    # 先統計過濾後的總句數，供靜態進度條使用（會再讀一次檔案）。
    total_sents = 0
    for fp in tqdm(input_files, desc="Counting", unit="file"):
        sents_tmp = load_tokenized_sentences(fp, min_len=args.min_len)
        if args.max_len is not None:
            sents_tmp = [s for s in sents_tmp if len(s) <= args.max_len]
        total_sents += len(sents_tmp)
    global_pbar = tqdm(total=total_sents, desc="Preprocessing", unit="sent")

    for fp in input_files:
        sentences = load_tokenized_sentences(fp, min_len=args.min_len)
        if args.max_len is not None:
            sentences = [s for s in sentences if len(s) <= args.max_len]
        if not sentences:
            continue

        output_entries: List[Dict[str, Any]] = []

        if workers == 1:
            for tokens in sentences:
                output_entries.append(
                    build_entry_for_sentence(
                        tokens,
                        syn_dict,
                        related_dict,
                        pos_filter=pos_filter,
                        max_pos_per_idx=args.max_pos_per_idx,
                        max_neg_per_idx=args.max_neg_per_idx,
                    )
                )
                global_pbar.update(1)
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(
                    args.filter_mode,
                    args.model_name,
                    args.device,
                    args.word_threshold,
                    args.ctx_threshold,
                    args.prob_threshold,
                    args.topk,
                    syn_dict,
                    related_dict,
                    args.max_pos_per_idx,
                    args.max_neg_per_idx,
                ),
            ) as executor:
                results = executor.map(
                    _process_one_global,
                    sentences,
                    chunksize=args.chunksize,
                )
                for entry in results:
                    output_entries.append(entry)
                    global_pbar.update(1)

        if output_entries:
            target_path = out_dir / fp.name
            with target_path.open("w", encoding="utf-8") as f:
                json.dump(output_entries, f, ensure_ascii=False, indent=2)
    global_pbar.close()

    global_pbar.close()


if __name__ == "__main__":
    main()
