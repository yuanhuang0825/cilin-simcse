# dataset.py
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, Sampler


def load_preprocessed_dataset(
    path: str | Path,
    min_len: int = 3,
    max_len: int | None = None,
    recursive: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load preprocessed entries produced by dataset_preprocess.py.
    Each JSON should contain a list of dicts with keys: tokens, pos, hard_neg.
    """
    base = Path(path)
    if base.is_dir():
        files = sorted(base.rglob("*.json")) if recursive else sorted(base.glob("*.json"))
    else:
        files = [base]

    entries: List[Dict[str, Any]] = []
    print(f"Loading preprocessed dataset from {len(files)} files...")
    for fp in tqdm(files):
        if not fp.is_file():
            continue
        with fp.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        for item in data:
            tokens = item.get("tokens", [])
            if not isinstance(tokens, list):
                continue
            if len(tokens) < min_len:
                continue
            if max_len is not None and len(tokens) > max_len:
                continue
            entries.append(
                {
                    "tokens": [str(t) for t in tokens],
                    "pos": item.get("pos", {}),
                    "hard_neg": item.get("hard_neg", {}),
                }
            )
    return entries


def load_word2senses(path: str | Path) -> Dict[str, List[str]]:
    """
    Parse same.txt and build word -> sense_id list mapping.
    Each sense_id is derived from the leading code without trailing '=' or '#'.
    """
    word2senses: Dict[str, List[str]] = {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            head = parts[0]
            if not (head.endswith("=") or head.endswith("#")):
                continue
            sense_id = head[:-1]
            for word in parts[1:]:
                word2senses.setdefault(word, []).append(sense_id)
    for word, senses in word2senses.items():
        word2senses[word] = sorted(set(senses))
    return word2senses


class TwoStageSenseBatchSampler(Sampler[List[int]]):
    """
    Two-stage sampling: sample K sense groups, then m samples per group.
    Batch size is K * m.
    """

    def __init__(
        self,
        entries: List[Dict[str, Any]],
        word2senses: Dict[str, List[str]],
        batch_size: int,
        group_sample_k: int,
        seed: Optional[int] = None,
        include_no_sense: bool = True,
    ) -> None:
        if group_sample_k <= 0:
            raise ValueError("group_sample_k must be positive.")
        if batch_size % group_sample_k != 0:
            raise ValueError("batch_size must be divisible by group_sample_k.")
        if group_sample_k > batch_size:
            raise ValueError("group_sample_k must be <= batch_size.")

        self.entries = entries
        self.word2senses = word2senses
        self.batch_size = batch_size
        self.group_sample_k = group_sample_k
        self.samples_per_group = batch_size // group_sample_k
        self.include_no_sense = include_no_sense
        self._rng = random.Random(seed)

        self.sense_to_indices: Dict[str, List[int]] = {}
        self._build_index()
        self.sense_ids = sorted(self.sense_to_indices.keys())

        if not self.sense_ids:
            raise ValueError("No sense groups found for two-stage sampling.")

    def _build_index(self) -> None:
        sense_map: Dict[str, set[int]] = {}
        for idx, entry in enumerate(self.entries):
            tokens = entry.get("tokens", [])
            sense_ids: set[str] = set()
            for token in tokens:
                for sense_id in self.word2senses.get(token, []):
                    sense_ids.add(sense_id)
            if not sense_ids and self.include_no_sense:
                sense_ids.add("NO_SENSE")
            for sense_id in sense_ids:
                sense_map.setdefault(sense_id, set()).add(idx)
        self.sense_to_indices = {k: sorted(v) for k, v in sense_map.items() if v}

    def __len__(self) -> int:
        return len(self.entries) // self.batch_size

    def __iter__(self):
        num_groups = len(self.sense_ids)
        for _ in range(len(self)):
            if num_groups >= self.group_sample_k:
                chosen_groups = self._rng.sample(self.sense_ids, k=self.group_sample_k)
            else:
                chosen_groups = self._rng.choices(self.sense_ids, k=self.group_sample_k)
            batch: List[int] = []
            for sense_id in chosen_groups:
                pool = self.sense_to_indices[sense_id]
                if len(pool) >= self.samples_per_group:
                    picks = self._rng.sample(pool, k=self.samples_per_group)
                else:
                    picks = self._rng.choices(pool, k=self.samples_per_group)
                batch.extend(picks)
            yield batch


class SimCSECilinDataset(Dataset):
    """
    Dataset that consumes the offline-preprocessed format from dataset_preprocess.py.
      entry: {
        "tokens": [...],
        "pos": {idx: [tokens_after_replace, ...]},
        "hard_neg": {idx: [tokens_after_replace, ...]}
      }
    For training, a positive and (optionally) hard negative are sampled per item.
    For validation, anchor/positive are identical copies of tokens.
    """

    def __init__(
        self,
        tokenizer,
        entries: Optional[List[Dict[str, Any]]] = None,
        data_path: str | Path | None = None,
        max_len: int = 64,
        mode: str = "train",
        seed: Optional[int] = None,
        min_len: int = 3,
    ):
        assert mode in ("train", "val")
        assert entries is not None or data_path is not None, "Provide entries or data_path"
        self.entries = entries if entries is not None else load_preprocessed_dataset(data_path, min_len=min_len, max_len=None)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.entries)

    def _encode_with_word_ids(self, tokens: List[str]):
        return self.tokenizer(
            tokens,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            is_split_into_words=True,
        )

    @staticmethod
    def _word_to_token_mask(
        word_ids: List[Optional[int]], word_idx: int
    ) -> torch.Tensor:
        mask = [wid == word_idx for wid in word_ids]
        return torch.tensor(mask, dtype=torch.bool)

    @staticmethod
    def _span_to_token_idx(offsets: torch.Tensor, char_start: int, char_end: int) -> int:
        indices = []
        for i, (s, e) in enumerate(offsets.tolist()):
            if s == e == 0:
                continue  # [CLS], [SEP], padding
            if not (e <= char_start or s >= char_end):
                indices.append(i)
        if not indices:
            return 0
        return indices[0]

    def _build_sample_from_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        tokens: List[str] = list(entry.get("tokens", []))
        pos_map: Dict[str, List[List[str]]] = entry.get("pos", {}) or {}
        hard_map: Dict[str, List[List[str]]] = entry.get("hard_neg", {}) or {}

        # positive sampling
        pos_indices = [int(k) for k, v in pos_map.items() if v]
        if pos_indices:
            if self.mode == "train":
                idx_choice = self._rng.choice(pos_indices)
                pos_candidates = pos_map.get(str(idx_choice), [])
                pos_tokens = list(self._rng.choice(pos_candidates))
            else:
                # Validation uses a deterministic positive when available.
                idx_choice = min(pos_indices)
                pos_candidates = pos_map.get(str(idx_choice), [])
                pos_tokens = list(pos_candidates[0]) if pos_candidates else list(tokens)

            word_idx = idx_choice
            orig_word = tokens[word_idx] if 0 <= word_idx < len(tokens) else None
            repl_word = pos_tokens[word_idx] if 0 <= word_idx < len(pos_tokens) else None
            has_retrofit = int(
                word_idx >= 0 and orig_word and repl_word and orig_word != repl_word
            )
        else:
            pos_tokens = list(tokens)
            word_idx = -1
            orig_word = repl_word = None
            has_retrofit = 0

        # hard negative sampling
        neg_tokens = None
        if self.mode == "train":
            neg_indices = [int(k) for k, v in hard_map.items() if v]
            if neg_indices:
                idx_choice = self._rng.choice(neg_indices)
                neg_candidates = hard_map.get(str(idx_choice), [])
                neg_tokens = list(self._rng.choice(neg_candidates))

        return {
            "s_tokens": tokens,
            "s_pos_tokens": pos_tokens,
            "s_neg_tokens": neg_tokens,
            "word_idx": word_idx,
            "orig_word": orig_word,
            "repl_word": repl_word,
            "has_retrofit": has_retrofit,
        }

    def _encode_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        s_tokens = sample["s_tokens"]
        s_pos_tokens = sample["s_pos_tokens"]
        s_neg_tokens = sample["s_neg_tokens"]
        word_idx = sample["word_idx"]
        orig_word = sample["orig_word"]
        repl_word = sample["repl_word"]
        has_retrofit = sample["has_retrofit"]
        has_neg = int(s_neg_tokens is not None)

        enc1 = self._encode_with_word_ids(s_tokens)
        enc2 = self._encode_with_word_ids(s_pos_tokens)

        input_ids1 = enc1["input_ids"].squeeze(0)
        attn_mask1 = enc1["attention_mask"].squeeze(0)

        input_ids2 = enc2["input_ids"].squeeze(0)
        attn_mask2 = enc2["attention_mask"].squeeze(0)

        word_ids1 = enc1.word_ids(batch_index=0)
        word_ids2 = enc2.word_ids(batch_index=0)

        if has_retrofit:
            token_mask1 = self._word_to_token_mask(word_ids1, word_idx)
            token_mask2 = self._word_to_token_mask(word_ids2, word_idx)
        else:
            token_mask1 = torch.zeros(self.max_len, dtype=torch.bool)
            token_mask2 = torch.zeros(self.max_len, dtype=torch.bool)

        data: Dict[str, torch.Tensor] = {
            "input_ids1": input_ids1,
            "attention_mask1": attn_mask1,
            "input_ids2": input_ids2,
            "attention_mask2": attn_mask2,
            "token_mask1": token_mask1,
            "token_mask2": token_mask2,
            "has_retrofit": torch.tensor(has_retrofit, dtype=torch.long),
            "has_neg": torch.tensor(has_neg, dtype=torch.long),
        }

        pad_id = self.tokenizer.pad_token_id or 0
        if self.mode == "train":
            if has_neg:
                enc3 = self.tokenizer(
                    s_neg_tokens,
                    max_length=self.max_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                    is_split_into_words=True,
                )
                data.update(
                    {
                        "input_ids_neg": enc3["input_ids"].squeeze(0),
                        "attention_mask_neg": enc3["attention_mask"].squeeze(0),
                    }
                )
            else:
                data.update(
                    {
                        "input_ids_neg": torch.full(
                            (self.max_len,), pad_id, dtype=torch.long
                        ),
                        "attention_mask_neg": torch.zeros(
                            self.max_len, dtype=torch.long
                        ),
                    }
                )
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[idx]
        sample = self._build_sample_from_entry(entry)
        encoded = self._encode_sample(sample)
        return encoded

    @staticmethod
    def tokens_to_text(tokens: List[str]) -> str:
        return "".join(tokens)
