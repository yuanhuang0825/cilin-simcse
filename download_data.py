import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import jieba
from datasets import load_dataset
from opencc import OpenCC
from tqdm import tqdm


HARD_SENT_PUNCT = "。！？；"
SOFT_PUNCT = "，、"
SAFE_CONNECTORS = [
    "因此", "所以", "然而", "但", "不過", "此外", "同時", "另外", "於是",
    "因為", "由於", "如果", "雖然", "當", "為了",
    "也就是說", "即", "例如", "比如", "特別是",
]


def rough_sentence_split(text: str) -> List[str]:
    """
    Conservative sentence splitter: primarily split by strong punctuation,
    optionally keep ':' attached to the prefix (useful for Wikipedia definitions).
    """
    parts = re.split(rf"([{re.escape(HARD_SENT_PUNCT)}])", text)
    sents: List[str] = []
    buf = ""
    for p in parts:
        if not p:
            continue
        buf += p
        if p in HARD_SENT_PUNCT:
            s = buf.strip()
            if s:
                sents.append(s)
            buf = ""
    if buf.strip():
        sents.append(buf.strip())

    # Optional ':' handling for very long definition-like sentences
    out: List[str] = []
    for s in sents:
        if "：" in s and token_len(s) > 80:
            head, rest = s.split("：", 1)
            head = head.strip()
            rest = rest.strip()
            if head:
                out.append(head + "：")
            if rest:
                out.append(rest)
        else:
            out.append(s)
    return out


def token_len(s: str) -> int:
    return len(list(jieba.cut(s, cut_all=False)))


def find_protected_span(tokens: List[str], target: str, window: int = 12) -> Optional[Tuple[int, int]]:
    """
    Return a [L, R] token index span to protect around the first occurrence of target token.
    If not found, returns None.
    """
    for i, t in enumerate(tokens):
        if t == target:
            L = max(0, i - window)
            R = min(len(tokens) - 1, i + window)
            return (L, R)
    return None


def pick_split_index(tokens: List[str], max_tokens: int, protected: Optional[Tuple[int, int]] = None) -> Optional[int]:
    """
    Pick a split index 'cut' (1..len(tokens)-1), prefer semantic-safe boundaries and closeness to max_tokens.
    The cut index is the number of tokens to keep in the left chunk.
    """
    candidates: List[int] = []

    # 1) high-priority: before connectors
    for i in range(1, len(tokens)):
        if tokens[i] in SAFE_CONNECTORS:
            candidates.append(i)

    # 2) medium: after soft punctuation
    for i in range(1, len(tokens)):
        if tokens[i - 1] in SOFT_PUNCT:
            candidates.append(i)

    # 3) fallback: any position
    if not candidates:
        candidates = list(range(1, len(tokens)))

    # filter protected window
    if protected is not None:
        L, R = protected
        candidates = [c for c in candidates if not (L <= c <= R)]
        if not candidates:
            # if everything is inside protected window, we must allow splits;
            # return None to trigger caller fallback (usually "do not split" or soft split)
            return None

    left = [c for c in candidates if c <= max_tokens]
    if left:
        return max(left)

    right = [c for c in candidates if c > max_tokens]
    return min(right) if right else None


def semantic_length_split(
    sentence: str,
    max_tokens: int = 60,
    target_word: Optional[str] = None,
    protect_window: int = 12,
    min_tokens: int = 6,
) -> List[str]:
    """
    Split a sentence into chunks under max_tokens without breaking semantics too much.
    If target_word is provided, avoid splitting inside its context window when possible.
    """
    tokens = list(jieba.cut(sentence, cut_all=False))
    if len(tokens) <= max_tokens:
        return [sentence.strip()]

    protected = find_protected_span(tokens, target_word, window=protect_window) if target_word else None

    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        remaining = len(tokens) - start
        if remaining <= max_tokens:
            tail = "".join(tokens[start:]).strip()
            if tail:
                chunks.append(tail)
            break

        sub_tokens = tokens[start:]
        sub_prot: Optional[Tuple[int, int]] = None
        if protected is not None:
            L, R = protected
            subL, subR = L - start, R - start
            if subR >= 0 and subL < len(sub_tokens):
                sub_prot = (max(0, subL), min(len(sub_tokens) - 1, subR))

        cut = pick_split_index(sub_tokens, max_tokens, protected=sub_prot)

        # last resort: avoid infinite loops, but still keep chunks meaningful
        if cut is None or cut <= 0:
            cut = max_tokens

        chunk = "".join(sub_tokens[:cut]).strip()
        if chunk and token_len(chunk) >= min_tokens:
            chunks.append(chunk)

        start += cut

    return chunks


def split_text_for_sense_training(
    text: str,
    max_tokens: int = 60,
    target_word: Optional[str] = None,
    protect_window: int = 12,
    min_tokens: int = 6,
) -> List[str]:
    """
    Full pipeline: rough sentence split -> semantic length split for long sentences.
    Returns list of sentence/chunk strings.
    """
    sents = rough_sentence_split(text)
    out: List[str] = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        out.extend(
            semantic_length_split(
                s, max_tokens=max_tokens, target_word=target_word, protect_window=protect_window, min_tokens=min_tokens
            )
        )
    return out


def sanitize_filename(name: str, max_len: int = 120) -> str:
    # windows-safe & filesystem-safe
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        name = "untitled"
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name


@lru_cache(maxsize=1)
def _get_cc(to_trad: bool):
    return OpenCC("s2twp") if to_trad else None


def process_wiki_row(
    row: dict,
    out_dir: str,
    max_tokens: int,
    target_word: Optional[str],
    protect_window: int,
    min_tokens: int,
    to_trad: bool,
) -> Tuple[int, int]:
    title = row.get("title") or "untitled"
    text = row.get("text") or ""
    doc_id = row.get("id") or ""

    cc = _get_cc(to_trad)
    if cc is not None:
        title = cc.convert(title)
        text = cc.convert(text)

    lines = split_text_for_sense_training(
        text,
        max_tokens=max_tokens,
        target_word=target_word,
        protect_window=protect_window,
        min_tokens=min_tokens,
    )

    if not lines:
        return (0, 1)

    base = sanitize_filename(str(doc_id)) if doc_id else sanitize_filename(title)
    out_path = Path(out_dir) / f"{base}.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return (1, 0)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download datasets and export to txt files.\n"
            "- dataset=nli_zh: original behavior (save sentence1/sentence2 pairs per file)\n"
            "- dataset=wiki_zh: Wikipedia article -> sentence/chunk per line (jieba + semantic-safe splitting)\n"
        )
    )

    # one arg to switch dataset
    parser.add_argument(
        "--dataset",
        default="nli_zh",
        choices=["nli_zh", "wiki_zh"],
        help="Select which dataset pipeline to run.",
    )

    # ---- nli_zh args (kept for backward compatibility) ----
    parser.add_argument(
        "--config",
        default="ALL",
        choices=["ATEC", "BQ", "LCQMC", "PAWSX", "STS-B", "ALL"],
        help="Subset config of shibing624/nli_zh, or ALL to download every subset. (nli_zh only)",
    )
    parser.add_argument("--split", default="train", help="Dataset split to download (train/validation/test).")

    # ---- common output arg ----
    parser.add_argument("--out-dir", type=str, default="raw_data", help="Output directory for txt files.")

    # ---- Wikipedia args ----
    parser.add_argument(
        "--wiki-config",
        default="20231101.zh",
        help='HF config for wikimedia/wikipedia, e.g. "20231101.zh". (wiki_zh only)',
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=60,
        help="Max jieba token length per output line. (wiki_zh only)",
    )
    parser.add_argument(
        "--target-word",
        type=str,
        default=None,
        help="Protect this word's context window when splitting (exact match on jieba token). (wiki_zh only)",
    )
    parser.add_argument(
        "--protect-window",
        type=int,
        default=12,
        help="Token window size (+/-) around target word to avoid splitting within. (wiki_zh only)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=6,
        help="Drop chunks shorter than this many jieba tokens. (wiki_zh only)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for wiki_zh. Use <=1 for sequential.",
    )
    parser.add_argument(
        "--to-trad",
        action="store_true",
        help="Convert Simplified->Traditional (s2twp) before saving. Applies to both pipelines.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cc = OpenCC("s2twp") if args.to_trad else None

    if args.dataset == "nli_zh":
        # Keep original behavior as much as possible
        nli_out_dir = out_dir
        nli_out_dir.mkdir(parents=True, exist_ok=True)

        configs = ["ATEC", "BQ", "LCQMC", "PAWSX", "STS-B"] if args.config == "ALL" else [args.config]
        total_pairs = 0

        for cfg in configs:
            print(f"Loading dataset shibing624/nli_zh ({cfg}, {args.split}) ...")
            ds = load_dataset("shibing624/nli_zh", cfg, split=args.split)
            for i, row in enumerate(tqdm(ds, desc=f"Converting {cfg}", unit="pair")):
                s1 = row["sentence1"]
                s2 = row["sentence2"]
                if cc is not None:
                    s1 = cc.convert(s1)
                    s2 = cc.convert(s2)
                out_path = nli_out_dir / f"{cfg}_{i}.txt"
                out_path.write_text(f"{s1}\n{s2}\n", encoding="utf-8")
            total_pairs += len(ds)

        print(f"Saved {total_pairs} pairs to {nli_out_dir}")
        return

    # wiki_zh pipeline: one article per file, newline-separated sentences/chunks
    wiki_out_dir = out_dir
    wiki_out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading dataset wikimedia/wikipedia ({args.wiki_config}, {args.split}) ...')
    ds = load_dataset("wikimedia/wikipedia", args.wiki_config, split=args.split)

    # Expected columns: id, url, title, text (per HF dataset card/metadata)
    workers = args.workers if args.workers is not None else max(1, os.cpu_count() or 1)
    total_articles = 0
    skipped_empty = 0

    if workers <= 1:
        for row in tqdm(ds, desc="Exporting wiki articles", unit="article"):
            written, skipped = process_wiki_row(
                row,
                str(wiki_out_dir),
                args.max_tokens,
                args.target_word,
                args.protect_window,
                args.min_tokens,
                args.to_trad,
            )
            total_articles += written
            skipped_empty += skipped
    else:
        total = len(ds)
        pending = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            with tqdm(total=total, desc="Exporting wiki articles", unit="article") as pbar:
                for row in ds:
                    pending.append(
                        executor.submit(
                            process_wiki_row,
                            row,
                            str(wiki_out_dir),
                            args.max_tokens,
                            args.target_word,
                            args.protect_window,
                            args.min_tokens,
                            args.to_trad,
                        )
                    )
                    if len(pending) >= workers * 4:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for fut in done:
                            written, skipped = fut.result()
                            total_articles += written
                            skipped_empty += skipped
                            pbar.update(1)
                        pending = list(pending)

                for fut in as_completed(pending):
                    written, skipped = fut.result()
                    total_articles += written
                    skipped_empty += skipped
                    pbar.update(1)

    print(f"Saved {total_articles} wiki articles to {wiki_out_dir} (skipped empty: {skipped_empty})")


if __name__ == "__main__":
    main()
