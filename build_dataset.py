import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Iterable, Sequence, Set

from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = (BASE_DIR / "raw_data/wiki_zh").resolve()
TARGET_DIR = BASE_DIR / "data/wiki_zh"
SAME_PATH = BASE_DIR / "same.txt"

CWS_MODEL_REPO = os.getenv("CKIP_CWS_REPO_ID", "Xenova/bert-base-chinese-ws")
BATCH_SIZE = int(os.getenv("CKIP_CWS_BATCH_SIZE", "256"))
MAX_SEQ_LEN = int(os.getenv("CKIP_CWS_MAX_LEN", "128"))

MATCHER: re.Pattern[str] | None = None  # populated in regex workers
TOKENIZER = None  # populated in tokenizer workers (HF tokenizer)
ORT_SESSION = None  # populated in tokenizer workers (onnxruntime session)
ID2LABEL: dict[int, str] | None = None  # populated in tokenizer workers
SYN_SET: Set[str] | None = None  # populated in tokenizer/jieba workers
WORD2SENSES: dict[str, list[str]] | None = None  # synonym -> sense IDs
SENSE_IDS: list[str] | None = None  # ordered sense IDs


def load_synonyms_with_senses(path: Path) -> tuple[Set[str], dict[str, list[str]], list[str]]:
    synonyms: Set[str] = set()
    word2senses: dict[str, list[str]] = {}
    sense_ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip().lstrip("\ufeff")
            if not line:
                continue
            parts = line.split()
            head = parts[0]
            if not (head.endswith("=") or head.endswith("#")):
                continue
            sense_id = head[:-1]
            sense_ids.append(sense_id)
            for word in parts[1:]:
                synonyms.add(word)
                word2senses.setdefault(word, []).append(sense_id)
    return synonyms, word2senses, sense_ids


def load_sense_map_once(word2senses: dict[str, list[str]], sense_ids: list[str]) -> None:
    global WORD2SENSES, SENSE_IDS
    WORD2SENSES = word2senses
    SENSE_IDS = sense_ids


def load_tokenizer_once(synonyms: Set[str]) -> None:
    """Load CKIP BERT CWS (ONNX Runtime) once per process (thread-safe session)."""
    import onnxruntime as ort
    from transformers import AutoConfig, AutoTokenizer

    global TOKENIZER, ORT_SESSION, SYN_SET, ID2LABEL

    if TOKENIZER is not None and ORT_SESSION is not None and SYN_SET is not None:
        SYN_SET = synonyms  # refresh reference in case synonyms set changes
        return

    tokenizer = AutoTokenizer.from_pretrained(CWS_MODEL_REPO)
    config = AutoConfig.from_pretrained(CWS_MODEL_REPO)
    model_path = resolve_onnx_model_path()
    session = ort.InferenceSession(model_path, providers=select_ort_providers(ort))

    TOKENIZER = tokenizer
    ORT_SESSION = session
    SYN_SET = synonyms
    ID2LABEL = {int(k): v for k, v in config.id2label.items()}


def load_jieba_once(synonyms: Set[str]) -> None:
    """Initialize shared synonym set for jieba workers."""
    global SYN_SET
    SYN_SET = synonyms


def tokenize_with_cws_batch(sentences: Sequence[str]) -> list[list[str]]:
    """Run CKIP BERT CWS in batches via ONNX Runtime and decode word tokens."""
    assert TOKENIZER is not None and ORT_SESSION is not None

    output_tokens: list[list[str]] = []
    idx = 0
    current_bs = max(1, BATCH_SIZE)

    while idx < len(sentences):
        bs = min(current_bs, len(sentences) - idx)
        batch = list(sentences[idx : idx + bs])
        try:
            output_tokens.extend(_run_cws_forward(batch))
            idx += bs
            # If we had to shrink before, gently ramp back toward configured batch size.
            if current_bs < BATCH_SIZE and current_bs * 2 <= BATCH_SIZE:
                current_bs *= 2
        except Exception as exc:  # noqa: BLE001 - handle OOM-like failures
            message = str(exc)
            if bs > 1 and ("Allocate" in message or "allocate" in message or "memory" in message):
                current_bs = max(1, bs // 2)
                continue
            raise
    return output_tokens


def _run_cws_forward(batch: list[str]) -> list[list[str]]:
    encoded = TOKENIZER(
        batch,
        return_offsets_mapping=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    offsets_batch = encoded.pop("offset_mapping")

    ort_inputs: dict[str, Any] = {}
    for input_def in ORT_SESSION.get_inputs():
        name = input_def.name
        if name in encoded:
            ort_inputs[name] = encoded[name]
        elif name == "token_type_ids":
            ort_inputs[name] = encoded["input_ids"] * 0
        else:
            raise KeyError(f"Missing required input {name} for ONNX model.")

    logits = ORT_SESSION.run(None, ort_inputs)[0]
    pred_ids = logits.argmax(axis=-1)

    decoded: list[list[str]] = []
    for row_idx, sent in enumerate(batch):
        labels_row = pred_ids[row_idx].tolist()
        offsets_row = offsets_batch[row_idx].tolist()
        decoded.append(decode_tokens(sent, labels_row, offsets_row))
    return decoded


def decode_tokens(sentence: str, label_ids: Sequence[int], offsets: Sequence[Sequence[int]]) -> list[str]:
    tokens: list[str] = []
    current = ""
    for label_id, (start, end) in zip(label_ids, offsets):
        if start == 0 and end == 0:
            continue  # padding or special tokens
        piece = sentence[start:end]
        if not piece:
            continue
        label = ID2LABEL.get(int(label_id), "S") if ID2LABEL else "S"
        tag = label[:1].upper()
        if tag == "S":
            if current:
                tokens.append(current)
                current = ""
            tokens.append(piece)
        elif tag == "B":
            if current:
                tokens.append(current)
            current = piece
        elif tag in ("M", "I"):
            current = f"{current}{piece}" if current else piece
        elif tag == "E":
            current = f"{current}{piece}" if current else piece
            tokens.append(current)
            current = ""
        else:
            if current:
                tokens.append(current)
                current = ""
            tokens.append(piece)
    if current:
        tokens.append(current)
    return tokens


def resolve_onnx_model_path() -> str:
    """Return a local path to the CKIP CWS ONNX model, trying common filenames."""
    from huggingface_hub import hf_hub_download

    env_override = os.getenv("CKIP_CWS_ONNX_PATH")
    if env_override:
        override_path = Path(env_override).expanduser()
        if override_path.exists():
            return str(override_path)
        raise FileNotFoundError(f"CKIP_CWS_ONNX_PATH set but file not found: {override_path}")

    candidates = (
        "model.onnx",
        "onnx/model.onnx",
        "onnx/model_quantized.onnx",
    )
    last_error: Exception | None = None
    for filename in candidates:
        try:
            return hf_hub_download(repo_id=CWS_MODEL_REPO, filename=filename)
        except Exception as exc:  # noqa: BLE001 - propagate only after all attempts
            last_error = exc
    raise RuntimeError(
        f"Unable to download CKIP CWS ONNX model for {CWS_MODEL_REPO}; "
        f"tried {', '.join(candidates)}."
    ) from last_error


def select_ort_providers(ort_module: Any) -> list[str]:
    """Prefer CUDA if available; allow override via CKIP_CWS_PROVIDERS env (comma-separated)."""
    override = os.getenv("CKIP_CWS_PROVIDERS")
    if override:
        providers = [p.strip() for p in override.split(",") if p.strip()]
        if providers:
            return providers
    available = set(ort_module.get_available_providers())
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def process_file_regex(path_str: str) -> tuple[int, int, int, dict[str, int]]:
    assert MATCHER is not None and WORD2SENSES is not None
    source_path = Path(path_str)
    target_path = TARGET_DIR / f"{source_path.stem}.json"

    matches: list[str] = []
    sense_counts: dict[str, int] = {}
    total_sentences = 0

    with source_path.open(encoding="utf-8") as handle:
        for raw in handle:
            sentence = raw.strip()
            if not sentence:
                continue
            total_sentences += 1
            if MATCHER.search(sentence):
                matches.append(sentence)
                for match in MATCHER.finditer(sentence):
                    word = match.group(0)
                    for sense_id in WORD2SENSES.get(word, []):
                        sense_counts[sense_id] = sense_counts.get(sense_id, 0) + 1

    if matches:
        target_path.write_text("\n".join(matches) + "\n", encoding="utf-8")
        return total_sentences, len(matches), 1, sense_counts

    if target_path.exists():
        target_path.unlink()
    return total_sentences, 0, 0, sense_counts


def process_file_tokenizer(path_str: str) -> tuple[int, int, int, dict[str, int]]:
    assert TOKENIZER is not None and SYN_SET is not None and ORT_SESSION is not None
    assert WORD2SENSES is not None
    source_path = Path(path_str)
    target_path = TARGET_DIR / f"{source_path.stem}.json"

    matches_tokens: list[list[str]] = []
    sense_counts: dict[str, int] = {}
    total_sentences = 0

    sentences: list[str] = []
    with source_path.open(encoding="utf-8") as handle:
        for raw in handle:
            sentence = raw.strip()
            if not sentence:
                continue
            sentences.append(sentence)

    total_sentences = len(sentences)
    if sentences:
        tokenized = tokenize_with_cws_batch(sentences)
        for tokens in tokenized:
            if any(token in SYN_SET for token in tokens):
                matches_tokens.append(tokens)
                for token in tokens:
                    for sense_id in WORD2SENSES.get(token, []):
                        sense_counts[sense_id] = sense_counts.get(sense_id, 0) + 1

    if matches_tokens:
        target_path.write_text(
            json.dumps(matches_tokens, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return total_sentences, len(matches_tokens), 1, sense_counts

    if target_path.exists():
        target_path.unlink()
    return total_sentences, 0, 0, sense_counts


def process_file_jieba(path_str: str) -> tuple[int, int, int, dict[str, int]]:
    assert SYN_SET is not None and WORD2SENSES is not None
    import jieba

    source_path = Path(path_str)
    target_path = TARGET_DIR / f"{source_path.stem}.json"

    matches_tokens: list[list[str]] = []
    sense_counts: dict[str, int] = {}
    total_sentences = 0

    with source_path.open(encoding="utf-8") as handle:
        for raw in handle:
            sentence = raw.strip()
            if not sentence:
                continue
            total_sentences += 1
            tokens = list(jieba.cut(sentence, cut_all=False))
            if any(token in SYN_SET for token in tokens):
                matches_tokens.append(tokens)
                for token in tokens:
                    for sense_id in WORD2SENSES.get(token, []):
                        sense_counts[sense_id] = sense_counts.get(sense_id, 0) + 1

    if matches_tokens:
        target_path.write_text(
            json.dumps(matches_tokens, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return total_sentences, len(matches_tokens), 1, sense_counts

    if target_path.exists():
        target_path.unlink()
    return total_sentences, 0, 0, sense_counts


def plot_rare_senses(
    sense_counts: dict[str, int],
    sense_ids: Sequence[str],
    output_path: Path,
    ratio: float = 0.1,
) -> None:
    if not sense_ids:
        print("No sense IDs found; skip histogram.")
        return
    if ratio <= 0:
        return

    counts_by_id = {sense_id: sense_counts.get(sense_id, 0) for sense_id in sense_ids}
    rare_count = max(1, int(len(sense_ids) * ratio))
    rare_items = sorted(counts_by_id.items(), key=lambda item: (item[1], item[0]))[:rare_count]
    if not rare_items:
        print("No sense counts to plot; skip histogram.")
        return

    ids = [sense_id for sense_id, _ in rare_items]
    counts = [count for _, count in rare_items]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - surface missing matplotlib cleanly
        raise RuntimeError("matplotlib is required for plotting rare senses.") from exc

    width = max(8, min(24, len(ids) * 0.4))
    fig, ax = plt.subplots(figsize=(width, 4.5))
    ax.bar(range(len(ids)), counts, color="#4C78A8")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=90, fontsize=8)
    ax.set_ylabel("Count")
    ax.set_xlabel("Sense ID")
    ax.set_title("Least-used 10% senses")
    fig.tight_layout()
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved rare-sense histogram to {output_path}")


def filter_sentences(
    workers: int | None = None,
    chunksize: int = 200,
    mode: str = "tokenizer",
    progress: bool = True,
    count_files: bool = False,
    sample_limit: int | None = None,
    seed: int | None = None,
    plot_rare_senses_after: bool = True,
    plot_path: str | Path = "./rare_senses_hist.png",
) -> None:
    def format_eta(seconds: float | None) -> str:
        if seconds is None or seconds < 0:
            return "unknown"
        seconds_int = int(seconds)
        hours, rem = divmod(seconds_int, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    synonyms, word2senses, sense_ids = load_synonyms_with_senses(SAME_PATH)
    if not synonyms:
        raise RuntimeError("No synonyms found in same.txt.")
    load_sense_map_once(word2senses, sense_ids)

    TARGET_DIR.mkdir(exist_ok=True)
    max_workers = workers or max(1, min(cpu_count(), 32))

    total_sentences = 0
    kept_sentences = 0
    written_files = 0
    sense_counts: dict[str, int] = {}

    files_list = list(SOURCE_DIR.glob("*.txt"))
    if seed is not None:
        random.seed(seed)
    random.shuffle(files_list)

    # Optional pre-count for accurate ETA; can be slow on huge dirs, so behind a flag.
    total_files = len(files_list) if (progress and count_files) else None

    if mode == "regex":
        pattern_text = "|".join(re.escape(word) for word in sorted(synonyms, key=len, reverse=True))
        global MATCHER
        MATCHER = re.compile(pattern_text)
        worker_fn = process_file_regex
        executor_class = ThreadPoolExecutor
        executor_kwargs = dict(max_workers=max_workers)
    elif mode == "tokenizer":
        load_tokenizer_once(synonyms)
        worker_fn = process_file_tokenizer
        executor_class = ThreadPoolExecutor
        executor_kwargs = dict(max_workers=max_workers)
    elif mode == "jieba":
        load_jieba_once(synonyms)
        worker_fn = process_file_jieba
        executor_class = ThreadPoolExecutor
        executor_kwargs = dict(max_workers=max_workers)
    else:
        raise ValueError(f"Unsupported mode {mode}, choose 'regex', 'tokenizer', or 'jieba'.")

    with executor_class(**executor_kwargs) as executor:
        pending = set()
        file_iter = iter(files_list)

        def submit_next(n: int = 1) -> None:
            for _ in range(n):
                try:
                    next_path = next(file_iter)
                except StopIteration:
                    return
                pending.add(executor.submit(worker_fn, str(next_path)))

        submit_next(max_workers * 2)

        bar = tqdm(total=total_files, desc="Files", unit="file") if progress else None
        stop_early = False

        while pending:
            for future in as_completed(pending):
                tot, kept, wrote, sense_delta = future.result()
                total_sentences += tot
                kept_sentences += kept
                written_files += wrote
                for sense_id, count in sense_delta.items():
                    sense_counts[sense_id] = sense_counts.get(sense_id, 0) + count
                if bar is not None:
                    bar.update(1)
                    eta_seconds = bar.format_dict.get("remaining")
                    eta_text = format_eta(eta_seconds)
                    if sample_limit:
                        remaining = max(sample_limit - kept_sentences, 0)
                        bar.set_postfix_str(f"remain={remaining} eta={eta_text}")
                    else:
                        bar.set_postfix_str(f"eta={eta_text}")
                pending.remove(future)
                submit_next()
                if sample_limit and kept_sentences >= sample_limit:
                    stop_early = True
                    break
            if stop_early:
                for fut in pending:
                    fut.cancel()
                break

        if bar is not None:
            bar.close()

    print(
        f"Processed {total_sentences} sentences; "
        f"kept {kept_sentences} sentences across {written_files} files "
        f"using {max_workers} workers in {mode} mode."
    )
    if plot_rare_senses_after and SENSE_IDS is not None:
        plot_rare_senses(sense_counts, SENSE_IDS, Path(plot_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dataset by matching synonyms.")
    parser.add_argument(
        "--mode",
        choices=("tokenizer", "jieba", "regex"),
        default="tokenizer",
        help="tokenizer=CKIP BERT CWS (ONNX Runtime), jieba=jieba cut, regex=faster substring match.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes.")
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200,
        help="Batch size per worker map call (tune for throughput).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar (useful in logs or when tqdm is noisy).",
    )
    parser.add_argument(
        "--count-files",
        action="store_true",
        help="Pre-count files for accurate progress total (slower on huge dirs).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=1_000_000,
        help="Stop after collecting this many matched sentences (0 = process all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for shuffling file order before sampling (default: random).",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=str(TARGET_DIR / "rare_senses_hist.png"),
        help="Path for rare-sense histogram image.",
    )
    parser.add_argument(
        "--no-plot-rare-senses",
        action="store_true",
        help="Disable histogram for least-used 10% senses.",
    )
    args = parser.parse_args()
    limit = None if args.sample_limit and args.sample_limit <= 0 else args.sample_limit
    filter_sentences(
        workers=args.workers,
        chunksize=args.chunksize,
        mode=args.mode,
        progress=not args.no_progress,
        count_files=args.count_files,
        sample_limit=limit,
        seed=args.seed,
        plot_rare_senses_after=not args.no_plot_rare_senses,
        plot_path=args.plot_path,
    )
