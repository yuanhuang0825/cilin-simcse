# Cilin-SimCSE

Chinese contextual sense embedding training with Cilin synonym constraints.

This repository trains a SimCSE-style Chinese encoder with supervision derived from
`same.txt` / Cilin-style synonym groups. It supports synonym-based positive pairs,
related-word hard negatives, contextual retrofitting, word-level retrieval metrics,
and sense prototype construction for downstream word sense assignment.

## Features

- Train Chinese SimCSE embeddings with `hfl/chinese-roberta-wwm-ext` or another Hugging Face BERT encoder.
- Parse Cilin-style entries where `=` marks synonym groups and `#` marks related groups.
- Build positive samples by replacing words with context-compatible synonyms.
- Build hard negatives from related-word groups.
- Add contextual retrofitting loss on replaced token embeddings.
- Evaluate sentence retrieval, word-level retrieval, contextual synonym consistency, and synonym margin.
- Build sense-group prototypes from trained contextual word embeddings.

## Repository Layout

```text
.
├── config.yaml                 # Training configuration
├── same.txt                    # Cilin-style synonym resource
├── download_data.py            # Download/export NLI or Wikipedia text
├── build_dataset.py            # Match Cilin words and tokenize raw text
├── dataset_preprocess.py       # Build positive and hard-negative training entries
├── dataset.py                  # PyTorch dataset and sampler
├── model.py                    # SimCSE model and losses
├── metric.py                   # Validation metrics
├── train.py                    # Main training script
├── prototype.py                # Build Cilin sense prototypes
└── valid_prototype.py          # Validate prototype assignments
```

Large generated directories such as `raw_data/`, `data/`, `dataset/`, `checkpoints/`,
`prototypes/`, and `wandb/` are experiment artifacts. For a public GitHub release,
consider excluding them unless you intentionally publish small sample files.

## Installation

Python 3.10+ is recommended.

```bash
pip install torch transformers datasets tqdm pyyaml numpy jieba opencc-python-reimplemented
pip install onnxruntime huggingface_hub matplotlib wandb
```

If you want CKIP CWS acceleration on GPU, install the ONNX Runtime GPU package that
matches your CUDA environment:

```bash
pip install onnxruntime-gpu
```

## Data Format

### Cilin / `same.txt`

The Cilin file is expected to be whitespace-separated:

```text
Aa01A02= 人類 生人 全人類
Aa01B03# 良民 順民
Aa01A05@ 二人 三人 兩人 雙人
```

- `=`: synonym group, used for positive pairs and prototypes.
- `#`: related-word group, used as hard negatives.
- `@`: ignored by the current training pipeline.

### Tokenized Corpus

Tokenized corpus files should be JSON files containing `List[List[str]]`:

```json
[
  ["這", "是", "一個", "例子", "。"],
  ["模型", "會", "讀取", "已", "斷詞", "句子", "。"]
]
```

### Preprocessed Training Entries

`dataset_preprocess.py` converts tokenized sentences into entries like:

```json
[
  {
    "tokens": ["這", "是", "一個", "例子", "。"],
    "pos": {
      "3": [["這", "是", "一個", "範例", "。"]]
    },
    "hard_neg": {
      "3": [["這", "是", "一個", "案例", "。"]]
    }
  }
]
```

## Pipeline

### 1. Download Raw Text

Wikipedia:

```bash
python download_data.py \
  --dataset wiki_zh \
  --out-dir raw_data/wiki_zh \
  --to-trad \
  --workers 8
```

NLI-style sentence pairs:

```bash
python download_data.py \
  --dataset nli_zh \
  --config ALL \
  --split train \
  --out-dir raw_data/nli_zh
```

### 2. Build Tokenized Cilin-Matched Corpus

`build_dataset.py` currently uses these default paths:

- input: `raw_data/wiki_zh`
- output: `data/wiki_zh`
- Cilin file: `same.txt`

Run CKIP BERT CWS mode:

```bash
python build_dataset.py \
  --mode tokenizer \
  --workers 8 \
  --sample-limit 1000000
```

Faster alternatives:

```bash
python build_dataset.py --mode jieba --workers 8
python build_dataset.py --mode regex --workers 8
```

### 3. Preprocess Training Entries

```bash
python dataset_preprocess.py \
  --input data/wiki_zh \
  --cilin same.txt \
  --output dataset/wiki_zh \
  --model-name hfl/chinese-bert-wwm-ext \
  --device cuda \
  --filter-mode embed \
  --workers 4 \
  --max-len 64
```

Use `--filter-mode mlm` for MLM probability filtering. It is usually slower.

### 4. Configure Training

Edit `config.yaml`:

```yaml
paths:
  cilin_path: "same.txt"
  train_corpus_path: "dataset/wiki_zh"
  save_dir: "checkpoints/run_name"

model:
  encoder_name: "hfl/chinese-roberta-wwm-ext"
  pooling: "mean"
  temp: 0.05
  use_hard_neg: true
  use_retrofit: true

training:
  batch_size: 384
  num_epochs: 10
  max_len: 64
  lr: 3e-5
  device: "auto"
  use_amp: true
```

### 5. Train

```bash
python train.py
```

Checkpoints are written to `paths.save_dir`:

- `simcse_cilin_epoch*.pt`: per-epoch model weights.
- `simcse_cilin_latest.pt`: resumable checkpoint with optimizer, scheduler, scaler, and metadata.
- `simcse_cilin_best.pt`: best model by validation word-level retrieval MRR.
- `simcse_cilin_final.pt`: final model weights.

To resume training, set `training.resume_path` in `config.yaml` to a latest checkpoint.

## Build Sense Prototypes

After training, build Cilin `=` group prototypes:

```bash
python prototype.py \
  --cilin same.txt \
  --data data/wiki_zh \
  --ckpt checkpoints/run_name/simcse_cilin_best.pt \
  --out prototypes/cilin_simcse.pt \
  --device cuda \
  --max-len 64
```

The output is a `torch.save` dictionary:

```python
{
    "prototypes": {group_id: tensor},
    "meta": {group_id: {...}},
    "params": {...}
}
```

## Validate Prototypes

```bash
python valid_prototype.py \
  --input raw_data/wiki_zh \
  --model checkpoints/run_name/simcse_cilin_best.pt \
  --prototype prototypes/cilin_simcse.pt \
  --encoder hfl/chinese-roberta-wwm-ext \
  --cilin same.txt \
  --output valid_prototype_results.json \
  --max-sentences 10000 \
  --min-sim 0.5 \
  --cilin-only
```

The validation output maps predicted words to Cilin prototype groups and includes
summary metadata under `_meta`.

## Metrics

Training reports:

- `val/loss`: SimCSE validation loss.
- `val/retrieval_mrr` and `val/retrieval_r@k`: sentence-pair retrieval.
- `val/word_retrieval_mrr` and `val/word_retrieval_r@k`: word-level retrieval.
- `val/local_uniformity`: local embedding uniformity.
- `val/CSC`: contextual synonym consistency.
- `val/CSM_*`: synonym-vs-related-word margin metrics.

If `training.use_wandb: true`, these metrics are also logged to Weights & Biases.

## Citation and Acknowledgements

This project builds on the following models, methods, and datasets:

- HFL Chinese BERT/RoBERTa WWM models: `hfl/chinese-bert-wwm-ext` and
  `hfl/chinese-roberta-wwm-ext`.
- SimCSE contrastive learning.
- Wikimedia Wikipedia, if you use the `download_data.py --dataset wiki_zh` pipeline.
- `shibing624/nli_zh`, if you use the `download_data.py --dataset nli_zh` pipeline.
- Cilin / `same.txt`, from the archived HIT IR-Lab Tongyici Cilin Extended
  repository: https://github.com/One-sixth/HIT-IR-Lab-Tongyici-Cilin-Extended

If you use this repository in academic work, please cite the relevant upstream work:

```bibtex
@inproceedings{cui-etal-2020-revisiting,
  title = {Revisiting Pre-Trained Models for Chinese Natural Language Processing},
  author = {Cui, Yiming and Che, Wanxiang and Liu, Ting and Qin, Bing and Wang, Shijin and Hu, Guoping},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
  year = {2020},
  pages = {657--668},
  doi = {10.18653/v1/2020.findings-emnlp.58}
}

@inproceedings{gao-etal-2021-simcse,
  title = {SimCSE: Simple Contrastive Learning of Sentence Embeddings},
  author = {Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  year = {2021},
  pages = {6894--6910},
  doi = {10.18653/v1/2021.emnlp-main.552}
}
```

Please also follow the licenses and attribution requirements of the data you use.
The Hugging Face `wikimedia/wikipedia` dataset card lists Wikipedia text under
CC BY-SA 3.0 and GFDL. The `shibing624/nli_zh` dataset aggregates multiple Chinese
semantic matching datasets; check its dataset card and the original dataset sources
before redistribution. The `same.txt` file used by this project comes from the
archived HIT IR-Lab Tongyici Cilin Extended repository:
https://github.com/One-sixth/HIT-IR-Lab-Tongyici-Cilin-Extended. Verify its license
before publishing the file or checkpoints trained from it.
