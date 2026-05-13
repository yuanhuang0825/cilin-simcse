# train.py
import os
import random
import yaml
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from model import SimCSEConfig, SimCSEModel
from dataset import (
    load_preprocessed_dataset,
    SimCSECilinDataset,
    load_word2senses,
    TwoStageSenseBatchSampler,
)
from dataset_preprocess import parse_cilin
from metric import (
    contextual_synonym_consistency,
    contextual_synonym_margin,
    contextual_synonym_consistency_from_entries,
    contextual_synonym_margin_from_entries,
    local_uniformity,
    paired_retrieval_metrics,
    word_level_retrieval_metrics_from_entries,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_cfg: str):
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def main():
    # ===== 讀 config.yaml =====
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)

    paths = cfg_yaml["paths"]
    mcfg = cfg_yaml["model"]
    tcfg = cfg_yaml["training"]

    device = get_device(tcfg.get("device", "auto"))
    set_seed(tcfg.get("seed", 42))
    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    use_amp = bool(tcfg.get("use_amp", False)) and is_cuda
    if bool(tcfg.get("use_amp", False)) and not is_cuda:
        print("AMP requested but CUDA is unavailable; falling back to FP32.")

    os.makedirs(paths["save_dir"], exist_ok=True)

    use_wandb = bool(tcfg.get("use_wandb", False))
    log_interval = int(tcfg.get("log_interval", 50))
    wandb_run_id = tcfg.get("wandb_id")
    wandb_resume = bool(tcfg.get("wandb_resume", True))
    if use_wandb:
        try:
            import wandb
        except ImportError:
            print("wandb not installed; disabling wandb logging.")
            use_wandb = False
    # ===== 1. Cilin =====
    print("Parsing Cilin...")
    syn_dict, related_dict = parse_cilin(paths["cilin_path"])
    print(f"Synonym entries: {len(syn_dict)}, related entries: {len(related_dict)}")

    # ===== 2. Preprocessed dataset & 8:2 split =====

    entries = load_preprocessed_dataset(
        paths["train_corpus_path"],
        min_len=tcfg.get("min_len", 3),
        max_len=tcfg.get("max_len", None),
    )
    print(f"Loaded {len(entries)} entries.")
    if not entries:
        raise ValueError(
            f"No preprocessed entries found under {paths['train_corpus_path']}. "
            "Ensure dataset_preprocess.py output path is correct or point train_corpus_path to the directory containing JSON files."
        )

    def train_valid_split_entries(data, valid_ratio=0.2, seed=42):
        rng = random.Random(seed)
        idx = list(range(len(data)))
        rng.shuffle(idx)
        split = int(len(data) * (1 - valid_ratio))
        train_idx = idx[:split]
        valid_idx = idx[split:]
        train_entries = [data[i] for i in train_idx]
        valid_entries = [data[i] for i in valid_idx]
        return train_entries, valid_entries

    train_entries, valid_entries = train_valid_split_entries(
        entries, valid_ratio=tcfg.get("val_ratio", 0.2), seed=tcfg.get("seed", 42)
    )
    print(f"Train entries: {len(train_entries)}, Valid entries: {len(valid_entries)}")

    # ===== 3. Tokenizer / Augmenter / Dataset / DataLoader =====
    tokenizer = AutoTokenizer.from_pretrained(mcfg["encoder_name"])

    train_dataset = SimCSECilinDataset(
        tokenizer=tokenizer,
        entries=train_entries,
        max_len=tcfg["max_len"],
        mode="train",
        seed=tcfg.get("seed", 42),
    )

    valid_dataset = SimCSECilinDataset(
        tokenizer=tokenizer,
        entries=valid_entries,
        max_len=tcfg["max_len"],
        mode="val",
    )

    use_two_stage = bool(tcfg.get("two_stage_sampling", False))
    if use_two_stage:
        word2senses = load_word2senses(paths["cilin_path"])
        group_k = int(tcfg.get("group_sample_k", tcfg["batch_size"]))
        sampler = TwoStageSenseBatchSampler(
            train_entries,
            word2senses,
            batch_size=tcfg["batch_size"],
            group_sample_k=group_k,
            seed=tcfg.get("seed", 42),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=tcfg["num_workers"],
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=tcfg["batch_size"],
            shuffle=True,
            num_workers=tcfg["num_workers"],
            drop_last=True,
        )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=tcfg["num_workers"],
        drop_last=False,
    )

    # ===== 4. Model / Optim / Scheduler =====
    simcse_cfg = SimCSEConfig(
        encoder_name=mcfg["encoder_name"],
        pooling=mcfg["pooling"],
        temp=mcfg["temp"],
        distance_metric=mcfg.get("distance_metric", "euclidean"),
        use_hard_neg=mcfg["use_hard_neg"],
        hard_neg_weight=mcfg["hard_neg_weight"],
        margin=mcfg["margin"],
        use_retrofit=mcfg["use_retrofit"],
        retrofit_weight=mcfg["retrofit_weight"],
    )

    model = SimCSEModel(simcse_cfg).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    lr = float(tcfg.get("lr", 3e-5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * tcfg["num_epochs"]
    warmup_steps = int(total_steps * tcfg["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ===== 5. Training Loop =====
    global_step = 0
    best_val_word_mrr = float("-inf")
    best_ckpt_path = None
    start_epoch = 0
    resume_path = tcfg.get("resume_path")
    if resume_path:
        if os.path.isfile(resume_path):
            ckpt = torch.load(resume_path, map_location=device)
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
                if "optimizer_state" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                if "scheduler_state" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler_state"])
                if "scaler_state" in ckpt:
                    scaler.load_state_dict(ckpt["scaler_state"])
                if "epoch" in ckpt:
                    start_epoch = int(ckpt["epoch"]) + 1
                if "global_step" in ckpt:
                    global_step = int(ckpt["global_step"])
                if "best_val_word_mrr" in ckpt:
                    best_val_word_mrr = float(ckpt["best_val_word_mrr"])
                elif "best_val_mrr" in ckpt:
                    # Backward compatibility with previous runs where best metric was sentence-level MRR.
                    best_val_word_mrr = float(ckpt["best_val_mrr"])
                elif "best_val_loss" in ckpt:
                    # Backward compatibility with older checkpoints (previously selected by loss).
                    best_val_word_mrr = float("-inf")
                if "best_ckpt_path" in ckpt:
                    best_ckpt_path = ckpt["best_ckpt_path"]
                if "wandb_run_id" in ckpt and not wandb_run_id:
                    wandb_run_id = ckpt["wandb_run_id"]
                print(f"Resumed training from {resume_path} at epoch {start_epoch}.")
            else:
                model.load_state_dict(ckpt)
                print(
                    f"Loaded model weights from {resume_path} (optimizer/scheduler not resumed)."
                )
        else:
            raise FileNotFoundError(f"resume_path not found: {resume_path}")

    if use_wandb:
        run_name = tcfg.get("run_name") or f"simcse-{mcfg['encoder_name']}"
        wandb_kwargs = {
            "project": tcfg.get("wandb_project", "simcse-cilin"),
            "name": run_name,
            "config": {"paths": paths, "model": mcfg, "training": tcfg},
        }
        if wandb_run_id:
            wandb_kwargs.update(
                {
                    "id": wandb_run_id,
                    "resume": "allow" if wandb_resume else "never",
                }
            )
        wandb.init(**wandb_kwargs)
        wandb_run_id = wandb.run.id
        wandb.watch(model, log="all")
        run_step = getattr(wandb.run, "step", None)
        if run_step is not None and run_step > global_step:
            global_step = run_step
            print(f"Synced global_step to wandb step: {global_step}")

    for epoch in range(start_epoch, tcfg["num_epochs"]):
        # ---------- Train ----------
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{tcfg['num_epochs']} [train]")
        for batch in pbar:
            input_ids1 = batch["input_ids1"].to(device)
            attn_mask1 = batch["attention_mask1"].to(device)
            input_ids2 = batch["input_ids2"].to(device)
            attn_mask2 = batch["attention_mask2"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                # 句子 embedding + token-level hidden
                z1, h1 = model.encode(input_ids1, attn_mask1, output_hidden=True)
                z2, h2 = model.encode(input_ids2, attn_mask2, output_hidden=True)

                loss = model.simcse_loss(z1, z2)

                # Hard Negative
                if simcse_cfg.use_hard_neg:
                    has_neg = batch["has_neg"].to(device).bool()
                    if has_neg.any():
                        input_ids_neg = batch["input_ids_neg"].to(device)[has_neg]
                        attn_mask_neg = batch["attention_mask_neg"].to(device)[has_neg]
                        z1_neg = z1[has_neg]
                        z2_neg = z2[has_neg]
                        z_neg = model.encode(input_ids_neg, attn_mask_neg, output_hidden=False)
                        hard_loss = model.hard_neg_loss(z1_neg, z2_neg, z_neg)
                        loss = loss + simcse_cfg.hard_neg_weight * hard_loss

                # Contextual Retrofitting（只在有替換的 sample 上）
                if simcse_cfg.use_retrofit:
                    has_retrofit = batch["has_retrofit"].to(device).bool()
                    if has_retrofit.any():
                        idx = has_retrofit.nonzero(as_tuple=True)[0]
                        token_mask1_all = batch["token_mask1"].to(device)
                        token_mask2_all = batch["token_mask2"].to(device)
                        token_mask1 = token_mask1_all[idx].float()
                        token_mask2 = token_mask2_all[idx].float()

                        h1_sel = h1[idx]
                        h2_sel = h2[idx]
                        denom1 = token_mask1.sum(dim=1, keepdim=True).clamp_min(1.0)
                        denom2 = token_mask2.sum(dim=1, keepdim=True).clamp_min(1.0)
                        syn_emb1 = (h1_sel * token_mask1.unsqueeze(-1)).sum(dim=1) / denom1
                        syn_emb2 = (h2_sel * token_mask2.unsqueeze(-1)).sum(dim=1) / denom2

                        retro_loss = model.retrofit_loss(syn_emb1, syn_emb2)
                        loss = loss + simcse_cfg.retrofit_weight * retro_loss

            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()

            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            if use_wandb and (global_step % log_interval == 0):
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "epoch": epoch + 1,
                    },
                    step=global_step,
                )

        # ---------- Validation (SimCSE loss on val set) ----------
        model.eval()
        val_losses = []
        lu_k = int(tcfg.get("local_uniformity_k", 5))
        lu_max_samples = int(tcfg.get("local_uniformity_max_samples", 2048))
        retrieval_topk = tuple(int(k) for k in tcfg.get("retrieval_topk", [1, 5]))
        word_retrieval_topk = tuple(int(k) for k in tcfg.get("word_retrieval_topk", list(retrieval_topk)))
        val_embs = None
        val_embs2 = None
        val_seen = 0
        rng = random.Random(tcfg.get("seed", 42) + epoch + 1)
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{tcfg['num_epochs']} [val]"):
                input_ids1 = batch["input_ids1"].to(device)
                attn_mask1 = batch["attention_mask1"].to(device)
                input_ids2 = batch["input_ids2"].to(device)
                attn_mask2 = batch["attention_mask2"].to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    # val 模式下，input1/input2 是同一句（不同 dropout），只看 InfoNCE
                    z1 = model.encode(input_ids1, attn_mask1, output_hidden=False)
                    z2 = model.encode(input_ids2, attn_mask2, output_hidden=False)
                    val_loss = model.simcse_loss(z1, z2)
                val_losses.append(val_loss.item())
                if lu_max_samples > 0:
                    z_cpu = z1.detach().cpu()
                    z2_cpu = z2.detach().cpu()
                    bsz = z_cpu.size(0)
                    if val_embs is None:
                        val_embs = torch.empty(
                            (lu_max_samples, z_cpu.size(1)), dtype=z_cpu.dtype
                        )
                        val_embs2 = torch.empty(
                            (lu_max_samples, z2_cpu.size(1)), dtype=z2_cpu.dtype
                        )
                    for i in range(bsz):
                        val_seen += 1
                        if val_seen <= lu_max_samples:
                            val_embs[val_seen - 1].copy_(z_cpu[i])
                            val_embs2[val_seen - 1].copy_(z2_cpu[i])
                        else:
                            j = rng.randrange(val_seen)
                            if j < lu_max_samples:
                                val_embs[j].copy_(z_cpu[i])
                                val_embs2[j].copy_(z2_cpu[i])

        mean_val_loss = float(sum(val_losses) / len(val_losses))
        if val_embs is not None:
            used = min(val_seen, lu_max_samples)
            all_val_embs = val_embs[:used]
            all_val_embs2 = val_embs2[:used]
        else:
            all_val_embs = None
            all_val_embs2 = None
        lu_score = (
            local_uniformity(all_val_embs, k=lu_k, max_samples=lu_max_samples)
            if all_val_embs is not None
            else 0.0
        )
        retrieval = (
            paired_retrieval_metrics(all_val_embs, all_val_embs2, topk=retrieval_topk)
            if (all_val_embs is not None and all_val_embs2 is not None)
            else {"mrr": 0.0, "n": 0.0, **{f"r@{k}": 0.0 for k in retrieval_topk}}
        )
        print(f"[Epoch {epoch+1}] Validation SimCSE loss: {mean_val_loss:.4f}")
        print(f"[Epoch {epoch+1}] Local Uniformity@{lu_k}: {lu_score:.4f}")
        topk_msg = ", ".join(
            [f"R@{k}={retrieval.get(f'r@{k}', 0.0):.4f}" for k in retrieval_topk]
        )
        print(
            f"[Epoch {epoch+1}] Val Retrieval ({int(retrieval.get('n', 0))} pairs): "
            f"{topk_msg}, MRR={retrieval.get('mrr', 0.0):.4f}"
        )
        word_retrieval = word_level_retrieval_metrics_from_entries(
            model=model,
            tokenizer=tokenizer,
            entries=valid_entries,
            device=device,
            num_pairs=int(tcfg.get("val_word_num_pairs", tcfg.get("val_num_pairs", 200))),
            max_len=tcfg["max_len"],
            topk=word_retrieval_topk,
        )
        word_topk_msg = ", ".join(
            [f"R@{k}={word_retrieval.get(f'r@{k}', 0.0):.4f}" for k in word_retrieval_topk]
        )
        print(
            f"[Epoch {epoch+1}] Val Word Retrieval ({int(word_retrieval.get('n', 0))} pairs): "
            f"{word_topk_msg}, MRR={word_retrieval.get('mrr', 0.0):.4f}"
        )

        # ---------- Contextual Cilin-based metrics ----------
        print("Running contextual synonym metrics...")
        val_num_pairs = int(tcfg.get("val_num_pairs", 200))
        csc = contextual_synonym_consistency_from_entries(
            model=model,
            tokenizer=tokenizer,
            entries=valid_entries,
            device=device,
            num_pairs=val_num_pairs,
            max_len=tcfg["max_len"],
        )
        csm = contextual_synonym_margin_from_entries(
            model=model,
            tokenizer=tokenizer,
            entries=valid_entries,
            device=device,
            num_pairs=val_num_pairs,
            max_len=tcfg["max_len"],
        )
        # Fallback to template-based metrics when validation split lacks usable pos/hard_neg examples.
        if csc == 0.0:
            csc = contextual_synonym_consistency(
                model=model,
                tokenizer=tokenizer,
                syn_dict=syn_dict,
                device=device,
                num_pairs=val_num_pairs,
                max_len=tcfg["max_len"],
            )
        if csm["mu_same"] == 0.0 and csm["mu_related"] == 0.0:
            csm = contextual_synonym_margin(
                model=model,
                tokenizer=tokenizer,
                syn_dict=syn_dict,
                related_dict=related_dict,
                device=device,
                num_pairs=val_num_pairs,
                max_len=tcfg["max_len"],
            )
        print(f"[Epoch {epoch+1}] CSC (synonym consistency): {csc:.4f}")
        print(
            f"[Epoch {epoch+1}] CSM: mu_same={csm['mu_same']:.4f}, "
            f"mu_related={csm['mu_related']:.4f}, margin={csm['margin']:.4f}"
        )
        if use_wandb:
            wandb.log(
                {
                    "val/loss": mean_val_loss,
                    "val/local_uniformity": lu_score,
                    "val/retrieval_mrr": retrieval.get("mrr", 0.0),
                    "val/word_retrieval_mrr": word_retrieval.get("mrr", 0.0),
                    "val/CSC": csc,
                    "val/CSM_mu_same": csm["mu_same"],
                    "val/CSM_mu_related": csm["mu_related"],
                    "val/CSM_margin": csm["margin"],
                    "epoch": epoch + 1,
                },
                step=global_step,
            )
            for k in retrieval_topk:
                wandb.log(
                    {f"val/retrieval_r@{k}": retrieval.get(f"r@{k}", 0.0), "epoch": epoch + 1},
                    step=global_step,
                )
            for k in word_retrieval_topk:
                wandb.log(
                    {
                        f"val/word_retrieval_r@{k}": word_retrieval.get(f"r@{k}", 0.0),
                        "epoch": epoch + 1,
                    },
                    step=global_step,
                )

        # ---------- Save checkpoint ----------
        os.makedirs(paths["save_dir"], exist_ok=True)
        ckpt_path = os.path.join(paths["save_dir"], f"simcse_cilin_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
        latest_ckpt_path = os.path.join(paths["save_dir"], "simcse_cilin_latest.pt")
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_val_word_mrr": best_val_word_mrr,
                "best_ckpt_path": best_ckpt_path,
                "wandb_run_id": wandb_run_id,
            },
            latest_ckpt_path,
        )

        # Select best model by validation word-level retrieval MRR
        curr_val_word_mrr = float(word_retrieval.get("mrr", 0.0))
        if curr_val_word_mrr > best_val_word_mrr:
            best_val_word_mrr = curr_val_word_mrr
            best_ckpt_path = os.path.join(paths["save_dir"], "simcse_cilin_best.pt")
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"New best model saved to: {best_ckpt_path} (word_MRR={best_val_word_mrr:.4f})")
            if use_wandb:
                wandb.summary["best_val_word_mrr"] = best_val_word_mrr

    final_path = os.path.join(paths["save_dir"], "simcse_cilin_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training finished. Final model saved to: {final_path}")
    if best_ckpt_path:
        print(f"Best model (by val word retrieval MRR): {best_ckpt_path} (MRR={best_val_word_mrr:.4f})")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
