# model.py
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


@dataclass
class SimCSEConfig:
    encoder_name: str = "hfl/chinese-roberta-wwm-ext"
    pooling: str = "cls"           # "cls" 或 "mean"
    temp: float = 0.05
    distance_metric: str = "euclidean"  # "euclidean" 或 "cosine"

    # Hard Negative
    use_hard_neg: bool = False
    hard_neg_weight: float = 1.0
    margin: float = 0.3            # Triplet loss margin

    # Retrofitting
    use_retrofit: bool = False
    retrofit_weight: float = 0.1   # retrofit loss 權重


class SimCSEModel(nn.Module):
    def __init__(self, cfg: SimCSEConfig):
        super().__init__()
        self.bert = BertModel.from_pretrained(cfg.encoder_name)
        self.pooling = cfg.pooling
        self.temp = cfg.temp
        self.cfg = cfg

    def encode(self, input_ids, attention_mask, output_hidden: bool = False):
        """
        若 output_hidden=True:
            回傳 (sentence_emb, last_hidden_state)
        否則:
            回傳 sentence_emb
        """
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden,
            return_dict=True,
        )
        last_hidden = out.last_hidden_state   # [B, L, H]
        cls = out.pooler_output               # [B, H]

        if self.pooling == "cls":
            emb = cls
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
            sum_emb = (last_hidden * mask).sum(dim=1)
            len_emb = mask.sum(dim=1).clamp(min=1)
            emb = sum_emb / len_emb
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        emb = F.normalize(emb, p=2, dim=-1)

        if output_hidden:
            return emb, last_hidden
        else:
            return emb

    def forward(self, input_ids, attention_mask):
        return self.encode(input_ids, attention_mask, output_hidden=False)

    def _pairwise_distance(self, a, b):
        """
        a, b: [*, H]
        Returns a distance value per row according to cfg.distance_metric.
        - euclidean: squared L2 distance
        - cosine: 1 - cosine similarity
        """
        metric = self.cfg.distance_metric.lower()
        if metric == "euclidean":
            return (a - b).pow(2).sum(dim=-1)
        if metric == "cosine":
            return 1 - F.cosine_similarity(a, b, dim=-1)
        raise ValueError(f"Unknown distance metric: {self.cfg.distance_metric}")

    # ===== SimCSE InfoNCE loss =====
    def simcse_loss(self, z1, z2):
        """
        z1, z2: [B, H]
        """
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)              # [2B, H]
        sim = torch.matmul(z, z.T) / self.temp      # [2B, 2B]

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, torch.finfo(sim.dtype).min)

        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # [2B]

        loss = nn.CrossEntropyLoss()(sim, labels)
        return loss

    # ===== Hard Negative Triplet Loss =====
    def hard_neg_loss(self, anchor, pos, neg):
        """
        Triplet loss: L = max(0, margin + d(a,p) - d(a,n))
        """
        d_ap = self._pairwise_distance(anchor, pos)
        d_an = self._pairwise_distance(anchor, neg)
        loss = torch.relu(self.cfg.margin + d_ap - d_an).mean()
        return loss

    # ===== Contextual Retrofitting Loss =====
    def retrofit_loss(self, syn_emb1, syn_emb2):
        """
        syn_emb1, syn_emb2: [M, H]
        表示同一句話中，同一個位置（原詞 vs 同義詞）在上下文下的 embedding。
        L_retro = mean || e(w|S) - e(s|S⁺) ||^2
        """
        if syn_emb1 is None or syn_emb2 is None or syn_emb1.numel() == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        assert syn_emb1.shape == syn_emb2.shape, \
            f"retrofit pair shape mismatch: {syn_emb1.shape} vs {syn_emb2.shape}"

        dist = self._pairwise_distance(syn_emb1, syn_emb2)
        loss = dist.mean()
        return loss
