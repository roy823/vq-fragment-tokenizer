"""Neural VQ tokenizer modules for peak, fragment, and event codes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    layers: int,
    dropout: float,
    *,
    normalize_output: bool = False,
) -> nn.Sequential:
    blocks: list[nn.Module] = []
    dim = input_dim
    for _ in range(max(layers - 1, 0)):
        blocks.append(nn.Linear(dim, hidden_dim))
        blocks.append(nn.GELU())
        if dropout > 0:
            blocks.append(nn.Dropout(dropout))
        dim = hidden_dim
    blocks.append(nn.Linear(dim, output_dim))
    if normalize_output:
        blocks.append(nn.LayerNorm(output_dim))
    return nn.Sequential(*blocks)


@dataclass(frozen=True)
class VQOutput:
    quantized: torch.Tensor
    codes: torch.Tensor
    loss: torch.Tensor
    perplexity: torch.Tensor
    entropy: torch.Tensor


class VectorQuantizer(nn.Module):
    """Straight-through VQ layer with random codebook initialization."""

    def __init__(self, codebook_size: int, code_dim: int, beta: float = 0.25) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.code_dim = int(code_dim)
        self.beta = float(beta)
        self.embedding = nn.Embedding(self.codebook_size, self.code_dim)
        nn.init.uniform_(self.embedding.weight, -1.0, 1.0)

    def forward(self, z: torch.Tensor) -> VQOutput:
        z_norm = F.normalize(z, dim=-1)
        z_flat = z_norm.reshape(-1, self.code_dim)
        codebook = F.normalize(self.embedding.weight, dim=-1)
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ codebook.t()
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        soft_probs = torch.softmax(-distances, dim=1).mean(dim=0)
        soft_entropy = -(soft_probs * torch.log(soft_probs.clamp_min(1e-12))).sum()
        codes = torch.argmin(distances, dim=1)
        z_q = codebook[codes].view_as(z)

        codebook_loss = F.mse_loss(z_q, z_norm.detach())
        commitment_loss = F.mse_loss(z_norm, z_q.detach())
        loss = codebook_loss + self.beta * commitment_loss
        quantized = z_norm + (z_q - z_norm).detach()

        with torch.no_grad():
            counts = torch.bincount(codes, minlength=self.codebook_size).float()
            probs = counts / counts.sum().clamp_min(1.0)
            active = probs > 0
            entropy = -(probs[active] * torch.log(probs[active])).sum()
            perplexity = torch.exp(entropy)

        return VQOutput(
            quantized=quantized,
            codes=codes.view(z.shape[:-1]),
            loss=loss,
            perplexity=perplexity,
            entropy=soft_entropy,
        )


class VQFragmentTokenizer(nn.Module):
    """Three-codebook tokenizer for peak, fragment, and pseudo-event units."""

    def __init__(
        self,
        *,
        peak_dim: int = 5,
        fragment_dim: int = 36,
        event_dim: int = 20,
        condition_dim: int = 12,
        hidden_dim: int = 128,
        code_dim: int = 64,
        codebook_size: int = 128,
        encoder_layers: int = 3,
        decoder_layers: int = 3,
        dropout: float = 0.0,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.peak_dim = int(peak_dim)
        self.fragment_dim = int(fragment_dim)
        self.event_dim = int(event_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.code_dim = int(code_dim)
        self.codebook_size = int(codebook_size)

        self.peak_encoder = _mlp(peak_dim, hidden_dim, code_dim, encoder_layers, dropout, normalize_output=True)
        self.fragment_encoder = _mlp(fragment_dim, hidden_dim, code_dim, encoder_layers, dropout, normalize_output=True)
        self.event_encoder = _mlp(event_dim, hidden_dim, code_dim, encoder_layers, dropout, normalize_output=True)

        self.peak_vq = VectorQuantizer(codebook_size, code_dim, beta=beta)
        self.fragment_vq = VectorQuantizer(codebook_size, code_dim, beta=beta)
        self.event_vq = VectorQuantizer(codebook_size, code_dim, beta=beta)

        decoder_input_dim = code_dim + condition_dim
        self.peak_decoder = _mlp(decoder_input_dim, hidden_dim, peak_dim, decoder_layers, dropout)
        self.fragment_decoder = _mlp(decoder_input_dim, hidden_dim, fragment_dim, decoder_layers, dropout)
        self.event_decoder = _mlp(decoder_input_dim, hidden_dim, event_dim, decoder_layers, dropout)

    def config_dict(self) -> dict:
        return {
            "peak_dim": self.peak_dim,
            "fragment_dim": self.fragment_dim,
            "event_dim": self.event_dim,
            "condition_dim": self.condition_dim,
            "hidden_dim": self.hidden_dim,
            "code_dim": self.code_dim,
            "codebook_size": self.codebook_size,
        }

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        peak_z = self.peak_encoder(batch["peak_x"])
        fragment_z = self.fragment_encoder(batch["fragment_x"])
        event_z = self.event_encoder(batch["event_x"])
        return {
            "peak": self.peak_vq(peak_z).codes,
            "fragment": self.fragment_vq(fragment_z).codes,
            "event": self.event_vq(event_z).codes,
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cond = batch["cond_x"]

        peak_vq = self.peak_vq(self.peak_encoder(batch["peak_x"]))
        fragment_vq = self.fragment_vq(self.fragment_encoder(batch["fragment_x"]))
        event_vq = self.event_vq(self.event_encoder(batch["event_x"]))

        peak_rec = self.peak_decoder(torch.cat([peak_vq.quantized, cond], dim=-1))
        fragment_rec = self.fragment_decoder(torch.cat([fragment_vq.quantized, cond], dim=-1))
        event_rec = self.event_decoder(torch.cat([event_vq.quantized, cond], dim=-1))

        peak_recon_loss = F.mse_loss(peak_rec, batch["peak_x"])
        fragment_recon_loss = F.mse_loss(fragment_rec, batch["fragment_x"])
        event_recon_loss = F.mse_loss(event_rec, batch["event_x"])
        recon_loss = peak_recon_loss + fragment_recon_loss + event_recon_loss
        vq_loss = peak_vq.loss + fragment_vq.loss + event_vq.loss

        return {
            "loss": recon_loss + vq_loss,
            "recon_loss": recon_loss.detach(),
            "vq_loss": vq_loss.detach(),
            "peak_recon_loss": peak_recon_loss.detach(),
            "fragment_recon_loss": fragment_recon_loss.detach(),
            "event_recon_loss": event_recon_loss.detach(),
            "peak_rec": peak_rec,
            "fragment_rec": fragment_rec,
            "event_rec": event_rec,
            "peak_codes": peak_vq.codes,
            "fragment_codes": fragment_vq.codes,
            "event_codes": event_vq.codes,
            "peak_perplexity": peak_vq.perplexity,
            "fragment_perplexity": fragment_vq.perplexity,
            "event_perplexity": event_vq.perplexity,
            "peak_entropy": peak_vq.entropy,
            "fragment_entropy": fragment_vq.entropy,
            "event_entropy": event_vq.entropy,
        }


def entropy_regularizer(outputs: dict[str, torch.Tensor], codebook_size: int) -> torch.Tensor:
    """Return a normalized entropy bonus to subtract from the training loss."""
    denom = torch.log(torch.tensor(float(codebook_size), device=outputs["loss"].device))
    entropies = torch.stack(
        [
            outputs["peak_entropy"],
            outputs["fragment_entropy"],
            outputs["event_entropy"],
        ]
    )
    return entropies.mean() / denom.clamp_min(1e-6)


@torch.no_grad()
def initialize_codebooks_from_loader(
    model: VQFragmentTokenizer,
    loader,
    *,
    device: torch.device,
    num_batches: int = 4,
    noise_std: float = 1e-3,
) -> None:
    """Seed codebooks from random encoder states on the data manifold.

    This remains a pure unsupervised VQ initialization: it samples encoder
    outputs from the current dataset and does not use motif rules or labels.
    """

    model.eval()
    collected = {"peak": [], "fragment": [], "event": []}
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        collected["peak"].append(F.normalize(model.peak_encoder(batch["peak_x"]), dim=-1))
        collected["fragment"].append(F.normalize(model.fragment_encoder(batch["fragment_x"]), dim=-1))
        collected["event"].append(F.normalize(model.event_encoder(batch["event_x"]), dim=-1))

    def _init_one(vq: VectorQuantizer, chunks: list[torch.Tensor]) -> None:
        if not chunks:
            return
        z = torch.cat(chunks, dim=0)
        if z.shape[0] == 0:
            return
        choice = torch.randint(0, z.shape[0], (vq.codebook_size,), device=z.device)
        values = z[choice]
        if noise_std > 0:
            values = F.normalize(values + torch.randn_like(values) * noise_std, dim=-1)
        vq.embedding.weight.copy_(values)

    _init_one(model.peak_vq, collected["peak"])
    _init_one(model.fragment_vq, collected["fragment"])
    _init_one(model.event_vq, collected["event"])


class SetVQEncoderBlock(nn.Module):
    """Cross-attention block from learned slots to an unordered peak set."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, slots: torch.Tensor, memory: torch.Tensor, peak_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~peak_mask.bool()
        cross, _ = self.cross_attn(slots, memory, memory, key_padding_mask=key_padding_mask, need_weights=False)
        slots = self.cross_norm(slots + cross)
        self_attn, _ = self.self_attn(slots, slots, slots, need_weights=False)
        slots = self.self_norm(slots + self_attn)
        slots = self.ffn_norm(slots + self.ffn(slots))
        return slots


class SetVQSpectrumTokenizer(nn.Module):
    """Observation-only SetVQ tokenizer for whole MS/MS spectra."""

    def __init__(
        self,
        *,
        peak_dim: int = 3,
        condition_dim: int = 12,
        hidden_dim: int = 128,
        code_dim: int = 64,
        codebook_size: int = 256,
        num_slots: int = 16,
        num_bins: int = 1501,
        encoder_layers: int = 2,
        decoder_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.0,
        beta: float = 0.25,
        bce_weight: float = 1.0,
        intensity_weight: float = 1.0,
        cosine_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.peak_dim = int(peak_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.code_dim = int(code_dim)
        self.codebook_size = int(codebook_size)
        self.num_slots = int(num_slots)
        self.num_bins = int(num_bins)
        self.encoder_layers = int(encoder_layers)
        self.decoder_layers = int(decoder_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.beta = float(beta)
        self.bce_weight = float(bce_weight)
        self.intensity_weight = float(intensity_weight)
        self.cosine_weight = float(cosine_weight)

        self.peak_encoder = _mlp(peak_dim, hidden_dim, hidden_dim, 2, dropout, normalize_output=True)
        self.slots = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
        self.encoder_blocks = nn.ModuleList(
            [SetVQEncoderBlock(hidden_dim, num_heads, dropout) for _ in range(encoder_layers)]
        )
        self.to_code = _mlp(hidden_dim, hidden_dim, code_dim, 2, dropout, normalize_output=True)
        self.vq = VectorQuantizer(codebook_size, code_dim, beta=beta)

        decoder_input_dim = num_slots * code_dim + condition_dim
        self.decoder = _mlp(decoder_input_dim, hidden_dim, num_bins * 2, decoder_layers, dropout)

    def config_dict(self) -> dict:
        return {
            "peak_dim": self.peak_dim,
            "condition_dim": self.condition_dim,
            "hidden_dim": self.hidden_dim,
            "code_dim": self.code_dim,
            "codebook_size": self.codebook_size,
            "num_slots": self.num_slots,
            "num_bins": self.num_bins,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "beta": self.beta,
            "bce_weight": self.bce_weight,
            "intensity_weight": self.intensity_weight,
            "cosine_weight": self.cosine_weight,
        }

    def encode_slots(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        memory = self.peak_encoder(batch["peak_x"])
        peak_mask = batch["peak_mask"].bool()
        slots = self.slots.unsqueeze(0).expand(memory.shape[0], -1, -1)
        for block in self.encoder_blocks:
            slots = block(slots, memory, peak_mask)
        return self.to_code(slots)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.vq(self.encode_slots(batch)).codes

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cond = batch["cond_x"]
        target_presence = batch["target_presence"]
        target_intensity = batch["target_intensity"]

        vq = self.vq(self.encode_slots(batch))
        decoder_in = torch.cat([vq.quantized.reshape(vq.quantized.shape[0], -1), cond], dim=-1)
        decoded = self.decoder(decoder_in)
        presence_logits, intensity_logits = decoded.chunk(2, dim=-1)
        presence_prob = torch.sigmoid(presence_logits)
        intensity_pred = torch.sigmoid(intensity_logits)
        binned_recon = presence_prob * intensity_pred

        pos = target_presence.sum().clamp_min(1.0)
        neg = (target_presence.numel() - target_presence.sum()).clamp_min(1.0)
        pos_weight = torch.clamp(neg / pos, min=1.0, max=50.0)
        bce_loss = F.binary_cross_entropy_with_logits(
            presence_logits,
            target_presence,
            pos_weight=pos_weight,
        )
        intensity_weights = 1.0 + 4.0 * target_presence
        intensity_loss = ((intensity_pred - target_intensity).pow(2) * intensity_weights).mean()
        cosine = F.cosine_similarity(binned_recon, target_intensity, dim=-1, eps=1e-8)
        spectral_cosine_loss = 1.0 - cosine.mean()
        recon_loss = (
            self.bce_weight * bce_loss
            + self.intensity_weight * intensity_loss
            + self.cosine_weight * spectral_cosine_loss
        )
        loss = recon_loss + vq.loss

        return {
            "loss": loss,
            "recon_loss": recon_loss.detach(),
            "vq_loss": vq.loss.detach(),
            "bce_loss": bce_loss.detach(),
            "intensity_loss": intensity_loss.detach(),
            "spectral_cosine_loss": spectral_cosine_loss.detach(),
            "presence_logits": presence_logits,
            "presence_prob": presence_prob,
            "intensity_pred": intensity_pred,
            "binned_recon": binned_recon,
            "slot_codes": vq.codes,
            "code_perplexity": vq.perplexity,
            "code_entropy": vq.entropy,
        }


def setvq_entropy_regularizer(outputs: dict[str, torch.Tensor], codebook_size: int) -> torch.Tensor:
    denom = torch.log(torch.tensor(float(codebook_size), device=outputs["loss"].device))
    return outputs["code_entropy"] / denom.clamp_min(1e-6)


@torch.no_grad()
def initialize_setvq_codebook_from_loader(
    model: SetVQSpectrumTokenizer,
    loader,
    *,
    device: torch.device,
    num_batches: int = 4,
    noise_std: float = 1e-3,
) -> None:
    """Seed the SetVQ codebook from observation-only encoder states."""

    model.eval()
    chunks = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        chunks.append(F.normalize(model.encode_slots(batch), dim=-1).reshape(-1, model.code_dim))
    if not chunks:
        return
    z = torch.cat(chunks, dim=0)
    if z.shape[0] == 0:
        return
    choice = torch.randint(0, z.shape[0], (model.codebook_size,), device=z.device)
    values = z[choice]
    if noise_std > 0:
        values = F.normalize(values + torch.randn_like(values) * noise_std, dim=-1)
    model.vq.embedding.weight.copy_(values)
