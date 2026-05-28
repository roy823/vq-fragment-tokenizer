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
