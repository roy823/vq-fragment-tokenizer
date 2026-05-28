import torch

from vqfrag.model import SetVQSpectrumTokenizer, VQFragmentTokenizer, entropy_regularizer, setvq_entropy_regularizer


def test_vq_fragment_tokenizer_forward_shapes():
    model = VQFragmentTokenizer(hidden_dim=32, code_dim=8, codebook_size=16)
    batch = {
        "peak_x": torch.rand(4, 5),
        "fragment_x": torch.rand(4, 36),
        "event_x": torch.rand(4, 20),
        "cond_x": torch.rand(4, 12),
    }

    outputs = model(batch)

    assert outputs["peak_codes"].shape == (4,)
    assert outputs["fragment_codes"].shape == (4,)
    assert outputs["event_codes"].shape == (4,)
    assert outputs["peak_rec"].shape == (4, 5)
    assert outputs["loss"].ndim == 0
    bonus = entropy_regularizer(outputs, model.codebook_size)
    assert bonus.ndim == 0


def test_setvq_spectrum_tokenizer_forward_and_backward():
    model = SetVQSpectrumTokenizer(
        hidden_dim=32,
        code_dim=8,
        codebook_size=16,
        num_slots=4,
        num_bins=21,
        num_heads=4,
        encoder_layers=1,
    )
    batch = {
        "peak_x": torch.rand(3, 5, 3),
        "peak_mask": torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, False, False, False],
                [True, True, True, True, False],
            ]
        ),
        "cond_x": torch.rand(3, 12),
        "target_presence": torch.zeros(3, 21),
        "target_intensity": torch.zeros(3, 21),
    }
    batch["target_presence"][:, [2, 7, 11]] = 1.0
    batch["target_intensity"][:, [2, 7, 11]] = 0.8

    outputs = model(batch)

    assert outputs["slot_codes"].shape == (3, 4)
    assert outputs["binned_recon"].shape == (3, 21)
    assert outputs["presence_logits"].shape == (3, 21)
    assert outputs["loss"].ndim == 0
    assert setvq_entropy_regularizer(outputs, model.codebook_size).ndim == 0
    outputs["loss"].backward()
