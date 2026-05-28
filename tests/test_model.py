import torch

from vqfrag.model import VQFragmentTokenizer, entropy_regularizer


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
