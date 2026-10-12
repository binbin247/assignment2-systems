"""Regression tests for language-model generation."""

import pytest
import torch
from cs336_basics.model import BasicsTransformerLM


@pytest.fixture
def model():
    torch.manual_seed(336)
    return BasicsTransformerLM(
        vocab_size=32, context_length=8, d_model=32, num_layers=1, num_heads=4, d_ff=64
    ).eval()


@pytest.mark.parametrize("prompt_length", [7, 8, 11])
def test_generate_preserves_new_tokens_after_context_truncation(model, prompt_length):
    prompt = torch.randint(0, 32, (1, prompt_length))
    generated = model.generate(prompt, max_new_tokens=5)
    assert generated.shape == (1, 5)
    assert ((generated >= 0) & (generated < 32)).all()


@pytest.mark.parametrize("batch_size", [1, 2])
def test_generate_top_k_one_is_greedy(model, batch_size, monkeypatch):
    prompt = torch.tensor([[1, 2, 3]]).expand(batch_size, -1)
    with torch.no_grad():
        greedy = model(prompt)[:, -1].argmax(-1, keepdim=True)

    sample = torch.multinomial

    def sample_top_one(probabilities, num_samples):
        # Sampling must assign nonzero probability only to the best token.
        assert torch.equal((probabilities > 0).sum(-1), torch.ones(batch_size, dtype=torch.long))
        return sample(probabilities, num_samples)

    monkeypatch.setattr(torch, "multinomial", sample_top_one)
    generated = model.generate(prompt, max_new_tokens=1, top_k=1)
    assert torch.equal(generated, greedy)
