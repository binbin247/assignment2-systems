import torch


def softmax(x, dim=-1):
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)


def log_softmax(x, dim=-1):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x = x - x_max
    return x - torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))


def cross_entropy(inputs, targets):
    negative_log_softmax_logits = -log_softmax(inputs)
    return torch.mean(torch.gather(negative_log_softmax_logits, -1, targets.unsqueeze(-1)))


def clip_gradient(parameters, max_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    norm = torch.tensor(0.0, device=grads[0].device)

    for g in grads:
        norm += (g**2).sum()

    norm = torch.sqrt(norm)
    clip_coef = min(1, max_norm / (norm + 1e-6))
    for g in grads:
        g *= clip_coef


def my_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Compute scaled dot-product attention using explicit PyTorch operations.
    """
    score = Q@K.transpose(-1,-2)/(Q.size(-1)**0.5)

    if is_causal:
        mask = torch.ones(
            Q.size(-2), K.size(-2), dtype=torch.bool, device=Q.device
        ).tril()
        score = score.masked_fill(~mask, float("-inf"))

    p = torch.nn.functional.softmax(
        score, dim = -1
    ) 

    return p@V
