import pytest
import torch
import torch.nn.functional as F

from verl_speco.integration.full_vocab_opd import reverse_kl_from_hidden


@pytest.mark.parametrize("cap", [None, 2.0])
def test_chunked_reverse_kl_matches_dense_loss_and_gradients(cap):
    torch.manual_seed(42)
    hidden = torch.randn(5, 4, requires_grad=True)
    teacher = torch.randn(5, 4, requires_grad=True)
    weight = torch.randn(8, 4, requires_grad=True)
    teacher_weight = torch.randn(8, 4, requires_grad=True)
    contract = {"vocab_size": 7, "soft_cap": cap, "logit_scale": 0.7}
    loss = reverse_kl_from_hidden(
        hidden, teacher, weight, teacher_weight, contract, chunk_rows=2
    )
    actual = torch.autograd.grad(loss, (hidden, weight), retain_graph=True)
    q = F.linear(teacher.detach(), teacher_weight.detach())
    if cap is not None:
        q = cap * torch.tanh(q / cap)
    logq = (q[:, :7] * 0.7).log_softmax(-1)
    logp = F.linear(hidden, weight)[:, :7].log_softmax(-1)
    reference = (logp.exp() * (logp - logq)).sum(-1).mean()
    expected = torch.autograd.grad(reference, (hidden, weight))
    torch.testing.assert_close(loss, reference)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    loss.backward()
    assert teacher.grad is None and teacher_weight.grad is None
