import torch
import torch.nn.functional as F

CHUNK = 4096


class _ChunkedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits2d, labels1d, denom):
        grad = torch.empty_like(logits2d)
        total = torch.zeros((), dtype=torch.float32, device=logits2d.device)
        for i in range(0, logits2d.shape[0], CHUNK):
            lp = F.log_softmax(logits2d[i:i + CHUNK].float(), dim=-1)
            tg = labels1d[i:i + CHUNK]
            valid = tg != -100
            safe = tg.clamp_min(0)
            total -= lp.gather(1, safe.unsqueeze(1)).squeeze(1)[valid].sum()
            g = lp.exp_()
            g[torch.arange(g.shape[0], device=g.device), safe] -= 1.0
            g[~valid] = 0.0
            grad[i:i + CHUNK] = (g / denom).to(logits2d.dtype)
            del lp, g
        ctx.save_for_backward(grad)
        return total / denom

    @staticmethod
    def backward(ctx, grad_out):
        (grad,) = ctx.saved_tensors
        return grad * grad_out, None, None


def chunked_causal_lm_loss(logits, labels, vocab_size, num_items_in_batch=None, ignore_index=-100,
                           shift_labels=None, **kwargs):
    assert ignore_index == -100
    if shift_labels is None:
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()
    logits2d = logits.view(-1, vocab_size)
    labels1d = shift_labels.view(-1).to(logits2d.device)
    if num_items_in_batch is not None:
        denom = (num_items_in_batch.to(logits2d.device).float() if torch.is_tensor(num_items_in_batch)
                 else torch.tensor(float(num_items_in_batch), device=logits2d.device))
    else:
        denom = (labels1d != -100).sum().float().clamp_min(1)
    return _ChunkedCE.apply(logits2d, labels1d, denom)
