from __future__ import annotations

import mlx.core as mx


def compute_token_entropy(logits: mx.array) -> mx.array:
    """Compute Shannon entropy per token position from logits.

    Args:
        logits: shape (..., vocab_size) or (1, seq_len, vocab_size)

    Returns:
        Entropy with shape logits.shape[:-1]
    """

    # Softmax with numerical stability via logsumexp
    log_z = mx.logsumexp(logits, axis=-1, keepdims=True)
    log_p = logits - log_z
    p = mx.exp(log_p)

    # H(p) = -sum p * log2(p) = -sum p * log(p) / log(2)
    entropy_nats = -mx.sum(p * log_p, axis=-1)
    return entropy_nats / mx.log(mx.array(2.0))

