"""The IOI circuit-performance metric: IO logit minus subject logit at the answer position.

Higher logit difference means the model correctly prefers the indirect object over the subject.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing torch at module load time
    import torch

# Default batch size for batched evaluation. Mirrors ``batch_size`` in
# ``configs/circuit_discovery.yaml``; production callers pass the configured value through.
DEFAULT_BATCH_SIZE = 32


def logit_difference(
    logits: "torch.Tensor",
    io_token_list: list[int],
    subj_token_list: list[int],
    position_list: list[int],
) -> "torch.Tensor":
    """Compute IO_logit - S_logit at each prompt's prediction position.

    Parameters
    ----------
    logits : torch.Tensor of shape (batch, seq_len, vocab_size)
    io_token_list : list[int]
        Token id of the indirect object (correct answer) for each prompt.
    subj_token_list : list[int]
        Token id of the subject (distractor) for each prompt.
    position_list : list[int]
        Sequence index at which to read the logits for each prompt.

    Returns
    -------
    torch.Tensor of shape (batch,) -- one scalar logit difference per prompt.
    """
    import torch

    batch = logits.shape[0]
    batch_index = torch.arange(batch, device=logits.device)
    positions = torch.as_tensor(position_list, device=logits.device, dtype=torch.long)
    io_tokens = torch.as_tensor(io_token_list, device=logits.device, dtype=torch.long)
    subj_tokens = torch.as_tensor(subj_token_list, device=logits.device, dtype=torch.long)

    final_logits = logits[batch_index, positions, :]  # (batch, vocab_size)
    io_logits = final_logits[batch_index, io_tokens]  # (batch,)
    subj_logits = final_logits[batch_index, subj_tokens]  # (batch,)
    return io_logits - subj_logits


def mean_logit_difference(
    model,
    prompt_list: list[str],
    io_token_list: list[int],
    subj_token_list: list[int],
    position_list: list[int],
    batch_size: int = 32,
) -> float:
    """Run ``model`` over ``prompt_list`` in batches and return the mean logit difference.

    This is the scalar circuit-performance metric used by circuit discovery and patching.
    Prompts are right-padded so that each per-prompt ``position`` still indexes the real
    trailing token rather than a pad token.
    """
    import torch

    per_prompt_diffs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(prompt_list), batch_size):
            stop = start + batch_size
            batch_prompts = prompt_list[start:stop]
            tokens = model.to_tokens(batch_prompts, padding_side="right")
            logits = model(tokens)
            diffs = logit_difference(
                logits,
                io_token_list[start:stop],
                subj_token_list[start:stop],
                position_list[start:stop],
            )
            per_prompt_diffs.append(diffs)

    return float(torch.cat(per_prompt_diffs).mean().item())
