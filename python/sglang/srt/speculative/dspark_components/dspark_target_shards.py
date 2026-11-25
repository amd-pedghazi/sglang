"""DSpark verify on the target's vocabulary shards.

With ``LogitsProcessor.defer_verify_gather`` a target verify returns this rank's LM-head
shard ``[rows, V / tp]`` instead of the gathered rows. The greedy accept reduces each
shard row to int64 keys and exchanges only those: the largest key over the ranks is the
first maximum of the gathered FP32 row, as ``torch.argmax`` picks it, except that a NaN
never wins. The key encoding is sglang#47200's (``speculative/sharded_verify.py``).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl


class TargetShard(NamedTuple):
    """Columns [0, valid) of a rank's shard are the global token ids start + column."""

    start: int
    valid: int
    vocab_size: int


def target_shard_of(lm_head, vocab_size: int) -> TargetShard:
    """This rank's columns of the target's vocab-parallel LM head."""
    indices = lm_head.shard_indices
    return TargetShard(
        int(indices.org_vocab_start_index), int(indices.num_org_elements), int(vocab_size)
    )


def target_shards_refusal(
    *,
    switch: bool,
    tp_size: int,
    model_type: Optional[str],
    num_logits_processors: int,
    simulate_acc_len: float,
    observers_read_logits: bool,
    local_width: int,
    shard: TargetShard,
    draft_shard=None,
) -> Optional[str]:
    """None when the DSpark worker may keep the target-verify logits on the vocabulary
    shards (SGLANG_DSPARK_TARGET_VOCAB_SHARDS); otherwise why not. Every input is the same
    on every TP rank, so all ranks take the same collectives."""
    if not switch:
        return "switch off"
    if tp_size <= 1:
        return "TP 1 has no gather"
    if model_type != "deepseek_v41":
        return f"target {model_type} is not validated"
    if num_logits_processors != 1:
        return f"{num_logits_processors} logits processors"
    if simulate_acc_len > 0:
        return "SGLANG_SIMULATE_ACC_LEN samples its bonus from the full target row"
    if observers_read_logits:
        return "a verify observer reads the full target logits"
    if local_width * tp_size != shard.vocab_size:
        return "the LM-head shards are padded"
    if draft_shard is not None and (draft_shard.start, draft_shard.width) != (
        shard.start,
        shard.valid,
    ):
        return "the draft's vocabulary shard differs from the LM head's"
    return None


def keeps_target_shards(
    *,
    shard: Optional[TargetShard],
    has_grammar: bool,
    return_logprob: bool,
    sampling_info,
    adjustments_noop: bool,
    sampling_on_shards: bool,
) -> bool:
    """Whether this verify step accepts on the shards. If not, the full logits are
    gathered before the logit adjustments and the step runs today's path."""
    if shard is None or has_grammar or return_logprob:
        return False
    if sampling_info is None:
        return True
    if getattr(sampling_info, "sampling_mask_batch_indices", None) is not None:
        return False
    if not adjustments_noop:
        return False
    return sampling_info.is_all_greedy or sampling_on_shards


def shard_of_logits(
    logits: Optional[torch.Tensor], shard: Optional[TargetShard]
) -> Optional[TargetShard]:
    """``shard`` when ``logits`` holds a vocabulary shard, None for full rows (for
    example from a verify graph captured before the shards were switched on)."""
    if shard is None or logits is None or logits.shape[-1] >= shard.vocab_size:
        return None
    return shard


def full_logits_unless_kept(logits: torch.Tensor, processor, keep: bool) -> torch.Tensor:
    """The verify logits for today's path: a deferred shard becomes the full FP32 rows,
    unless the step keeps the shards. Anything else comes back as it is."""
    if keep or processor is None or not processor.is_deferred_shard(logits):
        return logits
    return processor.finish_deferred_gather(logits)


def sampling_region_allowed(shard: Optional[TargetShard]) -> bool:
    """The sampling-accept graph region reads full target rows, so it stays off under
    target shards."""
    return shard is None


# key of (-inf, token id 0xFFFFFFFF): below every real column's key
_KEY_FLOOR = -2139095041 << 32
_KEY_BLOCK = 2048


@triton.jit
def _shard_key_kernel(
    logits_ptr,
    stride_row,
    keys_ptr,
    start,
    valid,
    vocab_size,
    span,
    KEY_FLOOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # program (row, part): columns [part * span, (part + 1) * span) of the row's shard
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    lo = part * span
    hi = tl.minimum(lo + span, valid)
    best = tl.full([BLOCK], KEY_FLOOR, tl.int64)
    for off in range(lo, hi, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        gid = start + cols
        mask = (cols < hi) & (gid < vocab_size)
        x = tl.load(
            logits_ptr + row * stride_row + cols, mask=mask, other=float("-inf")
        ).to(tl.float32)
        # NaN never wins; -0.0 ties +0.0 as in torch.argmax
        x = tl.where(x == x, x, float("-inf")) + 0.0
        bits = x.to(tl.int32, bitcast=True)
        # monotone in x as a signed int32; the low word prefers the smaller token id
        ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)
        key = (ordered << 32) | (4294967295 - gid.to(tl.int64))
        best = tl.maximum(best, tl.where(mask, key, KEY_FLOOR))
    tl.store(keys_ptr + row * tl.num_programs(1) + part, tl.max(best, axis=0))


def key_splits(width: int) -> int:
    """Column ranges per row, from the rank-uniform shard width."""
    n = -(-width // _KEY_BLOCK)
    return n + (n & 1)


def _shard_keys_torch(logits: torch.Tensor, shard: TargetShard, splits: int):
    gid = shard.start + torch.arange(shard.valid, device=logits.device)
    x = logits[:, : shard.valid].float()
    x = torch.where(x == x, x, float("-inf")) + 0.0
    bits = x.view(torch.int32)
    ordered = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(torch.int64)
    keys = (ordered << 32) | (0xFFFFFFFF - gid)
    keys = torch.where(gid < shard.vocab_size, keys, _KEY_FLOOR)
    span = -(-shard.valid // splits)
    out = torch.full(
        (logits.shape[0], splits), _KEY_FLOOR, dtype=torch.int64, device=logits.device
    )
    for part in range(splits):
        block = keys[:, part * span : (part + 1) * span]
        if block.shape[1]:
            out[:, part] = block.amax(dim=1)
    return out


def shard_keys(logits: torch.Tensor, shard: TargetShard, splits: int) -> torch.Tensor:
    """``[rows, splits]`` int64 keys of this rank's shard: (order-preserving FP32 bits of
    a column range's best value) << 32 | (0xFFFFFFFF - its token id)."""
    assert logits.dim() == 2 and logits.stride(-1) == 1
    if not logits.is_cuda:
        return _shard_keys_torch(logits, shard, splits)
    keys = torch.empty(
        (logits.shape[0], splits), dtype=torch.int64, device=logits.device
    )
    if logits.shape[0]:
        _shard_key_kernel[(logits.shape[0], splits)](
            logits,
            logits.stride(0),
            keys,
            shard.start,
            shard.valid,
            shard.vocab_size,
            -(-shard.valid // splits),
            KEY_FLOOR=_KEY_FLOOR,
            BLOCK=_KEY_BLOCK,
            num_warps=8,
        )
    return keys


def keys_to_tokens(gathered: torch.Tensor, world: int, splits: int) -> torch.Tensor:
    """Rank-major ``[world * rows * splits]`` keys -> ``[rows]`` int64 token ids."""
    best = gathered.view(world, -1, splits).amax(dim=(0, 2))
    return 0xFFFFFFFF - (best & 0xFFFFFFFF)


def sharded_target_tokens(
    logits: torch.Tensor, shard: TargetShard, group
) -> torch.Tensor:
    """``[rows]`` greedy token ids over the whole vocabulary from every rank's shard."""
    splits = key_splits(logits.shape[-1])
    keys = shard_keys(logits, shard, splits)
    gathered = torch.empty(
        group.world_size * keys.numel(), dtype=torch.int64, device=keys.device
    )
    group.all_gather_into_tensor(gathered, keys.view(-1))
    return keys_to_tokens(gathered, group.world_size, splits)


def gather_full_logits(logits: torch.Tensor, shard: TargetShard, group) -> torch.Tensor:
    """The FP32 ``[rows, vocab_size]`` logits the stock verify builds from the shards:
    all-gather, cut to the vocabulary, cast."""
    rows = logits.shape[0]
    out = torch.empty(
        (group.world_size,) + tuple(logits.shape),
        dtype=logits.dtype,
        device=logits.device,
    )
    group.all_gather_into_tensor(out.view(-1), logits.contiguous().view(-1))
    full = out.movedim(0, 1).reshape(rows, -1)
    return full[:, : shard.vocab_size].float()
