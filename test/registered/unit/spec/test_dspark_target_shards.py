"""DSpark verify on the target's vocabulary shards: the CPU half (the Triton kernels need a GPU)."""

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.srt.speculative.dspark_components.dspark_target_shards import (
    TargetShard,
    full_logits_unless_kept,
    gather_full_logits,
    keeps_target_shards,
    key_splits,
    sampling_region_allowed,
    shard_keys,
    shard_of_logits,
    sharded_target_tokens,
    target_shard_of,
    target_shards_refusal,
)

WORLD, WIDTH = 4, 3000


# The model-side modules (logits processor, DSpark verify) probe the GPU when a ROCm torch
# imports them, so their tests run on a GPU host; they still launch no GPU kernel.
needs_gpu_host = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the ROCm build's model modules probe the GPU at import"
)


def _import(name):
    return importlib.import_module(name)


def _assert_same(got, want):
    torch.testing.assert_close(got, want, rtol=0, atol=0, equal_nan=True)


class _FakeGroup:
    """all_gather_into_tensor over tensors of every rank held on one device, rank order."""

    def __init__(self, per_rank):
        self.per_rank, self.world_size = per_rank, len(per_rank)

    def all_gather_into_tensor(self, output, input):
        output.copy_(torch.cat([t.reshape(-1) for t in self.per_rank]))


def _shards(full, vocab):
    return [
        (full[:, r * WIDTH : (r + 1) * WIDTH].contiguous(),
         TargetShard(r * WIDTH, max(0, min(WIDTH, vocab - r * WIDTH)), vocab))
        for r in range(WORLD)
    ]


def _logits(rows, vocab, dtype):
    gen = torch.Generator().manual_seed(0)
    full = torch.randn(rows, WORLD * WIDTH, generator=gen) * 4
    full[0, 5] = full[0, 3 * WIDTH + 7] = full[0].max() + 1  # tie across shards: token 5
    full[1].fill_(1.5)  # constant row: token 0
    full[2].zero_()
    full[2, 0] = -0.0  # -0.0 ties +0.0: token 0
    full[3, 10] = float("nan")  # a NaN never wins
    full[4, vocab:] = 1e4  # padding past the vocabulary never wins
    return full.to(dtype)


def _want(full, vocab):
    rows = full[:, :vocab].float()
    return torch.argmax(torch.nan_to_num(rows, nan=float("-inf")), dim=-1)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=needs_gpu_host)],  # cuda runs the Triton kernel
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [WORLD * WIDTH, WORLD * WIDTH - 10])
def test_sharded_target_tokens_match_argmax(device, dtype, vocab):
    full = _logits(9, vocab, dtype).to(device)
    shards = _shards(full, vocab)
    keys = [shard_keys(s, t, key_splits(WIDTH)) for s, t in shards]
    for s, t in shards:
        got = sharded_target_tokens(s, t, _FakeGroup(keys))
        assert torch.equal(got, _want(full, vocab)), (t, got)


def test_gather_full_logits_matches_stock():
    vocab = WORLD * WIDTH - 10
    full = _logits(5, vocab, torch.bfloat16)
    shards = _shards(full, vocab)
    group = _FakeGroup([s for s, _ in shards])
    got = gather_full_logits(shards[0][0], shards[0][1], group)
    assert got.dtype == torch.float32
    _assert_same(got, full[:, :vocab].float())


def _greedy_case(bs, slots, vocab):
    full = _logits(bs * slots, vocab, torch.float32)
    predict = _want(full, vocab).view(bs, slots)
    candidates = torch.zeros(bs, slots, dtype=torch.int64)
    for b in range(bs):
        accepted = b % slots  # 0 .. slots - 1 drafts match the target, then one does not
        candidates[b, 1 : 1 + accepted] = predict[b, :accepted]
        if accepted < slots - 1:
            candidates[b, 1 + accepted] = (predict[b, accepted] + 1) % vocab
    return full, candidates


@needs_gpu_host
def test_greedy_accept_on_shards_equals_full_rows():
    accept_draft_tokens = _import(
        "sglang.srt.speculative.dspark_components.dspark_verify"
    ).accept_draft_tokens

    bs, slots, vocab = 7, 6, WORLD * WIDTH
    full, candidates = _greedy_case(bs, slots, vocab)
    common = dict(
        candidates=candidates,
        draft_block=SimpleNamespace(greedy_mask=None),
        sampling_info=None,
        draft_input=None,
        gamma=slots - 1,
        verify_num_draft_tokens=slots,
    )
    want = accept_draft_tokens(target_logits=full, **common)
    shards = _shards(full, vocab)
    keys = [shard_keys(s, t, key_splits(WIDTH)) for s, t in shards]
    for s, t in shards:
        got = accept_draft_tokens(
            target_logits=s, target_shard=t, tp_group=_FakeGroup(keys), **common
        )
        for g, w in zip(got, want):
            assert torch.equal(g, w), (t, got, want)
    assert want[0].tolist() == [b % slots for b in range(bs)]


class _Recorder:
    def __init__(self):
        self.args = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.args = args

        return launch


@pytest.mark.parametrize("on_shard", [False, True])
def test_sharded_sampling_reads_the_target_shard(on_shard):
    from sglang.kernels.ops.speculative.dspark import sharded_accept_hip as m

    bs, slots, width, start = 2, 6, 3000, 6000
    target = torch.zeros(bs * slots, width if on_shard else 4 * width)
    stats, accept, finish = _Recorder(), _Recorder(), _Recorder()
    with (
        patch.object(m, "_stats_kernel", stats),
        patch.object(m, "_accept_kernel", accept),
        patch.object(m, "_markov_sample_finish_kernel", finish),
    ):
        m.accept_sampling_sharded(
            candidates=torch.zeros(bs, slots, dtype=torch.int64),
            target_logits=target,
            draft_logits=torch.zeros(bs, slots - 1, width),
            temperatures=torch.ones(bs),
            vocab_start=start,
            width=width,
            gather_stacked=lambda x: x,
            key=1,
            offset=torch.zeros(1, dtype=torch.int64),
            target_start=0 if on_shard else None,
        )
    target_start = 0 if on_shard else start
    # (row stride, target column of vocab_start, vocab_start) in both launches
    assert stats.args[5:8] == (target.stride(0), target_start, start)
    assert accept.args[11:14] == (target.stride(0), target_start, start)


def _processor(**attrs):
    LogitsProcessor = _import("sglang.srt.layers.logits_processor").LogitsProcessor
    lp = LogitsProcessor.__new__(LogitsProcessor)
    nn.Module.__init__(lp)
    base = dict(
        defer_verify_gather=True,
        final_logit_softcapping=None,
        do_tensor_parallel_all_gather=True,
        use_attn_tp_group=False,
        do_tensor_parallel_all_gather_dp_attn=False,
        vocab_size=WORLD * WIDTH - 10,
    )
    for k, v in {**base, **attrs}.items():
        setattr(lp, k, v)
    lp._can_use_tp_lm_head_all_to_all = lambda *args: False
    return lp


@needs_gpu_host
@pytest.mark.parametrize(
    "attrs,verify,want",
    [
        ({}, True, True),
        ({}, False, False),
        ({"defer_verify_gather": False}, True, False),
        ({"final_logit_softcapping": 30.0}, True, False),
        ({"do_tensor_parallel_all_gather": False}, True, False),
        ({"use_attn_tp_group": True}, True, False),
        ({"do_tensor_parallel_all_gather_dp_attn": True}, True, False),
    ],
)
def test_plain_verify_gather_deferral(attrs, verify, want):
    metadata = SimpleNamespace(forward_mode=SimpleNamespace(is_target_verify=lambda: verify))
    logits = torch.zeros(3, WIDTH)
    assert _processor(**attrs)._defers_plain_verify_gather(logits, logits, None, metadata) is want


@needs_gpu_host
def test_finish_deferred_gather_without_softcap_is_the_stock_gather():
    lp = _processor()
    full = _logits(5, lp.vocab_size, torch.bfloat16)
    shards = [s for s, _ in _shards(full, lp.vocab_size)]
    lp._logits_gatherer = lambda local: torch.cat(shards, dim=-1)
    got = lp.finish_deferred_gather(shards[0])
    assert got.dtype == torch.float32
    _assert_same(got, full[:, : lp.vocab_size].float())


# --- worker opt-in, per-step fallback and the sampling-accept region (light helpers) ---

SHARD = TargetShard(start=2 * WIDTH, valid=WIDTH, vocab_size=WORLD * WIDTH)
OPT_IN = dict(
    switch=True,
    tp_size=WORLD,
    model_type="deepseek_v41",
    num_logits_processors=1,
    simulate_acc_len=0.0,
    observers_read_logits=False,
    local_width=WIDTH,
    shard=SHARD,
    draft_shard=SimpleNamespace(start=2 * WIDTH, width=WIDTH),
)


@pytest.mark.parametrize(
    "change,reason",
    [
        ({}, None),
        ({"draft_shard": None}, None),
        ({"switch": False}, "switch off"),
        ({"tp_size": 1}, "TP 1"),
        ({"model_type": "deepseek_v4"}, "not validated"),
        ({"num_logits_processors": 2}, "logits processors"),
        ({"simulate_acc_len": 2.5}, "SIMULATE_ACC_LEN"),
        ({"observers_read_logits": True}, "observer"),
        ({"local_width": WIDTH + 8}, "padded"),
        ({"draft_shard": SimpleNamespace(start=WIDTH, width=WIDTH)}, "draft"),
    ],
)
def test_opt_in_rules(change, reason):
    got = target_shards_refusal(**{**OPT_IN, **change})
    assert (got is None) if reason is None else (reason in got), got


def test_target_shard_of_reads_the_lm_head_columns():
    lm_head = SimpleNamespace(
        shard_indices=SimpleNamespace(org_vocab_start_index=2 * WIDTH, num_org_elements=WIDTH)
    )
    assert target_shard_of(lm_head, WORLD * WIDTH) == SHARD


def _sampling(**kw):
    base = dict(is_all_greedy=True, sampling_mask_batch_indices=None)
    return SimpleNamespace(**{**base, **kw})


@pytest.mark.parametrize(
    "change,want",
    [
        ({}, True),
        ({"sampling_info": None}, True),
        ({"shard": None}, False),
        ({"has_grammar": True}, False),
        ({"return_logprob": True}, False),
        ({"sampling_info": _sampling(sampling_mask_batch_indices=torch.tensor([0]))}, False),
        ({"adjustments_noop": False}, False),
        ({"sampling_info": _sampling(is_all_greedy=False)}, False),
        ({"sampling_info": _sampling(is_all_greedy=False), "sampling_on_shards": True}, True),
    ],
)
def test_per_step_keep_or_gather(change, want):
    step = dict(
        shard=SHARD,
        has_grammar=False,
        return_logprob=False,
        sampling_info=_sampling(),
        adjustments_noop=True,
        sampling_on_shards=False,
    )
    assert keeps_target_shards(**{**step, **change}) is want


class _Processor:
    def __init__(self, vocab_size):
        self.vocab_size, self.gathers = vocab_size, 0

    def is_deferred_shard(self, logits):
        return logits.shape[-1] < self.vocab_size

    def finish_deferred_gather(self, logits):
        self.gathers += 1
        return torch.zeros(logits.shape[0], self.vocab_size)


def test_fallback_gathers_a_shard_the_step_does_not_keep():
    processor = _Processor(WORLD * WIDTH)
    shard = torch.zeros(6, WIDTH, dtype=torch.bfloat16)
    full = full_logits_unless_kept(shard, processor, keep=False)
    assert processor.gathers == 1 and tuple(full.shape) == (6, WORLD * WIDTH)
    assert full_logits_unless_kept(shard, processor, keep=True) is shard
    assert processor.gathers == 1


def test_shard_of_logits_tells_shard_rows_from_full_rows():
    assert shard_of_logits(torch.zeros(6, WIDTH), SHARD) == SHARD
    assert shard_of_logits(torch.zeros(6, WORLD * WIDTH), SHARD) is None
    assert shard_of_logits(torch.zeros(6, WIDTH), None) is None
    assert shard_of_logits(None, SHARD) is None


def test_sampling_region_is_off_under_shards():
    assert sampling_region_allowed(None) is True
    assert sampling_region_allowed(SHARD) is False


def test_switch_off_leaves_todays_path_untouched():
    """Switch off: the opt-in refuses, no processor gathers or defers, the verify logits
    come back as the same tensor object, the step keeps nothing and the sampling-accept
    region is allowed, so the worker hands target_shard=None to today's accept."""
    assert target_shards_refusal(**{**OPT_IN, "switch": False}) == "switch off"
    full = torch.randn(6, WORLD * WIDTH)
    processor = _Processor(WORLD * WIDTH)
    for p in (None, processor):
        for keep in (False, True):
            assert full_logits_unless_kept(full, p, keep) is full
    assert processor.gathers == 0
    step = dict(
        has_grammar=False,
        return_logprob=False,
        sampling_info=_sampling(),
        adjustments_noop=True,
        sampling_on_shards=True,
    )
    assert keeps_target_shards(shard=None, **step) is False
    assert sampling_region_allowed(None) is True
    assert shard_of_logits(full, None) is None


def test_switch_defaults_off():
    try:
        from sglang.srt.environ import envs
    except Exception as e:  # the ROCm build's environ may probe the GPU
        pytest.skip(f"environ import: {e}")
    assert envs.SGLANG_DSPARK_TARGET_VOCAB_SHARDS.get() is False


@needs_gpu_host
def test_worker_opt_in_switch_off_touches_nothing(monkeypatch):
    worker = _import("sglang.srt.speculative.dspark_components.dspark_worker_v2")
    monkeypatch.delenv("SGLANG_DSPARK_TARGET_VOCAB_SHARDS", raising=False)
    fake = SimpleNamespace(target_worker=None)  # never read with the switch off
    owner = next(v for v in vars(worker).values() if hasattr(v, "_keep_target_verify_shards"))
    assert owner._keep_target_verify_shards(fake) is None


@needs_gpu_host
@pytest.mark.parametrize("temperature", [0.6, 1.0])
def test_sharded_sampling_on_the_shard_equals_the_full_row(temperature):
    """The sharded chain rule reads the same values from the shard (target_start 0) as
    from its slice of the full row: bitwise the same accept lengths and bonus tokens."""
    from sglang.kernels.ops.speculative.dspark.sharded_accept_hip import (
        accept_sampling_sharded,
    )

    bs, slots, start = 8, 6, 2 * WIDTH
    gen = torch.Generator().manual_seed(1)
    full = (torch.randn(bs * slots, WORLD * WIDTH, generator=gen) * 3).cuda()
    draft = (torch.randn(bs, slots - 1, WIDTH, generator=gen) * 3).cuda()
    candidates = torch.randint(start, start + WIDTH, (bs, slots), generator=gen).cuda()
    common = dict(
        candidates=candidates,
        draft_logits=draft,
        temperatures=torch.full((bs,), temperature).cuda(),
        vocab_start=start,
        width=WIDTH,
        gather_stacked=lambda x: x,
        key=1234,
        offset=torch.zeros(1, dtype=torch.int64).cuda(),
    )
    want = accept_sampling_sharded(target_logits=full, **common)
    got = accept_sampling_sharded(
        target_logits=full[:, start : start + WIDTH].contiguous(), target_start=0, **common
    )
    for g, w in zip(got, want):
        assert torch.equal(g, w)
