"""attlas_1120 item A: one-launch MoE routing for DeepSeek-V4.1-Flash on gfx950, behind SGLANG_ATTLAS_ROUTE_A=1.

For MoE calls of 257-2048 rows (DSpark verify steps), one FlyDSL launch (route_sort_quant) replaces aiter topk_gating,
the shared-expert append, fused_moe's Opus sort and its stage-1 MXFP8 quant + scale sort. Wired as a consume-once
handoff, like route_quant_handoff:

    DeepseekV2MoE.forward_normal   stage(hidden_states) before self.topk, clear() after the experts call
    select_experts (sqrtsoftplus)  try_route(...) runs the kernel and returns the top-7 ids / weights (shared slot
                                   included), or None to fall back to the unfused chain
    aiter fused_moe                its decode sort+quant hook takes the kernel's sorted ids and MXFP8 input when the
                                   call's activations and topk_ids are the ones the kernel saw; else AITER's own path

Every step falls back: another model or shape, a miss in the hook, or the switch off leave today's chain untouched.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)
_ENABLED = os.environ.get("SGLANG_ATTLAS_ROUTE_A", "0") == "1"
_MIN_ROWS = int(os.environ.get("SGLANG_ATTLAS_ROUTE_A_MIN_ROWS", "257"))
_MAX_ROWS = min(2048, int(os.environ.get("SGLANG_ATTLAS_ROUTE_A_MAX_ROWS", "2048")))
# debug: "graph" routes only inside a CUDA-graph capture, "eager" only outside one, "all" both
_MODE = os.environ.get("SGLANG_ATTLAS_ROUTE_A_MODE", "all")
# debug: compare every routed call with today's chain (top-k, sort, MXFP8 quant), inside graphs too; counters logged
# from eager calls (at most every 20 s)
_CHECK = os.environ.get("SGLANG_ATTLAS_ROUTE_A_CHECK", "0") == "1"
_CTR_NAMES = ("calls", "rows", "topk_bad_rows", "sort_bad_calls", "a1_bad_rows", "scale_bad_bytes")
_LOG_FIRST = 8  # log lines per event kind and process: proof the path ran, without flooding the server log
_state = {"x": None, "out": None, "tried": False, "router": None, "bias": {}, "routed": 0, "fed": 0, "unused": 0,
          "skipped": 0, "unreached": 0}


def _log_first(kind, msg, *args):
    _state[kind] = _state.get(kind, 0) + 1
    if _state[kind] <= _LOG_FIRST:
        logger.info("attlas_route_a: " + msg + " (%s #%d)", *args, kind, _state[kind])


def enabled() -> bool:
    return _ENABLED


def set_decode(flag: bool) -> None:
    """The MoE layer's batch is a decode or target-verify step (item A routes only those, not prefill)."""
    _state["decode"] = flag


def stage(x: torch.Tensor) -> None:
    _state["x"] = x
    _state["out"] = None
    _state["tried"] = False


def note_skip(m: int, why) -> None:
    """A staged MoE call of m rows that the kernel will not route, with the failed conditions."""
    _state["tried"] = True
    if _MIN_ROWS <= m <= _MAX_ROWS:
        _log_first("skipped: " + ", ".join(why), "%d rows not routed", m)


def clear() -> None:
    x = _state["x"]
    if x is not None and not _state["tried"] and _MIN_ROWS <= x.shape[0] <= _MAX_ROWS:
        _log_first("unreached", "select_experts did not reach the route-A branch (%d rows)", x.shape[0])
    if _state["out"] is not None:
        _log_first("unused", "routing result not taken by fused_moe (%d rows); AITER sorted again",
                   _state["out"]["topk_ids"].shape[0])
    _state["x"] = None
    _state["out"] = None


def prepare(device) -> None:
    """Build the kernel's buffers and compile both sort-block variants eagerly (model init, not inside a graph
    capture): RouteA zeroes its workspace once, and FlyDSL compiles on a kernel's first launch."""
    if not _ENABLED or _state["router"] is not None:
        return
    import sys

    sys.path.insert(0, os.environ.get("SGLANG_ATTLAS_ROUTE_A_PATH", "/sgl-workspace/attlas_route_a"))
    if _CHECK:
        _state["ctr"] = torch.zeros((2, len(_CTR_NAMES)), dtype=torch.int64, device=device)  # [graph, eager]
        _state["ctr_t"] = 0.0
    import route_a

    r = route_a.RouteA(device)
    bias = torch.zeros(384, dtype=torch.float32, device=device)
    for m in (300, 1100):  # sort block 32 and 64
        r(torch.zeros((m, 384), dtype=torch.float32, device=device), bias,
          torch.zeros((m, 5120), dtype=torch.bfloat16, device=device))
    torch.cuda.synchronize(device)
    _state["router"] = r
    _install_aiter_hook()
    logger.info("attlas_route_a: on for %d-%d rows, mode %s; kernel compiled for sort blocks 32 and 64", _MIN_ROWS,
                _MAX_ROWS, _MODE)


def _router():
    return _state["router"]


def _log_counters():
    import time

    if time.time() - _state["ctr_t"] < 20:
        return
    _state["ctr_t"] = time.time()
    torch.cuda.synchronize()
    c = _state["ctr"].tolist()
    logger.info("attlas_route_a CHECK graph %s | eager %s", dict(zip(_CTR_NAMES, c[0])), dict(zip(_CTR_NAMES, c[1])))


def _check_topk(ctr, r, ref_fn, m, num_token_non_padded):
    rw, ri = ref_fn()
    k = rw.shape[1]
    rows = torch.arange(m, device=rw.device)
    nv = num_token_non_padded.reshape(1).to(torch.int64) if num_token_non_padded is not None else m
    real = rows < nv
    bad = ((ri.to(torch.int32) != r["topk_ids"][:, :k]) | (rw.to(torch.float32) != r["topk_w"][:, :k])).any(1) & real
    ctr[0].add_(1)
    ctr[1].add_(real.sum())
    ctr[2].add_(bad.sum())
    r["check_real"] = real


def try_route(hidden_states: torch.Tensor, router_logits: torch.Tensor, correction_bias: Optional[torch.Tensor],
              renormalize: bool, routed_scaling_factor: Optional[float], shared_w: float,
              num_token_non_padded: Optional[torch.Tensor], ref_fn=None):
    """(topk_weights [M, 7] f32, topk_ids [M, 7] i32) from the one-launch kernel, or None (caller runs the chain)."""
    m = hidden_states.shape[0]
    _state["tried"] = True
    if _CHECK and "ctr" in _state and not torch.cuda.is_current_stream_capturing():
        _log_counters()
    if not (_ENABLED and _state["router"] is not None and _MIN_ROWS <= m <= _MAX_ROWS):
        return None
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        is_in_breakable_cuda_graph,
    )

    why = [name for name, bad in (
        ("not decode / verify", not _state.get("decode", False)),
        ("breakable prefill graph", is_in_breakable_cuda_graph()),
        ("not staged", _state["x"] is not hidden_states), ("renormalize off", not renormalize),
        ("no correction bias", correction_bias is None),
        ("logits dtype", router_logits.dtype not in (torch.float32, torch.bfloat16)),
        ("experts", router_logits.shape[1] != 384), ("x dtype", hidden_states.dtype != torch.bfloat16),
        ("hidden size", hidden_states.shape[1] != 5120),
        ("not contiguous", not (hidden_states.is_contiguous() and router_logits.is_contiguous()))) if bad]
    capturing = torch.cuda.is_current_stream_capturing()
    if (_MODE == "graph" and not capturing) or (_MODE == "eager" and capturing):
        why.append(f"mode {_MODE}")
    if why:
        note_skip(m, why)
        return None
    # above 64 rows AITER's router GEMM returns bf16 logits; the kernel reads fp32 (bf16 -> fp32 is exact)
    if router_logits.dtype != torch.float32:
        router_logits = router_logits.float()
    # A tensor made inside a CUDA-graph capture is only filled when that graph replays, so the fp32 bias is cached
    # from eager calls only; inside a capture it is converted in the graph itself.
    key = (correction_bias.data_ptr(), correction_bias.dtype)
    bias = _state["bias"].get(key)
    if bias is None:
        bias = correction_bias.float().contiguous()
        if capturing:
            _log_first("bias in capture", "fp32 bias converted inside the graph (layer bias %#x)", key[0])
        else:
            _state["bias"][key] = bias
    r = _router()(router_logits, bias, hidden_states, 1.0 if routed_scaling_factor is None else routed_scaling_factor,
                  shared_w, num_token_non_padded)
    _state["out"] = r
    if _CHECK and ref_fn is not None:
        r["check_ctr"] = _state["ctr"][0 if capturing else 1]
        _check_topk(r["check_ctr"], r, ref_fn, m, num_token_non_padded)
    _log_first("routed, graph capture" if capturing else "routed, eager", "one-launch routing of %d rows", m)
    return r["topk_w"], r["topk_ids"]


def _check_sort_quant(r, fm, x, topk_ids, topk_w, model_dim, global_experts, block_m, moebuf_dtype):
    ctr, real, m = r["check_ctr"], r["check_real"], x.shape[0]
    sid, sw, seid, nv, _ = fm.moe_sorting(topk_ids, topk_w, global_experts, model_dim, moebuf_dtype, block_m, None,
                                          None, 0, accumulate=False)[:5]
    n = min(seid.numel(), r["sorted_eids"].numel())
    blk = torch.arange(n, device=x.device) < (nv[0] // block_m)
    bad_sort = (nv[0] != r["num_valid"][0]) | ((seid[:n] != r["sorted_eids"][:n]) & blk).any()
    ctr[3].add_(bad_sort.to(torch.int64))
    # what fused_moe's own a8w4 path feeds stage 1 for this sort
    a1, a1s = fm.fused_dynamic_mxfp8_quant_moe_sort(x, sorted_ids=r["sorted_ids"], num_valid_ids=r["num_valid"],
                                                    token_num=m, topk=topk_ids.shape[1], block_size=block_m,
                                                    sorted_weights=r["sorted_w"], num_experts_upper_bound=global_experts)
    if a1.numel() == r["a1"].numel():  # shapes are host-side facts: no check that could raise inside a capture
        ctr[4].add_(((a1.view(torch.uint8).reshape(m, -1) != r["a1"].view(torch.uint8).reshape(m, -1)).any(1)
                     & real).sum())
    else:
        _log_first("check shape", "a1 reference has %d elements, kernel %d", a1.numel(), r["a1"].numel())
    want = a1s.view(torch.uint8).reshape(-1)
    got = r["a1_scale"].view(torch.uint8).reshape(-1)
    n = min(want.numel(), got.numel())
    ctr[5].add_(((want[:n] != got[:n]) & (want[:n] != 0)).sum())


def _install_aiter_hook():
    import aiter.fused_moe as fm

    if getattr(fm, "_attlas_route_a_hooked", False):
        return
    orig_enabled, orig_sort_quant = fm._is_fused_decode_sort_quant_enabled, fm._fused_decode_sort_quant

    def _hit(hidden_states, topk_ids):
        r = _state["out"]
        return (r is not None and _state["x"] is not None and hidden_states.data_ptr() == _state["x"].data_ptr()
                and topk_ids.data_ptr() == r["topk_ids"].data_ptr())

    def enabled_hook(hidden_states, topk_ids, topk_weights, **kw):
        if _hit(hidden_states, topk_ids) and kw.get("block_m") == _state["out"]["block_m"] \
                and kw.get("expert_mask") is None and kw.get("num_local_tokens") is None:
            return True
        return orig_enabled(hidden_states, topk_ids, topk_weights, **kw)

    def sort_quant_hook(hidden_states, topk_ids, topk_weights, *, model_dim, global_experts, block_m, moebuf_dtype,
                        accumulate, compact_scale=False):
        if not _hit(hidden_states, topk_ids):
            return orig_sort_quant(hidden_states, topk_ids, topk_weights, model_dim=model_dim,
                                   global_experts=global_experts, block_m=block_m, moebuf_dtype=moebuf_dtype,
                                   accumulate=accumulate, compact_scale=compact_scale)
        r = _state["out"]
        _state["out"] = None  # consume once
        _log_first("fed", "fused_moe took the kernel's sorted ids and MXFP8 input (%d rows, block_m %d)",
                   hidden_states.shape[0], block_m)
        if block_m != r["block_m"] or global_experts != 385 or compact_scale:
            raise RuntimeError(f"attlas_route_a: fused_moe wants block_m {block_m}, kernel sorted for {r['block_m']}")
        buf = (torch.zeros((hidden_states.shape[0], model_dim), dtype=moebuf_dtype, device=hidden_states.device)
               if accumulate else torch.empty((0, 0), dtype=moebuf_dtype, device=hidden_states.device))
        if "check_ctr" in r:
            _check_sort_quant(r, fm, hidden_states, topk_ids, topk_weights, model_dim, global_experts, block_m,
                              moebuf_dtype)
        return r["sorted_ids"], r["sorted_w"], r["sorted_eids"], r["num_valid"], buf, r["a1"], r["a1_scale"]

    fm._is_fused_decode_sort_quant_enabled = enabled_hook
    fm._fused_decode_sort_quant = sort_quant_hook
    fm._attlas_route_a_hooked = True
