"""The HIP low-ratio prefill indexer with SGLANG_OPT_DSV4_INDEXER_TP_SHARD: each attention-TP rank selects its
share of the rows and all-gathers the selections. Every rank must end with the replicated path's page indices,
raw indices and candidate publication on the real FlyDSL logits, top-k and candidate kernels. The top-k and the
candidate selection emit a row's selection in no fixed order, so rows compare as sets."""

import itertools
import random
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

PAGE, TOPK, HEADS, TOPK_BLOCKS, BLOCK = 64, 512, 32, 2048, 8


def _aiter_fp4_indexer_available() -> bool:
    if not (is_hip() and is_gfx95_supported()):
        return False
    try:
        from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4_prefill  # noqa: F401
    except ImportError:
        return False
    return True


class _Core:
    def __init__(self, rows):
        self.page = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
        self.raw = torch.full_like(self.page, -1)

    def sparse_page_indices(self, ratio):
        return self.page

    def sparse_raw_indices(self, ratio):
        return self.raw


def _batch(ratio, seq_lens, extend_lens, seed, pad_to=0):
    """A batch in the pool's real layout: random FP4 index-K pages per request, a packed FP4 query.
    Rows past the requests up to pad_to belong to no request, as in a padded batch."""
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import pack_fp4_query_flydsl
    from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
        refresh_low_ratio_prefill_workspaces,
    )

    g = torch.Generator().manual_seed(seed)
    lcs = [s // ratio for s in seq_lens]
    npages = [max(1, -(-lc // PAGE)) for lc in lcs]
    perm = torch.randperm(sum(npages), generator=g).tolist()
    starts = list(itertools.accumulate([0, *npages]))
    req_pages = [perm[a:b] for a, b in zip(starts, starts[1:])]
    payload = torch.randint(
        0, 256, (len(perm), 1, 4, PAGE, 16), dtype=torch.uint8, generator=g
    )
    scale = torch.randint(
        118, 127, (len(perm), 1, 4, PAGE), dtype=torch.uint8, generator=g
    )
    rows = max(sum(extend_lens), pad_to)
    page_table = torch.zeros((rows, max(npages)), dtype=torch.int32)
    pos = torch.zeros(rows, dtype=torch.int64)
    t = 0
    for b, (s, e) in enumerate(zip(seq_lens, extend_lens)):
        page_table[t : t + e, : npages[b]] = torch.tensor(
            req_pages[b], dtype=torch.int32
        )
        pos[t : t + e] = torch.arange(s - e, s)
        t += e
    req_to_token = torch.zeros(
        (len(seq_lens), max(seq_lens) + ratio), dtype=torch.int32
    )
    for b, lc in enumerate(lcs):
        j = torch.arange(lc)
        slots = torch.tensor(req_pages[b])[j // PAGE] * PAGE + j % PAGE
        req_to_token[b, j * ratio] = (slots * ratio).to(torch.int32)
    q = torch.randn((rows, HEADS, 128), generator=g).to(torch.bfloat16).cuda()
    weights = (torch.randn((rows, HEADS), generator=g) * 0.1).to(torch.bfloat16).cuda()
    meta = SimpleNamespace(
        page_table=page_table.cuda(),
        compressed_seq_lens=((pos + 1) // ratio).to(torch.int32).cuda(),
        compressed_page_size=PAGE,
    )
    payload = payload.cuda().view(torch.float4_e2m1fn_x2)
    scale = scale.cuda()
    return SimpleNamespace(
        ratio=ratio,
        seq_lens=seq_lens,
        extend_lens=extend_lens,
        rows=rows,
        pos=pos.cuda(),
        meta=meta,
        pool=SimpleNamespace(
            get_index_k_fp4_payload_buffer=lambda layer_id: payload,
            get_index_k_fp4_scale_buffer=lambda layer_id: scale,
            low_ratio_index_k_page8=lambda layer_id: False,
        ),
        workspaces=refresh_low_ratio_prefill_workspaces(
            {ratio: meta}, None, extend_lens=extend_lens, max_context_len=max(seq_lens)
        ),
        req_to_token=req_to_token.cuda(),
        inputs=(*pack_fp4_query_flydsl(q), weights),
    )


def _index(batch, kind, candidate_masks, row_shard):
    """One index layer (plain, source or consumer): (page indices, raw indices, publication)."""
    from sglang.srt.layers.attention.dsv4 import low_ratio_backend_hip

    core = _Core(batch.rows)
    backend = SimpleNamespace(
        token_to_kv_pool=batch.pool,
        forward_metadata=SimpleNamespace(
            core_metadata=core,
            late_layer_tail=None,
            low_ratio_indexer_metadata=lambda ratio: batch.meta,
            fp4_low_ratio_prefill_workspaces=batch.workspaces,
        ),
        low_ratio_identity_skip=True,
        candidate_masks=candidate_masks,
        req_to_token=batch.req_to_token,
        max_context_len=max(batch.seq_lens),
    )
    layer = SimpleNamespace(
        compress_ratio=batch.ratio,
        layer_id=0,
        indexer=SimpleNamespace(
            index_topk=TOPK,
            is_candidate_source=kind == "source",
            uses_candidates=kind == "consumer",
            candidate_topk_blocks=TOPK_BLOCKS,
            candidate_block_size=BLOCK,
        ),
    )
    forward_batch = SimpleNamespace(
        seq_lens_cpu=torch.tensor(batch.seq_lens),
        extend_seq_lens_cpu=batch.extend_lens,
        req_pool_indices=torch.arange(len(batch.seq_lens), device="cuda"),
    )
    # the projection stays replicated under the shard; only the scoring is split
    low_ratio_backend_hip._indexer_inputs = lambda *args: batch.inputs
    low_ratio_backend_hip.low_ratio_index_topk_hip_extend(
        backend, layer, None, None, batch.pos, forward_batch, row_shard=row_shard
    )
    return core.page, core.raw, backend.candidate_masks if kind == "source" else None


class _CaptureShares:
    """Pass one: records the share each gather is handed; its result is not used."""

    def __init__(self):
        self.shares = []

    def all_gather_into_tensor(self, out, local):
        self.shares.append(local.clone())
        out.fill_(-1)


class _ReplayShares:
    """Pass two: every gather returns all ranks' recorded shares in rank order, as the
    attention-TP all-gather does."""

    def __init__(self, captures):
        self.captures, self.calls = captures, 0

    def all_gather_into_tensor(self, out, local):
        out.copy_(torch.cat([c.shares[self.calls] for c in self.captures]))
        self.calls += 1


def _in_raw_order(page, raw):
    """Each row sorted by raw index, its page indices carried along."""
    raw, order = raw.sort(dim=1)
    return page.gather(1, order), raw


def _row_logits(batch, rows):
    """The rows' full-context logits; the FP4 logits of a row do not depend on the batch."""
    from sglang.srt.layers.attention.dsv4 import low_ratio_backend_hip

    q_fp4, q_scale, weights = batch.inputs
    return low_ratio_backend_hip.aiter_fp4_paged_mqa_logits(
        q_fp4=q_fp4[rows],
        q_scale=q_scale[rows],
        k_payload=batch.pool.get_index_k_fp4_payload_buffer(0),
        k_scale=batch.pool.get_index_k_fp4_scale_buffer(0),
        weights=weights[rows],
        page_table=batch.meta.page_table[rows],
        c4_seq_lens=batch.meta.compressed_seq_lens[rows],
        weight_scale=1.0,
        page_table_bucket=low_ratio_backend_hip.LOW_RATIO_PAGE_TABLE_BUCKET,
        is_decode=False,
        prefill_workspace=None,
    )


def _on_every_rank(batch, kinds, world):
    """Each rank's outputs per layer, every share scored by the code under test."""
    from sglang.srt.layers.attention.dsv4.v41_indexer.types import RowShard

    def shard(rank, group):
        return RowShard(group=group, rank=rank, world=world, row_align=128 // HEADS)

    published = [None] * world
    layers = []
    for kind in kinds:
        captures = [_CaptureShares() for _ in range(world)]
        for rank in range(world):
            _index(batch, kind, published[rank], shard(rank, captures[rank]))
        outs = [
            _index(batch, kind, published[rank], shard(rank, _ReplayShares(captures)))
            for rank in range(world)
        ]
        if kind == "source":
            published = [o[2] for o in outs]
        layers.append(outs)
    return layers


class TestShardPays(CustomTestCase):
    def _pays(self, ratio, seq_lens, extend_lens, tail_lens=None, world=4):
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            indexer_row_shard_pays,
        )

        tail = (
            None
            if tail_lens is None
            else SimpleNamespace(extend_seq_lens_cpu=tail_lens)
        )
        backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(late_layer_tail=tail)
        )
        forward_batch = SimpleNamespace(
            seq_lens_cpu=torch.tensor(seq_lens), extend_seq_lens_cpu=extend_lens
        )
        return indexer_row_shard_pays(backend, ratio, forward_batch, world)

    def test_mean_context_per_row(self):
        # one 16,384-row chunk ending at 16k / 24k / 40k tokens, ratio 2: its rows see
        # 4,096 / 8,192 / 16,384 positions on average
        self.assertFalse(self._pays(2, [16384], [16384]))
        # the chunk's final length alone (12,288 positions) would pass
        self.assertFalse(self._pays(2, [24576], [16384]))
        self.assertTrue(self._pays(2, [40960], [16384]))
        # bounded-replay tails count their own rows, the last of each request
        self.assertTrue(self._pays(1, [92160] * 8, [16384] * 8, tail_lens=[128] * 8))
        # rows weigh in: 15,000 rows at about 4,250 positions and 1,000 at about
        # 45,830 average about 6,850
        self.assertFalse(self._pays(2, [16000, 92160], [15000, 1000]))

    def test_floor_per_tp_size(self):
        # 16,384 rows seeing 16,384 positions on average pay at TP4 and TP8, not at TP2,
        # where a rank saves only half the scoring; 28,672 pay at TP2 too
        self.assertTrue(self._pays(2, [40960], [16384], world=8))
        self.assertTrue(self._pays(2, [40960], [16384], world=4))
        self.assertFalse(self._pays(2, [40960], [16384], world=2))
        self.assertTrue(self._pays(2, [65536], [16384], world=2))

    def test_positions_seen_match_the_rows(self):
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            _compressed_positions_seen,
        )

        for ratio in (1, 2, 4):
            for n in (0, 1, 2, 3, 7, 128, 1000, 4097):
                self.assertEqual(
                    _compressed_positions_seen(n, ratio),
                    sum(q // ratio for q in range(1, n + 1)),
                    (n, ratio),
                )


class TestClipRequests(CustomTestCase):
    def test_matches_reference(self):
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            _clip_requests,
        )

        rng = random.Random(0)
        for _ in range(500):
            lens = [rng.choice([0, 1, 3, 64, 500]) for _ in range(rng.randint(1, 7))]
            total = sum(lens)
            lo = rng.randint(0, total)
            rows = slice(lo, rng.randint(lo, total))
            starts = list(itertools.accumulate([0, *lens]))
            want = [
                max(0, min(e, rows.stop) - max(s, rows.start))
                for s, e in zip(starts, starts[1:])
            ]
            got, req_lo, req_hi, offsets = _clip_requests(lens, rows)
            self.assertEqual(got, want, (lens, rows))
            # the requests with rows in the slice lie inside [req_lo, req_hi)
            self.assertTrue(
                all(n == 0 for n in got[:req_lo] + got[req_hi:]), (lens, rows)
            )
            for b, n in enumerate(got):
                if n:
                    self.assertEqual(starts[b] + offsets[b], max(starts[b], rows.start))


@unittest.skipUnless(
    _aiter_fp4_indexer_available(), "gfx950 with aiter's FlyDSL FP4 indexer"
)
class TestRowShardMatchesReplicated(CustomTestCase):
    def _assert_same_selection(self, batch, got_raw, want_raw):
        # a tie at the k-th logit may be broken either way, also between two replicated calls
        differ = (got_raw != want_raw).any(dim=1).nonzero().squeeze(1)
        if differ.numel() == 0:
            return
        logits = _row_logits(batch, differ)
        for j, i in enumerate(differ.tolist()):
            got = logits[j, got_raw[i][got_raw[i] >= 0].long()].sort().values
            want = logits[j, want_raw[i][want_raw[i] >= 0].long()].sort().values
            self.assertTrue(torch.equal(got, want), f"raw indices, row {i}")

    def _check(self, ratio, seq_lens, extend_lens, kinds, pad_to=0):
        batch = _batch(
            ratio, seq_lens, extend_lens, seed=sum(seq_lens) + ratio, pad_to=pad_to
        )
        replicated, published = [], None
        for kind in kinds:
            page, raw, pub = _index(batch, kind, published, None)
            replicated.append((page.clone(), raw.clone(), pub))
            if kind == "source":
                published = pub
        for world in (2, 3, 4):
            for li, outs in enumerate(_on_every_rank(batch, kinds, world)):
                want = replicated[li]
                want_page, want_raw = _in_raw_order(want[0], want[1])
                for rank, got in enumerate(outs):
                    with self.subTest(world=world, layer=kinds[li], rank=rank):
                        got_page, got_raw = _in_raw_order(got[0], got[1])
                        self._assert_same_selection(batch, got_raw, want_raw)
                        same = (got_raw == want_raw).all(dim=1)
                        self.assertTrue(
                            torch.equal(got_page[same], want_page[same]), "page indices"
                        )
                        if want[2] is None:
                            self.assertIsNone(got[2])
                            continue
                        self.assertEqual(
                            [c is None for c in got[2]], [c is None for c in want[2]]
                        )
                        for x, y in zip(got[2], want[2]):
                            if y is None:
                                continue
                            self.assertTrue(
                                torch.equal(
                                    x.ids.sort(dim=1).values, y.ids.sort(dim=1).values
                                ),
                                "ids",
                            )
                            for field in ("compact_lens", "compact_page_table"):
                                self.assertTrue(
                                    torch.equal(getattr(x, field), getattr(y, field)),
                                    field,
                                )
                            self.assertEqual(x.compact_page_size, y.compact_page_size)

    def test_plain_layer(self):
        self._check(2, [20000], [2048], ["plain"])

    def test_candidate_source_then_consumer(self):
        self._check(1, [20000], [2048], ["source", "consumer"])

    def test_requests_cut_by_shares_and_identity_requests(self):
        # requests 1 and 3 fit index_topk (identity); the shares cut requests 0 and 2
        self._check(
            1, [6000, 300, 9000, 512], [1200, 300, 600, 512], ["source", "consumer"]
        )
        self._check(2, [6000, 1024, 9000], [1200, 1024, 600], ["plain"])

    def test_ragged_requests_with_an_empty_one(self):
        # 517 rows, just above the 512-row floor, over short requests that the shares
        # cut, one of them empty
        self._check(
            1, [1024, 768, 896, 640], [129, 0, 193, 195], ["source", "consumer"]
        )

    def test_rows_past_the_requests(self):
        # the shares are cut from every row the call has, so a padded batch's shares
        # are wider than its requests' rows alone would give
        self._check(2, [20000], [1500], ["plain"], pad_to=2048)
        self._check(1, [9000, 5000], [900, 600], ["source", "consumer"], pad_to=2048)


if __name__ == "__main__":
    unittest.main()
