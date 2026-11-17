import sys
import types
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.models.deepseek_common.amd import deepseek_v4_hip
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _cfg(*, mult=4, hidden=5120, cp_prefill=False):
    return mhc.HcConfig(
        mult=mult,
        sinkhorn_iters=20,
        eps=1e-6,
        rms_eps=1e-6,
        hidden=hidden,
        pre_from_prev=True,
        cp_prefill=cp_prefill,
    )


class _Norm:
    def __init__(self):
        self.weight = torch.ones(4)
        self.variance_epsilon = 1e-6


def _sublayer(cfg):
    return mhc.HcSubLayer(
        cfg=cfg,
        fn=torch.zeros(1),
        scale=torch.zeros(1),
        base=torch.zeros(1),
        norm=_Norm(),
    )


class TestMegaMhcGate(unittest.TestCase):
    def test_gate(self):
        # (flag, gfx950, config, expected)
        cases = [
            (False, True, _cfg(), False),
            (True, True, _cfg(), True),
            (True, True, _cfg(mult=2), False),
            (True, True, _cfg(hidden=4096), False),
            (True, True, _cfg(cp_prefill=True), False),
            (True, False, _cfg(), False),
        ]
        for flag, gfx95, cfg, expected in cases:
            with (
                self.subTest(flag=flag, gfx95=gfx95, cfg=cfg),
                envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(flag),
                mock.patch.object(mhc, "_is_hip", True),
                mock.patch.object(mhc, "_is_gfx95_supported", gfx95),
            ):
                self.assertEqual(mhc._can_use_mega_mhc(cfg), expected)


class TestHipFlydslDispatch(unittest.TestCase):
    def _boundary(self, cfg):
        hc = _sublayer(cfg)
        return mhc.HcNextBoundary(
            norm=hc.norm,
            accepts_mxfp8=False,
            norm_fusable=True,
            hc=hc,
        )

    def test_calls_flydsl_bf16_and_packs_next_stats(self):
        cfg = _cfg()
        y = torch.zeros(2, 4)
        residual = torch.zeros(2, 4, 4)
        pre = torch.zeros(2, 4)
        post_mix = torch.zeros(2, 4)
        comb = torch.zeros(2, 4, 4)
        updated = torch.ones(2, 4, 4)
        normalized = torch.full((2, 4), 3.0)
        next_post = torch.full((2, 4, 1), 4.0)
        next_comb = torch.full((2, 4, 4), 5.0)
        next_pre = torch.full((2, 4), 6.0)
        seen = {}

        def fake_flydsl(*args, **kwargs):
            seen["out_dtype"] = kwargs["out_dtype"]
            seen["post_ndim"] = kwargs["post_layer_mix"].ndim
            return updated, next_post, next_comb, normalized, next_pre

        flydsl = types.ModuleType("aiter.ops.flydsl")
        flydsl.flydsl_mega_mhc = fake_flydsl
        ops = types.ModuleType("aiter.ops")
        ops.flydsl = flydsl
        aiter = types.ModuleType("aiter")
        aiter.ops = ops
        with (
            envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(True),
            mock.patch.object(mhc, "_is_hip", True),
            mock.patch.object(mhc, "_is_gfx95_supported", True),
            mock.patch.dict(
                sys.modules,
                {
                    "aiter": aiter,
                    "aiter.ops": ops,
                    "aiter.ops.flydsl": flydsl,
                },
            ),
        ):
            state = mhc._post_fusion(
                _sublayer(cfg), y, residual, (pre, post_mix, comb), self._boundary(cfg)
            )

        self.assertEqual(seen["out_dtype"], "bf16")
        self.assertEqual(seen["post_ndim"], 3)
        self.assertIsInstance(state.input, mhc.HcNormed)
        self.assertTrue(torch.equal(state.input.rows, normalized))
        self.assertTrue(torch.equal(state.streams, updated))
        self.assertEqual(state.stats[0].shape, (2, 4))
        self.assertEqual(state.stats[1].shape, (2, 4))
        self.assertTrue(torch.equal(state.pre, pre))


class TestFusedBoundaryRouting(unittest.TestCase):
    def test_routing(self):
        layer = mock.Mock(hc_pre_from_prev_sublayer=True, hc_mult=4)
        with (
            envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(True),
            mock.patch.object(deepseek_v4_hip, "_is_gfx95_supported", True),
            mock.patch.object(
                deepseek_v4_hip.gfx95_dense,
                "fused_rmsnorm_fp8_quant_eligible",
                return_value=False,
            ),
            mock.patch.object(
                deepseek_v4_hip.gfx95_dense,
                "fused_rmsnorm_fake_quant_eligible",
                return_value=False,
            ),
        ):
            deepseek_v4_hip.init_decoder_layer(layer, quant_config=None)
        # the AMD monokernel refuses a layer without it at attach
        self.assertTrue(layer.hc_boundary_fused)

        prefill = (ForwardMode.EXTEND, ForwardMode.MIXED)
        decode = (
            ForwardMode.DECODE,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DRAFT_EXTEND_V2,
        )
        # (prefill env, decode env, monokernel) -> (fused prefill, fused decode)
        cases = {
            (False, False, False): (True, True),
            (False, False, True): (True, True),
            (False, True, True): (True, True),
            (True, False, False): (False, False),
            (True, False, True): (False, True),
            (True, True, False): (False, False),
            (True, True, True): (False, False),
        }
        for flags, (fused_prefill, fused_decode) in cases.items():
            mega, mega_decode, monokernel = flags
            for modes, expected in ((prefill, fused_prefill), (decode, fused_decode)):
                for mode in modes:
                    with (
                        self.subTest(flags=flags, mode=mode.name),
                        envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(mega),
                        envs.SGLANG_OPT_DSV41_MEGA_MHC_DECODE.override(mega_decode),
                        mock.patch.object(
                            deepseek_v4_hip,
                            "_amd_monokernel_enabled",
                            return_value=monokernel,
                        ),
                    ):
                        fused = deepseek_v4_hip.fused_boundary_this_step(
                            types.SimpleNamespace(forward_mode=mode)
                        )
                        self.assertEqual(fused, expected)


if __name__ == "__main__":
    unittest.main()
