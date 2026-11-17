"""Unit tests for QuickReduce ahead of the custom all-reduce (SGLANG_QUICK_REDUCE_FIRST_MIN_BYTES)."""

import os
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

import torch

import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as ops
from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.device_communicators.quick_all_reduce import (
    MB,
    QuickAllReduce,
    QuickReduceRegime,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_FIRST_MIN = "SGLANG_QUICK_REDUCE_FIRST_MIN_BYTES"
_FIRST_MAX = "SGLANG_QUICK_REDUCE_FIRST_MAX_BYTES"


def _quick_reduce(flydsl: bool = True) -> QuickAllReduce:
    quick_reduce = QuickAllReduce.__new__(QuickAllReduce)
    quick_reduce.disabled = False
    quick_reduce._flydsl_int4 = Mock() if flydsl else None
    # 0: no HIP handle, so close() at garbage collection destroys nothing
    quick_reduce._ptr = 0
    quick_reduce.world_size = 4
    quick_reduce.qr_quant_level = QuickReduceRegime.INT4
    quick_reduce.use_fp16_kernels = 1
    quick_reduce.qr_max_size = 2048 * MB
    return quick_reduce


def _rows(n: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """n rows of a 5120-wide hidden state: n x 10 KiB in bf16."""
    return torch.empty(n, 5120, dtype=dtype)


def _window(first_min: int, first_max: int = 0):
    return patch.dict(
        os.environ, {_FIRST_MIN: str(first_min), _FIRST_MAX: str(first_max)}
    )


def _coordinator(quick_reduce: QuickAllReduce) -> parallel_state.GroupCoordinator:
    coordinator = parallel_state.GroupCoordinator.__new__(
        parallel_state.GroupCoordinator
    )
    coordinator.world_size = 4
    coordinator.unique_name = "test"
    coordinator.hpu_communicator = None
    coordinator.xpu_communicator = None
    coordinator.pymscclpp_comm = None
    coordinator.torch_symm_mem_comm = None
    coordinator.pcie_ipc_comm = None
    coordinator._fi_workspace_hint = None
    coordinator.qr_comm = quick_reduce
    coordinator.ca_comm = Mock(disabled=False)
    coordinator.ca_comm.should_custom_ar.return_value = True
    return coordinator


class TestQuickReduceFirstWindow(CustomTestCase):
    def test_off_unless_the_minimum_is_set(self):
        quick_reduce = _quick_reduce()
        with patch.dict(os.environ):
            os.environ.pop(_FIRST_MIN, None)
            self.assertFalse(quick_reduce.should_quick_allreduce_first(_rows(192)))
        with _window(0):
            self.assertFalse(quick_reduce.should_quick_allreduce_first(_rows(192)))

    def test_window_bounds_and_codec_minimum(self):
        """DSpark verify at c32 is 192 rows (1.875 MiB), below INT4's 2 MiB minimum."""
        quick_reduce = _quick_reduce()
        with _window(1 * MB, 4 * MB):
            self.assertFalse(quick_reduce.should_quick_allreduce_first(_rows(64)))
            self.assertTrue(quick_reduce.should_quick_allreduce_first(_rows(192)))
            self.assertTrue(quick_reduce.should_quick_allreduce_first(_rows(384)))
            self.assertFalse(quick_reduce.should_quick_allreduce_first(_rows(768)))
            # the fallback rule after the custom all-reduce keeps the codec minimum
            self.assertFalse(quick_reduce.should_quick_allreduce(_rows(192)))
            self.assertTrue(quick_reduce.should_quick_allreduce(_rows(384)))

    def test_no_maximum_means_the_quick_reduce_max_size(self):
        quick_reduce = _quick_reduce()
        with _window(1 * MB):
            self.assertTrue(quick_reduce.should_quick_allreduce_first(_rows(1536)))
            quick_reduce.qr_max_size = 8 * MB
            self.assertFalse(quick_reduce.should_quick_allreduce_first(_rows(1536)))

    def test_window_keeps_the_kernel_input_rules(self):
        quick_reduce = _quick_reduce()
        with _window(1 * MB):
            self.assertFalse(
                quick_reduce.should_quick_allreduce_first(_rows(192, torch.float16))
            )
            self.assertFalse(
                quick_reduce.should_quick_allreduce_first(_rows(384)[:, :2560])
            )
            quick_reduce.disabled = True
            self.assertFalse(quick_reduce.should_quick_allreduce_first(_rows(192)))

    def test_hip_backend_asks_the_ops_which_dtypes_it_reduces(self):
        quick_reduce = _quick_reduce(flydsl=False)
        with (
            _window(1 * MB),
            patch.object(
                ops, "qr_supports", return_value=False, create=True
            ) as supports,
        ):
            self.assertFalse(quick_reduce.should_quick_allreduce_first(_rows(192)))
            supports.assert_called_once_with(0, torch.bfloat16)


class TestQuickReduceFirstRouting(CustomTestCase):
    def test_resolution_order(self):
        coordinator = _coordinator(_quick_reduce())
        with patch.object(
            parallel_state,
            "is_in_tc_piecewise_cuda_graph",
            return_value=False,
            create=True,
        ):
            with _window(0):
                self.assertEqual(
                    coordinator._resolve_outplace_all_reduce_method(_rows(192)), "ca"
                )
            with _window(1 * MB, 4 * MB):
                self.assertEqual(
                    coordinator._resolve_outplace_all_reduce_method(_rows(192)), "qr"
                )
                self.assertEqual(
                    coordinator._resolve_outplace_all_reduce_method(_rows(64)), "ca"
                )
                self.assertEqual(
                    coordinator._resolve_outplace_all_reduce_method(_rows(768)), "ca"
                )

    def test_all_reduce_skips_symmetric_memory_inside_the_window(self):
        coordinator = _coordinator(_quick_reduce())
        pynccl = Mock()
        pynccl.change_state.return_value = nullcontext()
        coordinator.get_available_pynccl_comm = Mock(return_value=pynccl)
        coordinator.is_symmetric_memory_enabled = Mock(return_value=True)
        coordinator.debug_check_symmetric_mempool = Mock()
        coordinator.ca_comm.should_custom_ar.return_value = False
        input_ = _rows(192)
        quick_output = object()
        with (
            _window(1 * MB, 4 * MB),
            patch.object(input_.__class__, "is_cpu", new=False),
            patch.object(
                parallel_state.torch.compiler, "is_compiling", return_value=False
            ),
            patch.object(
                parallel_state, "outplace_all_reduce", return_value=quick_output
            ) as outplace,
        ):
            self.assertIs(coordinator.all_reduce(input_), quick_output)
        outplace.assert_called_once_with(
            input_, group_name="test", outplace_all_reduce_method="qr"
        )
        pynccl.all_reduce.assert_not_called()


class TestJitQuickAllReduceWiring(CustomTestCase):
    def test_default_ops_are_sgl_kernel_and_take_every_dtype(self):
        self.assertFalse(ops._uses_jit_quick_all_reduce())
        self.assertNotIn("quick_all_reduce_hip", ops.init_custom_qr.__module__)
        self.assertTrue(ops.qr_supports(0, torch.float16))


if __name__ == "__main__":
    unittest.main()
