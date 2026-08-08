import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scatter = _load_module(
    "_dsv4_dcp_scatter_under_test",
    REPO_ROOT / "python/sglang/srt/disaggregation/dcp_scatter.py",
)
build_dsv4_physical_row_transfer_plan = scatter.build_dsv4_physical_row_transfer_plan
build_dsv4_replicated_page_transfer_plan = (
    scatter.build_dsv4_replicated_page_transfer_plan
)
dsv4_kv_descriptor_specs = scatter.dsv4_kv_descriptor_specs


def _load_common_utils_without_sglang_import():
    module_names = [
        "sglang",
        "sglang.srt",
        "sglang.srt.observability",
        "sglang.srt.observability.trace",
    ]
    originals = {name: sys.modules.get(name) for name in module_names}
    try:
        for name in module_names[:-1]:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        trace = types.ModuleType(module_names[-1])

        class TraceNullContext:
            pass

        trace.TraceNullContext = TraceNullContext
        trace.TraceReqContext = TraceNullContext
        sys.modules[module_names[-1]] = trace
        return _load_module(
            "_dcp_common_utils_under_test",
            REPO_ROOT / "python/sglang/srt/disaggregation/common/utils.py",
        )
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


build_dcp_token_transfer_plan = (
    _load_common_utils_without_sglang_import().build_dcp_token_transfer_plan
)


class _MutableEnv:
    def __init__(self, value=False):
        self.value = value

    def get(self):
        return self.value


def _load_pd_hook_without_sglang_import():
    module_names = ["sglang", "sglang.srt", "sglang.srt.environ"]
    originals = {name: sys.modules.get(name) for name in module_names}
    try:
        for name in module_names[:-1]:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        environ = types.ModuleType(module_names[-1])
        environ.envs = types.SimpleNamespace(
            SGLANG_DISAGG_STAGING_BUFFER=_MutableEnv(),
            SGLANG_RUST_SERVER=_MutableEnv(),
        )
        sys.modules[module_names[-1]] = environ
        return _load_module(
            "_pd_disaggregation_hook_under_test",
            REPO_ROOT / "python/sglang/srt/arg_groups/pd_disaggregation_hook.py",
        )
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


pd_disaggregation_hook = _load_pd_hook_without_sglang_import()


def _pd_args(**overrides):
    values = {
        "disaggregation_transfer_backend": "mori",
        "disaggregation_ib_device": None,
        "disaggregation_mode": "decode",
        "dcp_size": 8,
        "disaggregation_decode_enable_radix_cache": False,
        "enable_hierarchical_cache": False,
        "enable_hisparse": False,
        "speculative_algorithm": None,
        "disable_radix_cache": False,
        "max_running_requests": 16,
        "dp_size": 1,
        "disaggregation_decode_extra_slots": None,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _chunks(total_pages: int, pages_per_chunk: int):
    for start in range(0, total_pages, pages_per_chunk):
        yield start, min(start + pages_per_chunk, total_pages)


def test_dsv4_descriptor_layout_matches_pool_order():
    ratios = [0, 4, 128, 4, 128, 0]
    specs = dsv4_kv_descriptor_specs(ratios)
    assert [(spec.kind, spec.compression_ratio) for spec in specs] == [
        ("main", 4),
        ("main", 4),
        ("indexer", 4),
        ("indexer", 4),
        ("main", 128),
        ("main", 128),
    ]

    pp_specs = dsv4_kv_descriptor_specs(ratios, start_layer=1, end_layer=4)
    assert [(spec.kind, spec.compression_ratio) for spec in pp_specs] == [
        ("main", 4),
        ("main", 4),
        ("indexer", 4),
        ("indexer", 4),
        ("main", 128),
    ]


@pytest.mark.parametrize("compression_ratio", [4, 128])
def test_dsv4_70k_multichunk_rows_are_owned_once_and_in_bounds(compression_ratio):
    page_size = 256
    dcp_size = 8
    total_tokens = 70_000
    total_pages = math.ceil(total_tokens / page_size)
    pages_per_chunk = 8192 // page_size

    # DSV4's hybrid-SWA allocator keeps the raw transfer page size unchanged.
    # Physical sharding happens inside each page at compressed-row granularity.
    src_pages = np.arange(1000, 1000 + total_pages, dtype=np.int32)
    dst_pages = np.arange(2000, 2000 + total_pages, dtype=np.int32)
    assert src_pages.size == dst_pages.size

    rows_per_page = page_size // compression_ratio
    all_src_rows = []
    all_dst_global_rows = []
    later_chunk_counts = []

    expected_src = (
        src_pages[:, None] * rows_per_page + np.arange(rows_per_page, dtype=np.int64)
    ).reshape(-1)
    expected_dst = (
        dst_pages[:, None] * rows_per_page + np.arange(rows_per_page, dtype=np.int64)
    ).reshape(-1)
    expected_src_for_dst = dict(zip(expected_dst.tolist(), expected_src.tolist()))

    dst_row_capacity = math.ceil(
        ((int(dst_pages.max()) + 1) * rows_per_page) / dcp_size
    )
    for start, end in _chunks(total_pages, pages_per_chunk):
        for rank in range(dcp_size):
            plan = build_dsv4_physical_row_transfer_plan(
                src_pages[start:end],
                dst_pages[start:end],
                physical_page_size=page_size,
                compression_ratio=compression_ratio,
                dcp_size=dcp_size,
                dcp_rank=rank,
            )
            assert plan.src_row_indices.size == plan.dst_row_indices.size
            assert plan.src_row_indices.size > 0
            assert np.all(np.diff(plan.dst_row_indices) == 1)
            assert int(plan.dst_row_indices.max()) < dst_row_capacity
            dst_global_rows = plan.dst_row_indices * dcp_size + rank
            np.testing.assert_array_equal(
                plan.src_row_indices,
                np.asarray(
                    [expected_src_for_dst[int(row)] for row in dst_global_rows],
                    dtype=np.int64,
                ),
            )
            if start:
                later_chunk_counts.append(plan.src_row_indices.size)

            all_src_rows.append(plan.src_row_indices)
            all_dst_global_rows.append(dst_global_rows)

    assert later_chunk_counts and min(later_chunk_counts) > 0

    np.testing.assert_array_equal(np.sort(np.concatenate(all_src_rows)), expected_src)
    np.testing.assert_array_equal(
        np.sort(np.concatenate(all_dst_global_rows)), expected_dst
    )


def test_dsv4_c4_indexer_pages_are_replicated_without_remap():
    src_pages = np.array([11, 12, 30, 31], dtype=np.int32)
    dst_pages = np.array([101, 102, 205, 206], dtype=np.int32)

    plan = build_dsv4_replicated_page_transfer_plan(src_pages, dst_pages)
    np.testing.assert_array_equal(plan.src_page_indices, src_pages)
    np.testing.assert_array_equal(plan.dst_page_indices, dst_pages)


def test_generic_dcp_second_prefill_chunk_uses_full_destination_array():
    page_size = 64
    dcp_size = 8
    total_tokens = 51_953
    first_chunk_tokens = 32_768
    total_src_pages = math.ceil(total_tokens / page_size)
    total_dst_pages = math.ceil(total_tokens / (page_size * dcp_size))

    src_pages = np.arange(500, 500 + total_src_pages, dtype=np.int32)
    dst_pages = np.arange(900, 900 + total_dst_pages, dtype=np.int32)
    second_start_page = first_chunk_tokens // page_size
    second_tokens = total_tokens - first_chunk_tokens

    for rank in range(dcp_size):
        plan = build_dcp_token_transfer_plan(
            src_pages[second_start_page:],
            dst_pages,
            physical_page_size=page_size,
            dcp_size=dcp_size,
            dcp_rank=rank,
            src_page_offset=second_start_page,
            decode_prefix_len=0,
            num_kv_tokens=second_tokens,
        )
        assert plan.src_token_indices.size > 0
        assert plan.src_token_indices.size == plan.dst_token_indices.size

        with pytest.raises(ValueError, match="Insufficient destination DCP pages"):
            build_dcp_token_transfer_plan(
                src_pages[second_start_page:],
                dst_pages[second_start_page:],
                physical_page_size=page_size,
                dcp_size=dcp_size,
                dcp_rank=rank,
                src_page_offset=second_start_page,
                decode_prefix_len=0,
                num_kv_tokens=second_tokens,
            )


def test_pd_decode_dcp_allows_mori_and_forces_chunk_cache(monkeypatch):
    monkeypatch.setattr(
        pd_disaggregation_hook.envs.SGLANG_DISAGG_STAGING_BUFFER,
        "value",
        True,
    )
    server_args = _pd_args()

    pd_disaggregation_hook.handle_pd_disaggregation(server_args)

    assert server_args.disable_radix_cache
    assert server_args.disaggregation_decode_extra_slots == 32


def test_pd_decode_dcp_rejects_unknown_transfer_backend():
    server_args = _pd_args(disaggregation_transfer_backend="fake")

    with pytest.raises(ValueError, match="mooncake or nixl or mori"):
        pd_disaggregation_hook.handle_pd_disaggregation(server_args)
