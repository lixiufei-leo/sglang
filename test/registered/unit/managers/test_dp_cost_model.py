import unittest
from types import SimpleNamespace

from sglang.srt.managers.dp_cost_model import DeepSeekV4PrefillCostModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _model_config():
    return SimpleNamespace(
        hf_text_config=SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"],
            index_n_heads=4,
            index_head_dim=128,
            index_topk=16,
            kv_lora_rank=512,
        ),
        compress_ratios=[4, 4, 128, 128, 0],
        num_attention_heads=8,
        num_hidden_layers=4,
        head_dim=512,
        qk_rope_head_dim=64,
        v_head_dim=512,
        window_size=8,
    )


def _model_config_with_swa_only_layer():
    config = _model_config()
    config.compress_ratios = [4, 128, 0]
    config.num_hidden_layers = 3
    return config


def _model(**overrides):
    kwargs = {
        "attn_tp_size": 2,
        "attention_tflops_per_gpu": 1000.0,
        "h2d_bandwidth_gbps": 100.0,
        "storage_bandwidth_gbps": 25.0,
    }
    kwargs.update(overrides)
    return DeepSeekV4PrefillCostModel.from_model_config(_model_config(), **kwargs)


class TestDeepSeekV4PrefillCostModel(CustomTestCase):
    def test_reads_layer_mix_from_checkpoint_config_and_roundtrips_spec(self):
        model = _model()
        self.assertEqual(model.num_csa_layers, 2)
        self.assertEqual(model.num_hca_layers, 2)
        self.assertEqual(model.num_swa_only_layers, 0)
        self.assertEqual(model.qk_head_dim, 512)
        self.assertEqual(model.kv_cache_bytes_per_slot, 584)

        restored = DeepSeekV4PrefillCostModel.from_spec(model.to_spec())
        self.assertEqual(restored.to_spec(), model.to_spec())

    def test_supports_swa_only_layer_and_ignores_nextn_padding(self):
        model = DeepSeekV4PrefillCostModel.from_model_config(
            _model_config_with_swa_only_layer(),
            attn_tp_size=2,
            attention_tflops_per_gpu=1000.0,
            h2d_bandwidth_gbps=100.0,
            storage_bandwidth_gbps=25.0,
        )
        self.assertEqual(model.num_csa_layers, 1)
        self.assertEqual(model.num_hca_layers, 1)
        self.assertEqual(model.num_swa_only_layers, 1)
        self.assertGreater(
            model.estimate(input_tokens=64).swa_attention_seconds,
            0,
        )

    def test_cache_bytes_match_separate_and_unified_layouts(self):
        separate = _model()
        unified = _model(unified_kv=True)

        self.assertEqual(separate.kv_cache_bytes_per_slot, 584)
        self.assertEqual(unified.kv_cache_bytes_per_slot, 1024)
        self.assertEqual(separate.full_cache_bytes_per_input_token, 367.125)
        self.assertEqual(unified.full_cache_bytes_per_input_token, 594.0)
        self.assertEqual(separate.swa_cache_bytes_per_input_token, 2336)
        self.assertEqual(unified.swa_cache_bytes_per_input_token, 4096)

    def test_all_attention_components_are_modeled(self):
        estimate = _model().estimate(input_tokens=64)
        self.assertGreater(estimate.csa_indexer_seconds, 0)
        self.assertGreater(estimate.csa_attention_seconds, 0)
        self.assertGreater(estimate.hca_attention_seconds, 0)
        self.assertGreater(estimate.swa_attention_seconds, 0)
        self.assertAlmostEqual(
            estimate.total_seconds,
            estimate.attention_seconds + estimate.prefetch_seconds,
        )

    def test_cached_context_changes_attention_work_for_same_query_count(self):
        model = _model()
        no_context = model.estimate(input_tokens=1)
        long_context = model.estimate(
            input_tokens=129,
            cached_context_tokens=128,
        )
        self.assertGreater(
            long_context.csa_indexer_seconds,
            no_context.csa_indexer_seconds,
        )
        self.assertGreater(
            long_context.hca_attention_seconds,
            no_context.hca_attention_seconds,
        )

    def test_cache_hit_reduces_compute_for_same_prompt(self):
        model = _model()
        cold = model.estimate(input_tokens=128)
        warm = model.estimate(input_tokens=128, cached_context_tokens=96)
        self.assertLess(warm.attention_seconds, cold.attention_seconds)

    def test_storage_and_h2d_bandwidths_are_independent(self):
        slow = _model(storage_bandwidth_gbps=25.0).estimate(
            input_tokens=101,
            cached_context_tokens=100,
            host_cache_tokens=100,
            storage_cache_tokens=100,
        )
        fast = _model(storage_bandwidth_gbps=50.0).estimate(
            input_tokens=101,
            cached_context_tokens=100,
            host_cache_tokens=100,
            storage_cache_tokens=100,
        )
        self.assertEqual(slow.h2d_seconds, fast.h2d_seconds)
        self.assertAlmostEqual(
            slow.storage_prefetch_seconds,
            2 * fast.storage_prefetch_seconds,
        )

    def test_swa_loadback_is_capped_to_window(self):
        model = _model()
        window = model.estimate(
            input_tokens=129,
            cached_context_tokens=128,
            swa_host_cache_tokens=8,
        )
        beyond_window = model.estimate(
            input_tokens=129,
            cached_context_tokens=128,
            swa_host_cache_tokens=128,
        )
        self.assertEqual(window.h2d_seconds, beyond_window.h2d_seconds)

    def test_rejects_non_dsv4_config(self):
        config = _model_config()
        config.hf_text_config.architectures = ["DeepseekV3ForCausalLM"]
        with self.assertRaisesRegex(ValueError, "supports only DeepSeek-V4"):
            DeepSeekV4PrefillCostModel.from_model_config(
                config,
                attn_tp_size=1,
                attention_tflops_per_gpu=1000.0,
                h2d_bandwidth_gbps=100.0,
                storage_bandwidth_gbps=25.0,
            )


if __name__ == "__main__":
    unittest.main()
