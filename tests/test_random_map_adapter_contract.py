from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
import unittest
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "experiments" / "train_gpt_random_map_adapter.py"
VERIFY_RUN = ROOT / "experiments" / "verify_run.py"
FIXTURE_LOG = ROOT / "tests" / "fixtures" / "sample_random_map_adapter.log"


@lru_cache(maxsize=1)
def load_random_map_module():
    stub = types.ModuleType("flash_attn_interface")
    stub.flash_attn_func = lambda q, k, v, causal=True: q.new_zeros(q.shape)
    sys.modules.setdefault("flash_attn_interface", stub)

    spec = importlib.util.spec_from_file_location("train_gpt_random_map_adapter_test", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec for {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RandomMapAdapterContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_random_map_module()
        cls.source = SCRIPT_PATH.read_text(encoding="utf-8")

    def test_source_declares_required_env_vars_and_logging_fields(self) -> None:
        for token in [
            "RANDOM_MAP_ADAPTER_ENABLED",
            "RANDOM_MAP_ADAPTER_RANK",
            "RANDOM_MAP_ADAPTER_LAYERS",
            "RANDOM_MAP_ADAPTER_TARGETS",
            "RANDOM_MAP_ADAPTER_SEED",
            "RANDOM_MAP_ADAPTER_SCALE_INIT",
            "random_map_adapter:",
            "random_map_adapter_params:",
            "random_map_buffer_params:",
        ]:
            self.assertIn(token, self.source)

    def test_disabled_config_defaults_to_adapter_off_parity(self) -> None:
        args = SimpleNamespace(
            random_map_adapter_enabled=False,
            random_map_adapter_rank=8,
            random_map_adapter_layers="0,1",
            random_map_adapter_targets="q,v",
            random_map_adapter_seed=1729,
            random_map_adapter_scale_init=0.0,
            num_layers=4,
        )
        config = self.module.build_random_map_adapter_config(args)
        self.assertFalse(config.enabled)
        self.assertEqual(config.rank, 0)
        self.assertEqual(config.layer_indices, ())
        self.assertEqual(config.targets, ())

    def test_malformed_config_is_rejected_loudly(self) -> None:
        base = dict(
            random_map_adapter_enabled=True,
            random_map_adapter_rank=4,
            random_map_adapter_layers="0",
            random_map_adapter_targets="q,v",
            random_map_adapter_seed=1729,
            random_map_adapter_scale_init=0.0,
            num_layers=4,
        )
        with self.assertRaisesRegex(ValueError, "RANDOM_MAP_ADAPTER_RANK"):
            self.module.build_random_map_adapter_config(SimpleNamespace(**(base | {"random_map_adapter_rank": 0})))
        with self.assertRaisesRegex(ValueError, "RANDOM_MAP_ADAPTER_LAYERS"):
            self.module.build_random_map_adapter_config(SimpleNamespace(**(base | {"random_map_adapter_layers": "0,,1"})))
        with self.assertRaisesRegex(ValueError, "out of range"):
            self.module.build_random_map_adapter_config(SimpleNamespace(**(base | {"random_map_adapter_layers": "4"})))
        with self.assertRaisesRegex(ValueError, "unsupported target"):
            self.module.build_random_map_adapter_config(SimpleNamespace(**(base | {"random_map_adapter_targets": "q,k"})))

    def _build_model(self, enabled: bool):
        config = self.module.RandomMapAdapterConfig(
            enabled=enabled,
            rank=4 if enabled else 0,
            layer_indices=(0,) if enabled else (),
            targets=("q", "v") if enabled else (),
            seed=17,
            scale_init=0.0,
        )
        return self.module.GPT(
            vocab_size=64,
            num_layers=2,
            model_dim=16,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.005,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.5,
            mtp_num_heads=0,
            mtp_loss_weight=0.0,
            bigram_vocab_size=0,
            bigram_dim=8,
            xsa_last_n=0,
            rope_dims=0,
            ln_scale=False,
            dtg=False,
            ve_enabled=False,
            ve_dim=8,
            ve_layers="",
            gated_attention=False,
            value_residual=False,
            random_map_adapter_config=config,
        )

    def test_enabled_model_keeps_random_maps_as_buffers_and_learned_parts_as_parameters(self) -> None:
        model = self._build_model(enabled=True)
        buffer_names = {name for name, _ in model.named_buffers()}
        param_names = {name for name, _ in model.named_parameters()}
        self.assertIn("blocks.0.attn.q_adapter.random_map", buffer_names)
        self.assertIn("blocks.0.attn.v_adapter.random_map", buffer_names)
        self.assertNotIn("blocks.0.attn.q_adapter.random_map", param_names)
        self.assertNotIn("blocks.0.attn.v_adapter.random_map", param_names)
        self.assertIn("blocks.0.attn.q_adapter.up_proj.weight", param_names)
        self.assertIn("blocks.0.attn.v_adapter.up_proj.weight", param_names)
        self.assertIn("blocks.0.attn.q_adapter.scale", param_names)
        self.assertGreater(model.random_map_adapter_parameter_count, 0)
        self.assertGreater(model.random_map_adapter_buffer_count, 0)

    def test_disabled_model_has_no_adapter_buffers_or_parameters(self) -> None:
        model = self._build_model(enabled=False)
        self.assertEqual(model.random_map_adapter_parameter_count, 0)
        self.assertEqual(model.random_map_adapter_buffer_count, 0)
        self.assertFalse(any(name.endswith(".random_map") for name, _ in model.named_buffers()))
        self.assertFalse(any(".q_adapter." in name or ".v_adapter." in name for name, _ in model.named_parameters()))

    def test_source_routes_adapter_parameters_through_replicated_adamw_path(self) -> None:
        self.assertIn('adapter_params = [', self.source)
        self.assertIn('scalar_params.extend(adapter_params)', self.source)
        self.assertIn('".q_adapter." in name or ".v_adapter." in name', self.source)

    def test_minimal_enabled_config_survives_unbank_and_quantization_bookkeeping(self) -> None:
        model = self._build_model(enabled=True)
        state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        unbanked = self.module._unbank_state_dict(state_dict, num_layers=2)
        self.assertIn("blocks.0.attn.q_adapter.random_map", unbanked)
        self.assertIn("blocks.0.attn.v_adapter.up_proj.weight", unbanked)
        quantized, meta = self.module.mixed_quantize_int6(unbanked, {"mlp", "attn"})
        self.assertIn("blocks.0.attn.q_adapter.random_map", quantized)
        self.assertIn("blocks.0.attn.v_adapter.up_proj.weight", quantized)
        self.assertEqual(meta["blocks.0.attn.q_adapter.random_map"], "passthrough")

    def test_fixture_log_preserves_verify_run_contract(self) -> None:
        result = subprocess.run(
            ["python", str(VERIFY_RUN), str(FIXTURE_LOG)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("chosen_metric: final_int6_sliding_window_s64", result.stdout)
        self.assertIn("val_bpb: 1.1500", result.stdout)


if __name__ == "__main__":
    unittest.main()
