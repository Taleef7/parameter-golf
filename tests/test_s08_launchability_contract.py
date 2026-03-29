from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import audit_submission_launchability as audit


class SubmissionLaunchabilityContractTests(unittest.TestCase):
    def make_script(
        self,
        directory: Path,
        *,
        extra_imports: list[str] | None = None,
        data_default: str = "./data/datasets/fineweb10B_sp1024",
        tokenizer_default: str = "./data/tokenizers/fineweb_1024_bpe.model",
        name: str = "train_gpt.py",
    ) -> Path:
        imports = "\n".join(extra_imports or [])
        body = f"""from __future__ import annotations
import os
{imports}

class Hyperparameters:
    data_path = os.environ.get(\"DATA_PATH\", \"{data_default}\")
    tokenizer_path = os.environ.get(\"TOKENIZER_PATH\", \"{tokenizer_default}\")
"""
        path = directory / name
        path.write_text(body, encoding="utf-8")
        return path

    def make_downloader(self, directory: Path, *, imports: list[str] | None = None) -> Path:
        body = "\n".join(imports or ["from huggingface_hub import hf_hub_download", "hf_hub_download"])
        path = directory / "cached_challenge_fineweb.py"
        path.write_text(body + "\n", encoding="utf-8")
        return path

    def make_log(
        self,
        directory: Path,
        *,
        name: str = "seed.log",
        world_size: int = 8,
        stopping_reason: str = "wallclock_cap",
        train_time_ms: int = 600_128,
        include_world_size: bool = True,
        include_stopping_line: bool = True,
        include_legal_ttt_eval: bool = True,
        include_sliding_eval: bool = True,
        sliding_metric: str = "final_int6_sliding_window",
        sliding_eval_time_ms: int = 600_000,
        legal_ttt_eval_time_ms: int = 600_000,
        size_bytes: int = 15_990_006,
        tokenizer_path: str = "/root/parameter-golf/data/tokenizers/fineweb_1024_bpe.model",
        dataset_path: str = "/root/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin",
        extra_lines: list[str] | None = None,
    ) -> Path:
        lines = [
            "logs/example.txt",
            f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={tokenizer_path}",
            "train_loader:dataset:fineweb10B_sp1024 train_shards:80",
            f"val_loader:shards pattern={dataset_path} tokens:62021632",
        ]
        if include_world_size:
            lines.append(f"world_size:{world_size} grad_accum_steps:1")
        lines.extend(
            [
                "train_batch_tokens:786432 train_seq_len:2048 iterations:9000 warmup_steps:20 max_wallclock_seconds:600.000",
                "seed:1337",
                f"step:7179/9000 val_loss:1.9214 val_bpb:1.1379 train_time:{train_time_ms}ms step_avg:83.59ms",
            ]
        )
        if include_stopping_line:
            lines.append(
                f"stopping_early: {stopping_reason} train_time:{train_time_ms}ms step:7179/9000"
            )
        lines.extend(
            [
                "Total submission size int6+lzma: 15990006 bytes" if size_bytes is not None else "",
                "legal_ttt_exact val_loss:1.88976776 val_bpb:1.11922988",
            ]
        )
        if include_sliding_eval:
            lines.append(
                f"{sliding_metric} val_loss:1.8939 val_bpb:1.1217 stride:64 eval_time:{sliding_eval_time_ms}ms"
            )
        if include_legal_ttt_eval:
            lines.append(
                f"legal_ttt val_loss:1.8898 val_bpb:1.1192 eval_time:{legal_ttt_eval_time_ms}ms"
            )
        if size_bytes is not None:
            lines.append(f"Total submission size int6+lzma: {size_bytes} bytes")
        if extra_lines:
            lines.extend(extra_lines)
        path = directory / name
        path.write_text("\n".join(line for line in lines if line) + "\n", encoding="utf-8")
        return path

    def test_run_audit_reuses_submission_package_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            script = self.make_script(tmpdir)
            downloader = self.make_downloader(tmpdir)
            log_path = self.make_log(tmpdir)
            package_payload = {
                "accepted_metric_fallbacks": ["legal_ttt", "final_int6_sliding_window_s64"],
                "seed_audits": [
                    {
                        "log_path": str(log_path),
                        "chosen_metric": "legal_ttt",
                        "val_bpb": 1.1192,
                        "total_submission_bytes": 15_990_006,
                    }
                ],
                "aggregate": {"seed_count": 1},
                "provenance": {"scripts_match": True},
            }
            with mock.patch.object(audit.package_audit, "run_audit", return_value=package_payload) as patched:
                payload = audit.run_audit(
                    script=script,
                    logs=[log_path],
                    proven_script=script,
                    downloader_control=downloader,
                )

        patched.assert_called_once_with([log_path], promoted_script=script, proven_script=script)
        self.assertEqual(payload["package_audit"], package_payload)
        self.assertEqual(payload["seed_launchability"][0]["total_submission_bytes"], 15_990_006)

    def test_run_audit_accepts_known_wallclock_overshoot_and_exact_eval_caps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            script = self.make_script(tmpdir)
            downloader = self.make_downloader(tmpdir)
            logs = [
                self.make_log(tmpdir, name="seed1337.log", train_time_ms=600_128, legal_ttt_eval_time_ms=600_000, sliding_eval_time_ms=600_000),
                self.make_log(tmpdir, name="seed42.log", train_time_ms=600_110, legal_ttt_eval_time_ms=408_049, sliding_eval_time_ms=97_749),
                self.make_log(tmpdir, name="seed2025.log", train_time_ms=600_128, legal_ttt_eval_time_ms=407_984, sliding_eval_time_ms=73_676),
            ]

            payload = audit.run_audit(
                script=script,
                logs=logs,
                proven_script=script,
                downloader_control=downloader,
            )

        self.assertEqual(payload["launchability_aggregate"]["seed_count"], 3)
        self.assertEqual(payload["launchability_aggregate"]["max_train_time_overshoot_ms"], 128)
        self.assertEqual(payload["launchability_aggregate"]["max_legal_ttt_eval_time_ms"], 600_000)
        self.assertEqual(payload["launchability_aggregate"]["max_sliding_eval_time_ms"], 600_000)
        self.assertTrue(payload["script_launchability"]["no_network_proof"])
        self.assertEqual(payload["seed_launchability"][0]["sliding_eval_metric"], "final_int6_sliding_window")

    def test_missing_world_size_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            script = self.make_script(tmpdir)
            downloader = self.make_downloader(tmpdir)
            log_path = self.make_log(tmpdir, include_world_size=False)
            with self.assertRaisesRegex(audit.SubmissionLaunchabilityAuditError, "missing world_size line"):
                audit.run_audit(script=script, logs=[log_path], proven_script=script, downloader_control=downloader)

    def test_missing_stopping_line_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            script = self.make_script(tmpdir)
            downloader = self.make_downloader(tmpdir)
            log_path = self.make_log(tmpdir, include_stopping_line=False)
            with self.assertRaisesRegex(audit.SubmissionLaunchabilityAuditError, "missing stopping_early line"):
                audit.run_audit(script=script, logs=[log_path], proven_script=script, downloader_control=downloader)

    def test_missing_eval_timing_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            script = self.make_script(tmpdir)
            downloader = self.make_downloader(tmpdir)
            log_path = self.make_log(
                tmpdir,
                include_legal_ttt_eval=False,
                extra_lines=["legal_ttt val_loss:1.8898 val_bpb:1.1192"],
            )
            with self.assertRaisesRegex(audit.SubmissionLaunchabilityAuditError, "missing numeric legal_ttt eval_time line"):
                audit.run_audit(script=script, logs=[log_path], proven_script=script, downloader_control=downloader)

    def test_network_imports_in_promoted_script_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            script = self.make_script(tmpdir, extra_imports=["import requests"])
            downloader = self.make_downloader(tmpdir)
            log_path = self.make_log(tmpdir)
            with self.assertRaisesRegex(audit.SubmissionLaunchabilityAuditError, "unsupported network imports"):
                audit.run_audit(script=script, logs=[log_path], proven_script=script, downloader_control=downloader)

    def test_missing_local_script_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            script = self.make_script(tmpdir, tokenizer_default="https://huggingface.co/tokenizer.model")
            downloader = self.make_downloader(tmpdir)
            log_path = self.make_log(tmpdir)
            with self.assertRaisesRegex(audit.SubmissionLaunchabilityAuditError, "TOKENIZER_PATH must be local-only"):
                audit.run_audit(script=script, logs=[log_path], proven_script=script, downloader_control=downloader)

    def test_wallclock_overshoot_beyond_tolerance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            script = self.make_script(tmpdir)
            downloader = self.make_downloader(tmpdir)
            log_path = self.make_log(tmpdir, train_time_ms=601_500)
            with self.assertRaisesRegex(audit.SubmissionLaunchabilityAuditError, "wallclock overshoot exceeds tolerance"):
                audit.run_audit(script=script, logs=[log_path], proven_script=script, downloader_control=downloader)


if __name__ == "__main__":
    unittest.main()
