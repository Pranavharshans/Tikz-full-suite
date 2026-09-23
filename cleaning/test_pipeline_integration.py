"""Fault-injection integration tests for the production pipeline.

These tests drive the real controller, ledger, task files and worker loop in
threads with a deterministic fake engine.  No GPU, no network, no vLLM.
"""
import contextlib
import importlib.util
import io
import json
import os
import signal
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "build_dataset", Path(__file__).with_name("build_dataset.py"))
b = importlib.util.module_from_spec(SPEC)
sys.modules["build_dataset"] = b
SPEC.loader.exec_module(b)

from test_build_dataset import FakeEngine, FakePromptBuilder, make_work_fixture  # noqa: E402


class LocalWorkerHandle:
    """Thread-backed worker handle that mimics a process handle."""

    def __init__(self, thread, outcome):
        self.thread = thread
        self.outcome = outcome
        self.started_at = time.time()

    def poll(self):
        if self.thread.is_alive():
            return None
        return self.outcome.get("code", 137)  # died without a clean exit code

    @property
    def returncode(self):
        return self.poll()

    def terminate(self):
        self.outcome["terminated"] = True

    def kill(self):
        self.outcome["killed"] = True

    def wait(self, timeout=None):
        self.thread.join(timeout)
        return self.poll()

    def close(self):
        pass


class LocalSpawner:
    """Runs run_worker in a thread with an injected fake engine."""

    def __init__(self, engine_factory, prompt_builder_factory=None, on_spawn=None):
        self.engine_factory = engine_factory
        self.prompt_builder_factory = prompt_builder_factory or (lambda job: FakePromptBuilder(job))
        self.on_spawn = on_spawn
        self.handles = []

    def __call__(self, *, job_path, worker_index, device, args, work):
        outcome = {}

        def target():
            try:
                outcome["code"] = b.run_worker(
                    job_path, engine_factory=self.engine_factory,
                    prompt_builder_factory=self.prompt_builder_factory)
            except Exception as exc:  # pragma: no cover - defensive
                outcome["error"] = repr(exc)
                outcome["code"] = 99

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        handle = LocalWorkerHandle(thread, outcome)
        self.handles.append(handle)
        if self.on_spawn:
            self.on_spawn(job_path, worker_index)
        return handle


class GatedEngine(FakeEngine):
    """Fake engine that blocks until both workers are mid-batch, then until the stop marker."""

    def __init__(self, job, *, gate_path, barrier, log_path=None):
        super().__init__(job, log_path=log_path)
        self.gate_path = Path(gate_path)
        self.barrier = barrier

    def generate(self, items):
        self.barrier.wait(timeout=15)
        deadline = time.time() + 15
        while not self.gate_path.exists() and time.time() < deadline:
            time.sleep(0.005)
        return super().generate(items)


class LocalDeps(b.PipelineDeps):
    def __init__(self, spawner, gpu_names=None):
        self.spawner = spawner
        self.gpu_names = gpu_names or ["NVIDIA RTX PRO 6000 Blackwell Server Edition"] * 2

    def container_sha256(self, path, work):
        return "b" * 64

    def probe_gpus(self):
        return self.gpu_names

    def spawn_worker(self, **kwargs):
        return self.spawner(**kwargs)

    def sleep(self, seconds):
        time.sleep(min(max(seconds, 0.0), 0.02))


def run_args(work, *extra):
    return b.build_parser().parse_args([
        "run", "--work", str(work), "--allow-non-slurm", "--vllm-sif", "/tmp/vllm.sif",
        "--worker-restarts", "1", "--poll-seconds", "0.01", "--shutdown-grace-seconds", "3",
        "--retry-wait-seconds", "0", "--retry-backoff-base-seconds", "0.05",
        "--retry-backoff-cap-seconds", "0.05", "--warmup-samples", "0", *extra])


@contextlib.contextmanager
def slurm_env():
    with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "4242", "CUDA_VISIBLE_DEVICES": "0,1"}):
        yield


def read_states(work):
    with b.Ledger(work, read_only=True) as ledger:
        counts = ledger.counts()
        complete = ledger.complete_rows()
        attempts = ledger.attempts_for([row["row_id"] for row in complete])
    return counts["states"], complete, attempts


class FaultInjectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)
        self.meta, _rows = make_work_fixture(self.work, rows=12)
        self.entries = list(b.iter_manifest(self.work))
        self.log = self.work / "generated.log"

    def engine_factory(self, **kwargs):
        return lambda job: FakeEngine(job, log_path=self.log, **kwargs)

    def test_crash_mid_batch_resume_preserves_associations(self):
        crash_row = self.entries[5]["row_id"]
        spawner = LocalSpawner(self.engine_factory(crash_rows={crash_row}))
        with slurm_env():
            code = b.cmd_run(run_args(self.work), deps=LocalDeps(spawner))
        self.assertEqual(code, 3, "run 1 must report remaining work after a crash")

        states, complete, _ = read_states(self.work)
        self.assertGreater(states["complete"], 0, "rows outside the crashed batch must finish")
        self.assertGreaterEqual(states["retryable"], 1, "crashed batch rows must be reclaimed")
        self.assertEqual(states["running"], 0)
        self.assertEqual(states["pending"], 0)
        for row in complete:
            self.assertEqual(row["instruction"], f"Instruction for {row['row_id']}")
        run1_complete = {row["row_id"] for row in complete}
        run1_generated = set(self.log.read_text().split())
        self.assertTrue(run1_complete.issubset(run1_generated))
        self.assertNotIn(crash_row, run1_complete)

        # Resume with a healthy engine.
        self.log.write_text("")
        spawner = LocalSpawner(self.engine_factory())
        with slurm_env():
            code = b.cmd_run(run_args(self.work, "--retry-wait-seconds", "30"),
                             deps=LocalDeps(spawner))
        self.assertEqual(code, 0)

        states, complete, _ = read_states(self.work)
        self.assertEqual(states["complete"], 12)
        self.assertEqual(states["retryable"] + states["running"] + states["pending"], 0)
        run2_generated = set(self.log.read_text().split())
        self.assertEqual(run1_complete & run2_generated, set(),
                         "completed rows must never be regenerated")
        self.assertIn(crash_row, run2_generated)
        # Every instruction stays attached to exactly its own row.
        for row in complete:
            self.assertEqual(row["instruction"], f"Instruction for {row['row_id']}")
        ids = [row["row_id"] for row in complete]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {entry["row_id"] for entry in self.entries})
        # The crashed row consumed one transient attempt and was retried.
        with b.Ledger(self.work, read_only=True) as ledger:
            attempts = ledger.attempts_for([crash_row])[crash_row]
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts[0]["state"], "lost")
            self.assertEqual(attempts[1]["state"], "complete")

    def test_graceful_stop_releases_running_rows_without_budget(self):
        # Two rows per batch; a three-party barrier guarantees both workers are
        # inside their first batch before SIGTERM is delivered, so each finishes
        # exactly one batch and the rest is released without consuming budget.
        args = run_args(self.work, "--concurrency", "4")
        config = b.config_from_args(args, manifest_meta=self.meta, container_sha256="b" * 64,
                                    require_pinned=True)
        stop_flag = b.StopFlag()
        gate = self.work / "runtime" / "stop"
        barrier = threading.Barrier(3, timeout=15)

        def deliver_sigterm():
            barrier.wait()
            stop_flag.handle(signal.SIGTERM, None)

        threading.Thread(target=deliver_sigterm, daemon=True).start()
        spawner = LocalSpawner(
            lambda job: GatedEngine(job, gate_path=gate, barrier=barrier, log_path=self.log))
        with slurm_env():
            summary = b.run_pipeline(args, config, self.meta, LocalDeps(spawner), stop_flag)
        self.assertEqual(summary["states"]["complete"], 4)
        self.assertEqual(summary["states"]["retryable"], 8)
        self.assertEqual(summary["states"]["running"], 0)
        self.assertEqual(summary["exit_code"], 3)
        with b.Ledger(self.work, read_only=True) as ledger:
            self.assertEqual(ledger.rejected_rows(), [])
            released = [attempt for attempt in ledger.all_attempts()
                        if attempt["error_category"] == "run_interrupted"]
            self.assertEqual(len(released), 8)
            self.assertTrue(all(attempt["state"] == "lost" for attempt in released))

        # The interrupted rows were not charged a transient attempt: resume succeeds.
        spawner = LocalSpawner(self.engine_factory())
        with slurm_env():
            code = b.cmd_run(run_args(self.work, "--concurrency", "4",
                                      "--retry-wait-seconds", "30"),
                             deps=LocalDeps(spawner))
        self.assertEqual(code, 0)
        states, complete, _ = read_states(self.work)
        self.assertEqual(states["complete"], 12)
        for row in complete:
            self.assertEqual(row["instruction"], f"Instruction for {row['row_id']}")

    def test_completed_rows_are_never_regenerated(self):
        with slurm_env():
            self.assertEqual(b.cmd_run(run_args(self.work), deps=LocalDeps(
                LocalSpawner(self.engine_factory()))), 0)
        self.log.write_text("")
        with slurm_env():
            self.assertEqual(b.cmd_run(run_args(self.work), deps=LocalDeps(
                LocalSpawner(self.engine_factory()))), 0)
        self.assertEqual(self.log.read_text(), "")
        with b.Ledger(self.work, read_only=True) as ledger:
            counts = ledger.counts()
        self.assertEqual(counts["states"]["complete"], 12)
        self.assertEqual(counts["attempts"], 12)

    def test_chunk_controls_preserve_global_identity_and_resume(self):
        with slurm_env():
            code = b.cmd_run(run_args(self.work, "--max-rows-this-run", "3"),
                             deps=LocalDeps(LocalSpawner(self.engine_factory())))
        self.assertEqual(code, 3)
        with b.Ledger(self.work, read_only=True) as ledger:
            counts = ledger.counts()
            first = {row["row_id"] for row in ledger.complete_rows()}
        self.assertEqual(counts["states"]["complete"], 3)
        self.assertEqual(counts["states"]["pending"], 9)
        self.assertEqual(first, {entry["row_id"] for entry in self.entries[:3]})

        # An explicit index window selects exactly that slice and resumes the rest.
        with slurm_env():
            code = b.cmd_run(run_args(self.work, "--start-index", "9", "--end-index", "11"),
                             deps=LocalDeps(LocalSpawner(self.engine_factory())))
        self.assertEqual(code, 3)
        with b.Ledger(self.work, read_only=True) as ledger:
            counts = ledger.counts()
            selected = {row["source_row_index"] for row in ledger.complete_rows()}
        self.assertEqual(selected, set(range(0, 3)) | {9, 10, 11})
        self.assertEqual(counts["states"]["pending"], 6)

        with slurm_env():
            code = b.cmd_run(run_args(self.work), deps=LocalDeps(
                LocalSpawner(self.engine_factory())))
        self.assertEqual(code, 0)
        states, complete, _ = read_states(self.work)
        self.assertEqual(states["complete"], 12)
        self.assertEqual([row["source_row_index"] for row in complete], list(range(12)))

    def test_reprocess_rejected_is_explicit_and_preserves_history(self):
        args = run_args(self.work)
        config = b.config_from_args(args, manifest_meta=self.meta, container_sha256="b" * 64,
                                    require_pinned=True)
        index = b.manifest_index(self.work)
        target = self.entries[2]
        with b.Ledger(self.work).open() as ledger:
            ledger.initialize(b.ledger_identity_from_config(config, self.meta))
            ledger.seed(iter(b.iter_manifest(self.work)), rows_frozen=12)
            policy = b.RetryPolicy(backoff_base_seconds=0)
            claims = ledger.claim(b.balanced_assignment(
                [row for row in ledger.eligible() if row["row_id"] == target["row_id"]],
                index, workers=2), policy=policy)
            owner = next(worker for worker, rows in claims.items() if rows)
            ledger.record_result(dict(
                row_id=target["row_id"], source_row_index=target["source_row_index"],
                worker=owner, max_tokens=256, ok=False, instruction=None, text=None,
                finish_reason=None, prompt_tokens=None, completion_tokens=None,
                error_category="integrity", error_detail="image checksum mismatch",
                image_sha256="0" * 64,
                tikz_sha256=target["tikz_sha256"], model_revision=config.model.revision,
                prompt_sha256=config.prompt.sha256, config_hash=config.identity_sha256()),
                policy=policy)
        # A normal run leaves the rejected row alone.
        with slurm_env():
            self.assertEqual(b.cmd_run(run_args(self.work), deps=LocalDeps(
                LocalSpawner(self.engine_factory()))), 0)
        with b.Ledger(self.work, read_only=True) as ledger:
            self.assertEqual(ledger.get(target["row_id"])["state"], b.STATE_REJECTED)
        # The explicit flag reprocesses it and preserves the original attempt.
        with slurm_env():
            self.assertEqual(b.cmd_run(run_args(self.work, "--reprocess-rejected"),
                                       deps=LocalDeps(LocalSpawner(self.engine_factory()))), 0)
        with b.Ledger(self.work, read_only=True) as ledger:
            row = ledger.get(target["row_id"])
            self.assertEqual(row["state"], b.STATE_COMPLETE)
            self.assertEqual(row["instruction"], f"Instruction for {target['row_id']}")
            attempts = ledger.attempts_for([target["row_id"]])[target["row_id"]]
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts[0]["error_category"], "integrity")

    def test_run_identity_mismatch_refuses_resume(self):
        with slurm_env():
            self.assertEqual(b.cmd_run(run_args(self.work), deps=LocalDeps(
                LocalSpawner(self.engine_factory()))), 0)
        with slurm_env():
            with self.assertRaises(b.IdentityMismatch):
                b.cmd_run(run_args(self.work, "--concurrency", "32"),
                          deps=LocalDeps(LocalSpawner(self.engine_factory())))

    def test_truncation_escalates_once_then_rejects_end_to_end(self):
        target = self.entries[1]["row_id"]
        with slurm_env():
            code = b.cmd_run(run_args(self.work),
                             deps=LocalDeps(LocalSpawner(self.engine_factory(
                                 truncate_rows={target}))))
        self.assertEqual(code, 0)
        with b.Ledger(self.work, read_only=True) as ledger:
            row = ledger.get(target)
            self.assertEqual(row["state"], b.STATE_REJECTED)
            self.assertEqual(row["rejection_reason"], "truncated_at_ceiling")
            attempts = ledger.attempts_for([target])[target]
            self.assertEqual([attempt["max_tokens"] for attempt in attempts], [256, 384])
            self.assertEqual([attempt["finish_reason"] for attempt in attempts],
                             ["length", "length"])
            self.assertEqual(ledger.counts()["states"]["complete"], 11)
        # The truncated row's text is not attached to any completed row.
        with b.Ledger(self.work, read_only=True) as ledger:
            for row in ledger.complete_rows():
                self.assertEqual(row["instruction"], f"Instruction for {row['row_id']}")

    def test_invalid_instruction_is_rejected_without_blind_retry(self):
        target = self.entries[3]["row_id"]
        with slurm_env():
            code = b.cmd_run(run_args(self.work),
                             deps=LocalDeps(LocalSpawner(self.engine_factory(
                                 invalid_rows={target}))))
        self.assertEqual(code, 0)
        with b.Ledger(self.work, read_only=True) as ledger:
            row = ledger.get(target)
            self.assertEqual(row["state"], b.STATE_REJECTED)
            self.assertEqual(row["rejection_reason"], "invalid_instruction")
            self.assertIn("markdown_fence", row["last_error_detail"])
            attempts = ledger.attempts_for([target])[target]
            self.assertEqual(len(attempts), 1, "deterministic invalid output must not be retried")
            self.assertEqual(ledger.counts()["states"]["complete"], 11)

    def test_run_guards(self):
        spawner = LocalSpawner(self.engine_factory())
        # No --allow-non-slurm here: the production guard must refuse outright.
        guarded = b.build_parser().parse_args(
            ["run", "--work", str(self.work), "--vllm-sif", "/tmp/vllm.sif"])
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "", "CUDA_VISIBLE_DEVICES": "0,1"}):
            with self.assertRaisesRegex(SystemExit, "outside a Slurm allocation"):
                b.cmd_run(guarded, deps=LocalDeps(spawner))
        # Inside Slurm, the fixture manifest and GPU checks still apply.
        fixture = run_args(self.work)
        with slurm_env():
            with self.assertRaisesRegex(SystemExit, "exactly 2 visible GPUs"):
                b.cmd_run(fixture, deps=LocalDeps(spawner, gpu_names=["A100"]))
        with slurm_env():
            with self.assertRaisesRegex(SystemExit, "RTX PRO 6000"):
                b.cmd_run(fixture,
                          deps=LocalDeps(spawner, gpu_names=["NVIDIA A40"] * 2))


HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None


@unittest.skipUnless(HAS_PYARROW, "pyarrow is required for export tests")
class ExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)
        self.meta, _rows = make_work_fixture(self.work, rows=12)
        self.entries = list(b.iter_manifest(self.work))
        self.log = self.work / "generated.log"
        rejected_id = self.entries[7]["row_id"]
        with slurm_env():
            self.assertEqual(b.cmd_run(
                run_args(self.work),
                deps=LocalDeps(LocalSpawner(
                    lambda job: FakeEngine(job, log_path=self.log,
                                           invalid_rows={rejected_id})))), 0)
        self.rejected_id = rejected_id

    def read_shards(self, root):
        import pyarrow.parquet as parquet
        rows = []
        for path in sorted((root / "shards").glob("*.parquet")):
            rows.extend(parquet.read_table(path).to_pylist())
        return rows

    def test_export_is_deterministic_and_excludes_rejected(self):
        meta = b.export_dataset(self.work, shard_size=5)
        self.assertEqual(meta["rows"], 11)
        self.assertEqual(meta["shards"], 3)
        self.assertEqual(meta["rejected_rows"], 1)
        rows = self.read_shards(self.work / "export")
        self.assertEqual([row["source_row_index"] for row in rows], list(range(12))[:7] + list(range(8, 12)))
        for row in rows:
            self.assertEqual(row["instruction"], f"Instruction for {row['id']}")
            self.assertEqual(row["source_dataset"], b.DATASET_ID)
            self.assertEqual(row["source_revision"], "1" * 40)
            self.assertEqual(row["caption_model"], b.MODEL_ID)
            self.assertEqual(row["caption_model_revision"], "2" * 40)
            self.assertEqual(row["prompt_version"], "caption-v1")
            self.assertEqual(b.sha256_bytes(row["png_image"]), row["image_sha256"])
            self.assertEqual(b.sha256_text(row["tikz_code"]), row["tikz_sha256"])
        self.assertNotIn(self.rejected_id, {row["id"] for row in rows})

        import pyarrow.parquet as parquet
        rejected = parquet.read_table(self.work / "export" / "rejected.parquet").to_pylist()
        self.assertEqual([row["id"] for row in rejected], [self.rejected_id])
        self.assertEqual(rejected[0]["rejection_reason"], "invalid_instruction")
        attempts = parquet.read_table(self.work / "export" / "attempts.parquet").to_pylist()
        self.assertEqual(len(attempts), 12)
        checksums = json.loads((self.work / "export" / "checksums.json").read_text())
        self.assertEqual(len(checksums["shards"]), 3)
        self.assertTrue(all(shard["file_sha256"] for shard in checksums["shards"]))
        self.assertTrue((self.work / "export" / "export.meta.json").is_file())
        self.assertTrue((self.work / "export" / "run-metadata.json").is_file())
        self.assertTrue((self.work / "export" / "stats.json").is_file())
        report = json.loads((self.work / "export" / "validation-report.json").read_text())
        self.assertEqual(report["status"], "pass")
        card = (self.work / "export" / "dataset-card.md").read_text()
        self.assertIn("DRAFT", card)
        self.assertIn("not uploaded", card)
        self.assertIn(b.DATASET_ID, card)

    def test_export_is_idempotent(self):
        first = b.export_dataset(self.work, shard_size=5)
        first_checksums = json.loads((self.work / "export" / "checksums.json").read_text())
        second = b.export_dataset(self.work, shard_size=5)
        second_checksums = json.loads((self.work / "export" / "checksums.json").read_text())
        self.assertEqual(first["dataset_logical_sha256"], second["dataset_logical_sha256"])
        self.assertEqual([shard["logical_sha256"] for shard in first_checksums["shards"]],
                         [shard["logical_sha256"] for shard in second_checksums["shards"]])

    def test_export_removes_stale_shards_when_shard_size_changes(self):
        b.export_dataset(self.work, shard_size=5)
        self.assertEqual(len(list((self.work / "export" / "shards").glob("*.parquet"))), 3)
        meta = b.export_dataset(self.work, shard_size=100)
        self.assertEqual(meta["shards"], 1)
        self.assertEqual(len(list((self.work / "export" / "shards").glob("*.parquet"))), 1)

    def test_export_refuses_source_drift(self):
        entry = self.entries[0]
        image = self.work / entry["image"]
        image.write_bytes(image.read_bytes() + b"corruption")
        with self.assertRaisesRegex(b.ExportError, "Image checksum drift"):
            b.export_dataset(self.work, shard_size=5)

    def test_export_requires_a_run_record(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            make_work_fixture(work, rows=3)
            with self.assertRaisesRegex(b.ExportError, "nothing to export"):
                b.export_dataset(work)

    def test_export_cli_requires_slurm_guard(self):
        args = b.build_parser().parse_args(
            ["export", "--work", str(self.work), "--shard-size", "5"])
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": ""}):
            with self.assertRaisesRegex(SystemExit, "outside a Slurm allocation"):
                b.cmd_export(args)
        output = io.StringIO()
        with slurm_env(), contextlib.redirect_stdout(output):
            self.assertEqual(b.main(["export", "--work", str(self.work), "--shard-size", "5",
                                     "--allow-non-slurm"]), 0)
        self.assertIn("Export complete: 11 rows in 3 shards", output.getvalue())


if __name__ == "__main__":
    unittest.main()
