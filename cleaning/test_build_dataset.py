"""Local tests for the production dataset builder (no GPU, no network).

Synthetic fixtures only: tests never download the real dataset or model and
never start an inference engine.
"""
import contextlib
import base64
import dataclasses
import importlib.util
import io
import json
import sqlite3
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "build_dataset", Path(__file__).with_name("build_dataset.py"))
b = importlib.util.module_from_spec(SPEC)
sys.modules["build_dataset"] = b
SPEC.loader.exec_module(b)

PINNED = "a" * 40
CONTAINER = "b" * 64
RUNNER = "c" * 64
HELPERS = "d" * 64


def make_config(**overrides):
    prompt = overrides.pop("prompt", b.load_prompt())
    values = dict(
        dataset=b.DatasetSource(revision=PINNED),
        model=b.ModelSource(revision=PINNED, processor_revision=PINNED, path="/tmp/model"),
        prompt=prompt,
        inference=b.InferenceConfig(),
        validation=b.ValidationPolicy(),
        container_sha256=CONTAINER,
        runner_sha256=RUNNER,
        helper_sha256=HELPERS,
        git_commit="0" * 40,
    )
    values.update(overrides)
    return b.RunConfig(**values)


def mutate(config, **changes):
    """Apply nested dataclass replacements: mutate(cfg, inference__mtp=2)."""
    top = {}
    nested = {}
    for key, value in changes.items():
        if "__" in key:
            name, field = key.split("__", 1)
            nested.setdefault(name, {})[field] = value
        else:
            top[key] = value
    for name, fields in nested.items():
        top[name] = dataclasses.replace(getattr(config, name), **fields)
    return dataclasses.replace(config, **top)


class PromptTests(unittest.TestCase):
    def test_shipped_prompt_matches_registry(self):
        prompt = b.load_prompt()
        self.assertEqual(prompt.version, "caption-v1")
        self.assertTrue(b.is_sha256(prompt.sha256))
        registry = json.loads((b.PROMPTS_DIR / "registry.json").read_text())
        self.assertEqual(registry[prompt.version]["sha256"], prompt.sha256)
        self.assertIn("under 120 words", prompt.text)
        self.assertNotIn("\n", prompt.text)

    def test_caption_v2_is_registered_as_instruction_style_300_word_prompt(self):
        prompt = b.load_prompt("caption-v2")
        self.assertTrue(b.is_sha256(prompt.sha256))
        self.assertIn("imperative creation request", prompt.text)
        self.assertIn("authoritative ground truth", prompt.text)
        self.assertIn("Do not exceed 300 words", prompt.text)

    def test_normalization_is_wrap_insensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "caption-v9.txt").write_text("one two\nthree\n")
            normalized = "one two three"
            (directory / "registry.json").write_text(json.dumps({
                "caption-v9": {"file": "caption-v9.txt", "sha256": b.sha256_text(normalized)}}))
            first = b.load_prompt("caption-v9", directory)
            (directory / "caption-v9.txt").write_text("one\ntwo   three\n\n")
            second = b.load_prompt("caption-v9", directory)
            self.assertEqual(first.sha256, second.sha256)

    def test_changed_prompt_requires_new_version(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "caption-v9.txt").write_text("original text")
            (directory / "registry.json").write_text(json.dumps({
                "caption-v9": {"file": "caption-v9.txt", "sha256": b.sha256_text("original text")}}))
            (directory / "caption-v9.txt").write_text("edited text")
            with self.assertRaisesRegex(b.PromptError, "create a new version"):
                b.load_prompt("caption-v9", directory)

    def test_unknown_version_rejected(self):
        with self.assertRaisesRegex(b.PromptError, "Unknown prompt version"):
            b.load_prompt("caption-does-not-exist")


class IdentityTests(unittest.TestCase):
    def test_production_baseline_defaults(self):
        inference = b.InferenceConfig()
        self.assertEqual(inference.engine, "vllm-offline")
        self.assertEqual((inference.tensor_parallel, inference.replicas), (1, 2))
        self.assertEqual(inference.replicas_per_gpu, 1)
        self.assertEqual(inference.aggregate_concurrency, 64)
        self.assertEqual(inference.per_replica_concurrency(0), 32)
        self.assertEqual(inference.per_replica_concurrency(1), 32)
        self.assertEqual(inference.mtp, 1)
        self.assertFalse(inference.enable_thinking)
        self.assertTrue(inference.greedy)
        self.assertEqual(inference.batch_token_budget, 16384)
        self.assertEqual(inference.context, 32768)
        self.assertEqual(inference.max_output_tokens, 256)
        self.assertEqual(inference.truncation_retry_tokens, 384)
        self.assertFalse(inference.prefix_caching)
        self.assertFalse(inference.custom_all_reduce)
        self.assertEqual(inference.nccl_p2p, "disabled")
        self.assertAlmostEqual(inference.gpu_memory_utilization, 0.90)
        self.assertEqual(b.DATASET_ID, "nllg/DaTikZ-V4")
        self.assertEqual(b.MODEL_ID, "nvidia/Qwen3.8-27B-NVFP4")
        self.assertEqual((b.ROW_START, b.ROW_LIMIT), (0, 100_000))
        inference.validate()

    def test_identity_changes_for_every_scientific_input(self):
        base = make_config()
        baseline = base.identity_sha256()
        self.assertEqual(base.run_id(), baseline[:16])
        mutations = [
            dict(dataset__revision="e" * 40),
            dict(model__revision="f" * 40),
            dict(model__processor_revision="f" * 40),
            dict(prompt=b.Prompt(version="caption-v2", sha256=b.sha256_text("other"), text="other")),
            dict(runner_sha256="1" * 64),
            dict(helper_sha256="2" * 64),
            dict(container_sha256="3" * 64),
            dict(inference__aggregate_concurrency=32),
            dict(inference__mtp=2),
            dict(inference__batch_token_budget=32768),
            dict(inference__context=16384),
            dict(inference__max_output_tokens=512),
            dict(inference__truncation_retry_tokens=768),
            dict(inference__gpu_memory_utilization=0.5),
            dict(inference__nccl_p2p="auto"),
            dict(inference__enable_thinking=True),
            dict(inference__greedy=False),
            dict(inference__prefix_caching=True),
            dict(inference__custom_all_reduce=True),
            dict(inference__tensor_parallel=2, inference__replicas=1),
            dict(validation__min_chars=5),
            dict(validation__max_chars=100),
            dict(validation__max_words=50),
            dict(schema_version="dataset-v2"),
        ]
        for change in mutations:
            changed = mutate(base, **change)
            self.assertNotEqual(changed.identity_sha256(), baseline, msg=str(change))
            self.assertNotEqual(changed.run_id(), base.run_id(), msg=str(change))

    def test_identity_excludes_provenance_only_fields(self):
        base = make_config()
        self.assertEqual(mutate(base, git_commit="9" * 40).identity_sha256(),
                         base.identity_sha256())
        self.assertEqual(mutate(base, model__path="/elsewhere/model").identity_sha256(),
                         base.identity_sha256())

    def test_invalid_configurations_rejected(self):
        cases = [
            (dict(inference__enable_thinking=True), "Thinking must be explicitly disabled"),
            (dict(inference__replicas_per_gpu=2), "rejected topology"),
            (dict(inference__replicas=3), "one, two, or four replicas"),
            (dict(inference__greedy=False), "greedy"),
            (dict(inference__truncation_retry_tokens=256), "must exceed"),
            (dict(inference__context=1024), "at least 4096"),
            (dict(inference__nccl_p2p="sometimes"), "Unknown nccl_p2p"),
            (dict(validation__min_chars=0), "length policy"),
            (dict(schema_version="dataset-v9"), "Unknown output schema"),
        ]
        for change, message in cases:
            with self.assertRaisesRegex(b.ConfigError, message, msg=str(change)):
                mutate(make_config(), **change).validate()

    def test_require_pinned_revisions(self):
        config = make_config(dataset=b.DatasetSource(revision="main"))
        with self.assertRaisesRegex(b.ConfigError, "pinned 40-char"):
            config.require_pinned()
        make_config().require_pinned()

    def test_identity_diff_reports_nested_fields(self):
        base = make_config()
        changed = mutate(base, inference__mtp=3, prompt=b.Prompt(
            version="caption-v2", sha256=b.sha256_text("x"), text="x"))
        differences = b.identity_diff(base.identity(), changed.identity())
        self.assertTrue(any("inference.mtp" in line for line in differences))
        self.assertTrue(any("prompt.sha256" in line for line in differences))
        self.assertEqual(b.identity_diff(base.identity(), base.identity()), [])


class RunRecordTests(unittest.TestCase):
    def test_roundtrip_and_mismatch_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config()
            record = b.write_run_record(directory, config)
            self.assertEqual(record["run_id"], config.run_id())
            stored = b.verify_run_record(directory, config)
            self.assertEqual(stored["identity"], config.identity())
            changed = mutate(config, inference__aggregate_concurrency=128)
            with self.assertRaises(b.IdentityMismatch) as caught:
                b.verify_run_record(directory, changed)
            self.assertTrue(any("aggregate_concurrency" in line
                                for line in caught.exception.differences))
            with self.assertRaisesRegex(b.ConfigError, "run prepare first"):
                b.verify_run_record(Path(directory) / "missing", config)


class PlanTests(unittest.TestCase):
    def run_plan(self, *extra):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = b.main(["plan", *extra])
        self.assertEqual(code, 0)
        return json.loads(output.getvalue())

    def test_plan_reports_provisional_identity(self):
        plan = self.run_plan()
        self.assertEqual(plan["source"]["count"], 100_000)
        self.assertEqual(plan["source"]["rows"], "0-99999")
        self.assertEqual(plan["model"], b.MODEL_ID)
        self.assertEqual(plan["identity_provisional"]["dataset"]["revision"], "UNRESOLVED")
        self.assertEqual(plan["identity_provisional"]["inference"]["mtp"], 1)
        self.assertIn("prepare", plan["stages"])
        self.assertGreater(plan["baseline"]["estimated_generation_hours_at_baseline"], 5)

    def test_plan_honors_pinned_revisions(self):
        plan = self.run_plan("--dataset-revision", "1" * 40,
                             "--model-revision", "2" * 40, "--mtp", "2")
        identity = plan["identity_provisional"]
        self.assertEqual(identity["dataset"]["revision"], "1" * 40)
        self.assertEqual(identity["model"]["revision"], "2" * 40)
        self.assertEqual(identity["inference"]["mtp"], 2)

    def test_plan_identity_changes_with_configuration(self):
        first = self.run_plan()
        second = self.run_plan("--concurrency", "32")
        self.assertNotEqual(first["identity_sha256_provisional"],
                            second["identity_sha256_provisional"])


# ---------------------------------------------------------------------------
# Feature 2: manifest freeze
# ---------------------------------------------------------------------------

# A 1x1 transparent PNG used as the only real image fixture.
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def fake_row(index, tikz=None, image=TINY_PNG, **extra):
    row = {"file_id": f"file-{index}", "source": "arxiv",
           "tikz_code": f"\\begin{{tikzpicture}}\\draw (0,0) -- ({index},1);\\end{{tikzpicture}}" if tikz is None else tikz,
           "png_image": {"bytes": image}}
    row.update(extra)
    return row


class FakePrepareDeps(b.PrepareDeps):
    def __init__(self, rows, dataset_revision="1" * 40, model_revision="2" * 40,
                 model_path="/tmp/fake-model", pulled=None):
        self.rows = rows
        self.dataset_revision = dataset_revision
        self.model_revision = model_revision
        self.model_path = model_path
        self.pulled = pulled if pulled is not None else []

    def resolve_dataset_revision(self, args):
        return self.dataset_revision

    def resolve_model_revision(self, args):
        return self.model_revision

    def download_model(self, args, revision):
        return self.model_path

    def row_source(self, args, revision):
        for row in self.rows:
            self.pulled.append(row)
            yield row


def prepare_args(work, *extra):
    return b.build_parser().parse_args(
        ["prepare", "--work", str(work), "--allow-non-slurm", *extra])


class RowIdentityTests(unittest.TestCase):
    def test_stable_row_id_determinism(self):
        base = b.stable_row_id("d", "r" * 40, "train", 7, "t" * 64, "i" * 64)
        self.assertEqual(base, b.stable_row_id("d", "r" * 40, "train", 7, "t" * 64, "i" * 64))
        self.assertEqual(len(base), 32)
        for change in [("d2", "r" * 40, "train", 7, "t" * 64, "i" * 64),
                       ("d", "e" * 40, "train", 7, "t" * 64, "i" * 64),
                       ("d", "r" * 40, "validation", 7, "t" * 64, "i" * 64),
                       ("d", "r" * 40, "train", 8, "t" * 64, "i" * 64),
                       ("d", "r" * 40, "train", 7, "u" * 64, "i" * 64),
                       ("d", "r" * 40, "train", 7, "t" * 64, "j" * 64)]:
            self.assertNotEqual(base, b.stable_row_id(*change))

    def test_extract_image_bytes_shapes(self):
        self.assertEqual(b.extract_image_bytes(None), None)
        self.assertEqual(b.extract_image_bytes(TINY_PNG), TINY_PNG)
        self.assertEqual(b.extract_image_bytes({"bytes": TINY_PNG}), TINY_PNG)
        self.assertEqual(b.extract_image_bytes(base64.b64encode(TINY_PNG).decode()), TINY_PNG)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.png"
            path.write_bytes(TINY_PNG)
            self.assertEqual(b.extract_image_bytes({"path": str(path)}), TINY_PNG)
            self.assertEqual(b.extract_image_bytes(str(path)), TINY_PNG)
        self.assertIsNone(b.extract_image_bytes({"bytes": None, "path": "/missing.png"}))

    def test_image_inspection(self):
        limits = b.FreezeLimits()
        self.assertEqual(b.inspect_image(TINY_PNG, limits), None)
        self.assertEqual(b.inspect_image(b"", limits), "missing_image")
        self.assertEqual(b.inspect_image(b"not a png", limits), "invalid_image")
        self.assertEqual(b.inspect_image(TINY_PNG, b.FreezeLimits(max_image_bytes=4)),
                         "image_too_large")


class FreezeTests(unittest.TestCase):
    def setUp(self):
        # cmd_prepare performs a real disk-space check; tests run on small disks.
        patcher = mock.patch.object(
            b.shutil, "disk_usage",
            return_value=types.SimpleNamespace(free=10**12, total=10**12, used=0))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_invalid_early_rows_are_not_replaced(self):
        rows = [fake_row(0), fake_row(1), fake_row(2, tikz="   "), fake_row(3),
                fake_row(4, image=b"broken"), fake_row(5)]
        with tempfile.TemporaryDirectory() as directory:
            deps = FakePrepareDeps(rows)
            meta = b.freeze_source(deps.row_source(None, None), work=directory,
                                   dataset_id="d", revision="r" * 40, split="train",
                                   limits=b.FreezeLimits(), row_limit=5)
            self.assertEqual(len(deps.pulled), 5)  # no replacement row pulled
            self.assertEqual(meta["rows_frozen"], 5)
            self.assertEqual(meta["valid_rows"], 3)
            self.assertEqual(meta["rejected_rows"], 2)
            self.assertEqual(meta["rejection_counts"], {"empty_tikz": 1, "invalid_image": 1})
            entries = list(b.iter_manifest(directory))
            self.assertEqual([entry["source_row_index"] for entry in entries], [0, 1, 2, 3, 4])
            self.assertEqual(entries[2]["status"], "rejected")
            self.assertEqual(entries[2]["rejection_reason"], "empty_tikz")
            self.assertIsNone(entries[2]["image"])
            self.assertEqual(entries[3]["status"], "valid")
            self.assertEqual(entries[4]["rejection_reason"], "invalid_image")
            # Rejected rows still carry evidence: checksums of whatever was received.
            self.assertEqual(entries[4]["image_sha256"], b.sha256_bytes(b"broken"))
            # A valid row's image is on disk with the recorded checksum.
            image_path = Path(directory) / entries[3]["image"]
            self.assertEqual(b.sha256_file(image_path), entries[3]["image_sha256"])
            b.verify_manifest(directory, quick=False)

    def test_exact_first_100k_boundary(self):
        total = b.ROW_LIMIT + 10
        rows = (fake_row(i, tikz=f"\\draw ({i},0);") for i in range(total))
        with tempfile.TemporaryDirectory() as directory:
            meta = b.freeze_source(rows, work=directory, dataset_id="d", revision="r" * 40,
                                   split="train", limits=b.FreezeLimits(),
                                   write_images=False)
            self.assertEqual(meta["rows_frozen"], b.ROW_LIMIT)
            self.assertEqual(meta["valid_rows"], b.ROW_LIMIT)
            self.assertEqual(meta["rejected_rows"], 0)
            entries = list(b.iter_manifest(directory))
            self.assertEqual(len(entries), b.ROW_LIMIT)
            self.assertEqual(entries[0]["source_row_index"], 0)
            self.assertEqual(entries[-1]["source_row_index"], b.ROW_LIMIT - 1)
            self.assertEqual(len({entry["row_id"] for entry in entries}), b.ROW_LIMIT)
            excluded = b.stable_row_id("d", "r" * 40, "train", b.ROW_LIMIT,
                                       b.sha256_text(f"\\draw ({b.ROW_LIMIT},0);"),
                                       b.sha256_bytes(TINY_PNG))
            self.assertNotIn(excluded, {entry["row_id"] for entry in entries})

    def test_short_stream_is_a_hard_failure(self):
        rows = [fake_row(i) for i in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(b.ManifestError, "exactly 10"):
                b.freeze_source(iter(rows), work=directory, dataset_id="d", revision="r" * 40,
                                split="train", limits=b.FreezeLimits(), row_limit=10,
                                write_images=False)

    def test_interrupted_freeze_leaves_no_committed_manifest(self):
        def exploding():
            yield fake_row(0)
            yield fake_row(1)
            raise RuntimeError("simulated stream failure")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                b.freeze_source(exploding(), work=directory, dataset_id="d", revision="r" * 40,
                                split="train", limits=b.FreezeLimits(), row_limit=5)
            self.assertFalse(b.manifest_path(directory).exists())
            self.assertFalse(b.manifest_meta_path(directory).exists())
            self.assertEqual(list(Path(directory).glob(".staging-*")), [])
            with self.assertRaisesRegex(b.ManifestError, "authoritative only after"):
                b.verify_manifest(directory)

    def test_prepare_cleans_partial_artifacts_and_reuses_complete_manifest(self):
        rows = [fake_row(i) for i in range(6)]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            # Simulate a crash that left a manifest and images but no meta file.
            b.manifest_path(work).write_text('{"source_row_index": 0}\n')
            b.images_dir(work).mkdir()
            (b.images_dir(work) / "stale.png").write_bytes(b"stale")
            (work / ".staging-999").mkdir()
            deps = FakePrepareDeps(rows)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(b.cmd_prepare(prepare_args(work, "--row-limit", "6"), deps), 0)
            self.assertIn("Removed incomplete freeze artifacts", output.getvalue())
            self.assertFalse((b.images_dir(work) / "stale.png").exists())
            meta = b.verify_manifest(work, quick=False)
            self.assertEqual(meta["rows_frozen"], 6)
            first_meta = json.loads(b.manifest_meta_path(work).read_text())
            # A second prepare reuses the frozen manifest without touching it.
            deps2 = FakePrepareDeps(rows)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(b.cmd_prepare(prepare_args(work), deps2), 0)
            self.assertIn("Reusing frozen manifest", output.getvalue())
            self.assertEqual(deps2.pulled, [])
            self.assertEqual(json.loads(b.manifest_meta_path(work).read_text()), first_meta)
            # Requested revisions must agree with the frozen manifest.
            with self.assertRaisesRegex(b.ConfigError, "does not match frozen manifest"):
                b.cmd_prepare(prepare_args(work, "--dataset-revision", "9" * 40), deps2)

    def test_prepare_refuses_row_limit_without_fixture_mode(self):
        rows = [fake_row(i) for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            args = b.build_parser().parse_args(
                ["prepare", "--work", directory, "--row-limit", "3"])
            with mock.patch.dict("os.environ", {"SLURM_JOB_ID": "12345"}):
                with self.assertRaisesRegex(b.ConfigError, "synthetic-fixture control"):
                    b.cmd_prepare(args, FakePrepareDeps(rows))

    def test_disk_estimate_and_guard(self):
        rows = [fake_row(i) for i in range(10)]
        limits = b.FreezeLimits(estimate_rows=10)
        estimate = b.estimate_storage(rows, limits, row_limit=100)
        self.assertEqual(estimate["sample_rows"], 10)
        self.assertGreater(estimate["projected_bytes"], 100 * len(TINY_PNG))
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(b.shutil, "disk_usage",
                                   return_value=types.SimpleNamespace(free=0, total=0, used=0)):
                with self.assertRaisesRegex(SystemExit, "Insufficient free space"):
                    b.check_disk_space(directory, estimate, limits, row_limit=100)
            with mock.patch.object(b.shutil, "disk_usage",
                                   return_value=types.SimpleNamespace(free=10**12, total=0, used=0)):
                self.assertGreater(b.check_disk_space(directory, estimate, limits, row_limit=100), 0)

    def test_verify_detects_tampering(self):
        rows = [fake_row(i) for i in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            deps = FakePrepareDeps(rows)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(b.cmd_prepare(prepare_args(work, "--row-limit", "4"), deps), 0)
            entries = list(b.iter_manifest(work))
            image_path = work / entries[0]["image"]
            original = image_path.read_bytes()
            # Quick verify trusts existence; full verify checks hashes.
            image_path.write_bytes(b"corrupted")
            b.verify_manifest(work, quick=True)
            with self.assertRaisesRegex(b.ManifestError, "Image checksum mismatch"):
                b.verify_manifest(work, quick=False)
            image_path.write_bytes(original)
            image_path.unlink()
            with self.assertRaisesRegex(b.ManifestError, "Image missing"):
                b.verify_manifest(work, quick=True)
            image_path.write_bytes(original)
            manifest = b.manifest_path(work)
            text = manifest.read_text()
            manifest.write_text(text.replace(entries[1]["row_id"], "0" * 32))
            with self.assertRaisesRegex(b.ManifestError, "Stable id mismatch"):
                b.verify_manifest(work, quick=True)

    def test_prepare_redownloads_missing_model_snapshot(self):
        rows = [fake_row(index) for index in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            deps = FakePrepareDeps(rows, model_path=str(work / "model-ok"))
            Path(deps.model_path).mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(b.cmd_prepare(prepare_args(work, "--row-limit", "4"), deps), 0)
            # Simulate a moved or removed cache between runs.
            meta = json.loads(b.manifest_meta_path(work).read_text())
            meta["model_path"] = str(work / "gone")
            b.manifest_meta_path(work).write_text(json.dumps(meta))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(b.cmd_prepare(prepare_args(work, "--row-limit", "4"), deps), 0)
            self.assertEqual(json.loads(b.manifest_meta_path(work).read_text())["model_path"],
                             deps.model_path)

    def test_manifest_hash_mismatch_detected(self):
        rows = [fake_row(i) for i in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                b.cmd_prepare(prepare_args(work, "--row-limit", "4"), FakePrepareDeps(rows))
            manifest = b.manifest_path(work)
            manifest.write_text(manifest.read_text() + "\n")
            with self.assertRaises(b.ManifestError):
                b.verify_manifest(work, quick=True)


# ---------------------------------------------------------------------------
# Feature 3: durable state ledger
# ---------------------------------------------------------------------------

LEDGER_IDENTITY = dict(
    schema_version=b.LEDGER_SCHEMA_VERSION,
    run_id="run-0000000000000000",
    identity_sha256="f" * 64,
    manifest_sha256="a" * 64,
    dataset_revision="1" * 40,
    model_revision="2" * 40,
    prompt_sha256="3" * 64,
)


def manifest_entries(count=6, invalid=()):
    entries = []
    for index in range(count):
        tikz = f"\\draw ({index},0);"
        status = "rejected" if index in invalid else "valid"
        entries.append({
            "row_id": b.stable_row_id("d", "r" * 40, "train", index,
                                      b.sha256_text(tikz), b.sha256_bytes(TINY_PNG)),
            "source_row_index": index,
            "tikz_code": tikz,
            "tikz_sha256": b.sha256_text(tikz),
            "image": f"images/{index:06d}.png",
            "image_sha256": b.sha256_bytes(TINY_PNG),
            "status": status,
            "rejection_reason": "invalid_image" if status == "rejected" else None,
        })
    return entries


def open_seeded_ledger(work, entries, *, identity=None, rows_frozen=None):
    ledger = b.Ledger(work).open()
    ledger.initialize(identity or LEDGER_IDENTITY)
    ledger.seed(entries, rows_frozen=rows_frozen if rows_frozen is not None else len(entries))
    return ledger


def ok_result(row_id, index, *, instruction="Draw a circle with a radius label",
              max_tokens=256, worker="w0", finish="stop"):
    return dict(row_id=row_id, source_row_index=index, worker=worker, max_tokens=max_tokens,
                ok=finish == "stop", instruction=instruction if finish == "stop" else None,
                finish_reason=finish, prompt_tokens=11, completion_tokens=22,
                error_category=None, error_detail=None, model_revision="2" * 40,
                prompt_sha256="3" * 64, config_hash="f" * 64)


def fail_result(row_id, index, *, category="engine_transient", detail="cuda oops",
                max_tokens=256, worker="w0", finish=None):
    return dict(row_id=row_id, source_row_index=index, worker=worker, max_tokens=max_tokens,
                ok=False, instruction=None, finish_reason=finish, prompt_tokens=None,
                completion_tokens=None, error_category=category, error_detail=detail,
                model_revision="2" * 40, prompt_sha256="3" * 64, config_hash="f" * 64)


class LedgerIdentityTests(unittest.TestCase):
    def test_identity_roundtrip_and_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            with b.Ledger(directory).open() as ledger:
                ledger.initialize(LEDGER_IDENTITY)
                ledger.verify_identity(LEDGER_IDENTITY)
                with self.assertRaisesRegex(b.LedgerError, "different run identity"):
                    ledger.verify_identity(dict(LEDGER_IDENTITY, run_id="other"))
            with self.assertRaisesRegex(b.LedgerError, "No ledger"):
                b.Ledger(Path(directory) / "empty", read_only=True).open()

    def test_seed_is_idempotent_and_verifies_manifest(self):
        entries = manifest_entries(5)
        with tempfile.TemporaryDirectory() as directory:
            ledger = open_seeded_ledger(directory, entries)
            self.assertEqual(ledger.seed(entries, rows_frozen=5)["existing"], 5)
            self.assertEqual(ledger.counts()["states"]["pending"], 5)
            changed = [dict(entry) for entry in entries]
            changed[2]["image_sha256"] = "0" * 64
            with self.assertRaisesRegex(b.LedgerError, "does not match the frozen manifest"):
                ledger.seed(changed, rows_frozen=5)
            ledger.close()
            with b.Ledger(directory).open() as fresh:
                fresh.initialize(LEDGER_IDENTITY)
                with self.assertRaisesRegex(b.LedgerError, "do not belong together"):
                    fresh.seed(entries[:3], rows_frozen=3)

    def test_seed_records_invalid_rows_as_rejected(self):
        entries = manifest_entries(4, invalid=(1, 3))
        with tempfile.TemporaryDirectory() as directory:
            ledger = open_seeded_ledger(directory, entries)
            counts = ledger.counts()
            self.assertEqual(counts["states"]["pending"], 2)
            self.assertEqual(counts["states"]["rejected"], 2)
            self.assertEqual(counts["rejection_reasons"], {"invalid_image": 2})
            self.assertEqual(ledger.get(entries[1]["row_id"])["last_error_category"],
                             "input_invalid")
            ledger.close()


class LedgerTransitionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.entries = manifest_entries(6)
        self.ledger = open_seeded_ledger(self.directory.name, self.entries)
        self.addCleanup(self.ledger.close)
        self.policy = b.RetryPolicy()

    def ids(self, *indexes):
        return [self.entries[index]["row_id"] for index in indexes]

    def claim_one(self, index, worker="w0", now=None):
        claimed = self.ledger.claim({worker: self.ids(index)}, policy=self.policy, now=now)
        return claimed[worker][0]

    def test_claim_opens_attempt_and_marks_running(self):
        claim = self.claim_one(0)
        self.assertEqual(claim["max_tokens"], 256)
        self.assertEqual(claim["attempt_kind"], "normal")
        self.assertEqual(claim["attempt_no"], 1)
        row = self.ledger.get(self.ids(0)[0])
        self.assertEqual(row["state"], b.STATE_RUNNING)
        self.assertEqual(row["worker"], "w0")
        self.assertEqual(row["attempt_count"], 1)
        attempts = self.ledger.attempts_for(self.ids(0))[self.ids(0)[0]]
        self.assertEqual([(a["state"], a["attempt_no"]) for a in attempts], [("running", 1)])
        # A second claim on the same row is a hard conflict.
        with self.assertRaisesRegex(b.LedgerError, "Claim conflict"):
            self.ledger.claim({"w1": self.ids(0)}, policy=self.policy)

    def test_eligible_excludes_running_complete_rejected(self):
        self.claim_one(0)
        self.ledger.record_result(ok_result(*self.ids(1), 1), policy=self.policy)
        eligible = {row["source_row_index"] for row in self.ledger.eligible()}
        self.assertEqual(eligible, {2, 3, 4, 5})

    def test_complete_is_transactional_and_idempotent(self):
        row_id = self.ids(0)[0]
        self.claim_one(0)
        state = self.ledger.record_result(ok_result(row_id, 0), policy=self.policy)
        self.assertEqual(state, b.STATE_COMPLETE)
        row = self.ledger.get(row_id)
        self.assertEqual(row["instruction"], "Draw a circle with a radius label")
        self.assertEqual(row["finish_reason"], "stop")
        self.assertEqual((row["prompt_tokens"], row["completion_tokens"]), (11, 22))
        self.assertIsNotNone(row["completed_at"])
        # Re-ingesting the identical result is a no-op (crash before consumption).
        self.assertEqual(self.ledger.record_result(ok_result(row_id, 0), policy=self.policy),
                         b.STATE_COMPLETE)
        # A different instruction must never overwrite a completed caption.
        with self.assertRaisesRegex(b.LedgerError, "different instruction"):
            self.ledger.record_result(
                ok_result(row_id, 0, instruction="Something else entirely"),
                policy=self.policy)
        # Completed rows are never claimable again.
        with self.assertRaisesRegex(b.LedgerError, "Claim conflict"):
            self.ledger.claim({"w0": [row_id]}, policy=self.policy)

    def test_transient_retry_budget_and_backoff(self):
        row_id = self.ids(0)[0]
        expected = [(1, 30.0, b.STATE_RETRYABLE), (2, 60.0, b.STATE_RETRYABLE),
                    (3, None, b.STATE_REJECTED)]
        for attempt, backoff, state in expected:
            claim_time = 1000.0 * attempt
            claimed = self.ledger.claim({"w0": [row_id]}, policy=self.policy, now=claim_time)
            self.assertEqual(claimed["w0"][0]["attempt_no"], attempt)
            result_state = self.ledger.record_result(fail_result(row_id, 0),
                                                     policy=self.policy, now=claim_time)
            self.assertEqual(result_state, state)
            row = self.ledger.get(row_id)
            if backoff is None:
                self.assertEqual(row["rejection_reason"], "transient_exhausted")
            else:
                self.assertEqual(row["last_error_category"], "engine_transient")
                self.assertAlmostEqual(row["not_before"], claim_time + backoff)
                self.assertNotIn(row_id, [r["row_id"]
                                          for r in self.ledger.eligible(now=claim_time)])
                self.assertIn(row_id, [r["row_id"]
                                       for r in self.ledger.eligible(now=claim_time + backoff + 1)])
        attempts = self.ledger.attempts_for([row_id])[row_id]
        self.assertEqual([a["state"] for a in attempts],
                         ["retryable", "retryable", "rejected"])
        self.assertEqual(len(attempts), 3)

    def test_truncation_escalates_from_256_to_384_then_rejects(self):
        row_id = self.ids(0)[0]
        claim = self.claim_one(0)
        self.assertEqual(claim["max_tokens"], 256)
        state = self.ledger.record_result(
            ok_result(row_id, 0, instruction=None, finish="length"), policy=self.policy)
        self.assertEqual(state, b.STATE_RETRYABLE)
        claim = self.ledger.claim({"w0": [row_id]}, policy=self.policy)
        self.assertEqual(claim["w0"][0]["max_tokens"], 384)
        self.assertEqual(claim["w0"][0]["attempt_kind"], "truncation_retry")
        state = self.ledger.record_result(
            ok_result(row_id, 0, instruction=None, finish="length", max_tokens=384),
            policy=self.policy)
        self.assertEqual(state, b.STATE_REJECTED)
        self.assertEqual(self.ledger.get(row_id)["rejection_reason"], "truncated_at_ceiling")
        attempts = self.ledger.attempts_for([row_id])[row_id]
        self.assertEqual([a["max_tokens"] for a in attempts], [256, 384])
        self.assertEqual([a["finish_reason"] for a in attempts], ["length", "length"])

    def test_stale_claims_do_not_consume_retry_budget(self):
        row_id = self.ids(0)[0]
        for cycle in range(4):
            claim_time = 200.0 + 10 * cycle
            self.ledger.claim({"w0": [row_id]}, policy=self.policy, now=claim_time)
            reclaimed = self.ledger.reclaim_stale(5.0, policy=self.policy, now=claim_time + 10)
            self.assertEqual(reclaimed, 1)
            row = self.ledger.get(row_id)
            self.assertEqual(row["state"], b.STATE_RETRYABLE)
            self.assertEqual(row["last_error_category"], "stale_claim")
        attempts = self.ledger.attempts_for([row_id])[row_id]
        self.assertEqual(len(attempts), 4)
        self.assertTrue(all(attempt["state"] == "lost" for attempt in attempts))

    def test_worker_loss_consumes_budget_and_eventually_rejects(self):
        row_id = self.ids(0)[0]
        for cycle in range(3):
            self.ledger.claim({"w0": [row_id]}, policy=self.policy, now=1000.0 * (cycle + 1))
            self.ledger.release_running(worker="w0", category="worker_lost",
                                        policy=self.policy, now=1000.0 * (cycle + 1))
        self.assertEqual(self.ledger.get(row_id)["state"], b.STATE_REJECTED)
        self.assertEqual(self.ledger.get(row_id)["rejection_reason"], "transient_exhausted")

    def test_run_interruption_does_not_consume_budget(self):
        row_id = self.ids(0)[0]
        for cycle in range(5):
            now = 1000.0 * (cycle + 1)
            self.ledger.claim({"w0": [row_id]}, policy=self.policy, now=now)
            self.ledger.release_running(category="run_interrupted", policy=self.policy, now=now)
        self.assertEqual(self.ledger.get(row_id)["state"], b.STATE_RETRYABLE)

    def test_rejected_rows_stay_rejected_until_explicit_reprocessing(self):
        entries = manifest_entries(4, invalid=(0,))
        with tempfile.TemporaryDirectory() as directory:
            ledger = open_seeded_ledger(directory, entries)
            row_id = entries[1]["row_id"]
            ledger.claim({"w0": [row_id]}, policy=self.policy)
            ledger.record_result(fail_result(row_id, 1, category="integrity",
                                             detail="image checksum mismatch"),
                                 policy=self.policy)
            self.assertEqual(ledger.get(row_id)["state"], b.STATE_REJECTED)
            self.assertEqual({row["source_row_index"] for row in ledger.eligible()}, {2, 3})
            moved = ledger.reprocess_rejected(policy=self.policy)
            self.assertEqual(moved, 1)  # input_invalid row is not reprocessed
            self.assertEqual(ledger.get(row_id)["state"], b.STATE_PENDING)
            self.assertEqual({row["source_row_index"] for row in ledger.eligible()}, {1, 2, 3})
            self.assertEqual(ledger.get(entries[0]["row_id"])["state"], b.STATE_REJECTED)
            # Attempt history survives reprocessing.
            self.assertEqual(len(ledger.attempts_for([row_id])[row_id]), 1)
            ledger.close()

    def test_late_result_for_rejected_row_is_ignored(self):
        row_id = self.ids(0)[0]
        self.claim_one(0)
        self.ledger.record_result(fail_result(row_id, 0, category="integrity",
                                              detail="image checksum mismatch"),
                                  policy=self.policy)
        self.assertEqual(self.ledger.get(row_id)["state"], b.STATE_REJECTED)
        self.assertEqual(self.ledger.record_result(ok_result(row_id, 0), policy=self.policy),
                         b.STATE_REJECTED)
        self.assertEqual(self.ledger.get(row_id)["state"], b.STATE_REJECTED)
        self.assertIsNone(self.ledger.get(row_id)["instruction"])

    def test_checkpoint_backup_is_readable(self):
        row_id = self.ids(0)[0]
        self.claim_one(0)
        self.ledger.record_result(ok_result(row_id, 0), policy=self.policy)
        target = self.ledger.checkpoint()
        self.assertTrue(target.is_file())
        backup = b.Ledger(Path(self.directory.name) / "unused", read_only=False)
        backup.conn = sqlite3.connect(str(target))
        backup.conn.row_factory = sqlite3.Row
        try:
            self.assertEqual(backup.counts()["states"]["complete"], 1)
        finally:
            backup.close()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(b.main(["checkpoint", "--work", self.directory.name]), 0)
        self.assertIn("Ledger checkpoint:", output.getvalue())

    def test_crash_between_inference_and_commit_recovers(self):
        row_id = self.ids(0)[0]
        # The claim happened, then the process died before any result arrived.
        self.ledger.claim({"w0": [row_id]}, policy=self.policy, now=100.0)
        self.ledger.close()
        recovered = b.Ledger(self.directory.name).open()
        self.addCleanup(recovered.close)
        recovered.initialize(LEDGER_IDENTITY)
        self.assertEqual(recovered.reclaim_stale(10.0, policy=self.policy, now=200.0), 1)
        self.assertEqual(recovered.get(row_id)["state"], b.STATE_RETRYABLE)
        # Resume completes the row exactly once.
        recovered.claim({"w0": [row_id]}, policy=self.policy, now=201.0)
        self.assertEqual(recovered.record_result(ok_result(row_id, 0), policy=self.policy,
                                                 now=202.0), b.STATE_COMPLETE)
        self.assertEqual(recovered.get(row_id)["attempt_count"], 2)

    def test_recovered_result_after_restart_is_accepted(self):
        row_id = self.ids(0)[0]
        self.ledger.claim({"w0": [row_id]}, policy=self.policy, now=100.0)
        self.ledger.reclaim_stale(10.0, policy=self.policy, now=200.0)
        # A result file written before the crash is ingested after restart.
        state = self.ledger.record_result(ok_result(row_id, 0, worker="w0"),
                                          policy=self.policy, now=201.0)
        self.assertEqual(state, b.STATE_COMPLETE)
        attempts = self.ledger.attempts_for([row_id])[row_id]
        self.assertEqual([a["attempt_kind"] for a in attempts], ["normal", "recovered"])

    def test_counts_aggregate(self):
        self.claim_one(0)
        self.ledger.record_result(ok_result(*self.ids(1), 1), policy=self.policy)
        self.ledger.record_result(fail_result(*self.ids(2), 2), policy=self.policy)
        counts = self.ledger.counts()
        self.assertEqual(counts["total"], 6)
        self.assertEqual(counts["states"], {"pending": 3, "running": 1, "retryable": 1,
                                            "complete": 1, "rejected": 0})
        self.assertEqual(counts["attempts"], 3)
        self.assertEqual(counts["prompt_tokens"], 11)
        self.assertEqual(counts["completion_tokens"], 22)
        self.assertIn("engine_transient", counts["error_categories"])


class ValidationRuleTests(unittest.TestCase):
    def setUp(self):
        self.policy = b.ValidationPolicy()

    def check(self, text, **overrides):
        return b.validate_instruction(text, b.ValidationPolicy(**overrides))

    def test_valid_instruction_passes(self):
        ok, rule, _ = self.check("Draw a circle labelled r with a dashed radius line.")
        self.assertTrue(ok, rule)

    def test_rule_violations_are_classified(self):
        cases = [
            ("", "empty"),
            ("   \n  ", "empty"),
            ("\x00Draw a circle", "control_characters"),
            ("<think>private</think> Draw a circle", "thinking_leak"),
            ("Draw a circle</think>", "thinking_leak"),
            ("Use the <source> code", "source_tags"),
            ("Draw a circle\n```\ncode\n```", "markdown_fence"),
            ("Draw \\begin{tikzpicture} then stop", "tikz_code"),
            ("Draw the shape from \\draw (0,0);", "tikz_code"),
            ("Place a \\node at the origin", "tikz_code"),
            ("Recreate the supplied image as TikZ.", "task_reference"),
            ("Convert the source code into a diagram.", "task_reference"),
            ("Use the provided inputs to draw this.", "task_reference"),
            ("Return only a caption for this task.", "task_reference"),
            ("short", "too_short"),
            ("word " * 121, "too_many_words"),
        ]
        for text, expected in cases:
            ok, rule, detail = self.check(text)
            self.assertFalse(ok, msg=text[:40])
            self.assertEqual(rule, expected, msg=f"{text[:40]!r} -> {rule} ({detail})")

    def test_limits_are_configurable(self):
        text = "Draw a small circle with one radius line"
        self.assertTrue(self.check(text)[0])
        self.assertFalse(self.check(text, min_chars=1000)[0])
        self.assertFalse(self.check(text, max_words=3)[0])
        self.assertFalse(self.check(text, max_chars=5)[0])
        ok, rule, _ = self.check(text, max_words=3)
        self.assertEqual(rule, "too_many_words")

    def test_word_count_is_the_120_word_ceiling(self):
        exactly = "word " * 119 + "word"
        self.assertEqual(len(exactly.split()), 120)
        self.assertTrue(self.check(exactly)[0])
        over = exactly + " extra"
        ok, rule, _ = self.check(over)
        self.assertFalse(ok)
        self.assertEqual(rule, "too_many_words")

    def test_rule_table_is_part_of_the_identity(self):
        before_rules = b.validation_rules_sha256()
        before_identity = make_config().identity_sha256()
        with mock.patch.object(b, "FORBIDDEN_REFERENCES",
                               b.FORBIDDEN_REFERENCES + ("banana",)):
            self.assertNotEqual(b.validation_rules_sha256(), before_rules)
            self.assertNotEqual(make_config().identity_sha256(), before_identity)
        self.assertEqual(b.validation_rules_sha256(), before_rules)


# ---------------------------------------------------------------------------
# Feature 4: identity-safe inference
# ---------------------------------------------------------------------------


class FakeEngine:
    """Deterministic fake engine: text is derived from the prompt's row id."""

    def __init__(self, job, *, crash_rows=(), fail_rows=(), prompt_mismatch=False,
                 short_batch=False, log_path=None, truncate_rows=(), invalid_rows=(),
                 reverse_outputs=False, fail_first_call=False, no_prompt_echo=False,
                 fail_once_rows=()):
        self.job = job
        self.crash_rows = set(crash_rows)
        self.fail_rows = set(fail_rows)
        self.fail_once_rows = fail_once_rows if isinstance(fail_once_rows, set) \
            else set(fail_once_rows)
        self.prompt_mismatch = prompt_mismatch
        self.short_batch = short_batch
        self.log_path = log_path
        self.truncate_rows = set(truncate_rows)
        self.invalid_rows = set(invalid_rows)
        self.reverse_outputs = reverse_outputs
        self.fail_first_call = fail_first_call
        self.no_prompt_echo = no_prompt_echo
        self.calls = 0

    def _log(self, row_id):
        if self.log_path:
            with open(self.log_path, "a") as handle:
                handle.write(row_id + "\n")

    def generate(self, items):
        self.calls += 1
        if self.fail_first_call and self.calls == 1:
            raise RuntimeError("simulated warmup engine failure")
        outputs = []
        for index, item in enumerate(items):
            row_id = item["row_id"]
            if row_id in self.crash_rows:
                raise SystemExit(137)  # simulate SIGKILL: no results are written
            if row_id in self.fail_rows:
                raise RuntimeError("simulated CUDA failure")
            if row_id in self.fail_once_rows:
                self.fail_once_rows.discard(row_id)
                raise RuntimeError("simulated one-time engine failure")
            self._log(row_id)
            if row_id in self.truncate_rows:
                outputs.append(b.EngineOutput(prompt=item["prompt"], text="Partial caption",
                                              finish_reason="length", prompt_tokens=12,
                                              completion_tokens=item["max_tokens"]))
                continue
            text = f"Instruction for {row_id}"
            if row_id in self.invalid_rows:
                text = "Here it is:\n```tikz\n\\draw (0,0);\n```"
            outputs.append(b.EngineOutput(prompt=None if self.no_prompt_echo else item["prompt"],
                                          text=text,
                                          finish_reason="stop", prompt_tokens=12,
                                          completion_tokens=len(text.split())))
        if self.short_batch:
            outputs = outputs[:-1]
        if self.reverse_outputs:
            outputs = list(reversed(outputs))
        if self.prompt_mismatch and outputs:
            outputs[0].prompt = "wrong prompt"
        return outputs


class FakePromptBuilder:
    def __init__(self, job):
        self.job = job

    def __call__(self, task, config):
        prompt = f"PROMPT::{task['row_id']}::{config['prompt_sha256'][:8]}"
        return prompt, None, task.get("length", 10)


class LengthBuilder:
    """Test builder that reports a fixed input length."""

    def __init__(self, length):
        self.length = length

    def __call__(self, task, config):
        return "prompt", None, self.length


def make_work_fixture(work, rows=4, revision="1" * 40):
    """Small frozen manifest with pinned model metadata (synthetic fixture)."""
    entries = [fake_row(index) for index in range(rows)]
    meta = b.freeze_source(iter(entries), work=work, dataset_id=b.DATASET_ID, revision=revision,
                           split="train", limits=b.FreezeLimits(), row_limit=rows,
                           meta_extra=dict(model_id=b.MODEL_ID, model_revision="2" * 40,
                                           model_path="/fake/model"))
    return meta, entries


def worker_task(work, index, *, image_bytes=TINY_PNG, tikz=None, max_tokens=256):
    tikz = tikz if tikz is not None else f"\\draw ({index},0);"
    image = Path(work) / f"image-{index}.png"
    image.write_bytes(image_bytes)
    return dict(row_id=f"row-{index:04d}", source_row_index=index, image=str(image),
                image_sha256=b.sha256_bytes(image_bytes), tikz_code=tikz,
                tikz_sha256=b.sha256_text(tikz), max_tokens=max_tokens, attempt_no=1)


def make_worker_job(work, tasks, *, per_replica=2, warmup=0, job_name="job.json",
                    require_prompt_echo=True):
    job = dict(run_id="run-test", worker_index=0, generation=1,
               tasks_path=str(Path(work) / "tasks.jsonl"),
               results_dir=str(Path(work) / "results"),
               heartbeat_path=str(Path(work) / "heartbeat.json"),
               stop_path=str(Path(work) / "stop"),
               engine_options_path=str(Path(work) / "engine-options.json"),
               versions_path=str(Path(work) / "versions.json"),
               config=dict(model_path="/fake/model", model_revision="2" * 40,
                           prompt="system prompt", prompt_version="caption-v1",
                           prompt_sha256="3" * 64, context=32768,
                           per_replica_concurrency=per_replica, batch_token_budget=16384,
                           mtp=1, gpu_memory_utilization=0.9, max_output_tokens=256,
                           truncation_retry_tokens=384, warmup_samples=warmup,
                           require_prompt_echo=require_prompt_echo,
                           config_hash="f" * 64))
    with (Path(work) / "tasks.jsonl").open("w") as handle:
        for task in tasks:
            handle.write(json.dumps(task) + "\n")
    job_path = Path(work) / job_name
    b.bench.dump(job_path, job)
    return job_path


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)

    def run_worker(self, tasks, *, engine=None, builder=None, per_replica=2, warmup=0,
                   require_prompt_echo=True):
        job_path = make_worker_job(self.work, tasks, per_replica=per_replica, warmup=warmup,
                                   require_prompt_echo=require_prompt_echo)
        engine_factory = (lambda job: engine) if engine is not None else \
            (lambda job: FakeEngine(job))
        builder_factory = (lambda job: builder) if builder is not None else \
            (lambda job: FakePromptBuilder(job))
        code = b.run_worker(job_path, engine_factory=engine_factory,
                            prompt_builder_factory=builder_factory)
        results = []
        for path in sorted((self.work / "results").glob("*.json")):
            results.extend(json.loads(path.read_text())["results"])
        return code, {result["row_id"]: result for result in results}

    def test_worker_associates_results_by_stable_id(self):
        tasks = [worker_task(self.work, index) for index in range(3)]
        code, results = self.run_worker(tasks, per_replica=2)
        self.assertEqual(code, 0)
        self.assertEqual(sorted(results), [f"row-{i:04d}" for i in range(3)])
        for row_id, result in results.items():
            self.assertTrue(result["ok"])
            self.assertEqual(result["instruction"], f"Instruction for {row_id}")
            self.assertEqual(result["finish_reason"], "stop")
            self.assertEqual(result["prompt_tokens"], 12)
            self.assertEqual(result["config_hash"], "f" * 64)

    def test_worker_flags_checksum_mismatches_without_generating(self):
        tasks = [worker_task(self.work, index) for index in range(2)]
        (Path(tasks[0]["image"])).write_bytes(b"corrupted")
        tasks[1]["tikz_code"] = tasks[1]["tikz_code"] + "tampered"
        code, results = self.run_worker(tasks, per_replica=2)
        self.assertEqual(code, 0)
        self.assertEqual(results[tasks[0]["row_id"]]["error_category"], "integrity")
        self.assertIn("image checksum", results[tasks[0]["row_id"]]["error_detail"])
        self.assertEqual(results[tasks[1]["row_id"]]["error_category"], "integrity")
        self.assertIn("tikz checksum", results[tasks[1]["row_id"]]["error_detail"])
        self.assertFalse(results[tasks[0]["row_id"]]["ok"])

    def test_worker_rejects_inputs_over_context(self):
        tasks = [worker_task(self.work, 0)]
        code, results = self.run_worker(tasks, builder=LengthBuilder(32768))
        self.assertEqual(code, 0)
        self.assertEqual(results[tasks[0]["row_id"]]["error_category"], "input_too_long")

    def test_worker_aborts_batch_on_prompt_mismatch(self):
        tasks = [worker_task(self.work, index) for index in range(2)]
        engine = FakeEngine(None, prompt_mismatch=True)
        code, results = self.run_worker(tasks, engine=engine, per_replica=2)
        self.assertEqual(code, 3)
        self.assertEqual(len(results), 2)
        for result in results.values():
            self.assertFalse(result["ok"])
            self.assertEqual(result["error_category"], "engine_transient")
            self.assertIn("does not match", result["error_detail"])

    def test_worker_aborts_batch_on_output_count_mismatch(self):
        tasks = [worker_task(self.work, index) for index in range(3)]
        engine = FakeEngine(None, short_batch=True)
        code, results = self.run_worker(tasks, engine=engine, per_replica=3)
        self.assertEqual(code, 3)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(not result["ok"] for result in results.values()))

    def test_worker_engine_failure_records_transient_for_whole_batch(self):
        tasks = [worker_task(self.work, index) for index in range(2)]
        engine = FakeEngine(None, fail_rows={tasks[1]["row_id"]})
        code, results = self.run_worker(tasks, engine=engine, per_replica=2)
        self.assertEqual(code, 3)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["error_category"] == "engine_transient"
                            for result in results.values()))

    def test_worker_stops_between_batches(self):
        tasks = [worker_task(self.work, index) for index in range(4)]
        (self.work / "stop").touch()
        code, results = self.run_worker(tasks, per_replica=2)
        self.assertEqual(code, 0)
        self.assertEqual(results, {})
        self.assertFalse(any((self.work / "results").glob("*.json")))

    def test_worker_detects_swapped_engine_outputs(self):
        """No positional cross-row assignment: reversed outputs abort the batch."""
        tasks = [worker_task(self.work, index) for index in range(3)]
        engine = FakeEngine(None, reverse_outputs=True)
        code, results = self.run_worker(tasks, engine=engine, per_replica=3)
        self.assertEqual(code, 3)
        self.assertEqual(len(results), 3)
        for result in results.values():
            self.assertFalse(result["ok"])
            self.assertEqual(result["error_category"], "engine_transient")
            self.assertIn("does not match", result["error_detail"])

    def test_worker_warmup_failure_exits_without_results(self):
        tasks = [worker_task(self.work, index) for index in range(2)]
        engine = FakeEngine(None, fail_first_call=True)
        code, results = self.run_worker(tasks, engine=engine, warmup=1)
        self.assertEqual(code, 3)
        self.assertEqual(results, {})
        self.assertFalse(any((self.work / "results").glob("*.json")))

    def test_worker_heartbeat_records_progress(self):
        tasks = [worker_task(self.work, index) for index in range(2)]
        self.run_worker(tasks, per_replica=2)
        heartbeat = json.loads((self.work / "heartbeat.json").read_text())
        self.assertEqual(heartbeat["rows_done"], 2)
        self.assertEqual(heartbeat["batch_index"], 0)

    def test_worker_requires_prompt_echo_by_default(self):
        tasks = [worker_task(self.work, index) for index in range(2)]
        engine = FakeEngine(None, no_prompt_echo=True)
        code, results = self.run_worker(tasks, engine=engine, per_replica=2)
        self.assertEqual(code, 3)
        self.assertEqual(len(results), 2)
        for result in results.values():
            self.assertEqual(result["error_category"], "engine_transient")
            self.assertIn("did not echo", result["error_detail"])
        # The operator can explicitly accept engine ordering instead.
        code, results = self.run_worker(tasks, engine=FakeEngine(None, no_prompt_echo=True),
                                        per_replica=2, require_prompt_echo=False)
        self.assertEqual(code, 0)
        self.assertTrue(all(result["ok"] for result in results.values()))

    def test_input_budget_includes_reserved_output_tokens(self):
        tasks = [worker_task(self.work, 0, max_tokens=384)]
        code, results = self.run_worker(tasks, builder=LengthBuilder(32768 - 300))
        self.assertEqual(results[tasks[0]["row_id"]]["error_category"], "input_too_long")
        code, results = self.run_worker(tasks, builder=LengthBuilder(32768 - 384))
        self.assertTrue(results[tasks[0]["row_id"]]["ok"])

    def test_task_filtering_keeps_only_outstanding_rows(self):
        tasks = [worker_task(self.work, index) for index in range(4)]
        source = self.work / "all.jsonl"
        with source.open("w") as handle:
            for task in tasks:
                handle.write(json.dumps(task) + "\n")
        keep = {tasks[1]["row_id"], tasks[3]["row_id"]}
        destination = self.work / "filtered.jsonl"
        b.filter_task_file(source, destination, keep)
        kept = [json.loads(line)["row_id"] for line in destination.read_text().splitlines()]
        self.assertEqual(kept, [tasks[1]["row_id"], tasks[3]["row_id"]])

    def test_worker_last_activity_tracks_heartbeat(self):
        handle = types.SimpleNamespace(started_at=time.time() - 100)
        heartbeat = self.work / "heartbeat.json"
        self.assertLessEqual(b._worker_last_activity(handle, heartbeat), time.time() - 99)
        heartbeat.write_text("{}")
        self.assertGreaterEqual(b._worker_last_activity(handle, heartbeat), time.time() - 1)

    def test_stop_flag_records_signal_name(self):
        flag = b.StopFlag()
        self.assertFalse(flag.requested)
        flag.handle(signal.SIGTERM, None)
        self.assertTrue(flag.requested)
        self.assertEqual(flag.signal_name, "SIGTERM")


class RunArgumentTests(unittest.TestCase):
    def parse(self, *extra):
        return b.build_parser().parse_args(
            ["run", "--work", "/tmp/w", "--vllm-sif", "/tmp/v.sif", *extra])

    def test_chunk_bounds_are_validated(self):
        b.validate_run_args(self.parse())
        for extra in (["--start-index", "-1"], ["--end-index", "100000"],
                      ["--start-index", "10", "--end-index", "5"],
                      ["--max-rows-this-run", "0"], ["--max-runtime-minutes", "0"],
                      ["--worker-restarts", "-1"], ["--warmup-samples", "-1"],
                      ["--poll-seconds", "0"],
                      ["--worker-timeout", "0"], ["--max-transient-attempts", "0"],
                      ["--retry-backoff-base-seconds", "-1"]):
            with self.assertRaises(b.ConfigError, msg=str(extra)):
                b.validate_run_args(self.parse(*extra))

    def test_module_import_does_not_require_heavy_dependencies(self):
        """Importing the module must not pull in pyarrow, PIL, datasets or vLLM."""
        program = (
            "import importlib.util, sys;"
            "spec = importlib.util.spec_from_file_location('b', sys.argv[1]);"
            "module = importlib.util.module_from_spec(spec); sys.modules['b'] = module;"
            "spec.loader.exec_module(module);"
            "heavy = [name for name in ('pyarrow', 'PIL', 'datasets', 'vllm', 'torch')"
            " if name in sys.modules];"
            "print(','.join(heavy))"
        )
        result = subprocess.run([sys.executable, "-c", program,
                                 str(Path(b.__file__).resolve())],
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "")

    def test_manifest_verify_rejects_rejected_row_without_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            make_work_fixture(work, rows=3)
            lines = b.manifest_path(work).read_text().splitlines()
            entry = json.loads(lines[1])
            entry["status"] = "rejected"
            entry["rejection_reason"] = None
            lines[1] = json.dumps(entry, ensure_ascii=False, sort_keys=True)
            b.manifest_path(work).write_text("\n".join(lines) + "\n")
            meta = json.loads(b.manifest_meta_path(work).read_text())
            meta["manifest_sha256"] = b.sha256_file(b.manifest_path(work))
            b.manifest_meta_path(work).write_text(json.dumps(meta))
            with self.assertRaisesRegex(b.ManifestError, "no rejection reason"):
                b.verify_manifest(work, quick=True)


class AssignmentTests(unittest.TestCase):
    def test_assignment_is_deterministic_and_balanced(self):
        index = {f"row-{i}": dict(tikz_chars=i * 10, source_row_index=i) for i in range(10)}
        eligible = [dict(row_id=f"row-{i}", source_row_index=i) for i in range(10)]
        first = b.balanced_assignment(eligible, index, workers=2)
        second = b.balanced_assignment(list(reversed(eligible)), index, workers=2)
        self.assertEqual(first, second)
        self.assertEqual(len(first["worker-0"]) + len(first["worker-1"]), 10)
        self.assertEqual(sorted(first["worker-0"] + first["worker-1"]),
                         sorted(row["row_id"] for row in eligible))
        loads = []
        for worker, rows in first.items():
            loads.append(sum(index[row_id]["tikz_chars"] + 1024 for row_id in rows))
        self.assertLessEqual(abs(loads[0] - loads[1]), 1024 + 90)


class ResultIdentityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)
        self.meta, _fixture_rows = make_work_fixture(self.work, rows=4)
        args = b.build_parser().parse_args(
            ["run", "--work", str(self.work), "--allow-non-slurm", "--vllm-sif", "/tmp/vllm.sif"])
        self.config = b.config_from_args(args, manifest_meta=self.meta,
                                         container_sha256="b" * 64, require_pinned=True)
        self.index = b.manifest_index(self.work)
        self.entries = list(b.iter_manifest(self.work))
        self.policy = b.RetryPolicy(backoff_base_seconds=0)
        self.ledger = b.Ledger(self.work).open()
        self.addCleanup(self.ledger.close)
        self.ledger.initialize(b.ledger_identity_from_config(self.config, self.meta))
        self.ledger.seed(iter(b.iter_manifest(self.work)), rows_frozen=4)

    def claim_all(self):
        assignment = b.balanced_assignment(
            self.ledger.eligible(), self.index, workers=2)
        return self.ledger.claim(assignment, policy=self.policy)

    def write_result_file(self, worker, results, name="batch-1-00000.json", run_id=None):
        directory = self.work / "runtime" / "results" / worker
        directory.mkdir(parents=True, exist_ok=True)
        payload = dict(run_id=run_id or self.config.run_id(), worker=worker,
                       worker_index=int(worker.split("-")[1]), generation=1, batch_index=0,
                       results=results)
        (directory / name).write_text(json.dumps(payload))

    def result_for(self, entry, *, worker="worker-0", ok=True, instruction=None):
        return dict(row_id=entry["row_id"], source_row_index=entry["source_row_index"],
                    worker=worker, batch_index=0, max_tokens=256, ok=ok,
                    instruction=instruction if instruction is not None else f"Draw {entry['row_id']}",
                    text=None, finish_reason="stop" if ok else None,
                    prompt_tokens=10, completion_tokens=20, error_category=None,
                    error_detail=None, image_sha256=entry["image_sha256"],
                    tikz_sha256=entry["tikz_sha256"], model_revision=self.config.model.revision,
                    prompt_sha256=self.config.prompt.sha256,
                    config_hash=self.config.identity_sha256())

    def test_ingest_commits_by_stable_id(self):
        claims = self.claim_all()
        worker = "worker-0" if claims["worker-0"] else "worker-1"
        entries = [entry for entry in self.entries
                   if entry["row_id"] in {row["row_id"] for row in claims[worker]}]
        self.write_result_file(worker, [self.result_for(entry, worker=worker)
                                        for entry in entries])
        ingested = b.ingest_result_files(ledger=self.ledger, work=self.work, config=self.config,
                                         index=self.index, policy=self.policy)
        self.assertEqual(ingested[worker], {entry["row_id"] for entry in entries})
        for entry in entries:
            row = self.ledger.get(entry["row_id"])
            self.assertEqual(row["state"], b.STATE_COMPLETE)
            self.assertEqual(row["instruction"], f"Draw {entry['row_id']}")

    def quarantined(self):
        return sorted((self.work / "runtime" / "quarantine").glob("*"))

    def test_ingest_rejects_unknown_row_id(self):
        self.claim_all()
        bogus = dict(self.result_for(self.entries[0]), row_id="row-unknown")
        self.write_result_file("worker-0", [bogus])
        ingested = b.ingest_result_files(ledger=self.ledger, work=self.work, config=self.config,
                                         index=self.index, policy=self.policy)
        self.assertEqual(ingested, {})
        self.assertEqual(len(self.quarantined()), 1)
        self.assertEqual(self.ledger.counts()["states"]["complete"], 0)

    def test_ingest_rejects_foreign_run_id(self):
        self.claim_all()
        self.write_result_file("worker-0", [self.result_for(self.entries[0])],
                               run_id="run-someone-else")
        b.ingest_result_files(ledger=self.ledger, work=self.work, config=self.config,
                              index=self.index, policy=self.policy)
        self.assertEqual(len(self.quarantined()), 1)
        self.assertEqual(self.ledger.counts()["states"]["complete"], 0)

    def test_ingest_rejects_worker_ownership_mismatch(self):
        self.claim_all()
        entry = self.entries[0]
        self.write_result_file("worker-1", [self.result_for(entry, worker="worker-1")])
        # The row was claimed by whichever worker the assignment chose; find it.
        owner = self.ledger.get(entry["row_id"])["worker"]
        other = "worker-1" if owner == "worker-0" else "worker-0"
        # Rewrite the file as if it came from the other worker.
        for path in (self.work / "runtime" / "results").glob("worker-*/*.json"):
            payload = json.loads(path.read_text())
            payload["worker"] = other
            payload["worker_index"] = int(other.split("-")[1])
            payload["results"][0]["worker"] = other
            path.write_text(json.dumps(payload))
        b.ingest_result_files(ledger=self.ledger, work=self.work, config=self.config,
                              index=self.index, policy=self.policy)
        self.assertEqual(len(self.quarantined()), 1)
        self.assertEqual(self.ledger.get(entry["row_id"])["state"], b.STATE_RUNNING)

    def test_ingest_rejects_config_and_prompt_drift(self):
        self.claim_all()
        for field, value in (("config_hash", "0" * 64), ("prompt_sha256", "0" * 64),
                             ("model_revision", "9" * 40)):
            result = dict(self.result_for(self.entries[0]), **{field: value})
            self.write_result_file("worker-0", [result], name=f"batch-{field}.json")
        b.ingest_result_files(ledger=self.ledger, work=self.work, config=self.config,
                              index=self.index, policy=self.policy)
        self.assertEqual(len(self.quarantined()), 3)
        self.assertEqual(self.ledger.counts()["states"]["complete"], 0)

    def test_ingest_turns_source_checksum_drift_into_integrity_rejection(self):
        self.claim_all()
        entry = self.entries[0]
        result = dict(self.result_for(entry), image_sha256="0" * 64)
        self.write_result_file("worker-0", [result])
        b.ingest_result_files(ledger=self.ledger, work=self.work, config=self.config,
                              index=self.index, policy=self.policy)
        row = self.ledger.get(entry["row_id"])
        self.assertEqual(row["state"], b.STATE_REJECTED)
        self.assertEqual(row["rejection_reason"], "integrity")

    def test_ingest_is_order_independent(self):
        claims = self.claim_all()
        payloads = []
        for worker, rows in claims.items():
            entries = [entry for entry in self.entries
                       if entry["row_id"] in {row["row_id"] for row in rows}]
            payloads.append((worker, [self.result_for(entry, worker=worker)
                                      for entry in entries]))
        for worker, results in reversed(payloads):
            self.write_result_file(worker, results, name=f"batch-{worker}.json")
        b.ingest_result_files(ledger=self.ledger, work=self.work, config=self.config,
                              index=self.index, policy=self.policy)
        for entry in self.entries:
            row = self.ledger.get(entry["row_id"])
            self.assertEqual(row["state"], b.STATE_COMPLETE)
            self.assertEqual(row["instruction"], f"Draw {entry['row_id']}")

    def test_duplicate_result_inside_file_is_quarantined(self):
        self.claim_all()
        result = self.result_for(self.entries[0])
        self.write_result_file("worker-0", [result, dict(result)])
        b.ingest_result_files(ledger=self.ledger, work=self.work, config=self.config,
                              index=self.index, policy=self.policy)
        self.assertEqual(len(self.quarantined()), 1)
        self.assertEqual(self.ledger.counts()["states"]["complete"], 0)

    def test_reapplied_result_file_does_not_double_count_retries(self):
        """A crash between commit and file removal must not burn the retry budget."""
        self.claim_all()
        entry = self.entries[0]
        payload = dict(run_id=self.config.run_id(), worker="worker-0", worker_index=0,
                       generation=1, batch_index=0,
                       results=[dict(self.result_for(entry, ok=False, instruction=None),
                                     finish_reason=None, error_category="engine_transient",
                                     error_detail="simulated CUDA failure")])
        self.write_result_file("worker-0", payload["results"])
        path = self.work / "runtime" / "results" / "worker-0" / "batch-1-00000.json"
        original = path.read_bytes()
        b.ingest_result_files(ledger=self.ledger, work=self.work, config=self.config,
                              index=self.index, policy=self.policy)
        self.assertEqual(self.ledger.get(entry["row_id"])["state"], b.STATE_RETRYABLE)
        self.assertEqual(len(self.ledger.attempts_for([entry["row_id"]])[entry["row_id"]]), 1)
        # Simulate a crash before the file was moved: put the identical content back.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(original)
        b.ingest_result_files(ledger=self.ledger, work=self.work, config=self.config,
                              index=self.index, policy=self.policy)
        attempts = self.ledger.attempts_for([entry["row_id"]])[entry["row_id"]]
        self.assertEqual(len(attempts), 1, "identical content must not be counted twice")
        self.assertEqual(self.ledger.get(entry["row_id"])["state"], b.STATE_RETRYABLE)
        self.assertEqual(self.ledger.get(entry["row_id"])["attempt_count"], 1)


# ---------------------------------------------------------------------------
# Feature 8: Slurm integration
# ---------------------------------------------------------------------------


class SlurmScriptTests(unittest.TestCase):
    def build(self, *extra):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = b.main(["slurm-script", "--work", "/tmp/tikz-prod",
                           "--vllm-sif", "/tmp/vllm.sif", "--wall-time", "12:00:00", *extra])
        self.assertEqual(code, 0)
        return output.getvalue()

    def test_script_requests_the_production_allocation(self):
        script = self.build()
        self.assertIn("#SBATCH --partition=rtxpro6k", script)
        self.assertIn("#SBATCH --gres=gpu:rtxpro6k:2", script)
        self.assertIn("#SBATCH --nodes=1", script)
        self.assertIn("#SBATCH --ntasks=1", script)
        self.assertIn("#SBATCH --cpus-per-task=32", script)
        self.assertIn("#SBATCH --time=12:00:00", script)
        self.assertIn("#SBATCH --export=NONE", script)
        self.assertIn("#SBATCH --output=slurm-tikz-prod-%j.out", script)
        self.assertNotIn("sbatch ", script.splitlines()[-1])
        self.assertTrue(all("sbatch" not in line or line.startswith("#")
                            for line in script.splitlines() if "sbatch" in line))
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_script_supports_an_isolated_one_gpu_twenty_row_pilot(self):
        script = self.build(
            "--gpus", "1", "--replicas", "1", "--row-limit", "20",
            "--end-index", "19", "--prompt-version", "caption-v2",
            "--concurrency", "20", "--max-output-tokens", "512",
            "--truncation-retry-tokens", "768", "--max-instruction-words", "300")
        self.assertIn("#SBATCH --gres=gpu:rtxpro6k:1", script)
        self.assertIn("prepare --work", script)
        self.assertIn("--row-limit 20 --replicas 1", script)
        self.assertIn("--end-index 19", script)
        self.assertIn("--prompt-version caption-v2", script)
        self.assertIn("--max-instruction-words 300", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_script_supports_four_independent_tp1_replicas(self):
        script = self.build(
            "--gpus", "4", "--replicas", "4", "--concurrency", "128",
            "--prompt-version", "caption-v2", "--max-output-tokens", "512",
            "--truncation-retry-tokens", "768", "--max-instruction-words", "300")
        self.assertIn("#SBATCH --gres=gpu:rtxpro6k:4", script)
        self.assertIn("--replicas 4", script)
        self.assertIn("--concurrency 128", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_script_rejects_gpu_replica_mismatch(self):
        args = b.build_parser().parse_args([
            "slurm-script", "--work", "/tmp/w", "--vllm-sif", "/tmp/v.sif",
            "--wall-time", "00:20:00", "--gpus", "1", "--replicas", "2"])
        with self.assertRaisesRegex(b.ConfigError, "gpus must equal"):
            b.build_slurm_script(args)

    def test_script_stages_prepare_run_export_and_audit(self):
        script = self.build()
        self.assertIn("== status before ==", script)
        self.assertIn("== prepare (downloads stay inside the allocation) ==", script)
        self.assertIn("prepare_rc=0", script)
        self.assertIn("after publishing the verified manifest; continuing", script)
        self.assertIn("== run ==", script)
        self.assertIn("== status after ==", script)
        self.assertIn("== export ==", script)
        self.assertIn("== audit ==", script)
        # prepare, export and audit run inside the container; run is on the host.
        self.assertIn("apptainer exec --nv", script)
        self.assertIn(" prepare --work", script)
        self.assertIn(" export --work", script)
        self.assertIn(" audit --work", script)
        run_line = next(line for line in script.splitlines() if " run --work" in line)
        self.assertNotIn("apptainer", run_line)
        self.assertTrue(run_line.startswith("python3 "))
        # No dependency installation and no secret material.
        self.assertNotIn("pip install", script)
        self.assertNotIn("HF_TOKEN=", script)

    def test_script_propagates_configuration_and_derives_runtime_budget(self):
        script = self.build("--concurrency", "32", "--mtp", "2",
                            "--max-rows-this-run", "5000", "--start-index", "10",
                            "--end-index", "99", "--shard-size", "250",
                            "--max-transient-attempts", "2")
        self.assertIn("--concurrency 32", script)
        self.assertIn("--mtp 2", script)
        self.assertIn("--max-rows-this-run 5000", script)
        self.assertIn("--start-index 10", script)
        self.assertIn("--end-index 99", script)
        self.assertIn("--shard-size 250", script)
        self.assertIn("--max-transient-attempts 2", script)
        # 12h wall time minus the 30 minute reserve.
        self.assertIn("--max-runtime-minutes 690", script)
        explicit = self.build("--max-runtime-minutes", "600")
        self.assertIn("--max-runtime-minutes 600", explicit)

    def test_script_requires_wall_time_and_container(self):
        with self.assertRaisesRegex(b.ConfigError, "wall-time is required"):
            b.build_slurm_script(b.build_parser().parse_args(
                ["slurm-script", "--work", "/tmp/w", "--vllm-sif", "/tmp/v.sif"]))
        with self.assertRaisesRegex(b.ConfigError, "vllm-sif is required"):
            b.build_slurm_script(b.build_parser().parse_args(
                ["slurm-script", "--work", "/tmp/w", "--wall-time", "1:00:00"]))

    def test_wall_time_parsing(self):
        self.assertEqual(b.parse_wall_time("12:00:00"), 720)
        self.assertEqual(b.parse_wall_time("00:30:00"), 30)
        self.assertEqual(b.parse_wall_time("1-00:00:00"), 1440)
        self.assertEqual(b.parse_wall_time("2"), 2)
        self.assertEqual(b.parse_wall_time("90"), 90)
        self.assertEqual(b.parse_wall_time("5:00"), 5)
        for invalid in ("", "abc", "12:99:00", "0", "1-25:00:00", "1:2:3:4"):
            with self.assertRaises(b.ConfigError, msg=invalid):
                b.parse_wall_time(invalid)

    def test_prepare_python_bind_and_model_cache_bind(self):
        script = self.build("--prepare-python", "/envs/prep/bin/python",
                            "--model-cache-dir", "/shared/hf")
        self.assertIn("--bind /envs/prep", script)
        self.assertIn("--bind /shared/hf", script)
        self.assertIn("/envs/prep/bin/python", script)

    def test_historical_slurm_script_flag_alias(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(b.main(["--slurm-script", "--work", "/tmp/w",
                                     "--vllm-sif", "/tmp/v.sif",
                                     "--wall-time", "1:00:00"]), 0)
        self.assertIn("#SBATCH --gres=gpu:rtxpro6k:2", output.getvalue())


if __name__ == "__main__":
    unittest.main()
