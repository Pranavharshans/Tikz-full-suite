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
import sys
import tempfile
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
            (dict(inference__replicas=4), "TP1 with exactly two replicas"),
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


if __name__ == "__main__":
    unittest.main()
