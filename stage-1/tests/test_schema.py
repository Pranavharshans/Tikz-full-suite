"""Tests for the minimal JSON-Schema validator and the shipped schemas."""
import unittest
from pathlib import Path

from stage1 import schema
from stage1.errors import SchemaError


class ValidatorTests(unittest.TestCase):
    def test_accepts_conforming_value(self):
        value = {"name": "x", "count": 2, "tags": ["a"]}
        spec = {
            "type": "object",
            "required": ["name", "count"],
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "count": {"type": "integer", "minimum": 0},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        }
        schema.validate_artifact(value, spec, "test")

    def test_reports_missing_required_with_path(self):
        with self.assertRaises(SchemaError) as context:
            schema.validate_artifact({}, {"type": "object", "required": ["a"]}, "x")
        self.assertIn("missing required property 'a'", str(context.exception))

    def test_rejects_bool_where_integer_expected(self):
        with self.assertRaises(SchemaError):
            schema.validate_artifact({"a": True},
                                     {"type": "object",
                                      "properties": {"a": {"type": "integer"}}},
                                     "x")

    def test_rejects_unknown_property_when_closed(self):
        with self.assertRaises(SchemaError):
            schema.validate_artifact({"b": 1},
                                     {"type": "object", "additionalProperties": False},
                                     "x")

    def test_pattern_and_enum(self):
        spec = {"type": "string", "pattern": "^[0-9a-f]{8}$"}
        schema.validate_artifact("deadbeef", spec, "x")
        with self.assertRaises(SchemaError):
            schema.validate_artifact("nope", spec, "x")
        with self.assertRaises(SchemaError):
            schema.validate_artifact("x", {"enum": ["y"]}, "x")

    def test_unknown_schema_name(self):
        with self.assertRaises(SchemaError):
            schema.load_schema("does-not-exist.schema.json")


class ShippedSchemaTests(unittest.TestCase):
    def test_all_shipped_schemas_load(self):
        names = [
            "split-manifest.schema.json", "dataset-report.schema.json",
            "run.schema.json", "preflight.schema.json",
            "gate-evidence.schema.json", "training-metrics.schema.json",
            "eval.schema.json", "artifact.schema.json",
        ]
        for name in names:
            loaded = schema.load_schema(name)
            self.assertEqual(loaded.get("type"), "object", name)

    def test_no_orphan_schemas(self):
        """Every shipped schema must be exercised by this test suite."""
        schema_dir = Path(__file__).resolve().parents[1] / "schemas"
        shipped = sorted(path.name for path in schema_dir.glob("*.schema.json"))
        self.assertEqual(shipped, [
            "artifact.schema.json", "dataset-report.schema.json",
            "eval.schema.json", "gate-evidence.schema.json",
            "preflight.schema.json", "run.schema.json",
            "split-manifest.schema.json", "training-metrics.schema.json",
        ])

    def test_artifact_meta_matches_schema(self):
        from stage1 import checkpointing
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            checkpointing.write_artifact_meta(
                Path(directory), kind="checkpoint", identity_sha256="a" * 64,
                model_id="test/Model-A", model_revision="b" * 40,
                adapter="minicpm5", gate="smoke-1000", global_step=5,
                supervised_tokens_seen=100, epochs_completed=0.5, run_id="run1")
            meta = (Path(directory) / checkpointing.CHECKPOINT_META_NAME)
            schema.validate_artifact(json.loads(meta.read_text()),
                                     schema.load_schema("artifact.schema.json"),
                                     "artifact meta")


if __name__ == "__main__":
    unittest.main()
