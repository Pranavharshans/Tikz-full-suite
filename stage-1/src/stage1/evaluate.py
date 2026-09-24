"""Evaluation metrics for base-model and checkpoint outputs.

Pure metric computation over ``GenerationRecord`` objects, with TeX compilation
and image similarity injected so the module stays testable without a GPU. The
GPU path lives in ``scripts/evaluate.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import data
from .data import LengthAccumulator, percentile
from .util import sha256_text, utc_now_iso

EVAL_SCHEMA_VERSION = "stage1-eval-v1"


def normalize_text(text) -> str:
    return " ".join((text or "").split())


def ngram_hashes(text: str, n: int) -> set:
    tokens = normalize_text(text).split()
    if len(tokens) < n:
        return {sha256_text(" ".join(tokens))} if tokens else set()
    return {sha256_text(" ".join(tokens[index:index + n]))
            for index in range(len(tokens) - n + 1)}


class NearDuplicateIndex:
    """Inverted index over sampled training targets for overlap detection."""

    def __init__(self, samples, n: int):
        self.n = n
        self.entries = []           # (row_id, gram_set)
        self.inverted = {}          # gram hash -> list of entry indices
        for row_id, text in samples:
            grams = ngram_hashes(text, n)
            if not grams:
                continue
            index = len(self.entries)
            self.entries.append((row_id, grams))
            for gram in grams:
                self.inverted.setdefault(gram, []).append(index)

    def query(self, text: str) -> tuple:
        """Return ``(best_overlap, best_row_id)``; 0.0 when nothing overlaps."""
        grams = ngram_hashes(text, self.n)
        if not grams:
            return 0.0, None
        candidates = {}
        for gram in grams:
            for index in self.inverted.get(gram, ()):
                candidates[index] = candidates.get(index, 0) + 1
        best_overlap, best_id = 0.0, None
        for index in candidates:
            row_id, other = self.entries[index]
            union = len(grams) + len(other) - candidates[index]
            overlap = candidates[index] / union if union else 0.0
            if overlap > best_overlap:
                best_overlap, best_id = overlap, row_id
        return best_overlap, best_id


@dataclass(frozen=True)
class SimilarityResult:
    status: str            # ok | not_available | error
    similarity: float | None
    reason: str | None

    def to_jsonable(self) -> dict:
        return {"status": self.status, "similarity": self.similarity,
                "reason": self.reason}


def image_similarity(render_png, reference_bytes, *, size: int = 64) -> SimilarityResult:
    """Grayscale normalized-L1 similarity in [0, 1] using Pillow only."""
    try:
        from PIL import Image
    except ImportError:
        return SimilarityResult("not_available", None, "Pillow is not installed")
    try:
        import io
        with Image.open(render_png) as image:
            render = image.convert("L").resize((size, size))
        with Image.open(io.BytesIO(reference_bytes)) as image:
            reference = image.convert("L").resize((size, size))
        render_pixels = list(render.getdata())
        reference_pixels = list(reference.getdata())
        if len(render_pixels) != len(reference_pixels):
            return SimilarityResult("error", None, "pixel count mismatch")
        difference = sum(abs(a - b) for a, b in zip(render_pixels, reference_pixels))
        similarity = 1.0 - difference / (255.0 * len(render_pixels))
        return SimilarityResult("ok", round(similarity, 6), None)
    except Exception as exc:
        return SimilarityResult("error", None, f"{type(exc).__name__}: {exc}")


def select_evaluation_ids(manifest: dict, split: str, *, limit, seed: int,
                          eligibility: dict) -> list:
    """Evaluation rows for a split, excluding that model's quarantined rows."""
    return data.select_split_ids(manifest, split, limit=limit, seed=seed,
                                 exclude=eligibility["quarantined"])


def select_memorization_ids(manifest: dict, *, limit, seed: int,
                            eligibility: dict) -> list:
    """Training-target sample for memorization checks, excluding quarantine."""
    return data.select_split_ids(manifest, "train", limit=limit, seed=seed,
                                 exclude=eligibility["quarantined"])


def _summary(values) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 6),
        "median": ordered[len(ordered) // 2],
        "p10": percentile(ordered, 0.10),
        "max": max(ordered),
    }


def evaluate_records(records, *, evaluation, compile_fn=None,
                     compile_enabled: bool = True, reference_images=None,
                     train_targets=None) -> dict:
    """Compute the full evaluation metric set.

    ``compile_fn(text) -> CompileResult`` and ``reference_images`` (row id ->
    PNG bytes) are injected; ``train_targets`` is an iterable of
    ``(row_id, tikz_code)`` used for duplicate/memorization checks.

    When ``compile_fn`` is not injected, compilation runs in a fresh temporary
    directory tree that is removed before this function returns; callers that
    want to keep TeX evidence should inject their own ``compile_fn`` with a
    persistent workdir (``scripts/evaluate.py`` does).
    """
    import shutil
    import tempfile
    from pathlib import Path

    if compile_fn is not None:
        return _evaluate_records_impl(
            records, evaluation=evaluation, compile_fn=compile_fn,
            compile_enabled=compile_enabled, reference_images=reference_images,
            train_targets=train_targets)
    import itertools

    from .compile_tikz import compile_tikz

    tex_root = Path(tempfile.mkdtemp(prefix="stage1-eval-tex-"))
    counter = itertools.count()

    def temporary_compile(text):
        workdir = tex_root / f"row-{next(counter):05d}"
        return compile_tikz(
            text, engine=evaluation.compile.engine,
            timeout_seconds=evaluation.compile.timeout_seconds,
            render=evaluation.compile.render, workdir=workdir)

    try:
        return _evaluate_records_impl(
            records, evaluation=evaluation, compile_fn=temporary_compile,
            compile_enabled=compile_enabled, reference_images=reference_images,
            train_targets=train_targets)
    finally:
        shutil.rmtree(tex_root, ignore_errors=True)


def _evaluate_records_impl(records, *, evaluation, compile_fn,
                           compile_enabled: bool = True, reference_images=None,
                           train_targets=None) -> dict:
    """Compute the full evaluation metric set.

    ``compile_fn(text) -> CompileResult`` and ``reference_images`` (row id ->
    PNG bytes) are injected; ``train_targets`` is an iterable of
    ``(row_id, tikz_code)`` used for duplicate/memorization checks.
    """
    from .compile_tikz import extract_tikz

    records = list(records)
    if not records:
        raise ValueError("evaluate_records requires at least one record")

    generation = {
        "examples": len(records),
        "completed": 0,
        "truncated": 0,
        "errors": 0,
        "empty": 0,
    }
    extraction_counts = {"picture": 0, "document": 0, "none": 0}
    compile_categories = {}
    compile_attempts = 0
    compile_success = 0
    first_errors = []
    output_tokens = LengthAccumulator()
    output_chars = LengthAccumulator()
    latencies = []
    total_output_tokens = 0

    normalized_outputs = {}
    rows = []
    similarities = []
    similarity_status = {"ok": 0, "not_available": 0, "error": 0}
    similarity_reasons = {}

    for record in records:
        normalized = normalize_text(record.raw_output)
        extraction = extract_tikz(record.raw_output)
        if record.error:
            generation["errors"] += 1
        elif record.finish_reason == "stop":
            generation["completed"] += 1
        elif record.finish_reason == "length":
            generation["truncated"] += 1
        if not normalized:
            generation["empty"] += 1
        extraction_counts[extraction.kind] = extraction_counts.get(extraction.kind, 0) + 1
        output_tokens.add(record.output_tokens)
        output_chars.add(len(record.raw_output))
        total_output_tokens += record.output_tokens
        latencies.append(record.latency_s)

        compiled = compile_fn(record.raw_output) if (compile_enabled and not record.error) else None
        if compiled is not None:
            compile_status = compiled.status
            compile_category = compiled.category
            compile_attempts += 1
            if compiled.compiled:
                compile_success += 1
            else:
                compile_categories[compile_category] = compile_categories.get(
                    compile_category, 0) + 1
                if compiled.error_message and len(first_errors) < 10:
                    first_errors.append({
                        "row_id": record.row_id,
                        "category": compile_category,
                        "message": compiled.error_message,
                        "line": compiled.error_line,
                    })
        else:
            compile_status = "skipped"
            compile_category = "disabled" if not compile_enabled else "generation_error"

        similarity = SimilarityResult("not_available", None, "no reference image")
        if (compiled is not None and compiled.compiled and compiled.render_path
                and evaluation.render_similarity.enabled
                and reference_images and record.row_id in reference_images):
            similarity = image_similarity(
                compiled.render_path, reference_images[record.row_id],
                size=evaluation.render_similarity.size)
        if compiled is not None and compiled.compiled:
            similarity_status[similarity.status] = similarity_status.get(
                similarity.status, 0) + 1
            if similarity.status == "ok":
                similarities.append(similarity.similarity)
            elif similarity.reason:
                similarity_reasons.setdefault(similarity.reason, 0)
                similarity_reasons[similarity.reason] += 1

        normalized_outputs.setdefault(sha256_text(normalized), []).append(record.row_id)
        rows.append({
            "row_id": record.row_id,
            "source_row_index": record.source_row_index,
            "finish_reason": record.finish_reason,
            "empty": not normalized,
            "extraction_kind": extraction.kind,
            "compile_status": compile_status,
            "compile_category": compile_category,
            "output_tokens": record.output_tokens,
            "output_chars": len(record.raw_output),
            "latency_s": round(record.latency_s, 4),
            "normalized_sha256": sha256_text(normalized),
            "similarity": similarity.similarity,
            "similarity_status": similarity.status,
        })

    duplicate_groups = {digest: ids for digest, ids in normalized_outputs.items()
                        if len(ids) > 1}
    duplicate_rows = sum(len(ids) - 1 for ids in duplicate_groups.values())

    # Memorization checks against training targets.
    memorization = {
        "train_targets_checked": 0,
        "exact_matches": 0,
        "near_matches": 0,
        "overlap_threshold": evaluation.duplicate_overlap_threshold,
        "ngram_size": evaluation.duplicate_ngram_size,
    }
    if train_targets:
        targets = list(train_targets)
        memorization["train_targets_checked"] = len(targets)
        exact_index = {sha256_text(normalize_text(text)) for _, text in targets}
        near_index = NearDuplicateIndex(targets, evaluation.duplicate_ngram_size)
        records_by_id = {record.row_id: record for record in records}
        for row in rows:
            if row["normalized_sha256"] in exact_index:
                row["memorized"] = "exact"
                memorization["exact_matches"] += 1
                continue
            overlap, match_id = near_index.query(records_by_id[row["row_id"]].raw_output)
            row["near_duplicate_overlap"] = round(overlap, 4)
            row["near_duplicate_of"] = match_id
            if overlap >= evaluation.duplicate_overlap_threshold:
                row["memorized"] = "near"
                memorization["near_matches"] += 1

    metrics = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "generation": generation,
        "extraction": extraction_counts,
        "compilation": {
            "attempted": compile_attempts,
            "success": compile_success,
            "success_rate": round(compile_success / compile_attempts, 6) if compile_attempts else None,
            "categories": compile_categories,
            "first_errors": first_errors,
        },
        "output_length": {
            "tokens": output_tokens.summary(),
            "chars": output_chars.summary(),
        },
        "latency": {
            **_summary(latencies),
            "total_s": round(sum(latencies), 3),
            "tokens_per_second": round(total_output_tokens / sum(latencies), 3)
            if sum(latencies) > 0 else None,
        },
        "duplicates": {
            "duplicate_groups": len(duplicate_groups),
            "duplicate_rows": duplicate_rows,
            "largest_group": max((len(ids) for ids in duplicate_groups.values()),
                                 default=0),
        },
        "memorization": memorization,
        "render_similarity": {
            "enabled": bool(evaluation.render_similarity.enabled),
            "status": "ok" if similarities else (
                "not_available" if similarity_status.get("not_available") else "no_data"),
            "compared": len(similarities),
            **_summary(similarities),
            "reasons": similarity_reasons,
        },
    }
    return {"metrics": metrics, "rows": rows}
