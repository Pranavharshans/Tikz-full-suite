"""Shared error types for Stage 1.

Exit-code contract (implemented by ``stage1.cli.run``):

- ``0`` success
- ``1`` a check or gate criterion failed (preflight, evaluation, audit-style)
- ``2`` handled error: configuration, identity, data, formatting, checkpoint
- ``3`` prerequisites missing: a gate refused to start without prior evidence
"""
from __future__ import annotations


class Stage1Error(RuntimeError):
    """Base class for handled Stage 1 errors."""


class ConfigError(Stage1Error, ValueError):
    """Configuration is invalid, inconsistent or incomplete."""


class IdentityMismatch(Stage1Error):
    """A run directory belongs to a different identity than requested."""

    def __init__(self, differences):
        self.differences = list(differences)
        detail = "\n".join(f"  - {line}" for line in self.differences)
        super().__init__(
            "Run identity mismatch.\n"
            f"{detail}\n"
            "Use a fresh --run-dir for a new run, or restore the matching "
            "inputs. Stage 1 never silently resumes across identity changes.")


class DataError(Stage1Error):
    """Dataset export verification, splitting or quarantine failure."""


class FormattingError(Stage1Error):
    """Chat-template rendering or assistant-boundary detection failure."""


class BatchError(Stage1Error):
    """A batch would violate the padding or loss-mask contract."""


class CheckpointError(Stage1Error):
    """A checkpoint is incomplete, corrupt or belongs to another run."""


class GateError(Stage1Error):
    """A gate refused to start because prerequisite evidence is missing."""


class GateFailed(Stage1Error):
    """A gate ran but did not meet its success criteria."""


class CompileError(Stage1Error):
    """TeX compilation could not be attempted safely."""


class PreflightError(Stage1Error):
    """The preflight could not run (distinct from a failed check)."""


class SchemaError(Stage1Error):
    """A machine-readable artifact does not match its schema."""
