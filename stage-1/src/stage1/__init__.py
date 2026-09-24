"""Stage 1 supervised fine-tuning infrastructure for text-to-TikZ.

The package is import-safe without torch, transformers, unsloth or pyarrow:
heavy dependencies are imported inside the functions that need them so that
configuration validation, dataset preparation, TeX compilation and unit tests
run on a login node or a laptop.
"""

__version__ = "0.1.0"
