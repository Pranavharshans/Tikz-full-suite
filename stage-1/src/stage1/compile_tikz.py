"""TikZ extraction and safe TeX compilation.

Safety rules, all enforced on every invocation:

- ``-no-shell-escape`` (no ``\\write18`` execution), plus
  ``-cnf-line=openin_any=p -cnf-line=openout_any=p`` to restrict file access.
- An explicit denylist of file/IO TeX primitives (``\\input``, ``\\include``,
  ``\\openout``, ``\\read``, ``\\write``, ``\\immediate``). A hit is reported
  as ``unsafe_primitive`` and nothing is executed.
- Compilation runs in a fresh isolated temporary directory with a minimal
  environment, a strict wall-clock timeout, and the full log captured.
- Timeout kills the whole process group; TeX processes can spawn children.

Compilation success is not a quality claim: it only means the extracted TikZ
compiled with this TeX installation.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .errors import CompileError
from .util import sha256_text

UNSAFE_PATTERN = re.compile(
    r"\\(input|include|openout|read|write|immediate)\b", re.IGNORECASE)
DRAWING_PATTERN = re.compile(
    r"\\(draw|node|path|fill|filldraw|shade|shadedraw|coordinate|foreach|clip|matrix)\b")
FENCE_PATTERN = re.compile(r"```(?:latex|tex|tikz)?\s*(.*?)```", re.DOTALL)
PICTURE_PATTERN = re.compile(
    r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", re.DOTALL)

PREAMBLE = "\\documentclass[border=2pt]{standalone}\n" \
           "\\usepackage[T1]{fontenc}\n" \
           "\\usepackage{tikz}\n" \
           "\\usepackage{amsmath,amssymb}\n" \
           "\\begin{document}\n"
POSTAMBLE = "\n\\end{document}\n"

LOG_TAIL_LINES = 40
LOG_TAIL_CHARS = 4000


@dataclass(frozen=True)
class Extraction:
    kind: str          # picture | document | none
    code: str
    note: str

    def to_jsonable(self) -> dict:
        return {"kind": self.kind, "note": self.note, "code_sha256":
                sha256_text(self.code) if self.code else None}


@dataclass(frozen=True)
class CompileResult:
    status: str        # success | no_tikz | unsafe_primitive | timeout |
                       # latex_error | unavailable | compile_error
    category: str
    extraction_kind: str
    extraction_note: str
    returncode: int | None
    duration_s: float
    error_line: int | None
    error_message: str | None
    log_tail: str
    artifacts_dir: str | None
    pdf_path: str | None
    render_path: str | None
    render_error: str | None

    @property
    def compiled(self) -> bool:
        return self.status == "success"

    def to_jsonable(self) -> dict:
        return {
            "status": self.status,
            "category": self.category,
            "extraction_kind": self.extraction_kind,
            "extraction_note": self.extraction_note,
            "returncode": self.returncode,
            "duration_s": round(self.duration_s, 4),
            "error_line": self.error_line,
            "error_message": self.error_message,
            "log_tail": self.log_tail,
            "pdf_path": self.pdf_path,
            "render_path": self.render_path,
            "render_error": self.render_error,
        }


def _extract_document(candidate: str):
    """Return a complete document including its preamble, or None."""
    begin = candidate.find("\\begin{document}")
    if begin == -1:
        return None
    end = candidate.find("\\end{document}", begin)
    if end == -1:
        return None
    start = candidate.rfind("\\documentclass", 0, begin)
    if start == -1:
        start = begin
    return candidate[start:end + len("\\end{document}")]


def extract_tikz(text) -> Extraction:
    """Deterministically locate the TikZ payload in model output."""
    if not isinstance(text, str) or not text.strip():
        return Extraction("none", "", "empty output")
    fenced = FENCE_PATTERN.findall(text)
    candidate = max(fenced, key=len) if fenced else text
    document = _extract_document(candidate)
    if document:
        return Extraction("document", document, "complete LaTeX document")
    picture = PICTURE_PATTERN.search(candidate)
    if picture:
        return Extraction("picture", picture.group(0), "tikzpicture environment")
    if DRAWING_PATTERN.search(candidate):
        return Extraction("picture", candidate.strip(),
                          "bare drawing commands wrapped in tikzpicture")
    return Extraction("none", "", "no TikZ picture found")


def build_document(extraction: Extraction) -> str:
    if extraction.kind == "document":
        return extraction.code if extraction.code.endswith("\n") else extraction.code + "\n"
    if extraction.kind == "picture":
        return PREAMBLE + extraction.code + POSTAMBLE
    raise CompileError("Cannot build a document from an empty extraction")


def check_engine_available(engine: str) -> str | None:
    return shutil.which(engine)


def _default_runner(command, cwd, timeout_seconds, log_path):
    """Run the engine, capture the log, and kill the whole group on timeout."""
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(cwd),
        "TEXMFVAR": str(Path(cwd) / "texmf-var"),
        "TEXMFCONFIG": str(Path(cwd) / "texmf-config"),
        "SOURCE_DATE_EPOCH": "0",
    }
    with open(log_path, "wb") as log_handle:
        process = subprocess.Popen(
            command, cwd=str(cwd), env=environment, stdout=log_handle,
            stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout_seconds)
            return returncode, False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:  # pragma: no cover - already gone
                pass
            process.wait()
            return None, True


def _parse_log(log_text: str) -> tuple:
    """Return ``(error_line, error_message)`` from a pdflatex log."""
    match = re.search(r"^(.+?):(\d+): (.*)$", log_text, re.MULTILINE)
    if match:
        return int(match.group(2)), match.group(3).strip()[:300]
    match = re.search(r"^! (.*)$", log_text, re.MULTILINE)
    if match:
        return None, match.group(1).strip()[:300]
    return None, None


def _tail(log_text: str) -> str:
    lines = log_text.strip().splitlines()
    return "\n".join(lines[-LOG_TAIL_LINES:])[-LOG_TAIL_CHARS:]


def compile_tikz(text, *, engine: str = "pdflatex", timeout_seconds: int = 20,
                 render: bool = False, keep_artifacts: bool = False,
                 workdir=None, runner=None, render_runner=None) -> CompileResult:
    """Extract and compile TikZ from model output; never raises for TeX errors."""
    import time
    started = time.monotonic()
    extraction = extract_tikz(text)

    def result(status, category, **overrides):
        payload = dict(
            status=status, category=category,
            extraction_kind=extraction.kind, extraction_note=extraction.note,
            returncode=None, duration_s=time.monotonic() - started,
            error_line=None, error_message=None, log_tail="",
            artifacts_dir=None, pdf_path=None, render_path=None,
            render_error=None)
        payload.update(overrides)
        return CompileResult(**payload)

    if extraction.kind == "none":
        return result("no_tikz", "no_tikz")
    unsafe = UNSAFE_PATTERN.search(extraction.code)
    if unsafe:
        return result(
            "unsafe_primitive", "unsafe_primitive",
            error_message=f"refused TeX primitive {unsafe.group(0)!r}")
    engine_path = check_engine_available(engine)
    if engine_path is None:
        return result("unavailable", "unavailable",
                      error_message=f"{engine} not found on PATH")

    if workdir is not None:
        base = Path(workdir).resolve()
        base.mkdir(parents=True, exist_ok=True)
        keep = True
    else:
        base = Path(tempfile.mkdtemp(prefix="stage1-tex-"))
        # A rendered PNG only stays usable while its directory exists, so
        # rendering implies keeping the artifacts. Callers doing bulk
        # evaluation should pass an explicit workdir they manage.
        keep = keep_artifacts or render
    try:
        document = build_document(extraction)
        tex_path = base / "main.tex"
        tex_path.write_text(document)
        log_path = base / "main.log.run"
        command = [
            engine_path,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-no-shell-escape",
            "-cnf-line=openin_any=p",
            "-cnf-line=openout_any=p",
            "main.tex",
        ]
        run = runner or _default_runner
        returncode, timed_out = run(command, base, timeout_seconds, log_path)
        log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
        if timed_out:
            return result("timeout", "timeout", log_tail=_tail(log_text),
                          artifacts_dir=str(base) if keep else None)
        if returncode != 0:
            error_line, error_message = _parse_log(log_text)
            return result("latex_error", "latex_error", returncode=returncode,
                          error_line=error_line, error_message=error_message,
                          log_tail=_tail(log_text),
                          artifacts_dir=str(base) if keep else None)
        pdf_path = base / "main.pdf"
        if not pdf_path.is_file():
            return result("compile_error", "compile_error", returncode=returncode,
                          error_message="engine exited 0 but produced no PDF",
                          log_tail=_tail(log_text),
                          artifacts_dir=str(base) if keep else None)
        render_path = None
        render_error = None
        if render:
            renderer = render_runner or render_pdf
            try:
                render_path = renderer(pdf_path, base / "render.png")
            except CompileError as exc:
                render_error = str(exc)
        return result("success", "success", returncode=returncode,
                      log_tail=_tail(log_text),
                      artifacts_dir=str(base) if keep else None,
                      pdf_path=str(pdf_path) if keep else None,
                      render_path=str(render_path) if (keep and render_path) else None,
                      render_error=render_error)
    finally:
        if not keep and workdir is None:
            shutil.rmtree(base, ignore_errors=True)


def render_pdf(pdf_path, output_png, *, dpi: int = 150) -> Path:
    """Render a PDF to PNG with pdftoppm; raises when unavailable."""
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        raise CompileError("pdftoppm not found on PATH; cannot render PDF")
    pdf_path = Path(pdf_path)
    output_png = Path(output_png)
    prefix = output_png.with_suffix("")
    completed = subprocess.run(
        [pdftoppm, "-r", str(dpi), "-png", "-singlefile", str(pdf_path),
         str(prefix)],
        capture_output=True, text=True, timeout=60)
    if completed.returncode != 0 or not output_png.is_file():
        raise CompileError(
            f"pdftoppm failed ({completed.returncode}): "
            f"{completed.stderr.strip()[:200]}")
    return output_png
