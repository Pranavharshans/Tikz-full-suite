"""Tests for TikZ extraction and safe TeX compilation.

Real pdflatex tests are skipped when TeX is unavailable; the safety tests
(denylist, timeout, unavailable engine) never need TeX.
"""
import unittest
from pathlib import Path

from tests import support

from stage1 import compile_tikz

VALID_PICTURE = ("Here is the code:\n```latex\n"
                 "\\begin{tikzpicture}\n"
                 "\\draw[thick, blue] (0,0) -- (2,1);\n"
                 "\\node at (1,1.5) {$x^2$};\n"
                 "\\end{tikzpicture}\n```\n")
INVALID_PICTURE = ("\\begin{tikzpicture}\n"
                   "\\draw (0,0) -- (1,1)\n"          # missing semicolon
                   "\\undefinedcommand\n"
                   "\\end{tikzpicture}\n")
FULL_DOCUMENT = ("\\documentclass{article}\n\\usepackage{tikz}\n"
                 "\\begin{document}\n\\begin{tikzpicture}\\draw (0,0) circle (1);"
                 "\\end{tikzpicture}\n\\end{document}\n")


class ExtractionTests(unittest.TestCase):
    def test_fenced_picture_is_extracted(self):
        extraction = compile_tikz.extract_tikz(VALID_PICTURE)
        self.assertEqual(extraction.kind, "picture")
        self.assertTrue(extraction.code.startswith("\\begin{tikzpicture}"))
        self.assertIn("blue", extraction.code)

    def test_full_document_is_detected(self):
        extraction = compile_tikz.extract_tikz(FULL_DOCUMENT)
        self.assertEqual(extraction.kind, "document")

    def test_bare_drawing_commands_are_wrapped(self):
        extraction = compile_tikz.extract_tikz("\\draw (0,0) -- (1,1);")
        self.assertEqual(extraction.kind, "picture")
        self.assertIn("\\draw", extraction.code)

    def test_no_tikz_is_reported(self):
        extraction = compile_tikz.extract_tikz("I cannot draw that.")
        self.assertEqual(extraction.kind, "none")

    def test_empty_output_is_reported(self):
        self.assertEqual(compile_tikz.extract_tikz("").kind, "none")
        self.assertEqual(compile_tikz.extract_tikz(None).kind, "none")

    def test_build_document_uses_a_standalone_preamble(self):
        extraction = compile_tikz.extract_tikz(VALID_PICTURE)
        document = compile_tikz.build_document(extraction)
        self.assertIn("standalone", document)
        self.assertIn("\\begin{document}", document)
        self.assertIn("\\end{document}", document)


class SafetyTests(unittest.TestCase):
    def test_file_io_primitives_are_refused_without_execution(self):
        payload = ("\\begin{tikzpicture}\\draw (0,0) -- (1,1);"
                   "\\immediate\\write18{echo pwned > /tmp/stage1-pwned}"
                   "\\end{tikzpicture}")
        result = compile_tikz.compile_tikz(payload)
        self.assertEqual(result.status, "unsafe_primitive")
        self.assertIn("immediate", result.error_message)
        self.assertFalse(Path("/tmp/stage1-pwned").exists())

    def test_input_primitive_is_refused(self):
        payload = ("\\begin{tikzpicture}\\draw (0,0) -- (1,1);"
                   "\\input{/etc/passwd}\\end{tikzpicture}")
        result = compile_tikz.compile_tikz(payload)
        self.assertEqual(result.status, "unsafe_primitive")

    def test_text_outside_the_picture_is_not_part_of_the_document(self):
        payload = ("\\begin{tikzpicture}\\draw (0,0) -- (1,1);\\end{tikzpicture}\n"
                   "\\immediate\\write18{echo pwned}")
        extraction = compile_tikz.extract_tikz(payload)
        self.assertNotIn("write18", extraction.code)
        document = compile_tikz.build_document(extraction)
        self.assertNotIn("write18", document)

    def test_no_tikz_short_circuits(self):
        result = compile_tikz.compile_tikz("no code here")
        self.assertEqual(result.status, "no_tikz")
        self.assertEqual(result.category, "no_tikz")

    def test_unavailable_engine_is_reported(self):
        result = compile_tikz.compile_tikz(VALID_PICTURE,
                                           engine="definitely-not-a-tex-engine")
        self.assertEqual(result.status, "unavailable")

    def test_timeout_is_reported_and_kills_the_runner(self):
        def fake_runner(command, cwd, timeout_seconds, log_path):
            Path(log_path).write_text("simulated hang")
            return None, True

        result = compile_tikz.compile_tikz(VALID_PICTURE, runner=fake_runner)
        self.assertEqual(result.status, "timeout")
        self.assertIn("simulated hang", result.log_tail)

    def test_keep_artifacts_retains_the_directory(self):
        def fake_runner(command, cwd, timeout_seconds, log_path):
            Path(cwd, "main.pdf").write_bytes(b"%PDF-1.4 fake")
            Path(log_path).write_text("ok")
            return 0, False

        result = compile_tikz.compile_tikz(VALID_PICTURE, runner=fake_runner,
                                           keep_artifacts=True)
        self.assertEqual(result.status, "success")
        self.assertTrue(Path(result.artifacts_dir).is_dir())
        self.assertIn("main.tex", [item.name for item in Path(result.artifacts_dir).iterdir()])


@support.requires_pdflatex
class RealLatexTests(unittest.TestCase):
    def test_valid_picture_compiles(self):
        result = compile_tikz.compile_tikz(
            VALID_PICTURE, timeout_seconds=60,
            render=support.HAS_PDFTOPPM)
        self.assertEqual(result.status, "success", result.log_tail)
        self.assertTrue(Path(result.pdf_path).is_file())
        if support.HAS_PDFTOPPM:
            self.assertIsNotNone(result.render_path, result.render_error)
            self.assertTrue(Path(result.render_path).is_file())

    def test_invalid_picture_reports_latex_error_with_line(self):
        result = compile_tikz.compile_tikz(INVALID_PICTURE, timeout_seconds=60)
        self.assertEqual(result.status, "latex_error")
        self.assertEqual(result.category, "latex_error")
        self.assertIsNotNone(result.error_message)
        self.assertTrue(result.log_tail)

    def test_full_document_compiles(self):
        result = compile_tikz.compile_tikz(FULL_DOCUMENT, timeout_seconds=60)
        self.assertEqual(result.status, "success", result.log_tail)


if __name__ == "__main__":
    unittest.main()
