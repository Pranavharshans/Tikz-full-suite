import importlib.util
import json
from pathlib import Path
import threading
import tempfile
import argparse
import builtins
import unittest
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("benchmark", Path(__file__).with_name("benchmark.py"))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class Tests(unittest.TestCase):
    def test_slurm_p2p_workaround(self):
        command = [sys.executable, str(Path(b.__file__)), "--slurm-script",
                   "--vllm-sif", "/tmp/vllm.sif", "--sglang-sif", "/tmp/sglang.sif"]
        for extra, expected in [([], "disabled"), (["--nccl-p2p", "auto"], "auto")]:
            script = subprocess.check_output(command + extra, text=True)
            self.assertIn(f"--nccl-p2p {expected}", script)
            self.assertIn("python3", script)
            subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_resume_without_dataset_dependencies(self):
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name in ("datasets", "huggingface_hub"):
                raise ModuleNotFoundError(name)
            return original_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            image.write_bytes(b"fixture")
            rows = [dict(image=str(image), image_sha256=b.hashlib.sha256(b"fixture").hexdigest()) for _ in range(100)]
            b.dump(Path(directory) / "dataset.json", dict(rows=rows))
            with patch("builtins.__import__", side_effect=guarded):
                b.prepare(argparse.Namespace(work=directory))
            image.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Image hash mismatch"):
                b.prepare(argparse.Namespace(work=directory))

    def test_matrix_constraints(self):
        counts = {"engines":9, "mtp":32, "topology":14, "scheduling":12}
        for phase, count in counts.items():
            configs = b.phase_configs(phase, b.config())
            self.assertEqual(len(configs), count)
            for cfg in configs:
                self.assertIn(cfg["tp"], (2, 4))
                self.assertLessEqual(cfg["tp"]*cfg["replicas"], 4)
                self.assertGreaterEqual(cfg["concurrency"], cfg["replicas"])

    def test_selection_excludes_failures_and_control(self):
        rows = [dict(config=b.config(mtp=m), successful=ok, failed=100-ok,
                     samples_per_gpu_hour=speed) for m, ok, speed in [(0,100,999),(2,99,900),(3,100,20)]]
        self.assertEqual(b.best(rows, True)["mtp"], 3)

    def test_reasoning_separation(self):
        self.assertEqual(b.split_reasoning("<think>private reasoning</think>Draw A"), ("private reasoning", "Draw A"))

    def test_sse_completion_and_truncation(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                assert payload["chat_template_kwargs"]["enable_thinking"] is True
                assert payload["max_tokens"] == 20000
                self.send_response(200)
                self.end_headers()
                for event in [dict(choices=[dict(delta={"reasoning_content":"analysis"})]),
                              dict(choices=[dict(delta={"content":"Draw a circle"}, finish_reason=self.server.finish)]),
                              dict(choices=[], usage={"prompt_tokens":12,"completion_tokens":25})]:
                    self.wfile.write(("data: "+json.dumps(event)+"\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.object(b, "messages", return_value=[]):
                for finish, expected in [("stop",True),("length",False)]:
                    server.finish = finish
                    row = b.http_request(server.server_port, {"id":"1"}, 20000, 5)
                    self.assertEqual(row["ok"], expected)
                    self.assertEqual(row["reasoning"], "analysis")
                    self.assertEqual(row["final"], "Draw a circle")
                    self.assertEqual(row["usage"]["completion_tokens"], 25)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
