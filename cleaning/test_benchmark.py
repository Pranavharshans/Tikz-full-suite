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
    def test_screen_sizes_and_nested_samples(self):
        expected = {1:16, 2:16, 4:16, 8:16, 16:32, 32:64, 64:100, 100:100}
        manifest = dict(rows=[dict(id=str(i), tikz_code="x" * i) for i in range(100)])
        for concurrency, count in expected.items():
            self.assertEqual(b.screen_size(concurrency), count)
            rows = b.screen_manifest(manifest, count)["rows"]
            self.assertEqual(len(rows), count)
            self.assertEqual(len({r["id"] for r in rows}), count)
            self.assertGreaterEqual(count, concurrency)
            self.assertEqual(rows, b.screen_manifest(manifest, 100)["rows"][:count])
        reversed_manifest = dict(rows=list(reversed(manifest["rows"])))
        self.assertEqual(b.screen_manifest(manifest, 16), b.screen_manifest(reversed_manifest, 16))
        strata = [int(r["id"]) // 25 for r in b.screen_manifest(manifest, 16)["rows"]]
        self.assertEqual([strata.count(i) for i in range(4)], [4]*4)

    def test_shortlist_never_treats_screen_as_final_winner(self):
        rows = [dict(config=b.config(mtp=m), samples=n, successful=ok, failed=n-ok,
                     samples_per_gpu_hour=speed) for m,n,ok,speed in
                [(0,16,16,999), (2,16,15,900), (2,16,16,80), (3,100,100,70)]]
        self.assertEqual(len(b.shortlist(rows)), 2)
        self.assertEqual(b.best(rows, True)["mtp"], 3)

    def test_split_slurm_flags(self):
        script = subprocess.check_output([
            sys.executable, str(Path(b.__file__)), "--slurm-script", "--split-screen",
            "--warmup-samples", "1", "--vllm-sif", "/tmp/vllm.sif",
            "--sglang-sif", "/tmp/sglang.sif"], text=True)
        self.assertIn("--split-screen", script)
        self.assertIn("--warmup-samples 1", script)
        self.assertIn("#SBATCH --gres=gpu:a40:4", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_non_thinking_throughput_slurm_flags(self):
        script = subprocess.check_output([
            sys.executable, str(Path(b.__file__)), "--slurm-script",
            "--throughput-screen", "--no-enable-thinking",
            "--max-output-tokens", "256", "--warmup-samples", "1",
            "--vllm-sif", "/tmp/vllm.sif", "--sglang-sif", "/tmp/sglang.sif"],
            text=True)
        self.assertIn("--throughput-screen", script)
        self.assertIn("--no-enable-thinking", script)
        self.assertIn("--max-output-tokens 256", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_rtx_throughput_slurm_flags(self):
        script = subprocess.check_output([
            sys.executable, str(Path(b.__file__)), "--slurm-script",
            "--rtx-throughput-screen", "--no-enable-thinking",
            "--max-output-tokens", "256", "--warmup-samples", "1",
            "--vllm-sif", "/tmp/vllm.sif", "--sglang-sif", "/tmp/sglang.sif"],
            text=True)
        self.assertIn("--rtx-throughput-screen", script)
        self.assertIn("#SBATCH --partition=rtxpro6k", script)
        self.assertIn("#SBATCH --gres=gpu:rtxpro6k:2", script)
        self.assertNotIn("#SBATCH --partition=a40", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_rtx_concurrency_slurm_flags(self):
        script = subprocess.check_output([
            sys.executable, str(Path(b.__file__)), "--slurm-script",
            "--rtx-concurrency-screen", "--no-enable-thinking",
            "--max-output-tokens", "256", "--load-samples", "512",
            "--throughput-mtp", "1", "--batch-token-budget", "32768",
            "--rtx-concurrencies", "64", "--greedy",
            "--model", "nvidia/Qwen3.8-27B-NVFP4",
            "--replicas-per-gpu", "2", "--gpu-memory-utilization", "0.45",
            "--vllm-sif", "/tmp/vllm.sif",
            "--sglang-sif", "/tmp/sglang.sif"], text=True)
        self.assertIn("--rtx-concurrency-screen", script)
        self.assertIn("--load-samples 512", script)
        self.assertIn("--throughput-mtp 1", script)
        self.assertIn("--batch-token-budget 32768", script)
        self.assertIn("--greedy", script)
        self.assertIn("--rtx-concurrencies 64", script)
        self.assertIn("--model nvidia/Qwen3.8-27B-NVFP4", script)
        self.assertIn("--replicas-per-gpu 2", script)
        self.assertIn("--gpu-memory-utilization 0.45", script)
        self.assertIn("#SBATCH --partition=rtxpro6k", script)
        self.assertIn("#SBATCH --gres=gpu:rtxpro6k:2", script)
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_concurrency_parser(self):
        self.assertEqual(b.parse_concurrencies("64, 96,128"), (64, 96, 128))
        for invalid in ("", "0", "64,64", "abc"):
            with self.assertRaises(argparse.ArgumentTypeError):
                b.parse_concurrencies(invalid)

    def test_load_manifest_has_unique_ids(self):
        manifest = {"rows": [{"id": "a", "tikz_code": "x"},
                             {"id": "b", "tikz_code": "y"}]}
        expanded = b.load_manifest(manifest, 512)
        self.assertEqual(len(expanded["rows"]), 512)
        self.assertEqual(len({row["id"] for row in expanded["rows"]}), 512)
        self.assertEqual(expanded["load_test_source_rows"], 2)
        self.assertEqual(expanded["load_test_requests"], 512)

    def test_generation_settings(self):
        thinking = b.generation_settings({"enable_thinking": True})
        instruct = b.generation_settings({"enable_thinking": False})
        self.assertEqual(thinking["temperature"], 1.0)
        self.assertEqual(instruct["temperature"], .7)
        self.assertEqual(instruct["presence_penalty"], 1.5)
        self.assertEqual(b.generation_settings({"greedy": True})["temperature"], 0.0)

    def test_vllm_runtime_workaround_and_metrics(self):
        opts = b.vllm_runtime_options()
        self.assertIs(opts["disable_custom_all_reduce"], True)
        self.assertIs(opts["disable_log_stats"], False)
        self.assertIs(opts["enforce_eager"], False)
        self.assertIn("--disable-custom-all-reduce", b.vllm_runtime_flags())

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

    def test_warmup_accepts_bounded_output_but_not_missing_output(self):
        self.assertTrue(b.warmup_passed({"final": "caption", "finish_reason": "stop"}))
        self.assertTrue(b.warmup_passed({"final": "caption", "finish_reason": "length"}))
        self.assertFalse(b.warmup_passed({"final": "", "finish_reason": "length"}))
        self.assertFalse(b.warmup_passed({"final": "caption", "finish_reason": None}))

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
                job = dict(enable_thinking=True, reasoning_effort="xhigh",
                           max_output_tokens=32768)
                for finish, expected in [("stop",True),("length",False)]:
                    server.finish = finish
                    row = b.http_request(server.server_port, {"id":"1"}, 20000, 5, job)
                    self.assertEqual(row["ok"], expected)
                    self.assertEqual(row["reasoning"], "analysis")
                    self.assertEqual(row["final"], "Draw a circle")
                    self.assertEqual(row["usage"]["completion_tokens"], 25)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
