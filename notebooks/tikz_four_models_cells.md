# Four TikZ models — Modal / Colab
Run the numbered cells in order. Four models × five distinct prompts = 20 generations.
Use an NVIDIA GPU; an A100 40/80 GB is a conservative BF16 starting point (capacity not benchmarked here). Models run one at a time. For smaller GPUs set `LOAD_IN_4BIT=True`; this changes inference precision. Allow disk space for all downloaded checkpoints.

The three Praha repositories are explicitly loaded as pinned upstream bases plus PEFT adapters. TikZilla is a complete checkpoint. Qwen uses `enable_thinking=False`; LFM uses the repository's direct-answer training template because its uploaded template unconditionally opens `<think>`. TikZilla has no thinking-mode switch.

This notebook has been syntax checked, not GPU-executed. Model output is requested as LaTeX only; invalid/truncated generations are recorded, not silently repaired. Rendering uses a timeout and disables shell escape; use a disposable notebook without sensitive mounted files when compiling generated TeX.


## Cell 1 — Python dependencies
Run before importing Transformers/PEFT. If these libraries were already imported, restart the kernel after this cell and continue with Cell 2. Keep the notebook image’s CUDA-enabled PyTorch.

```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformers==5.5.0", "peft==0.21.0", "accelerate>=1.12,<2",
    "bitsandbytes>=0.48,<1", "huggingface_hub>=1.0,<2", "pillow", "sentencepiece"], check=True)
```

## Cell 2 — Install the TikZ renderer
Requires a Debian/Ubuntu notebook image with root or sudo. On Modal, these packages can alternatively be installed in the notebook image.

```python
import os, shutil, subprocess
if not all(shutil.which(x) for x in ("pdflatex", "pdftoppm", "kpsewhich")):
    prefix = [] if os.geteuid() == 0 else ["sudo"]
    if prefix and not shutil.which("sudo"):
        raise RuntimeError("Add texlive-latex-extra, texlive-pictures, texlive-science, texlive-fonts-recommended and poppler-utils to the Modal image.")
    subprocess.run(prefix + ["apt-get", "update", "-qq"], check=True)
    subprocess.run(prefix + ["apt-get", "install", "-y", "-qq",
        "texlive-latex-extra", "texlive-pictures", "texlive-science",
        "texlive-fonts-recommended", "poppler-utils"], check=True)
for package in ("standalone.cls", "tikz.sty", "pgfplots.sty"):
    subprocess.run(["kpsewhich", package], check=True)
```

## Cell 3 — Settings, pinned models, and code-only system prompt

```python
import gc, json, re, time, html, zipfile
from pathlib import Path
import torch
from IPython.display import display, HTML, Image, FileLink
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3_5ForConditionalGeneration, BitsAndBytesConfig, GenerationConfig, set_seed
from peft import PeftConfig, PeftModel

assert torch.cuda.is_available(), "Select a GPU runtime before continuing."
LOAD_IN_4BIT = False
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
MAX_NEW_TOKENS = 4096
OUTPUT_DIR = Path("tikz_outputs").resolve()
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS = {
    "qwen9b": ("Praha-Labs/Qwen3.5-9B-TikZ-LoRA", "f5d92e9dbea683e5932a6b9a7c7aefa87905cea5", True),
    "qwen4b": ("Praha-Labs/Qwen3.5-4B-TikZ-LoRA", "1db8a84d2c285a22095c3d910320703cd717049d", True),
    "lfm": ("Praha-Labs/LFM2.5-2.6B-TikZ-LoRA", "7ef3da23fff1ebd6c8da353b9a4f1d41d354e1af", True),
    "tikzilla": ("nllg/TikZilla-3B-RL", "c1d197f509ba77e6cf0f3eda158060bfd70f50c7", False),
}
SYSTEM_PROMPT = r"""You are a TikZ code generation model. Convert the user's description into a precise, attractive TikZ figure.
Output ONLY a complete compilable LaTeX document, starting with \documentclass[tikz,border=5pt]{standalone} and ending with \end{document}.
Include all required packages and TikZ libraries. Use TikZ and pgfplots where appropriate. Use only standard installed TeX packages and no external files.
Do not output explanations, reasoning, thinking tags, Markdown fences, or any text outside the LaTeX document."""
print(torch.cuda.get_device_name(0), "| dtype:", DTYPE, "| 4-bit:", LOAD_IN_4BIT)
# Public repositories need no login. For access errors, use huggingface_hub.login()
# interactively; never paste a token into a saved notebook.
```

## Cell 4 — Five different prompts per model
These are 20 distinct examples, not a controlled same-prompt benchmark. Edit any descriptions here.

```python
PROMPTS = {
"qwen9b": [
    "Draw a Transformer architecture diagram: Input Tokens, Embedding, three stacked Transformer Blocks, Linear Head, and Output Tokens. Inside one enlarged block show Attention, Add & Norm, Feed Forward, Add & Norm, with residual arrows. Use blue and orange fills.",
    "Draw a scientific pgfplots line chart of training loss versus epochs 0 to 10. Show a blue training curve decreasing from 2.5 to 0.3 and an orange validation curve decreasing from 2.7 to 0.5. Include a legend and light gray grid.",
    "Draw a directed acyclic graph with six circular nodes A through F. Edges: A to B, A to C, B to D, C to D, C to E, D to F, E to F. Arrange in four left-to-right layers and label the edges with distinct integer weights.",
    "Draw a labeled right triangle with vertices A=(0,0), B=(4,0), C=(4,3). Mark the right angle at B, label side lengths 3, 4, 5, and mark angle theta at A. Use a pale blue fill.",
    "Draw a grouped bar chart comparing accuracy for models A, B, C on datasets X and Y. Values are A: 72 and 68, B: 81 and 77, C: 89 and 85 percent. Include labeled axes, legend, and values above bars."
],
"qwen4b": [
    "Draw a vertical flowchart: Start, Read number n, decision Is n even?, Yes branch Print Even, No branch Print Odd, both joining End. Use rounded terminals, rectangular processes, and a diamond decision.",
    "Draw a solar system schematic with the Sun and circular orbits for Mercury, Venus, Earth, and Mars. Label each planet. Use warm colors for the Sun and distinct colors for planets; no external images.",
    "Plot sin(x) and cos(x) from 0 to 2*pi on the same axes with pgfplots. Use blue solid and red dashed lines, a legend, labeled axes, and ticks at 0, pi/2, pi, 3pi/2, and 2pi.",
    "Draw a balanced binary tree with root 8, children 4 and 12, and leaves 2, 6, 10, 14. Use circular light green nodes and clean straight edges.",
    "Draw a water cycle diagram with four labeled stages Evaporation, Condensation, Precipitation, Collection in a clockwise cycle. Use blue arrows, simple cloud shapes, and a water basin."
],
"lfm": [
    "Draw three horizontally aligned boxes labeled Input, Process, Output, connected by arrows. Use pale blue, pale orange, and pale green fills.",
    "Draw a Venn diagram with two overlapping circles labeled A and B. Shade their intersection purple and label it A intersection B.",
    "Draw a square of side length 4 with both diagonals, center O, and vertices A, B, C, D. Label the bottom side 4 cm.",
    "Draw a horizontal timeline from 2020 to 2024 with five evenly spaced milestone dots. Label them Idea, Prototype, Testing, Launch, Growth, alternating above and below the line.",
    "Draw coordinate axes with vectors u=(3,1) in blue and v=(1,2) in red from the origin. Draw u+v=(4,3) in green and dashed parallelogram construction lines."
],
"tikzilla": [
    "Draw a neural network with 3 input neurons, 4 hidden neurons, and 2 output neurons. Connect adjacent layers fully, use different fill colors per layer, and label the three layers.",
    "Draw a pgfplots scatter plot with points (1,2), (2,2.8), (3,4.1), (4,4.9), (5,6.2) and a red fitted line y=x+1. Label axes x and y and include a legend.",
    "Draw a finite-state machine with states Idle, Running, Paused. Mark Idle as the start state. Add transitions Idle to Running labeled start, Running to Paused labeled pause, Paused to Running labeled resume, Running to Idle labeled stop.",
    "Draw a block diagram of a feedback control system: Reference enters a summing junction, then Controller, Plant, and Output. Feed Output through a Sensor block back to the negative input of the summing junction.",
    "Draw a unit circle centered at the origin with x and y axes. Mark a point at 45 degrees, its radius, dashed projections to both axes, and angle theta. Label the point (cos theta, sin theta)."
]}
assert all(len(p) == 5 for p in PROMPTS.values())
```

## Cell 5 — Explicit base + adapter loading and thinking disabled

```python
LFM_DIRECT_TEMPLATE = "{#\n  LFM2.5 single-turn Stage 1 template. Match the model's BOS and ChatML\n  boundary tokens, but do not open a <think> block: assistant targets are\n  direct TikZ, and literal thinking tags must remain unchanged.\n#}\n{{- bos_token -}}\n{%- for message in messages %}\n    {{- '<|im_start|>' + message['role'] + '\\n' }}\n    {{- message['content'] }}\n    {{- '<|im_end|>\\n' }}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"

def load_model(key):
    repo, revision, is_adapter = MODELS[key]
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    if key == "lfm":
        tokenizer.chat_template = LFM_DIRECT_TEMPLATE
    kwargs = dict(device_map={"": 0}, dtype=DTYPE, attn_implementation="sdpa")
    if LOAD_IN_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE, bnb_4bit_use_double_quant=True)
    if is_adapter:
        cfg = PeftConfig.from_pretrained(repo, revision=revision)
        if not cfg.revision:
            raise ValueError("Adapter has no pinned base revision.")
        # Preserve the multimodal model's module names to match the trained adapter.
        cls = Qwen3_5ForConditionalGeneration if key.startswith("qwen") else AutoModelForCausalLM
        base = cls.from_pretrained(cfg.base_model_name_or_path, revision=cfg.revision, **kwargs)
        model = PeftModel.from_pretrained(base, repo, revision=revision, is_trainable=False)
        assert "default" in model.peft_config
        assert any("lora_A" in n for n, _ in model.named_parameters()), "LoRA weights missing"
        print("Loaded base:", cfg.base_model_name_or_path, "revision:", cfg.revision)
        print("Active adapter:", model.active_adapters)
    else:
        model = AutoModelForCausalLM.from_pretrained(repo, revision=revision, **kwargs)
    model.eval()
    return model, tokenizer

def prompt_text(tokenizer, key, description):
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": description},
    ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tail = text.rsplit("<|im_start|>assistant", 1)[-1]
    if key.startswith("qwen"):
        assert "</think>" in tail, "Qwen template did not close thinking mode."
    else:
        assert "<think>" not in tail, "Unexpected thinking prefix."
    return text

def generate(model, tokenizer, key, description, seed):
    set_seed(seed)
    text = prompt_text(tokenizer, key, description)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda:0")
    eos_ids = {tokenizer.eos_token_id}
    if "<|im_end|>" in tokenizer.get_vocab():
        eos_ids.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    eos_ids.discard(None)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
        temperature=0.7, top_p=0.9, top_k=50, use_cache=True,
        eos_token_id=sorted(eos_ids), pad_token_id=pad_id)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=config)
    ids = output[0, inputs.input_ids.shape[1]:]
    raw = tokenizer.decode(ids, skip_special_tokens=True)
    return raw, len(ids), time.perf_counter()-started, bool(len(ids) >= MAX_NEW_TOKENS and ids[-1].item() not in eos_ids)
```

## Cell 6 — Compile LaTeX, display PNGs, save code and failure logs

```python
def extract_latex(raw):
    # Save raw output separately; strip wrappers only, never rewrite the figure.
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    match = re.search(r"\\documentclass(?:\s*\[[^\]]*\])?\s*\{.*?\\end\{document\}", text, re.S)
    if not match:
        raise ValueError("No complete LaTeX document found; inspect raw.txt for truncation or format errors.")
    return match.group(0)

def render_latex(latex, folder):
    (folder / "figure.tex").write_text(latex)
    env = dict(os.environ, openin_any="p", openout_any="p")
    proc = subprocess.run(["pdflatex", "-no-shell-escape", "-interaction=nonstopmode",
        "-halt-on-error", "figure.tex"], cwd=folder, env=env,
        capture_output=True, text=True, timeout=60)
    (folder / "compile.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode or not (folder / "figure.pdf").exists():
        raise RuntimeError("LaTeX compilation failed; see compile.log")
    subprocess.run(["pdftoppm", "-png", "-r", "140", "-singlefile", "-f", "1",
        "figure.pdf", "figure"], cwd=folder, capture_output=True, check=True, timeout=30)
    return folder / "figure.png"

RESULTS = []
def run_model(key):
    model = tokenizer = None
    try:
        model, tokenizer = load_model(key)
        for i, description in enumerate(PROMPTS[key], 1):
            folder = OUTPUT_DIR / key / f"example_{i}"
            folder.mkdir(parents=True, exist_ok=True)
            # Prevent stale successful render artifacts on reruns.
            for name in ("figure.tex", "figure.pdf", "figure.png", "compile.log", "raw.txt"):
                (folder / name).unlink(missing_ok=True)
            row = dict(model=key, repo=MODELS[key][0], revision=MODELS[key][1],
                       example=i, prompt=description, quantized=LOAD_IN_4BIT, seed=42+i)
            display(HTML(f"<h3>{html.escape(key)} — example {i}</h3><p>{html.escape(description)}</p>"))
            try:
                raw, count, seconds, truncated = generate(model, tokenizer, key, description, 42+i)
                (folder / "raw.txt").write_text(raw)
                row.update(tokens=count, seconds=round(seconds, 2), truncated=truncated)
                latex = extract_latex(raw)
                display(HTML("<details><summary>TikZ / LaTeX code</summary><pre>" + html.escape(latex) + "</pre></details>"))
                png = render_latex(latex, folder)
                display(Image(filename=str(png)))
                row["status"] = "rendered"
            except Exception as exc:
                row.update(status="failed", error=str(exc))
                print("FAILED:", exc)
                if (folder / "compile.log").exists():
                    print((folder / "compile.log").read_text()[-2500:])
            (folder / "result.json").write_text(json.dumps(row, indent=2))
            RESULTS.append(row)
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
```

## Cell 7 — qwen9b: generate and render five outputs

```python
run_model("qwen9b")
```

## Cell 8 — qwen4b: generate and render five outputs

```python
run_model("qwen4b")
```

## Cell 9 — lfm: generate and render five outputs

```python
run_model("lfm")
```

## Cell 10 — tikzilla: generate and render five outputs

```python
run_model("tikzilla")
```

## Cell 11 — Results and downloadable ZIP
Download the archive before stopping the notebook. It contains raw model responses, extracted TeX, successful PDFs/PNGs, and metadata/logs.

```python
rows = [json.loads(p.read_text()) for p in sorted(OUTPUT_DIR.glob("*/example_*/result.json"))]
for row in rows:
    print(f"{row['model']:10} example {row['example']}: {row['status']} | tokens={row.get('tokens', '-')} | truncated={row.get('truncated', '-')}")
(OUTPUT_DIR / "summary.json").write_text(json.dumps(rows, indent=2))
archive = Path("tikz_outputs.zip").resolve()
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for path in OUTPUT_DIR.rglob("*"):
        if path.is_file():
            z.write(path, path.relative_to(OUTPUT_DIR.parent))
display(FileLink(str(archive.relative_to(Path.cwd()))))
# Colab only: from google.colab import files; files.download(str(archive))
```

## Sources
- [Qwen 9B adapter configuration](https://huggingface.co/Praha-Labs/Qwen3.5-9B-TikZ-LoRA/blob/main/adapter_config.json)
- [Qwen 4B model card](https://huggingface.co/Praha-Labs/Qwen3.5-4B-TikZ-LoRA)
- [LFM adapter configuration](https://huggingface.co/Praha-Labs/LFM2.5-2.6B-TikZ-LoRA/blob/main/adapter_config.json)
- [LFM uploaded chat template](https://huggingface.co/Praha-Labs/LFM2.5-2.6B-TikZ-LoRA/blob/main/chat_template.jinja)
- [TikZilla model card](https://huggingface.co/nllg/TikZilla-3B-RL)
- [Qwen3.5 Transformers classes](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
