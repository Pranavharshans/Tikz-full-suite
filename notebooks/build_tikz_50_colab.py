"""Build the self-contained four-model Colab comparison notebook."""
import json
from pathlib import Path

OUT = Path(__file__).with_name("tikz_50_four_model_colab.ipynb")

categories = {
    "Charts and data": [
        "Plot two rainfall curves across January to June: City A [40,55,48,70,65,80] and City B [60,50,58,62,72,75] mm. Label months and include a legend.",
        "Draw a grouped bar chart of recycling rates for paper, glass and plastic in Town A [72,61,38] and Town B [65,74,45] percent, with values above bars.",
        "Create a scatter plot of study hours [1,2,3,4,5,6] versus scores [48,54,63,68,79,85], with a dashed upward trend line and labeled axes.",
        "Draw a horizontal stacked bar chart for three projects showing completed and remaining tasks: Alpha 7/3, Beta 4/6, Gamma 8/2. Include a legend.",
        "Plot a blue sine wave and orange cosine wave from 0 to 2 pi with grid, pi-based x ticks and a legend.",
        "Draw a box-and-whisker style schematic for three groups with medians at 4, 6 and 5; label quartiles and whiskers clearly.",
        "Draw a heatmap-like 4 by 4 grid for values 1 through 16, using increasingly dark blue cells and centered numeric labels.",
        "Plot battery charge over six hours at [100,86,70,55,36,18,5] percent, with points, a line and lightly shaded area under it.",
        "Draw a donut-style chart with four labeled sectors: Design 30%, Build 40%, Test 20%, Review 10%, using distinct colors.",
        "Create a histogram of ten observations [1,1,2,2,2,3,3,4,4,5] with integer bins and frequency labels."
    ],
    "Graphs and networks": [
        "Draw a directed graph with nodes A through F and edges A-B, A-C, B-D, C-D, D-E, C-F; highlight the path A-C-D-E in blue.",
        "Draw a balanced binary tree with root 12, children 6 and 18, and leaves 3,9,15,21; use circular nodes and level labels.",
        "Create a three-layer neural network with 4 input, 3 hidden and 2 output units; connect adjacent layers and label them.",
        "Draw a finite-state machine with Idle, Running, Paused and Done; label start, pause, resume and finish transitions.",
        "Draw a dependency DAG for tasks Plan, Design, Code, Test, Review, Ship; show Plan before Design and Code, both before Test, then Review and Ship.",
        "Draw a star network with central Router connected to five numbered devices; use dashed red link to device 5.",
        "Draw a small undirected weighted graph with nodes P,Q,R,S and edges P-Q:2, P-R:5, Q-R:1, Q-S:4, R-S:3.",
        "Draw a family-tree style hierarchy for an organization: Director, two Managers, and two Team members under each Manager.",
        "Draw a bipartite graph with left nodes U1,U2,U3, right nodes V1,V2,V3 and edges U1-V1, U1-V2, U2-V2, U3-V2, U3-V3.",
        "Draw a circular four-node feedback network A to B to C to D to A, plus a diagonal shortcut A to C."
    ],
    "Flowcharts and systems": [
        "Create a flowchart for checking whether an integer is even: Start, Read n, decision n mod 2 equals 0, print Even or Odd, End.",
        "Draw a feedback control loop with Reference, summing junction, Controller, Plant, Output and Sensor in the return path.",
        "Draw a three-stage data pipeline: Collect, Clean, Analyze, each in a colored rounded box with arrows and a final Report icon.",
        "Create a decision tree for weather: Is it raining? Yes leads to Umbrella; No asks Is it sunny? leading to Sunglasses or Jacket.",
        "Draw a client-server diagram with three clients sending requests to one server, which connects to a database; label request and response arrows.",
        "Draw a vertical cycle diagram for water: evaporation, condensation, precipitation, collection, with arrows forming a loop.",
        "Create a swimlane-style process with lanes User and Service: submit request, validate, process, receive result.",
        "Draw a queueing system with arriving tasks, a queue of four slots, one processor and outgoing completed tasks.",
        "Draw a simple compiler pipeline: Source Code, Lexer, Parser, Optimizer, Machine Code, with small intermediate labels.",
        "Create an experiment workflow with Prepare Sample, Run Trial, Record Data, decision Quality OK?, then Publish or Retry."
    ],
    "Geometry and mathematics": [
        "Draw a right triangle with side lengths 3, 4 and 5, labeled vertices A,B,C, a right-angle marker and angle theta.",
        "Draw a unit circle with axes, a 45-degree radius, dashed coordinate projections and labels cos theta and sin theta.",
        "Draw a square ABCD with both diagonals, center O, side length 4 cm and pale blue fill.",
        "Show two intersecting circles A and B, shade the overlap purple and label union and intersection regions.",
        "Draw a parabola y=x squared minus 2 from x=-3 to 3 with vertex, roots and labeled coordinate axes.",
        "Draw a coordinate vector diagram for u=(3,1), v=(1,2), and u+v=(4,3), including dashed parallelogram edges.",
        "Draw a regular hexagon with a circumscribed circle, center O and one highlighted radius.",
        "Draw a tangent line touching a circle at point T, center O, and mark the 90-degree angle between radius OT and tangent.",
        "Show a number line from -3 to 5 with closed dot at -1, open dot at 3, and a shaded interval between them.",
        "Draw a 3D-looking cube with front and rear squares, dashed hidden edges and three labeled edge directions x,y,z."
    ],
    "Scientific and conceptual": [
        "Draw a labeled cell membrane cross-section with two rows of phospholipid heads, tails, and one membrane protein.",
        "Draw a solar-system schematic with Sun and circular orbits for Mercury, Venus, Earth and Mars, each labeled.",
        "Draw a simple electric circuit with battery, resistor, switch and lamp in one loop, with current direction arrow.",
        "Draw an atom schematic with nucleus labeled protons and neutrons and three electrons on two orbital rings.",
        "Draw a food chain from Grass to Grasshopper to Frog to Snake to Hawk, with arrows and simple icon-like shapes.",
        "Draw a side-view landscape showing water table, soil layers, a well and groundwater-flow arrows.",
        "Draw a timeline from 2021 through 2025 with five milestone dots alternating above and below: Idea, Prototype, Pilot, Launch, Growth.",
        "Draw a comparison of three adjacent building silhouettes with heights 2, 4 and 3 units, including a scale line.",
        "Draw a camera pinhole diagram with object, aperture, image plane, and two crossing light rays.",
        "Draw a simple ecosystem cycle connecting Plants, Herbivores, Carnivores, Decomposers and Nutrients with directional arrows."
    ],
}

prompts = [dict(id=i, category=category, prompt=prompt) for i, (category, items) in enumerate(categories.items()) for prompt in items for _ in [None]]
for i, p in enumerate(prompts, 1): p["id"] = i
assert len(prompts) == 50 and all(len(x) == 10 for x in categories.values())

cells = []
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(True)})
def code(s): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.splitlines(True)})

md("# 50 prompt TikZ comparison: four models\nRun cells in order in Colab with an NVIDIA GPU. The same 50 original prompts are sent to each model (200 generations). Models load one at a time. A100 80 GB is a practical starting point for BF16; smaller GPUs may need 4-bit mode. Runtime fit is unverified. Download the ZIP before ending the session.\n")
md("## Cell 1 — install Python and TeX dependencies\nRun once before imports. If Transformers was imported already, restart the runtime after this cell.\n")
code('''import subprocess, sys, shutil, os
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.17.0", "peft==0.21.0", "accelerate>=1.12,<2", "bitsandbytes>=0.48,<1", "huggingface_hub>=1.0,<2", "pillow", "sentencepiece"], check=True)
if not all(shutil.which(x) for x in ("pdflatex", "pdftoppm", "kpsewhich")):
    prefix = [] if os.geteuid() == 0 else ["sudo"]
    subprocess.run(prefix + ["apt-get", "update", "-qq"], check=True)
    subprocess.run(prefix + ["apt-get", "install", "-y", "-qq", "texlive-latex-extra", "texlive-pictures", "texlive-science", "texlive-fonts-recommended", "poppler-utils"], check=True)
for package in ("standalone.cls", "tikz.sty", "pgfplots.sty"):
    subprocess.run(["kpsewhich", package], check=True)
''')
md("## Cell 2 — settings and model metadata\nGemma and Qwen are PEFT adapters. The loader reads their current `adapter_config.json` and uses its base model and revision. If Gemma is private, log in interactively with `huggingface_hub.login()`.\n")
code('''import gc, html, json, re, time, zipfile
from pathlib import Path
import torch
from IPython.display import display, HTML, Image, FileLink
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3_5ForConditionalGeneration, Gemma4UnifiedForConditionalGeneration, BitsAndBytesConfig, GenerationConfig, set_seed
from peft import PeftConfig, PeftModel
assert torch.cuda.is_available(), "Select a Colab GPU runtime"
OUTPUT_DIR = Path("tikz_50_results").resolve()
OUTPUT_DIR.mkdir(exist_ok=True)
LOAD_IN_4BIT = False  # Change to True before loading any model if VRAM is insufficient.
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
MAX_NEW_TOKENS = 3072
MODELS = {
    "gemma4_12b": ("Praha-Labs/Gemma-4-12B-TikZ-LoRA", True),
    "qwen35_9b": ("Praha-Labs/Qwen3.5-9B-TikZ-LoRA", True),
    "tikzilla_3b": ("nllg/TikZilla-3B-RL", False),
    "tikzilla_8b": ("nllg/TikZilla-8B-RL", False),
}
SYSTEM_PROMPT = r"""You generate precise TikZ scientific figures. Output only one complete, compilable LaTeX document beginning with \\documentclass[tikz,border=5pt]{standalone} and ending with \\end{document}. Include all packages and TikZ libraries needed. Use no external files, shell escape, explanations, reasoning, or Markdown fences."""
print(torch.cuda.get_device_name(0), DTYPE)
''')
md("## Cell 3 — 50 original prompts, ten in each category\nThese prompts were written for this comparison and were not sampled from a dataset. Every model receives the identical list in the same order.\n")
code("PROMPTS = " + json.dumps(prompts, ensure_ascii=False, indent=2) + "\nassert len(PROMPTS) == 50 and len({p['prompt'] for p in PROMPTS}) == 50\n(OUTPUT_DIR / 'prompts.txt').write_text('\\n\\n'.join(f\"{p['id']:02d}. [{p['category']}] {p['prompt']}\" for p in PROMPTS) + '\\n')\nprint('Saved', OUTPUT_DIR / 'prompts.txt')\n")
md("## Cell 4 — load, generate, compile, and save\nEach prompt gets the same system instruction. TikZilla's card uses a user-only chat message, so the instruction is included there in the user message. Raw responses and compile logs are retained for failures.\n")
code(r'''def load_model(key):
    repo, is_adapter = MODELS[key]
    tokenizer = AutoTokenizer.from_pretrained(repo)
    kwargs = dict(device_map={"": 0}, dtype=DTYPE, attn_implementation="sdpa")
    if LOAD_IN_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE, bnb_4bit_use_double_quant=True)
    if is_adapter:
        cfg = PeftConfig.from_pretrained(repo)
        base_repo = cfg.base_model_name_or_path
        if not base_repo:
            raise ValueError(f"{repo} has no base_model_name_or_path")
        cls = Gemma4UnifiedForConditionalGeneration if key == "gemma4_12b" else Qwen3_5ForConditionalGeneration
        base = cls.from_pretrained(base_repo, revision=cfg.revision or "main", **kwargs)
        model = PeftModel.from_pretrained(base, repo, is_trainable=False)
        if not any("lora_A" in name for name, _ in model.named_parameters()):
            raise RuntimeError("LoRA adapter weights were not attached")
        print(key, "base:", base_repo, "base revision:", cfg.revision, "adapter:", repo)
    else:
        model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
    model.eval()
    return model, tokenizer

def make_input(tokenizer, key, prompt):
    request = "Generate a complete LaTeX document with a TikZ figure for this description:\n" + prompt + "\nOnly output valid LaTeX code."
    if key.startswith("tikzilla"):
        messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n" + request}]
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": request}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if key == "qwen35_9b": kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)

def extract_latex(raw):
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if "</think>" in cleaned: cleaned = cleaned.split("</think>", 1)[1].strip()
    match = re.search(r"\\documentclass(?:\s*\[[^\]]*\])?\s*\{.*?\\end\{document\}", cleaned, re.S)
    if not match: raise ValueError("No complete LaTeX document in response")
    return match.group(0)

def render_latex(latex, folder, stem):
    tex = folder / f"{stem}.tex"
    tex.write_text(latex)
    env = dict(os.environ, openin_any="p", openout_any="p")
    proc = subprocess.run(["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", tex.name], cwd=folder, env=env, capture_output=True, text=True, timeout=90)
    (folder / f"{stem}.compile.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode or not (folder / f"{stem}.pdf").exists(): raise RuntimeError("LaTeX compilation failed")
    subprocess.run(["pdftoppm", "-f", "1", "-singlefile", "-png", "-r", "140", f"{stem}.pdf", stem], cwd=folder, capture_output=True, check=True, timeout=45)
    return folder / f"{stem}.png"

def run_model(key):
    repo, _ = MODELS[key]
    folder = OUTPUT_DIR / key
    folder.mkdir(exist_ok=True)
    model = tokenizer = None
    try:
        model, tokenizer = load_model(key)
        for item in PROMPTS:
            number = item["id"]
            stem = f"{number:02d}"
            result_path = folder / f"{stem}.json"
            if result_path.exists() and json.loads(result_path.read_text()).get("status") == "rendered" and (folder / f"{stem}.png").exists():
                print(key, stem, "already rendered; skipping")
                continue
            for suffix in ("png", "pdf", "tex", "compile.log", "raw.txt"):
                (folder / f"{stem}.{suffix}").unlink(missing_ok=True)
            row = dict(id=number, category=item["category"], prompt=item["prompt"], model=key, repo=repo, seed=1000+number, quantized=LOAD_IN_4BIT)
            display(HTML(f"<h4>{html.escape(key)} — {number:02d}/50</h4><p>{html.escape(item['prompt'])}</p>"))
            try:
                set_seed(1000 + number)
                prompt_text = make_input(tokenizer, key, item["prompt"])
                inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to("cuda:0")
                eos = [x for x in {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")} if isinstance(x, int) and x >= 0]
                pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7, top_p=0.9, eos_token_id=eos, pad_token_id=pad)
                started = time.perf_counter()
                with torch.inference_mode(): output = model.generate(**inputs, generation_config=config)
                ids = output[0, inputs.input_ids.shape[1]:]
                raw = tokenizer.decode(ids, skip_special_tokens=True)
                (folder / f"{stem}.raw.txt").write_text(raw)
                row.update(tokens=len(ids), seconds=round(time.perf_counter()-started, 2), truncated=bool(len(ids) >= MAX_NEW_TOKENS and ids[-1].item() not in eos))
                latex = extract_latex(raw)
                png = render_latex(latex, folder, stem)
                display(HTML("<details><summary>Generated LaTeX</summary><pre>" + html.escape(latex) + "</pre></details>"))
                display(Image(filename=str(png)))
                row["status"] = "rendered"
            except Exception as exc:
                row.update(status="failed", error=str(exc))
                print("FAILED:", exc)
                log = folder / f"{stem}.compile.log"
                if log.exists(): print(log.read_text()[-1500:])
            result_path.write_text(json.dumps(row, indent=2))
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
''')
for key in categories: pass
for key in ("gemma4_12b", "qwen35_9b", "tikzilla_3b", "tikzilla_8b"):
    md(f"## Run {key} — 50 prompts\nYou may rerun this cell; completed rendered images are skipped.\n")
    code(f'run_model("{key}")\n')
md("## Final cell — verify and download ZIP\nThe ZIP contains `prompts.txt`, a summary, and four separate model folders. Images are numbered `01.png` through `50.png` in each folder when rendering succeeds. Failed outputs retain numbered raw text, metadata and logs.\n")
code('''rows = [json.loads(p.read_text()) for p in sorted(OUTPUT_DIR.glob("*/*.json"))]
(OUTPUT_DIR / "summary.json").write_text(json.dumps(rows, indent=2))
for key in MODELS:
    matched = [r for r in rows if r["model"] == key]
    print(key, len(matched), "attempted,", sum(r["status"] == "rendered" for r in matched), "rendered")
archive = Path("tikz_50_four_model_results.zip").resolve()
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if path.is_file(): z.write(path, path.relative_to(OUTPUT_DIR.parent))
display(FileLink(str(archive)))
try:
    from google.colab import files
    files.download(str(archive))
except ImportError:
    pass
''')
md("## Sources\n- [TikZilla 3B model card](https://huggingface.co/nllg/TikZilla-3B-RL)\n- [TikZilla 8B model card](https://huggingface.co/nllg/TikZilla-8B-RL)\n- [Gemma 4 adapter](https://huggingface.co/Praha-Labs/Gemma-4-12B-TikZ-LoRA)\n- [Qwen 3.5 adapter](https://huggingface.co/Praha-Labs/Qwen3.5-9B-TikZ-LoRA)\n")
notebook = {"cells":cells,"metadata":{"colab":{"name":OUT.name},"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
print(OUT)
