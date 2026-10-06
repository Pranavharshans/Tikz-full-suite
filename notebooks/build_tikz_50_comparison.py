"""Create a self-contained HTML gallery from the 50-prompt result directory."""

import base64
import html
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "tikz_50_results"
OUTPUT = ROOT / "tikz_50_comparison.html"
MODELS = [
    ("gemma4_12b", "Gemma 4 12B"),
    ("qwen35_9b", "Qwen 3.5 9B"),
    ("tikzilla_3b", "TikZilla 3B RL"),
    ("tikzilla_8b", "TikZilla 8B RL"),
]


def esc(value):
    return html.escape(str(value), quote=True)


def image_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


records = {}
for key, _ in MODELS:
    records[key] = {
        int(path.stem): json.loads(path.read_text())
        for path in (RESULTS / key).glob("[0-9][0-9].json")
    }

prompt_rows = [records[MODELS[0][0]].get(i) for i in range(1, 51)]
if any(row is None for row in prompt_rows):
    raise RuntimeError("Expected 50 numbered prompts in the first model's results")

categories = list(dict.fromkeys(row["category"] for row in prompt_rows))
rendered = sum(
    row.get("status") == "rendered"
    for model_records in records.values()
    for row in model_records.values()
)

parts = ["""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>50 TikZ prompts · four-model comparison</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#17222f;background:#f2f5f8}
*{box-sizing:border-box}body{margin:0}header{padding:30px max(24px,calc((100vw - 1760px)/2));background:#152b3d;color:#fff}
h1{font-size:clamp(27px,3vw,40px);margin:0 0 8px}header p{margin:0;color:#cad8e2;line-height:1.5}
.toolbar{position:sticky;top:0;z-index:10;background:#fff;border-bottom:1px solid #d8e0e7;box-shadow:0 2px 12px #13263914;padding:12px max(24px,calc((100vw - 1760px)/2));display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.toolbar select,.toolbar input{border:1px solid #b8c8d5;background:white;border-radius:8px;font:inherit;padding:9px 12px;color:#17222f}
.toolbar input{min-width:240px;flex:1}.toolbar label{display:flex;gap:7px;align-items:center;white-space:nowrap}
#visible-count{color:#617487;font-size:14px;white-space:nowrap}main{max-width:1810px;margin:auto;padding:24px}
.row{background:#fff;border:1px solid #dce4eb;border-radius:14px;margin:0 0 25px;box-shadow:0 4px 20px #1b344409;overflow:hidden;scroll-margin-top:95px}
.prompt{padding:18px 20px;border-bottom:1px solid #e1e8ee;background:#f9fbfd}.prompt-top{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:8px}
.number{font-weight:800;color:#185b82}.category{font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:#667d8d;font-weight:700}.prompt p{margin:0;font-size:16px;line-height:1.55}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0}.card{border-right:1px solid #e1e8ee;min-width:0}.card:last-child{border-right:0}
.card-head{padding:12px 14px;display:flex;gap:8px;align-items:center;justify-content:space-between;border-bottom:1px solid #e8edf1;min-height:52px}.model{font-size:14px;font-weight:750}.status{font-size:11px;white-space:nowrap;padding:4px 7px;border-radius:99px;background:#e5f5ec;color:#176640;font-weight:700}.status.fail{background:#fff0e7;color:#a34718}
.image-box{height:270px;padding:16px;background:#fff;display:flex;align-items:center;justify-content:center;overflow:hidden}.image-box img{max-height:100%;max-width:100%;object-fit:contain;cursor:zoom-in}.image-box:has(img):hover{background:#f6fafc}.fail-box{padding:22px;color:#5e6d7a;line-height:1.45;text-align:center}.fail-box strong{display:block;color:#ab4f24;margin-bottom:8px}.meta{font-size:12px;color:#637587;padding:9px 14px;border-top:1px solid #edf1f5;min-height:36px}
dialog{border:0;border-radius:13px;padding:12px;box-shadow:0 20px 70px #0007;max-width:96vw;max-height:96vh}dialog::backdrop{background:#071420bb}dialog img{display:block;max-width:calc(96vw - 24px);max-height:calc(96vh - 76px);object-fit:contain}dialog button{float:right;border:0;background:#eaf0f5;border-radius:7px;padding:7px 11px;cursor:pointer;margin-bottom:7px}
@media(max-width:1050px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.card:nth-child(2){border-right:0}.card:nth-child(-n+2){border-bottom:1px solid #e1e8ee}}
@media(max-width:580px){.grid{grid-template-columns:1fr}.card{border-right:0;border-bottom:1px solid #e1e8ee}.card:last-child{border-bottom:0}.image-box{height:230px}main{padding:12px}}
</style></head><body>
<header><h1>Four-model TikZ comparison</h1><p>50 original prompts · 5 categories · 200 attempts · """ + str(rendered) + """ rendered images. Click an image to inspect it at full size. A rendered image confirms compilation, not prompt accuracy.</p></header>
<div class="toolbar"><select id="category"><option value="">All categories</option>"""]
parts.extend(f'<option value="{esc(c)}">{esc(c)}</option>' for c in categories)
parts.append("""</select><input id="search" type="search" placeholder="Search prompt or number" aria-label="Search prompts"><label><input id="complete" type="checkbox"> All four rendered</label><span id="visible-count"></span></div><main id="rows">""")

for prompt in prompt_rows:
    number = prompt["id"]
    category = prompt["category"]
    complete = all(
        records[key].get(number, {}).get("status") == "rendered"
        and (RESULTS / key / f"{number:02d}.png").exists()
        for key, _ in MODELS
    )
    parts.append(
        f'<section class="row" id="p{number:02d}" data-category="{esc(category)}" '
        f'data-search="{esc(str(number) + " " + prompt["prompt"].lower())}" data-complete="{str(complete).lower()}">'
        f'<div class="prompt"><div class="prompt-top"><span class="number">Prompt {number:02d}</span><span class="category">{esc(category)}</span></div>'
        f'<p>{esc(prompt["prompt"])}</p></div><div class="grid">'
    )
    for key, label in MODELS:
        row = records[key].get(number)
        png = RESULTS / key / f"{number:02d}.png"
        ok = row is not None and row.get("status") == "rendered" and png.exists()
        parts.append(f'<article class="card"><div class="card-head"><span class="model">{esc(label)}</span><span class="status {"" if ok else "fail"}">{"Rendered" if ok else "Failed"}</span></div>')
        if ok:
            uri = image_uri(png)
            parts.append(f'<div class="image-box"><img loading="lazy" src="{uri}" alt="{esc(label)} output for prompt {number:02d}" tabindex="0" data-title="Prompt {number:02d} · {esc(label)}"></div>')
            parts.append(f'<div class="meta">{esc(row.get("tokens", "?"))} tokens · {esc(row.get("seconds", "?"))} s</div>')
        else:
            reason = row.get("error", "No result recorded") if row else "No result recorded"
            parts.append(f'<div class="image-box fail-box"><div><strong>No rendered image</strong>{esc(reason)}</div></div>')
            parts.append(f'<div class="meta">{esc(row.get("tokens", "—") if row else "—")} tokens · {esc(row.get("seconds", "—") if row else "—")} s</div>')
        parts.append("</article>")
    parts.append("</div></section>")

parts.append("""</main><dialog id="viewer"><button id="close" aria-label="Close image">Close ×</button><div id="caption"></div><img id="large" alt="Enlarged model output"></dialog>
<script>
const category=document.getElementById('category'),search=document.getElementById('search'),complete=document.getElementById('complete');
const rows=[...document.querySelectorAll('.row')],count=document.getElementById('visible-count');
function filter(){let n=0;for(const row of rows){const show=(!category.value||row.dataset.category===category.value)&&(!complete.checked||row.dataset.complete==='true')&&(!search.value||row.dataset.search.includes(search.value.trim().toLowerCase()));row.hidden=!show;if(show)n++}count.textContent=n+' / 50 prompts'}
[category,search,complete].forEach(el=>el.addEventListener('input',filter));filter();
const viewer=document.getElementById('viewer'),large=document.getElementById('large');
function openImage(img){large.src=img.src;large.alt=img.alt;document.getElementById('caption').textContent=img.dataset.title;viewer.showModal()}
document.querySelectorAll('.image-box img').forEach(img=>{img.addEventListener('click',()=>openImage(img));img.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();openImage(img)}})});
document.getElementById('close').onclick=()=>viewer.close();viewer.addEventListener('click',e=>{if(e.target===viewer)viewer.close()});
</script></body></html>""")

OUTPUT.write_text("".join(parts))
print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024 / 1024:.2f} MiB)")
