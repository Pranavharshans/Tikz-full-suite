# Colab: 50 prompts × four TikZ models

Open `tikz_50_four_model_colab.ipynb` in Colab, select an NVIDIA GPU, and run the numbered cells in order. An A100 80 GB is a practical BF16 starting point, but this notebook has not been GPU tested. Set `LOAD_IN_4BIT=True` in Cell 2 if needed; that changes inference precision. The models load one at a time.

1. **Dependencies:** install Transformers, PEFT, and the TeX renderer. Restart the runtime if Transformers was already imported.
2. **Settings:** choose BF16 or 4-bit, output directory, and the four model repositories. If a Praha adapter is private, use interactive `huggingface_hub.login()` before loading it.
3. **Prompts:** save the 50 original prompts to `prompts.txt` (five categories, ten each).
4. **Functions:** load adapters with their matching base checkpoints, generate LaTeX, compile it, show images, and retain raw outputs and logs.
5. **Gemma 4 12B:** run all 50 prompts.
6. **Qwen 3.5 9B:** run the same 50 prompts.
7. **TikZilla 3B RL:** run the same 50 prompts.
8. **TikZilla 8B RL:** run the same 50 prompts.
9. **Final cell:** build and download `tikz_50_four_model_results.zip`.

The ZIP has `tikz_50_results/prompts.txt`, `summary.json`, and one folder for each model. Each successful image is numbered `01.png` through `50.png`; matching `.tex`, `.pdf`, `.raw.txt`, `.json`, and compile logs help inspect failures. A failed render has no PNG and is marked `failed` in its numbered JSON file. Rerunning a model cell skips only successful renders.
