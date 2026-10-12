# Terminal Cheatsheet

All commands assume you're in the repo root. Windows examples use PowerShell
paths (`.venv\Scripts\...`); on Linux/macOS use `.venv/bin/...` instead.

## 1. Environment setup (training / evaluation)

```powershell
# Create the main environment (Python 3.12 or 3.13) and install pinned dependencies
uv sync

# Activate it (optional — you can also just call .venv\Scripts\python.exe directly)
.venv\Scripts\activate

# Re-generate uv.lock after editing pyproject.toml
uv lock

# Add a new dependency
uv add <package-name>

# One-off install into the current venv without touching pyproject.toml
uv pip install <package-name>
```

`torch`/`torchaudio`/`torchvision` (2.11.0) are pulled from PyTorch's CUDA 12.8 index on
Windows/Linux automatically (see `pyproject.toml`'s `[tool.uv.sources]`), so
`uv sync` gives you a GPU-enabled install if you have an NVIDIA GPU + driver.
Check it worked:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Environment setup (feature extraction only)

`extract_features.py` needs a **separate Python 3.11 environment** —
`guitar-transcription-continuous` depends on `muda`, an unmaintained package
whose setup.py imports the `imp` stdlib module (removed in Python 3.12).

```powershell
uv venv --python 3.11 .venv-extract

uv pip install --python .venv-extract\Scripts\python.exe numpy librosa "setuptools<81" `
    "amt-tools @ git+https://github.com/cwitkowitz/amt-tools" `
    "guitar-transcription-continuous @ git+https://github.com/cwitkowitz/guitar-transcription-continuous"
```

This is a one-time setup; not needed again once you have feature-enriched
`.npz` files.

`setuptools<81` is required: `muda` (a `guitar-transcription-continuous`
dependency) still imports `pkg_resources`, which setuptools >=81 no longer
ships. Without the pin, `uv pip install` grabs the newest setuptools and
`import guitar_transcription_continuous` fails with
`ModuleNotFoundError: No module named 'pkg_resources'`. If that happens
anyway (e.g. setuptools got upgraded later), fix it with:

```powershell
uv pip install --python .venv-extract\Scripts\python.exe "setuptools<81" --reinstall
```

## 3. Running the pipeline

### 3a. Feature extraction (raw GuitarSet -> .npz)

```powershell
.venv-extract\Scripts\python.exe extract_features.py `
    --guitarset-root <path\to\raw\GuitarSet> `
    --output-dir <path\to\output> `
    [--stages 1,2,3] `
    [--verify]
```

| Flag | Meaning |
|---|---|
| `--guitarset-root` | Raw GuitarSet dataset root (required). |
| `--output-dir` | Base output dir; creates `ground_truth/`, `all_features/`, `enriched/` under it (required). |
| `--stages` | Comma-separated stages to run: `1`=ground truth, `2`=spectral features (HCQT/CQT/Mel/STFT/Power), `3`=auxiliary labels (fret region, stored as hand_position / string_activity / pitch_class). Default: all three. |
| `--verify` | Run a structural integrity check on the final `enriched/` directory. |

Point `train.py`'s `--data-root` at the resulting `enriched/` directory.

### 3b. Training

```powershell
.venv\Scripts\python.exe train.py `
    --config configs\cnn_mtl_config.yaml `
    --data-root <path\to\enriched\npz> `
    --output-root <path\to\outputs> `
    [--folds 0,1,2,3,4,5] `
    [--num-workers 0] `
    [--batch-size 1] `
    [--skip-eval]
```

| Flag | Meaning |
|---|---|
| `--config` | Path to a config YAML (required). `configs/cnn_mtl_config.yaml` is the paper's proposed HCQT+Mel Cascaded MT configuration; `configs/ablations/` holds one config per other reported run. |
| `--data-root` | Overrides the dataset's `data_root` from the config (required unless set in the config file). |
| `--output-root` | Overrides `paths.output_root` from the config (required unless set in the config file). Creates `outputs/<model>_<timestamp>/fold_<n>/...`. |
| `--folds` | Comma-separated 0-based fold indices to run, e.g. `0,1`. Default: all folds (six, per the paper's leave-one-player-out protocol). |
| `--num-workers` | Overrides `data.num_workers`. **Set to `0` on Windows** — multiprocessing DataLoader workers can segfault there. Pure I/O parallelism, never changes results. |
| `--batch-size` | Overrides `data.batch_size`. Each dataset item is a `framify_chunk_size`-frame chunk, so the paper's `batch_size=4` means ~2000 windowed samples per step (needs datacenter-class VRAM — the paper used an A100). Lower this for small local GPUs (e.g. `1`). **Unlike `--num-workers`, this can shift BatchNorm statistics slightly** — fine for local smoke-testing, don't use it for a run meant to reproduce the paper's numbers. |
| `--skip-eval` | By default, every fold just trained is **automatically evaluated** afterwards (calls `evaluate.py`'s `run_evaluation()` directly — same leak-free threshold search, writes `k-fold_test_results.txt`). Pass this to skip that and only train. |

Before the first fold, `train.py` runs the preflight checks in
`src/utils/preflight.py`: every player must share the same repertoire, no
feature file may be duplicated or malformed, every fold's train/validation/test
files must be disjoint, and the composite loss must match the configured
objective on a real batch. Any failure stops the run before training. The
dataset audit is written to `dataset_split_manifest.json` in the experiment
directory.

Config switches used by the ablation configs:

| Key | Meaning |
|---|---|
| `model.params.use_cascade_fusion` | `false` feeds the tablature head the bottleneck alone (auxiliary heads and losses stay). Default `true`. |
| `model.params.tab_head_hidden_size` | Hidden width of the tablature head. Default `256`; the parameter-matched single-task run uses `771`. |
| `loss.weighting_strategy` | `uncertainty` (learned) or `static` (tablature 1.0, auxiliaries `loss.<task>_loss.weight`). |
| `training.reseed_each_fold` | `true` reseeds at the start of every fold, as the component campaign did. Default `false` (seed once, as in the Table 1 runs). |

### 3c. Evaluation (produces the paper's actual reported numbers)

Runs automatically after `train.py` unless `--skip-eval` was passed. Also
available standalone, any time, to re-evaluate an existing experiment
directory (e.g. different `--folds`, or after moving/renaming `--data-root`)
without retraining:

```powershell
.venv\Scripts\python.exe evaluate.py `
    --experiment-path <path\to\outputs\cnn_mtl_TIMESTAMP> `
    [--config <path\to\config.yaml>] `
    [--data-root <path\to\enriched\npz>] `
    [--folds 1,2,3,4,5,6]
```

| Flag | Meaning |
|---|---|
| `--experiment-path` | Path to a `train.py` output dir containing `fold_*/model_best.pt` (required). |
| `--config` | Defaults to `<experiment-path>/config.yaml` (auto-copied there by `train.py`). |
| `--data-root` | Overrides the dataset's `data_root` from the config. |
| `--folds` | Comma-separated **1-based** fold numbers to evaluate, e.g. `1,2`. Default: every `fold_*` directory found. |

This is the leak-free evaluation path: threshold/duration grid search on the
*validation* player only, applied to the held-out *test* player. Writes
`fold_<n>/test_results.json` (including auxiliary-head accuracy and the
wrong-string breakdown) and `k-fold_test_results.txt` into `--experiment-path`.

### 3d. Result tables and paired tests

```powershell
.venv\Scripts\python.exe summarize_results.py `
    --run proposed=<path\to\outputs\run_A> `
    --run campaign_control=<path\to\outputs\run_B> `
    [--run LABEL=DIR ...] `
    [--reference campaign_control] `
    [--gcs-run campaign_control] `
    --output-dir <path\to\results>
```

| Flag | Meaning |
|---|---|
| `--run` | An evaluated experiment directory and the label to report it under (repeatable). Name the component campaign's runs after their configs (`campaign_control`, `A_no_fusion`, ...) so `make_figures.py` finds them. |
| `--reference` | Compare every other run against this one, fold by fold: mean difference, 95% t interval, Cohen's d_z, exact Wilcoxon p (repeatable). |
| `--gcs-run` | Run whose GCS curves are plotted; defaults to the first `--reference`. |
| `--output-dir` | Where the CSV files, `gcs_curves.pdf/.png` and `SUMMARY.md` are written. |

### 3e. Figures

```powershell
.venv\Scripts\python.exe make_figures.py `
    [--results-dir <path\to\results>] `
    [--gcs-run <path\to\outputs\run>] `
    [--overlay-run <path\to\outputs\run> --data-root <path\to\enriched\npz>] `
    [--fold 1] [--track 00_SS3-98-C_solo] [--start 985] [--end 1035] [--threshold 0.45] `
    [--out-dir figures]
```

`--results-dir` gives the string confusion and component figures,
`--gcs-run` the GCS curves, `--overlay-run` the tablature and multi-pitch
overlays (the threshold defaults to the one `evaluate.py` stored for that
fold). Each figure is written as PDF and PNG; figures whose inputs are not
given are skipped.

### 3f. Inference cost

```powershell
.venv\Scripts\python.exe benchmark_cost.py [--output cost.json] [--no-timing]
```

Parameters, MACs per frame and real-time factor (GPU if available, and CPU)
for TabCNN, the single-task, the parameter-matched and the proposed model.
`--no-timing` only counts parameters and MACs, which takes a few seconds.

## 4. Type checking

```powershell
uvx pyright --pythonpath .venv/Scripts/python.exe train.py evaluate.py extract_features.py summarize_results.py make_figures.py benchmark_cost.py src
```

## 5. Git basics for this repo

```powershell
git status
git add <files>
git commit -m "message"
git push origin <branch>
```

Remote is already configured:

```powershell
git remote -v
# origin  https://github.com/ErenReyhanlioglu/Guitar-Transcription.git
```

## 6. Windows-specific gotchas

- **`--num-workers 0` is required** for `train.py` on Windows — multiprocessing
  DataLoader workers reliably segfault otherwise (a known PyTorch+Windows
  limitation, unrelated to this repo's code).
- **`uv sync` prunes anything not declared in `pyproject.toml`** — including a
  manually `uv pip install`ed `pip`. If you see `No module named pip` warnings
  from `create_experiment_directory`'s environment-capture step, that's why;
  harmless (training still runs), just skips saving `environment.txt`.
- **Low-VRAM GPUs** (e.g. 4GB laptop GPUs) will hit `CUDA error: out of
  memory` with the paper's default `batch_size=4` (~2000 windowed samples per
  step). Use `--batch-size 1` for local smoke-testing on such hardware.
