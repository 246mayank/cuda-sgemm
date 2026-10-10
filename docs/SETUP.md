# SETUP.md — instructions for Claude Code

> **Claude Code: this file is addressed to you.** Set up the development environment for the
> `cuda-sgemm` project by working through the tasks below in order. `CLAUDE.md` in this folder is the
> project spec; read it first. **Do not write any CUDA kernel code in this task.** Kernels start in
> step 1 of `CLAUDE.md`, in a separate session.
>
> Ask me (Mayank) before anything that needs a password or login, installs system-wide software, or
> pushes to GitHub. Everything else you may do directly. Finish with the report in Task 9.

---

## Context you need

- Owner: Mayank Singh, GitHub user `246mayank`. Repo name: `cuda-sgemm` (public, MIT licence).
- **This machine has no NVIDIA GPU.** Kernels will be compiled and run on **Google Colab (Tesla T4,
  sm_75)**, with **Kaggle (T4 sm_75 / P100 sm_60)** as the fallback. Never claim anything compiled,
  ran, passed or achieved a number unless it actually happened on this machine or came from a CSV in
  `results/`.
- Deadline context: results feed a resume due Monday 12 Oct 2026, 09:00 IST. Keep setup fast. Aim for
  under 30 minutes, and don't gold-plate.

---

## Task 1 — Inspect the machine (read-only)

Detect the OS and report the version of each tool below, or "missing":

| Tool | Command | Needed for |
|---|---|---|
| git | `git --version` | everything |
| GitHub CLI | `gh --version` and `gh auth status` | creating the repo (optional) |
| Python 3.10+ | `python --version` / `python3 --version` | plots, tables |
| pip | `python -m pip --version` | |
| make | `make --version` | local builds (optional) |
| nvcc | `nvcc --version` | local compile check (optional) |
| unzip / zip | `unzip -v` | bringing Colab results back |

Do not install anything system-wide in this task. Just report.

## Task 2 — Repository

- If the current folder is **not** a git repo:
  - If `gh` is installed and authenticated, **ask me** to confirm, then run
    `gh repo create 246mayank/cuda-sgemm --public --license mit --clone=false` and add it as `origin`.
  - Otherwise tell me to create it on github.com (public, MIT licence, no README), then
    `git init -b main` and `git remote add origin https://github.com/246mayank/cuda-sgemm.git`.
- If it **is** already a repo with `origin` set, leave it alone and report the remote URL and branch.

## Task 3 — Folder layout

Create exactly this structure (empty folders get a `.gitkeep`):

```
src/kernels/
scripts/
results/
figures/
notebooks/
docs/
.claude/
```

Leave `CLAUDE.md` where it is. Move `CLAUDE_CODE_SETUP.md` (if present) into `docs/`.

## Task 4 — Repo hygiene files

1. **`.gitignore`**: CUDA/C++ build outputs (`*.o`, `*.obj`, `*.exe`, `bench`, `bandwidth`, `build/`),
   Python (`__pycache__/`, `.venv/`, `*.pyc`), OS junk (`.DS_Store`, `Thumbs.db`), editor folders,
   and `results.zip`. **Do not** ignore `results/*.csv`, `results/*.json` or `figures/*.png`: those
   are committed deliberately.
2. **`LICENSE`**: MIT, copyright 2026 Mayank Singh (skip if Task 2 already created one).
3. **`requirements.txt`**: `pandas` and `matplotlib` (unpinned is fine).
4. **`README.md`**: a placeholder only. Title, one sentence ("Step-by-step SGEMM optimization in
   CUDA C++ on an NVIDIA T4, verified against cuBLAS."), and "Results: in progress." No numbers.

## Task 5 — Python environment

- Create a virtual environment at `.venv` and install `requirements.txt` into it.
- Verify with `python -c "import pandas, matplotlib; print('ok')"` using the venv's interpreter.
- Report the exact command I use to activate it on this OS.

## Task 6 — Claude Code project settings

Create `.claude/settings.json`:

```json
{
  "permissions": {
    "allow": [
      "Bash(make:*)",
      "Bash(nvcc:*)",
      "Bash(python scripts/*)",
      "Bash(.venv/bin/python scripts/*)",
      "Bash(git status)",
      "Bash(git diff:*)",
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(git log:*)"
    ],
    "deny": [
      "Bash(git push:*)"
    ]
  }
}
```

On Windows, add the equivalent `.venv\\Scripts\\python` entry.

## Task 7 — Append environment facts to `CLAUDE.md`

Append this section to the **end** of `CLAUDE.md`, filling the brackets from Task 1:

```markdown
## Environment facts (for Claude Code)
- OS: [detected OS]. There is NO NVIDIA GPU on this machine.
- nvcc locally: [version | NOT available]. If not available, every summary must say
  "not compiled locally — compile on Colab".
- Kernels run only on Colab (T4, sm_75) or Kaggle (T4 sm_75 / P100 sm_60). Results come back as CSVs
  in results/. Never state that a kernel ran, passed, or achieved a number unless it is in those CSVs.
- Python: use the venv at .venv for scripts/.
- Mayank reviews and pushes every commit himself. Never push.
- End every step with: files changed, what Mayank should check, and the exact Colab cells to run.
```

Change nothing else in `CLAUDE.md`.

## Task 8 — Colab notebook

Create `notebooks/run_on_colab.ipynb` (valid nbformat 4 JSON) with a first markdown cell
("Runtime → Change runtime type → T4 GPU before running") and these code cells:

```python
# 1 — environment
!nvidia-smi
!nvcc --version
```
```python
# 2 — fresh code
!rm -rf cuda-sgemm && git clone https://github.com/246mayank/cuda-sgemm
%cd cuda-sgemm
```
```python
# 3 — build & run (works once step 1 of CLAUDE.md exists)
!make
!./bench
```
```python
# 4 — profiler probe: expected to be blocked on Colab; record the outcome either way
!ncu --metrics dram__bytes_read.sum ./bench --kernel v2 --size 2048 || echo "ncu blocked"
```
```python
# 5 — bring results home
!zip -r results.zip results
from google.colab import files; files.download("results.zip")
```

Add a final markdown cell: "Kaggle fallback: same cells; download results.zip from the Output panel
instead of cell 5. Use -arch=sm_60 if the GPU is a P100."

Validate the notebook is well-formed JSON (e.g. `python -c "import json; json.load(open('notebooks/run_on_colab.ipynb'))"`).

## Task 9 — Commit and report

1. `git add -A` and commit: `chore: project scaffold, settings, venv requirements, Colab notebook`.
2. **Do not push.** Tell me to review and push.
3. Report in this format:

```
SETUP REPORT
OS: ...
Tools: git ✓/✗ | gh ✓/✗ (auth ✓/✗) | python x.y | make ✓/✗ | nvcc ✓/✗ | zip ✓/✗
Repo: <remote URL>  branch: <name>  commit: <hash>
Created: <list of files/folders>
Venv: activate with `...`  — import check: ok/failed
Needs Mayank: <anything you could not do, e.g. create the GitHub repo, log in to gh, push>
Next: open a fresh session (/clear) and start CLAUDE.md step 1 in plan mode.
```

---

## Out of scope for this task

- No kernel, harness, Makefile or script code. That is step 1 of `CLAUDE.md`.
- No installing the CUDA Toolkit, Visual Studio or other large system software. If `nvcc` is missing,
  just report it. Mayank will decide whether it's worth installing.
- No pushing to GitHub.
