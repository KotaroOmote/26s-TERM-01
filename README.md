# 26s-TERM-01

Conversation-log based reproducibility package for:
- **N-Axis Attribution OoD (2026-02-16)**
- Scope: **axis-construction stage only** (`start -> axis construction`)

## What is included
- `docs/reproducibility_one_page_ja.md`
  - 提出版の1ページ要約（日本語）
- `docs/reproducibility_from_chat.md`
  - Full summary of decisions fixed during the chat
  - Repro runbook and troubleshooting notes
- `colab/colab_tfds_axis_builder.py`
  - TFDS + OpenCLIP + SparsePCA(CV + 1SE) axis builder
- `colab/colab_make_axis_figures.py`
  - Figure generator (scatter, density, loadings, CV+1SE)
- `colab/colab_eval_detectors.py`
  - (A) AUROC/TNR@95 comparison on `(z_u, z_c)` vs MSP/energy
- `colab/colab_quadrant_cases.py`
  - (B) Representative FP/FN extraction from `z_u-z_c` quadrants
- `colab/colab_run_robustness.py`
  - (C) Prompt/split/VLM robustness runner and axis-similarity summary
- `report/N-Axis_Attribution_OoD_2026-02-16.pdf`
  - Compiled report with figures
- `report/figs/`
  - `fig1_zu_zc_scatter.png`
  - `fig2_zu_zc_density.png`
  - `fig3_axis_loadings.png`
  - `fig4_cv_1se.png`

## Quick start (Colab)
1. Install dependencies (see `docs/reproducibility_from_chat.md`).
2. Run `colab/colab_tfds_axis_builder.py` to generate axis outputs.
3. Run `colab/colab_make_axis_figures.py` to create figures.
4. Use the included PDF report for submission/review.

## Expected reference outcome
- Selected model: `k=4`, `alpha=8.0`
- Feature vector: 4 variables (`d_conf_drop`, `d_entropy_gain`, `d_energy_gain`, `d_oodscore_gain`)
- Stable axis mapping across seeds (`42,43,44`)
- Final axis formulas are documented, but should be re-estimated after any feature-set change
