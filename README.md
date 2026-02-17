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
- `docs/experiment_update_2026-02-16.md`
  - 前日報告に対する当日追補（A/B/C結果を含む最新確定値）
- `colab/colab_tfds_axis_builder.py`
  - TFDS + OpenCLIP + SparsePCA(CV + 1SE) axis builder
- `colab/colab_make_axis_figures.py`
  - Figure generator (scatter, density, loadings, CV+1SE)
- `colab/colab_make_statistics_figures.py`
  - Statistical report generator (3 tables + 10 figures)
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
4. Run A/B/C validation scripts:
   - `colab/colab_eval_detectors.py`
   - `colab/colab_quadrant_cases.py`
   - `colab/colab_run_robustness.py`
5. Run `colab/colab_make_statistics_figures.py` for statistical tables + 10 publication figures.
6. See `docs/experiment_update_2026-02-16.md` for the latest validated metrics.

## Latest reference outcome (2026-02-16, 4-variable setting)
- Axis selection:
  - `k_selected=3`, `alpha_selected=1.0`
  - `axis_u_index=2`, `axis_c_index=0`
- Axis formulas:
  - `z_u = +1.0000*d_entropy_gain`
  - `z_c = +0.7071*d_conf_drop +0.7071*d_oodscore_gain`
- Detection benchmark (AUROC / TNR@95TPR):
  - `energy_single`: `0.996254 / 0.996000`
  - `logistic_2d`: `0.941065 / 0.687000`
  - `linear_svm`: `0.941045 / 0.687333`
  - `zsum_1d`: `0.907249 / 0.598667`
  - `msp_single`: `0.886241 / 0.578000`
- Robustness:
  - Prompt/seed/model sweep (`ViT-B-32`, `ViT-B-16`) produced axis cosine similarity `1.0` across all tested runs.
