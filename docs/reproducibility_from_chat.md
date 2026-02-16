# Reproducibility Summary From Conversation Log

This document summarizes the implementation decisions and execution steps that were fixed during the chat session, so another reader can reproduce the same axis-construction stage without relying on chat context.

## 1. Scope fixed in this work
- Objective: complete only `start -> axis construction` for Attribution OoD.
- Principle: `Step Separation` (axis design and detector design are strictly separated).
- This package does **not** include detector training/threshold optimization as part of axis selection.

## 2. Data design fixed in the session
Initial idea changed during execution:
- Rejected: very small pilot scale (`N=360`).
- Revised target: large enough scale for stable axis selection.
- Operational dataset plan finalized on Colab/TFDS:
  - ID: `food101`, `train[:10000]`
  - OoD: `cifar100`, `train[:5000]` + `imagenet_r`, `test[:5000]`
  - Total: 20,000 samples

Important incident and resolution:
- `sun397/tfds` failed due upstream URL `404`.
- The pipeline was updated to use `cifar100` as OoD replacement while keeping `imagenet_r`.

## 3. Axis selection policy fixed in the session
- Removed heuristic weighted objective `J(k)`.
- Adopted statistical selection:
  - SparsePCA grid search over `(k, alpha)`
  - Cross-validation reconstruction error (MSE)
  - One-Standard-Error (1SE) rule for conservative model choice
- Added variance-threshold preprocessing to remove near-constant features.

## 4. Implementation files and roles
- `colab/colab_tfds_axis_builder.py`
  - TFDS loading, OpenCLIP inference, feature deltas, variance filtering,
    SparsePCA CV + 1SE selection, axis fixation.
- `colab/colab_make_axis_figures.py`
  - Generates 4 key figures from axis outputs.
- `report/N-Axis_Attribution_OoD_2026-02-16.pdf`
  - Final report for sharing/review.
- `report/figs/*.png`
  - Figure artifacts used in the report.

## 5. Colab execution runbook (exact order)
### 5.1 Install
```bash
pip install -U open-clip-torch==2.26.1 ftfy tensorflow tensorflow-datasets scikit-learn pandas matplotlib scipy pillow tqdm
```

### 5.2 Build axes
```bash
python colab_tfds_axis_builder.py \
  --n-id 10000 \
  --n-ood-cifar 5000 \
  --n-ood-imagenetr 5000 \
  --batch-size 64 \
  --k-max 4 \
  --alpha-grid 0.5,1.0,2.0,4.0,8.0 \
  --output-dir /content/outputs/axis_build_k4 \
  --quiet
```

### 5.3 Multi-seed stability
```bash
for s in 42 43 44; do
  python colab_tfds_axis_builder.py \
    --seed $s \
    --n-id 10000 --n-ood-cifar 5000 --n-ood-imagenetr 5000 \
    --batch-size 64 --k-max 4 --alpha-grid 0.5,1.0,2.0,4.0,8.0 \
    --output-dir /content/outputs/axis_build_k4_s${s} \
    --quiet
done
```

### 5.4 Make figures
```bash
python colab_make_axis_figures.py \
  --axis-scores /content/outputs/axis_build_k4/axis_scores.csv \
  --axis-loadings /content/outputs/axis_build_k4/axis_loadings.csv \
  --cv-table /content/outputs/axis_build_k4/sparsepca_cv_table.csv \
  --summary-json /content/outputs/axis_build_k4/summary_axis_only.json \
  --out-dir /content/outputs/axis_build_k4/figs
```

## 6. Key expected results (reference values)
From the finalized run in the conversation:
- `k_selected = 4`
- `alpha_selected = 8.0`
- axis mapping:
  - `axis_u_index = 3`
  - `axis_c_index = 0`
- fixed-axis formulas:
  - `z_u = +0.9187*d_entropy_gain +0.3949*d_oodscore_gain`
  - `z_c = +0.6546*d_conf_drop +0.6546*d_msp_drop +0.3783*d_oodscore_gain`
  - because `msp == conf` in this implementation:
    - `z_c ~= +1.3092*d_conf_drop +0.3783*d_oodscore_gain`

Multi-seed stability (cosine similarity to seed42):
- seed42: `1.000000 / 1.000000`
- seed43: `0.999998 / 0.999999`
- seed44: `1.000000 / 1.000000`
- mean: `0.99999948 / 0.99999983`

## 7. Troubleshooting captured in the session
- `ModuleNotFoundError: open_clip`
  - install `open-clip-torch` and rerun.
- Excessive Colab output freezes UI
  - run with `--quiet` and redirect logs to file.
- LaTeX compile issue with `k^*`
  - use `k^{\ast}`.
- If figures are not rendered in TeX
  - place png files under `report/figs/` with exact names used in TeX.

## 8. Reproducibility checklist for this stage
- [x] Dataset identities and sample counts fixed.
- [x] Axis selection criterion fixed (CV MSE + 1SE).
- [x] Variance-threshold preprocessing included.
- [x] Random-seed stability evaluated.
- [x] Figure generation scripts included.
- [x] Report and compiled PDF included.
