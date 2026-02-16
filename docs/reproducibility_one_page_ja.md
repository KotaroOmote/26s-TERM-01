# 再現実装 1ページ要約（N-Axis Attribution OoD, 2026-02-16）

## 1. この実装で再現する範囲
- 対象: `最初 -> 軸作成` フェーズのみ
- 方針: **Step Separation**（軸設計と判定器設計を分離）
- 非対象: 判定器学習・しきい値最適化・最終性能比較

## 2. データ設計（会話で確定した設定）
- ID: `food101 train[:10000]`
- OoD: `cifar100 train[:5000]` + `imagenet_r test[:5000]`
- 合計: 20,000 サンプル
- 備考: `sun397/tfds` は配布元 404 のため代替として `cifar100` を採用

## 3. 軸選択の中核仕様
- 特徴: `conf / msp / entropy / energy / ood_score` から差分特徴を作成
- 前処理: Variance Threshold（定数特徴を除外）
- 軸抽出: SparsePCA
- モデル選択: CV再構成誤差（MSE） + **1SEルール**
- 探索例: `k in [1..4]`, `alpha in {0.5,1.0,2.0,4.0,8.0}`

## 4. Colab再現手順（最短）
### 4.1 依存導入
```bash
pip install -U open-clip-torch==2.26.1 ftfy tensorflow tensorflow-datasets scikit-learn pandas matplotlib scipy pillow tqdm
```

### 4.2 軸作成
```bash
python colab/colab_tfds_axis_builder.py \
  --n-id 10000 \
  --n-ood-cifar 5000 \
  --n-ood-imagenetr 5000 \
  --batch-size 64 \
  --k-max 4 \
  --alpha-grid 0.5,1.0,2.0,4.0,8.0 \
  --output-dir /content/outputs/axis_build_k4 \
  --quiet
```

### 4.3 seed安定性（3本）
```bash
for s in 42 43 44; do
  python colab/colab_tfds_axis_builder.py \
    --seed $s \
    --n-id 10000 --n-ood-cifar 5000 --n-ood-imagenetr 5000 \
    --batch-size 64 --k-max 4 --alpha-grid 0.5,1.0,2.0,4.0,8.0 \
    --output-dir /content/outputs/axis_build_k4_s${s} \
    --quiet
done
```

### 4.4 図生成（4種）
```bash
python colab/colab_make_axis_figures.py \
  --axis-scores /content/outputs/axis_build_k4/axis_scores.csv \
  --axis-loadings /content/outputs/axis_build_k4/axis_loadings.csv \
  --cv-table /content/outputs/axis_build_k4/sparsepca_cv_table.csv \
  --summary-json /content/outputs/axis_build_k4/summary_axis_only.json \
  --out-dir /content/outputs/axis_build_k4/figs
```

## 5. 期待される基準結果（会話実行の参照値）
- `k_selected = 4`
- `alpha_selected = 8.0`
- 軸対応: `axis_u_index = 3`, `axis_c_index = 0`
- 軸式:
  - `z_u = +0.9187*d_entropy_gain +0.3949*d_oodscore_gain`
  - `z_c = +0.6546*d_conf_drop +0.6546*d_msp_drop +0.3783*d_oodscore_gain`
  - 実装上 `msp == conf` なので
    - `z_c ~= +1.3092*d_conf_drop +0.3783*d_oodscore_gain`
- seed安定性（seed42基準の|cos|）
  - `z_u`: 0.99999948
  - `z_c`: 0.99999983

## 6. 成果物（このリポジトリ）
- 軸作成コード: `colab/colab_tfds_axis_builder.py`
- 図生成コード: `colab/colab_make_axis_figures.py`
- 詳細手順: `docs/reproducibility_from_chat.md`
- レポート: `report/N-Axis_Attribution_OoD_2026-02-16.tex` / `.pdf`
- 図: `report/figs/fig1~fig4_*.png`

## 7. よくあるエラー
- `ModuleNotFoundError: open_clip`:
  - `open-clip-torch` をインストールして再実行
- Colabが出力過多で固まる:
  - `--quiet` とログリダイレクトを併用
- TeX式エラー（`k^*`）:
  - `k^{\ast}` を使用
