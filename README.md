# 26s-TERM-01 (Colab-style Quickstart)

Colabのセル実行に合わせて、`Input -> Output` で順番に進める手順です。

## Input 1: Driveをマウントして作業フォルダへ移動

```python
from google.colab import drive
drive.mount("/content/drive")

import os
PROJECT_DIR = "/content/drive/MyDrive/pj-TERM"
os.makedirs(PROJECT_DIR, exist_ok=True)
%cd /content/drive/MyDrive/pj-TERM
```

## Output 1

- `Mounted at /content/drive`
- `/content/drive/MyDrive/pj-TERM` に移動できる

## Input 2: パイプライン本体を保存

```bash
# このリポジトリの natural_capital_pipeline.py を
# /content/drive/MyDrive/pj-TERM/natural_capital_pipeline.py に配置
```

## Output 2

- `/content/drive/MyDrive/pj-TERM/natural_capital_pipeline.py` が作成される

## Input 3: 設定ファイルを作成

```python
%%writefile /content/drive/MyDrive/pj-TERM/config_example.json
{
  "period": {
    "start_date": "2024-01-01",
    "end_date": "2025-12-31"
  },
  "data": {
    "cache_dir": "./cache",
    "outputs_subdir": "fujisawa_demo"
  },
  "model": {
    "lookback_weeks": 7,
    "garch_forecast_horizon": 8
  }
}
```

## Output 3

- `Overwriting /content/drive/MyDrive/pj-TERM/config_example.json`

## Input 4: 初回実行 (run)

```python
!python /content/drive/MyDrive/pj-TERM/natural_capital_pipeline.py run \
  --config /content/drive/MyDrive/pj-TERM/config_example.json
```

## Output 4 (例)

```json
{
  "n_weeks": 105,
  "last_index": 100.29045137932808,
  "last_return": 0.0030505616832941468,
  "last_garch_sigma2": 7.369311346586789e-06,
  "max_forecast_sigma2": 6.590885682940199e-06
}
```

- `[ok] outputs at: cache/fujisawa_demo`

## Input 5: キャッシュ実行 (from-cache)

```python
!python /content/drive/MyDrive/pj-TERM/natural_capital_pipeline.py from-cache \
  --config /content/drive/MyDrive/pj-TERM/config_example.json
```

## Output 5

- `run` と同じ形式のサマリJSON
- `[ok] outputs at: cache/fujisawa_demo`

## Input 6: 週次特徴量の可視化

```python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

feat = pd.read_csv(Path("/content/drive/MyDrive/pj-TERM/cache/fujisawa_demo/derived/features_weekly.csv"))
feat["week_start"] = pd.to_datetime(feat["week_start"])

num_cols = [c for c in feat.columns if c != "week_start"]
z = feat[num_cols].copy()
z = (z - z.mean()) / z.std().replace(0, 1)

plt.figure(figsize=(14, 5))
for c in num_cols:
    plt.plot(feat["week_start"], z[c], label=c, alpha=0.9)
plt.title("Weekly Features (Z-score normalized)")
plt.xlabel("Week")
plt.ylabel("Z-score")
plt.grid(alpha=0.3)
plt.legend(ncol=3, fontsize=9)
plt.show()
```

## Output 6

- `Weekly Features (Z-score normalized)` の折れ線グラフが表示される

## Input 7: スクリプトで同じ図を保存（任意）

```python
!python /content/drive/MyDrive/pj-TERM/plot_weekly_features.py \
  --features-csv /content/drive/MyDrive/pj-TERM/cache/fujisawa_demo/derived/features_weekly.csv \
  --save /content/drive/MyDrive/pj-TERM/cache/fujisawa_demo/derived/weekly_features_zscore.png
```

## Output 7

- `[ok] saved: /content/drive/MyDrive/pj-TERM/cache/fujisawa_demo/derived/weekly_features_zscore.png`
