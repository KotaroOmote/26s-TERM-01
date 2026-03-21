# A10. 保存したCSVを読み込んで、評価しやすい形にする
# ここから先は「結果を見る」段階

# CSVを読む
df = pd.read_csv(CSV_PATH)

# 採点列の名前をまとめておく
score_cols = [
    "b1_darkness",
    "b2_weather_visibility",
    "b3_quality_degradation",
    "b4_incompleteness_occlusion",
    "b5_other_residual",
]

# 数値として読めるようにしておく
for c in score_cols + ["confidence"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")


def split_path_info(p):
    # 画像パスから、
    # 「上位フォルダ名」「種名らしき部分」「ファイル名」を取り出す
    p = Path(p)

    try:
        rel = p.relative_to(DATA_ROOT)
        parts = rel.parts

        top_group = parts[0] if len(parts) >= 1 else "unknown"
        species   = parts[1] if len(parts) >= 2 else "unknown"
        filename  = parts[-1]

        return top_group, species, filename

    except Exception:
        # うまく分解できなかったら最低限ファイル名だけ返す
        return "unknown", "unknown", p.name


# 新しい列を3つ追加する
df[["top_group", "species", "filename"]] = df["image_path"].apply(
    lambda x: pd.Series(split_path_info(x))
)

print(df.head())
print("rows =", len(df))
