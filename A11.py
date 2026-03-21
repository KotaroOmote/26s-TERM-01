# A11. 各スコアがどれくらい出たかを棒グラフで見る
# たとえば b1_darkness で 0.50 が何枚あったか、などを数える

anchors = [0.00, 0.25, 0.50, 0.75, 1.00]

for col in score_cols:
    # その列の値を数える
    counts = df[col].value_counts().reindex(anchors, fill_value=0)

    # 棒グラフを描く
    plt.figure(figsize=(7, 4))
    plt.bar([str(a) for a in anchors], counts.values)
    plt.title(f"Distribution of {col}")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.grid(axis="y", alpha=0.3)
    plt.show()






# A12. confidence の分布と、各軸の平均値を見る
# どの軸が全体として高めなのかをざっくり確認できる

# confidence のヒストグラム
plt.figure(figsize=(7, 4))
plt.hist(df["confidence"].dropna(), bins=20)
plt.title("Confidence distribution")
plt.xlabel("confidence")
plt.ylabel("count")
plt.grid(alpha=0.3)
plt.show()

# 各スコア列の平均を計算する
mean_scores = df[score_cols].mean().sort_values(ascending=False)

# 平均値を棒グラフで表示する
plt.figure(figsize=(8, 4))
plt.bar(mean_scores.index, mean_scores.values)
plt.title("Mean score by axis")
plt.xlabel("axis")
plt.ylabel("mean score")
plt.ylim(0, 1.0)
plt.grid(axis="y", alpha=0.3)
plt.xticks(rotation=20)
plt.show()

print(mean_scores)





# A13. b1 と b4 の組み合わせを見る
# 「暗い画像は、同時に見切れも多いのか？」のような傾向を見るための表

anchors = [0.00, 0.25, 0.50, 0.75, 1.00]

# 行が b1、列が b4 のクロス集計表を作る
cross = pd.crosstab(df["b1_darkness"], df["b4_incompleteness_occlusion"])
cross = cross.reindex(index=anchors, columns=anchors, fill_value=0)

# ヒートマップとして表示する
plt.figure(figsize=(6, 5))
plt.imshow(cross.values, aspect="auto")
plt.title("b1_darkness vs b4_incompleteness_occlusion")
plt.xlabel("b4_incompleteness_occlusion")
plt.ylabel("b1_darkness")
plt.xticks(range(len(anchors)), [str(a) for a in anchors])
plt.yticks(range(len(anchors)), [str(a) for a in anchors])

# マスの中に件数を書く
for i in range(cross.shape[0]):
    for j in range(cross.shape[1]):
        plt.text(j, i, str(cross.iloc[i, j]), ha="center", va="center")

plt.colorbar(label="count")
plt.show()

print(cross)





# A14. 各スコアどうしの相関を見る
# たとえば「暗さが強いほど confidence は下がるか」などを確認する

# 相関係数表を作る
corr = df[score_cols + ["confidence"]].corr(numeric_only=True)

# 相関行列をヒートマップとして表示する
plt.figure(figsize=(7, 6))
plt.imshow(corr.values, aspect="auto")
plt.title("Correlation matrix")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)

# 各マスに相関係数を書き込む
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

plt.colorbar(label="correlation")
plt.tight_layout()
plt.show()

print(corr)
