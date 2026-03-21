# A05. 画像を集めて、10000枚だけ使う
# ここでは「どの画像を処理するか」を決める

def collect_images(root: Path) -> List[Path]:
    # rootフォルダの中をすべて見て、
    # 画像っぽい拡張子のファイルだけを集める関数
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)

    # 毎回同じ順番になるように並べる
    files.sort()
    return files


# まず全部の画像を集める
all_image_paths = collect_images(DATA_ROOT)
print("all images =", len(all_image_paths))

# 乱数の種を固定する
# これをしておくと、毎回ほぼ同じ10000枚を選べる
random.seed(RANDOM_SEED)

# もし画像が10000枚より多ければ、その中から10000枚だけ選ぶ
# 10000枚以下なら全部使う
if len(all_image_paths) > TARGET_N:
    image_paths = sorted(random.sample(all_image_paths, TARGET_N))
else:
    image_paths = all_image_paths

print("sampled images =", len(image_paths))

# 最初の10枚だけ見て確認する
for p in image_paths[:10]:
    print(p)
