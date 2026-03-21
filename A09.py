# A09. 10000枚を順番に処理して、結果を保存する
# ここが「大量処理」の本体

# すでに終わった画像を読み込む
done_set = load_done_set(JSONL_PATH)
print("already_done =", len(done_set))

# CSVがすでにあるか確認する
# なければ最初にヘッダ行を書きたい
csv_exists = CSV_PATH.exists()

# まだ処理していない画像だけにしぼる
target_image_paths = [p for p in image_paths if str(p) not in done_set]
print("target images =", len(target_image_paths))

# 3つの保存先を開く
# JSONL: 1画像1行のJSONで保存
# CSV  : 表形式で保存
# ERR  : エラーだけ保存
with open(JSONL_PATH, "a", encoding="utf-8") as jf, \
     open(CSV_PATH, "a", newline="", encoding="utf-8") as cf, \
     open(ERR_PATH, "a", encoding="utf-8") as ef:

    # CSVの列名を決める
    fieldnames = [
        "image_path",
        "b1_darkness",
        "b2_weather_visibility",
        "b3_quality_degradation",
        "b4_incompleteness_occlusion",
        "b5_other_residual",
        "main_evidence",
        "confidence",
        "raw_response",
    ]

    writer = csv.DictWriter(cf, fieldnames=fieldnames)

    # CSVが新規作成ならヘッダを書く
    if not csv_exists:
        writer.writeheader()

    # 進捗用の変数
    start = time.time()
    processed = 0
    ok_count = 0
    err_count = 0
    total_target = len(target_image_paths)

    # 画像を1枚ずつ処理する
    for image_path in target_image_paths:
        processed += 1

        try:
            # 1枚推論する
            result = infer_one_anchor5(image_path)

            # JSONLに保存する
            jf.write(json.dumps(result, ensure_ascii=False) + "\n")
            jf.flush()

            # CSV用に main_evidence を文字列に直す
            row = result.copy()
            row["main_evidence"] = " | ".join(row["main_evidence"])

            # CSVに保存する
            writer.writerow(row)
            cf.flush()

            ok_count += 1

        except Exception as e:
            # 失敗したらエラー内容だけ別ファイルに残す
            err_obj = {
                "image_path": str(image_path),
                "error": repr(e),
            }
            ef.write(json.dumps(err_obj, ensure_ascii=False) + "\n")
            ef.flush()
            err_count += 1

        # 100枚ごと、または最後に進捗を表示する
        if processed % PROGRESS_EVERY == 0 or processed == total_target:
            elapsed = time.time() - start
            print(
                f"[{processed}/{total_target}] "
                f"ok={ok_count} err={err_count} "
                f"elapsed_sec={elapsed:.1f}"
            )

print("finished")
print("saved jsonl =", JSONL_PATH)
print("saved csv   =", CSV_PATH)
print("saved err   =", ERR_PATH)
