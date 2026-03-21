# A08. 画像1枚をAIに見せて、JSON結果を返してもらう
# ここが1枚ぶんの推論本体

def infer_one_anchor5(image_path: Path) -> Dict[str, Any]:
    # 画像を開く
    img = load_pil_image(image_path)

    # モデルに渡す会話形式の入力を作る
    # system: 全体ルール
    # user  : 画像と採点指示
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    # Qwen用の会話テンプレートに変換する
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 文章と画像を、モデルに入れられるテンソルへ変換する
    inputs = processor(
        text=[text],
        images=[img],
        padding=True,
        return_tensors="pt",
    )

    # GPUに送れるものはGPUへ送る
    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    # 推論を実行する
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,   # 毎回なるべく安定した出力にしたいのでサンプリングしない
        )

    # 入力部分をのぞいて、生成された返答だけ取り出す
    gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]

    # トークン列を文字列へ戻す
    raw_text = processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    # 返答文字列からJSONを抜き出す
    obj = extract_json(raw_text)

    # 形を整えて返す
    result = normalize_result(obj, raw_text, image_path)
    return result
