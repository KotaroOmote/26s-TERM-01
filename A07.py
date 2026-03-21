# A07. プロンプトと、便利な小道具を用意する
# ここでは「指示文」と「結果を整える関数」をまとめて作る

SYSTEM_PROMPT = """
You are a strict annotator for camera-trap image condition scoring.
Return ONLY one JSON object.
Do not output markdown.
Do not output any explanation outside JSON.
"""

USER_PROMPT = """
Score this camera-trap image on five axes.

Allowed score values for each axis:
0.00, 0.25, 0.50, 0.75, 1.00

Use ONLY those five anchor values.
Do not output any other number for b1-b5.

Definitions:

- b1_darkness:
  severity of darkness.
  0.00 = sufficiently bright
  0.25 = slightly dark but details still visible
  0.50 = clearly dark and details reduced
  0.75 = very dark, silhouette-like in many regions
  1.00 = extremely dark, identification is severely hindered

- b2_weather_visibility:
  visibility degradation due to rain, fog, snow, droplets on lens, haze, glare, mist, or similar environmental effects.
  0.00 = absent
  0.25 = slight
  0.50 = moderate
  0.75 = severe
  1.00 = extremely severe

- b3_quality_degradation:
  blur, motion blur, sensor noise, compression artifacts, low resolution, focus failure, or severe exposure failure as image-quality degradation.
  0.00 = absent
  0.25 = slight
  0.50 = moderate
  0.75 = severe
  1.00 = extremely severe

- b4_incompleteness_occlusion:
  partial body, cut off by frame, or occluded by vegetation/objects.
  0.00 = target sufficiently visible
  0.25 = slightly incomplete
  0.50 = moderately incomplete
  0.75 = largely incomplete
  1.00 = only a small part visible or heavily occluded

- b5_other_residual:
  use only when the image difficulty is real but not well explained by b1-b4.
  Examples: unusual event, carcass/remains, semantic anomaly, strange artifact, or multiple conflicting issues not captured above.
  0.00 = unnecessary
  0.25 = slight need
  0.50 = moderate need
  0.75 = strong need
  1.00 = dominant unexplained difficulty

Important rules:
1. Be conservative.
2. Do not overuse 1.00.
3. Keep b5 low unless b1-b4 cannot explain the difficulty.
4. Ignore timestamp, temperature, brand watermark, and overlay text unless they directly interfere with visibility.
5. Judge image condition, not species identity.
6. If only part of the animal is visible, reflect that mainly in b4.
7. Output b1-b5 using ONLY the five anchor values above.

Return exactly this JSON schema:
{
  "b1_darkness": 0.00,
  "b2_weather_visibility": 0.00,
  "b3_quality_degradation": 0.00,
  "b4_incompleteness_occlusion": 0.00,
  "b5_other_residual": 0.00,
  "main_evidence": ["...", "..."],
  "confidence": 0.00
}

Rules:
- main_evidence must be short phrases, max 4 items
- confidence must be between 0.00 and 1.00
- b1-b5 must use ONLY 0.00, 0.25, 0.50, 0.75, 1.00
- return JSON only
"""

def load_pil_image(path: Path) -> Image.Image:
    # 画像を開いてRGBにそろえる
    # EXIFの回転情報も反映して、向きがおかしくならないようにする
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    return img


def extract_json(text: str) -> Dict[str, Any]:
    # モデルの返答からJSONだけを取り出す関数
    # 理想はそのままJSONだが、たまに ```json ... ``` で返ることもある
    text = text.strip()

    # 1. そのままJSONとして読めるか試す
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2. ```json ... ``` の形なら中身だけ抜く
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        return json.loads(m.group(1))

    # 3. とにかく { ... } を見つけて抜く
    m = re.search(r"(\{.*\})", text, flags=re.S)
    if m:
        return json.loads(m.group(1))

    # どれでもだめなら失敗
    raise ValueError(f"JSON parse failed: {text[:500]}")


def nearest_anchor(x: Any) -> float:
    # スコアを 0.00 / 0.25 / 0.50 / 0.75 / 1.00 のどれかに丸める
    # モデルが少しズレた数を返しても、最終的には5段階にそろえる
    try:
        v = float(x)
    except Exception:
        return 0.00

    v = max(0.0, min(1.0, v))
    return min(ANCHORS, key=lambda a: abs(a - v))


def clamp01(x: Any) -> float:
    # 数字を 0.0 ～ 1.0 の間におさめる
    try:
        v = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


def normalize_result(obj: Dict[str, Any], raw_text: str, image_path: Path) -> Dict[str, Any]:
    # モデルの出力を、保存しやすい形に整える
    evidence = obj.get("main_evidence", [])

    # main_evidence がリストでない場合でも壊れないようにする
    if not isinstance(evidence, list):
        evidence = [str(evidence)]

    return {
        "image_path": str(image_path),
        "b1_darkness": nearest_anchor(obj.get("b1_darkness", 0.00)),
        "b2_weather_visibility": nearest_anchor(obj.get("b2_weather_visibility", 0.00)),
        "b3_quality_degradation": nearest_anchor(obj.get("b3_quality_degradation", 0.00)),
        "b4_incompleteness_occlusion": nearest_anchor(obj.get("b4_incompleteness_occlusion", 0.00)),
        "b5_other_residual": nearest_anchor(obj.get("b5_other_residual", 0.00)),
        "main_evidence": [str(x)[:100] for x in evidence[:4]],
        "confidence": round(clamp01(obj.get("confidence", 0.00)), 2),
        "raw_response": raw_text,
    }


def load_done_set(jsonl_path: Path) -> set:
    # すでに終わった画像の一覧を読み込む
    # これがあると、途中で止まっても再開しやすい
    done = set()

    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                    done.add(obj["image_path"])
                except Exception:
                    # 壊れた行があっても全体を止めない
                    pass

    return done
