# A06. Qwen2.5-VL を読み込む
# ここでは「画像を見てJSONを返すAI本体」を用意する

# processor は、画像や文章をモデルに食べさせやすい形に整える係
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# model は実際に推論する本体
# GPUがbfloat16を使えるならbfloat16、そうでなければfloat16を使う
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)

# 学習ではなく推論だけするので eval モードにする
model.eval()

print("model loaded")
