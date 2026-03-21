# A04. 入出力設定
DATA_ROOT = Path("/content/drive/MyDrive/lila_camera_traps_10000/run_20260311_183115")
#lila_camera_trapsから、無作為に10000枚抽出
OUT_DIR = DATA_ROOT / "qwen_c_labels_anchor5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSONL_PATH = OUT_DIR / "c_pseudo_labels_anchor5.jsonl"
CSV_PATH   = OUT_DIR / "c_pseudo_labels_anchor5.csv"
ERR_PATH   = OUT_DIR / "c_pseudo_labels_anchor5_errors.jsonl"

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGET_N = 10000
RANDOM_SEED = 42
MAX_NEW_TOKENS = 220
PROGRESS_EVERY = 100

ANCHORS = [0.00, 0.25, 0.50, 0.75, 1.00]

print("DATA_ROOT:", DATA_ROOT)
print("OUT_DIR  :", OUT_DIR)
