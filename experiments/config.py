from pathlib import Path

# Project/root paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SA_DIR = DATA_DIR / "sa_canvas"
MIXED_DIR = DATA_DIR / "mixed_canvas"
DRAW_DIR = DATA_DIR / "draw_canvas"
RESULT_DIR = ROOT / "results"

# Model defaults
OPENAI_MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-small"

# Dataset sizes
N_SA_TRAIN = 800
N_SA_DEV = 100
N_SA_TEST = 100
