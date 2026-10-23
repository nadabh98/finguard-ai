from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"
EXPORTS = ROOT / "exports"

# Split temporel : 70 % passé / 15 % validation / 15 % test (le plus récent)
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15

# Hypothèses de coût (à justifier dans le README)
FALSE_ALERT_COST = 5.0  # coût d'une fausse alerte en euros (analyste + friction client)
