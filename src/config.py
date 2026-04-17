from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
EDA_DIR = OUTPUT_DIR / 'eda'
MODELS_DIR = OUTPUT_DIR / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'

TRAIN_XML = RAW_DIR / 'Restaurants_Train_v2.xml'
TEST_XML = RAW_DIR / 'Restaurants_Test_Gold.xml'

CATEGORY_ORDER = [
    'food',
    'service',
    'price',
    'ambience',
    'anecdotes/miscellaneous',
]
POLARITY_ORDER = ['positive', 'negative', 'neutral', 'conflict']

for path in [PROCESSED_DIR, EDA_DIR, MODELS_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
