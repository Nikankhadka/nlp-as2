"""Shared configuration — paths, constants, API settings, spaCy/NLTK loading.
These are loaded once at startup and used by all other modules."""

import sys, os, re
from pathlib import Path

# --- Environment variables (.env file) ---
_ENV_PATH = Path(__file__).resolve().parent.parent / '.env'
if _ENV_PATH.exists():
    with open(_ENV_PATH) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _key, _val = _line.split('=', 1)
                os.environ.setdefault(_key.strip(), _val.strip())

# --- Project paths (works on both Colab and local) ---
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = Path('/content/drive/MyDrive/absa')
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_XML = PROJECT_ROOT / 'data' / 'raw' / 'Restaurants_Train_v2.xml'
TEST_XML  = PROJECT_ROOT / 'data' / 'raw' / 'Restaurants_Test_Gold.xml'

# --- API config ---
LLM_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
LLM_MODEL = 'z-ai/glm-4.5-air'
LLM_API_URL = 'https://openrouter.ai/api/v1/chat/completions'
LLM_ENABLED = bool(LLM_API_KEY)

# --- Text processing constants ---
TOKEN_RE = re.compile(r'[A-Za-z][A-Za-z\-\']+')
CATEGORIES = ['food', 'service', 'price', 'ambience', 'miscellaneous']

# --- Load spaCy and NLTK (one-time, shared by all modules) ---
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

print('Loading spaCy and NLTK...', flush=True)
nlp = spacy.load('en_core_web_sm')
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
print(f'Loaded {len(STOPWORDS)} stopwords', flush=True)
