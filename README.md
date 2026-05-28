# RestaurantXpert

**Aspect-Based Sentiment Analysis chatbot for restaurant reviews.**

Built on SemEval-2014 Task 4 data, RestaurantXpert analyzes restaurant reviews, answers domain questions from real guest feedback, and explains its own ML pipeline — all from the CLI or a Jupyter notebook.

---

## Goal

Build an interactive, knowledge-grounded chatbot that:
- Extracts aspects (food, service, price, ambience) from restaurant reviews
- Classifies sentiment (positive/negative/neutral/conflict) per aspect
- Answers customer questions with pre-aggregated stats from 3,693 real annotations
- Optionally wraps responses through an LLM for natural conversation
- Runs on Google Colab (with Drive mount) or locally

---

## Architecture

```
                          ┌─────────────────────────┐
                          │     User Input           │
                          └───────────┬─────────────┘
                                      │
                          ┌───────────▼─────────────┐
                          │    Intent Router         │
                          │   (priority-ordered      │
                          │    7-intent tree)        │
                          └───┬───┬───┬───┬───┬─────┘
                              │   │   │   │   │
         ┌────────────────────┘   │   │   │   └──────────────┐
         ▼                        │   │   │                  ▼
┌─────────────────┐               │   │   │    ┌─────────────────────┐
│  Review Path    │               │   │   │    │  Domain Query Path  │
│                 │               │   │   │    │                     │
│  clean_text()   │               │   │   │    │  classify_domain()  │
│       │         │               │   │   │    │         │           │
│  spaCy POS +    │               │   │   │    │  Knowledge Base     │
│  noun chunks +  │               │   │   │    │  (3,693 annots)     │
│  learned lexicon│               │   │   │    │         │           │
│       │         │               │   │   │    │  answer_*_query()   │
│  predict cat    │               │   │   │    └─────────────────────┘
│  (keyword map)  │               │   │   │
│       │         │               │   │   │
│  [ASPECT] tag   │               │   │   │    ┌─────────────────────┐
│       │         │               │   │   │    │  Tech / Help /      │
│  TF-IDF vector  │               │   │   │    │  Off-domain Path    │
│       │         │               │   │   │    │                     │
│  Logistic Regr  │               │   │   │    │  Pre-written         │
│       │         │               │   │   │    │  templates          │
│  format_absa()  │               │   │   │    └─────────────────────┘
└────────┬────────┘               │   │   │
         │                        │   │   │
         └────────────────────────┼───┼───┘
                                  │   │
                    ┌─────────────┘   └──────────────┐
                    │                                 │
                    ▼                                 ▼
          ┌──────────────────┐           ┌──────────────────┐
          │  Template Output  │           │   LLM Wrapper    │
          │  (deterministic)  │           │  (OpenRouter API) │
          └──────────────────┘           │  natural rephrase │
                                         │  + guard rails    │
                                         └──────────────────┘

State tracked by ConversationMemory (3-turn history + last topic)
```

### Module Map

```
chatbot/
├── config.py       # Paths, env vars, spaCy/NLTK singletons
├── data_utils.py   # XML parsing, text cleaning, ParsedDataset
├── absa.py         # Aspect extraction, category mapping, TF-IDF+LR training
├── knowledge.py    # Domain stats aggregation, query answering
├── intents.py      # 7-intent priority router with lemmatized matching
├── responses.py    # Template responses for non-review intents
├── llm.py          # OpenRouter API wrapper + guard rails
├── memory.py       # 3-turn conversation memory
├── app.py          # Orchestrator: init, chat, eval, CLI, test harness
└── __init__.py
```

---

## Implementation

### Data
| Source | Sentences | Aspects | Categories |
|--------|-----------|---------|-------------|
| Train | 3,041 | 3,693 | 3,713 |
| Test  | 800 | ~1,134 | ~1,134 |

### Sentiment Model
```
Text → clean_text() → [ASPECT] feature tagging → TfidfVectorizer (ngram 1-2, 5000 feats)
     → SMOTE oversampling → Logistic Regression (max_iter=1000)
```
SMOTE balances the rare conflict class (~45 examples) against the dominant positive class.

### Aspect Extraction
Three cascading methods:
1. **spaCy POS tagging** — nouns and proper nouns (pasta, waiter, atmosphere)
2. **spaCy noun chunks** — multi-word phrases (wait staff, dining room)
3. **Learned lexicon** — domain terms spaCy might miss (tiramisu, sashimi)

### Category Mapping
Keyword-based classifier maps aspect terms to food/service/price/ambience/misc — 340× faster than BART zero-shot, fully explainable, zero GPU dependency.

### Intent Routing
Priority-ordered decision tree: specific tech topics → help → remaining tech → domain query → greeting → farewell → review → spacy fallback → off-domain.

### LLM Integration (optional)
- Model: `z-ai/glm-4.5-air` via OpenRouter
- Phase 1 generates a deterministic, factually correct response
- Phase 2 sends it to the LLM for natural rephrasing
- If the API is down or `--no-llm` flag is set, falls back to deterministic
- Guard rails block medical/legal/financial advice and enforce 2-strike off-domain limit

### Colab Support
`config.py` detects the Colab runtime and auto-mounts Google Drive, resolving paths to `/content/drive/MyDrive/absa`. Works identically on local machines.

---

## Results

| Metric | Value |
|--------|-------|
| Accuracy | **70.99%** |
| Weighted F1 | **0.715** |
| Positive F1 | 0.837 |
| Negative F1 | 0.684 |
| Neutral F1 | 0.451 |
| Conflict F1 | 0.213 |

The model excels at positive/negative classification but struggles with neutral and conflict labels — neutral reviews often have implicit sentiment, and conflict is severely underrepresented.

---

## Outcome

A fully functional, modular restaurant domain chatbot with:
- Deterministic review analysis via traditional ML (no GPU required)
- Pre-computed knowledge base for instant domain Q&A
- Optional LLM layer for natural conversation
- 50-question automated test harness
- Graceful degradation when LLM is unavailable
- Runs in CLI, Jupyter, or Google Colab

---

## Future Improvements

- **RAG integration**: Index guest reviews in a vector DB (Chroma/FAISS) to ground answers in real quotes rather than pre-aggregated percentages. Enables "show me an example" and "why?" follow-ups with actual review context.
- **Multi-turn conversational RAG**: Chain follow-up questions ("Why was the pizza bad?" → retrieve reviews mentioning pizza negatively → generate grounded response)
- **Sarcasm-aware classification**: Contrastive learning or prompt-based detection with few-shot examples
- **Web UI**: Streamlit/Gradio interface with drag-and-drop review upload and batch analysis
- **Unit tests and CI**: pytest suite covering intent routing, aspect extraction, and knowledge base queries
- **Model registry**: Track experiments (TF-IDF hyperparams, SMOTE ratios, feature engineering variants)

---

## Setup

```bash
# Clone and install
git clone <repo-url>
cd absa
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# (Optional) Set up LLM — get a free key at https://openrouter.ai/keys
cp .env.example .env
# Edit .env and add your key
```

## Usage

### CLI

```bash
# Full chatbot (LLM enabled if API key is set)
python run_chatbot.py

# Keyword-only mode — no LLM, faster, fully deterministic
python run_chatbot.py --no-llm

# Run evaluation + 50-question test harness only (skip interactive chat)
python run_chatbot.py --test-only
```

### Jupyter Notebook (local)

```bash
# Launch and open RestaurantXpert.ipynb
jupyter notebook RestaurantXpert.ipynb
```

The notebook runs the full pipeline cell by cell:
1. Auto-detects local vs Colab environment
2. Installs dependencies if needed, downloads spaCy model
3. Loads and explores the SemEval-2014 dataset
4. Trains the TF-IDF + SMOTE + LR sentiment model
5. Builds the domain knowledge base
6. Evaluates on the test set (accuracy, weighted F1, per-class report)
7. Demos aspect extraction on 4 example reviews
8. Runs a 50-question test harness to verify intent routing
9. Launches an **interactive chat loop** (skipped gracefully in automated runs)

The interactive chat cell uses `input()` and works in Jupyter, VSCode, and Colab.

### Google Colab

1. Upload the project to Google Drive into `MyDrive/absa/`
2. Open `RestaurantXpert.ipynb` in Colab
3. Run all cells — the notebook auto-mounts Drive and resolves all paths
4. Make sure your dataset files are at `MyDrive/absa/data/raw/`

### Automated / CI execution

```bash
# Execute the full notebook non-interactively (skips the chat loop)
jupyter nbconvert --to notebook --execute RestaurantXpert.ipynb \
    --output executed.ipynb --ExecutePreprocessor.timeout=120
```
