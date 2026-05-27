# Viva Presentation: Aspect-Based Sentiment Analysis Restaurant Chatbot

**3 presenters · 12 minutes max · Code-focused + Live Demo**

---

## Presentation Roles & Timing

| Presenter | Time | Focus |
|-----------|------|-------|
| **Presenter 1** | 0:00–4:00 | Introduction, Dataset, Live Demo |
| **Presenter 2** | 4:00–8:00 | Architecture & Code Walkthrough |
| **Presenter 3** | 8:00–12:00 | Evaluation, Results, Q&A Prep |

---

## Quick Reference – Source Code Map

| Component | File | Line |
|-----------|------|------|
| App entry point | `run_chatbot.py` | 1 |
| Main orchestrator | `chatbot/app.py` | 1 |
| Paths, env, spaCy/NLTK | `chatbot/config.py` | 1 |
| XML parser + text cleaning | `chatbot/data_utils.py` | 1 |
| Aspect extraction + training | `chatbot/absa.py` | 1 |
| Domain knowledge + query answers | `chatbot/knowledge.py` | 1 |
| Intent detection | `chatbot/intents.py` | 1 |
| Response templates | `chatbot/responses.py` | 1 |
| Conversation memory | `chatbot/memory.py` | 1 |
| LLM wrapper | `chatbot/llm.py` | 1 |
| Test harness (72 questions) | `test_chatbot_comprehensive.py` | 1 |

---

## Key Stats (Memorize)

| Metric | Value |
|--------|-------|
| Dataset | SemEval-2014 Task 4, 3,041 sentences, 3,693 aspects |
| Sentiment Model | TF-IDF + SMOTE + Logistic Regression |
| Accuracy | **70.99%** · Weighted F1 **0.715** |
| Positive F1 | 0.837 · Negative 0.656 · Neutral 0.451 · Conflict 0.213 |
| Category Mapper | Keyword rule engine (340x faster than BART) |
| LLM Wrapper | OpenRouter API, model `z-ai/glm-4.5-air` |
| Chatbot Passable Rate | **100%** (72/72 questions, 0 wrong) |
| Test Harness | 12 categories × 72 questions |

---

---

## PRESENTER 1 — Introduction + Demo (4 min)

### SLIDE 1: Title & Problem (1 min)

**"Aspect-Based Sentiment Analysis Restaurant Chatbot"**

**Say:** "We built a chatbot that analyses restaurant reviews aspect-by-aspect and answers customer questions using real guest feedback. Instead of one score per review, it breaks down sentiment for each thing mentioned — food, service, price, ambience — separately."

**Show:** Two modes:
1. **Review analysis** — user types a review, bot extracts aspects and classifies sentiment
2. **Domain Q&A** — user asks "Is the food good?", bot answers from pre-computed stats

---

### SLIDE 2: Dataset (30 sec)

**SemEval-2014 Task 4 Restaurant Corpus**

- 3,041 review sentences with 3,693 aspect annotations
- Each sentence tagged with aspect terms + categories + polarity
- 4 sentiment classes: positive, negative, neutral, conflict

**Source:** `chatbot/data_utils.py:71` (parse_restaurant_xml)

**Say:** "Same dataset serves two purposes — trains the sentiment classifier AND populates the knowledge base for Q&A."

---

### SLIDE 3: Live Demo (2.5 min)

**Switch to terminal — run:** `python3 run_chatbot.py`

#### Demo 1 — Greeting (natural response)
```
You: Hey, how are you doing?
Bot: I'm doing well, thanks for asking! I can share what our guests say about the
     food, service, price, and atmosphere. How can I help?
```

#### Demo 2 — Review Analysis (multi-aspect)
```
You: The steak was perfectly cooked but the service was incredibly slow
     and the bill was way too high.
Bot: [LLM] I found three things in your review:
     — Steak (food): positive — perfectly cooked
     — Service: negative — incredibly slow
     — Bill (price): negative — way too high
     Your experience was mixed — great food but let down by service.
```

#### Demo 3 — Domain Q&A (promotional tone)
```
You: Tell me about your restaurant
Bot: [LLM] We're known for our food — 7 out of 10 guests mention it positively!
     Our service is well-regarded too (54% positive), people love the atmosphere
     (61% positive), and guests feel they get good value. Popular dishes include
     food, service, prices. What catches your eye?
```

**Point out:** "See how it promotes ALL categories — food, service, ambience, value — like real staff. The `[LLM]` tag tells you the LLM is active."

#### Demo 4 — Tech Question
```
You: What model are you using?
Bot: [LLM] I use Logistic Regression with TF-IDF features, trained on 3,693
     aspect annotations from the SemEval-2014 corpus.
```

#### Demo 5 — Thinking indicator + Off-domain
```
You: What's the capital of France?
Bot: Thinking...
Bot: I'm here for restaurant questions! Want to know what guests love about
     our food, or how our service rates?
```

**Point out:** "The 'Thinking...' indicator shows when the LLM is processing. Graceful redirect for off-topic questions."

---

---

## PRESENTER 2 — Architecture & Code (4 min)

### SLIDE 4: System Architecture (1.5 min)

**Show diagram (draw or display):**

```
                    USER INPUT
                        │
               detect_intent()          ← chatbot/intents.py
                        │
    ┌──────┬──────┬─────┼─────┬──────┬──────┐
    ▼      ▼      ▼     │     ▼      ▼      ▼
 greeting farewell help  │  domain  tech   off_domain
                   _query   _questions
                         │
                    ┌────┴────┐
                    │  ABSA   │
                    │ Pipeline│       ← chatbot/absa.py
                    └────┬────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         extract      map to     classify
         aspects     category    sentiment
         (spaCy)    (keywords)  (TF-IDF+LR)
                         │
                    ┌────┴────┐
                    │   LLM   │       ← chatbot/llm.py
                    │ Wrapper │
                    └────┬────┘
                         ▼
                      RESPONSE
```

**Key source files to show:**

| Step | File | Function |
|------|------|----------|
| Intent routing | `chatbot/intents.py:115` | `detect_intent()` |
| Aspect extraction | `chatbot/absa.py:52` | `extract_aspects_spacy()` |
| Category mapping | `chatbot/absa.py:92` | `predict_category_fast()` |
| Feature engineering | `chatbot/absa.py:128` | `make_feature()` |
| Training | `chatbot/absa.py:140` | `train_model()` |
| Knowledge base | `chatbot/knowledge.py:121` | `compute_domain_knowledge()` |
| Main chat loop | `chatbot/app.py:125` | `chat()` |

---

### SLIDE 5: Intent Detection (1 min)

**Source:** `chatbot/intents.py:115`

**Say:** "The intent router uses a 11-step priority-ordered decision tree. Order matters — specific intents checked before general ones."

**Priority order:**
1. Specific tech topics (model, accuracy, training, limitations, sarcasm)
2. Help patterns
3. Remaining tech topics (how-it-works, compare)
4. Domain queries ("Is the food good?")
5. Complaint/popular keywords
6. Restaurant about-us ("tell me about your restaurant")
7. Greetings (token set intersection)
8. Farewells
9. Review (lemmatized 120+ term matching)
10. spaCy fallback (noun + copula pattern)
11. Off-domain

**Say:** "Lemmatization is key — 'waiters' lemmatizes to 'waiter' so plurals are matched correctly. This fixed several failures from the original system."

---

### SLIDE 6: ABSA Pipeline in Detail (1.5 min)

**Source:** `chatbot/absa.py`

**3-step extraction** (`extract_aspects_spacy()`, line 52):
1. **spaCy POS tagging** — finds nouns/proper nouns (pizza, waiter, atmosphere)
2. **spaCy noun chunks** — catches multi-word phrases (wait staff, dining room)
3. **Learned lexicon** — catches domain terms spaCy might miss (tiramisu, sashimi)

**Feature engineering** (`make_feature()`, line 128):
- Wraps aspect term in `[ASPECT]...[/ASPECT]` tags within the full sentence
- Example: "the pasta was cold" → `"the [ASPECT] pasta [/ASPECT] was cold"`
- Without tags, "cold" is just a word — with tags, the model learns context

**Training** (`train_model()`, line 140):
- TF-IDF converts tagged text to numeric vectors
- SMOTE balances classes (conflict has only ~45 examples)
- Logistic Regression classifies sentiment

---

---

## PRESENTER 3 — Results + Q&A (4 min)

### SLIDE 7: Performance (1.5 min)

**Source:** Evaluation in `chatbot/app.py:210` (evaluate_test_set)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Positive | 0.854 | 0.820 | **0.837** |
| Negative | 0.522 | 0.612 | 0.563 |
| Neutral | 0.483 | 0.424 | 0.451 |
| Conflict | 0.152 | 0.357 | 0.213 |
| **Accuracy** | | | **70.99%** |

**Chatbot test harness** (`test_chatbot_comprehensive.py`):
- 72 questions, 12 categories
- **100% passable, 0 wrong** after refactoring

**Say:** "Strong on positive (most common case at 60%+ of reviews). Weak on neutral and conflict — these are inherently ambiguous classes with few training examples."

---

### SLIDE 8: Key Design Decisions (1 min)

| Decision | Why |
|----------|-----|
| **Keywords not BART** for categories | 340x faster, 100% transparent, same accuracy |
| **Pre-computed stats** for domain Q&A | Every percentage traceable, no hallucination |
| **LLM as wrapper, not brain** | LLM formats naturally; facts come from deterministic computations |
| **6-path intent system** | Covers all interaction types an examiner expects |
| **3-turn memory** | Demonstrates stateful conversation within assignment scope |
| **"Thinking..." indicator** | Gives feedback during LLM API call (1.5s latency) |

---

### SLIDE 9: Limitations & Future Work (1 min)

**Limitations:**
1. No sarcasm detection — "Oh great, another cold meal" reads as positive
2. Cannot compare two specific restaurants — no identities in data
3. 2014 vocabulary — newer food terms unknown
4. Neutral/conflict F1 low (45%/21%)

**Future improvements:**
1. Replace TF-IDF with BERT/RoBERTa embeddings (~10% accuracy gain)
2. Hierarchical classification (positive vs not, then fine-grained)
3. Data augmentation for minority classes (neutral, conflict)

---

### SLIDE 10: Q&A Prep (30 sec)

**Say:** "Three things to remember:
1. **Same data, two purposes** — annotations train the classifier AND populate the knowledge base
2. **Transparency over complexity** — 5KB keyword list replaced 1.6GB BART model
3. **LLM is formatting, not facts** — if the API is down, `--no-llm` flag keeps the bot working"

**Common examiner questions:**
- "Why 70.99% accuracy?" → Neutral/conflict drag it down; 83.7% on positive
- "Why Logistic Regression?" → Interpretable, fast, competitive on this benchmark
- "What does SMOTE do?" → Creates synthetic examples for minority classes
- "How does conversation memory work?" → 3-turn deque, tracks last topic for follow-ups

---

## Presentation Tips

- Open `chatbot/` folder in your editor before presenting
- Have terminal ready with `python3 run_chatbot.py` for live demo
- Use `--no-llm` flag if API is down during viva
- Print this sheet for quick reference to line numbers
