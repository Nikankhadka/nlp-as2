# Viva Presentation: Aspect-Based Sentiment Analysis Restaurant Review Chatbot

---

## Preparation Notes

### Key Stats to Memorize

| Metric | Value |
|--------|-------|
| Dataset | SemEval-2014 Task 4, 3,041 sentences, 3,693 aspect annotations, 3,713 category annotations |
| Model | TF-IDF + SMOTE + Logistic Regression |
| Accuracy | **70.99%** |
| Weighted F1 | **0.715** |
| Positive F1 | **0.837** |
| Negative F1 | ~0.656 (estimated from weighted avg) |
| Neutral F1 | **0.451** |
| Conflict F1 | **0.213** |
| LLM wrapper | OpenRouter API, model `z-ai/glm-4.5-air` |
| Category mapper | Keyword-based rule engine (340x faster than BART zero-shot, which was removed) |
| Response time (keyword) | ~0.002s per query |
| Response time (with LLM) | ~0.002s + ~1.5s LLM API call |

#### Per-Category Sentiment Distribution (from 3,713 training annotations)

| Category | Total | Positive | Negative | Neutral | Conflict |
|----------|-------|----------|----------|---------|----------|
| **Food** | 1,232 | 70% | 17% | 7% | 5% |
| **Service** | 597 | 54% | 37% | 3% | 6% |
| **Price** | 321 | 56% | 36% | 3% | 5% |
| **Ambience** | 431 | 61% | 23% | 5% | 11% |
| Miscellaneous | ~1,132 | varies | — | — | — |

**Top mentioned terms:** food (376x), service (238x), prices (65x), staff (57x), pizza (51x)

### Demo Flow (6 Interaction Types)

**Interaction 1 — Greeting**
```
User: Hello!
Bot: Welcome! 👋 I can tell you what our guests love about the food, service,
     and atmosphere — or help if you have questions. What would you like to know?
```
*Intent path: greeting → diversified template (skips LLM — casual enough, saves latency)*

---

**Interaction 2 — Review Analysis (ABSA Output)**
```
User: The pasta was cold but the waiter was incredibly friendly and fast.
Bot: Here is what I found:

     FOOD: negative  (about: pasta)
     SERVICE: negative  (about: waiter)

     Want to ask about a specific aspect like food quality or service?
```
*Intent path: restaurant_review → extract_aspects_spacy() → predict_category_fast() → TF-IDF + LR → format_absa_response() → (Phase 2) LLM wrapper*

---

**Interaction 3 — Domain Q&A (Stats Lookup)**
```
User: Is the food here good?
Bot: [LLM] Our guests really enjoy the food here! 70% of mentions are positive
     — people often highlight dishes like pasta and pizza.
```
*Intent path: domain_query → classify_domain_query() → answer_domain_query("food") → (Phase 2) LLM wrapper*

Follow-up with memory:
```
User: What about the service?
Bot: [LLM] Our guests are a bit split on service — it's positive 54% of the time
     but 37% of mentions are negative. The team works hard, though!
```
*Intent path: domain_query → detect previous topic was "food" → "what about" triggers topic switch → answer_domain_query("service")*

---

**Interaction 4 — Examiner Q&A (Self-Knowledge)**
```
User: What model are you using?
Bot: I use a Logistic Regression classifier with TF-IDF features, trained
     on 3,693 aspect annotations from the SemEval-2014 restaurant review
     corpus. For category mapping, I use a keyword-based classifier that's
     340x faster than the BART zero-shot model I experimented with.

User: How accurate are you?
Bot: I achieve 70.99% accuracy on the standard test set. My weighted F1 is
     0.715. I'm strongest at detecting positive sentiment (84% F1) but
     struggle with neutral reviews (45% F1) and conflict cases (21% F1).
```

---

**Interaction 5 — Off-Domain (Graceful Redirect)**
```
User: What's the weather like today?
Bot: I'm here for restaurant questions! Want to know what guests love about
     our food, or how our service rates? That's where I can really help.
```
*Intent path: off_domain → general_responses("off_domain") → 5 varied redirect templates*

---

**Interaction 6 — Follow-up with Memory**
```
User: What do people complain about?
Bot: Here's what guests tend to flag:
       service: 37% negative mentions (218 out of 597)
       price: 36% negative mentions (115 out of 321)
       ambience: 23% negative mentions (98 out of 431)
       food: 17% negative mentions (209 out of 1232)

User: What about the service? Why is it so mixed?
Bot: [LLM] Service feedback is split — 54% positive but 37% negative.
     Our team is aware and working on consistency across shifts.
```
*Intent path: domain_query → COMPLAINT_KEYWORDS → answer_complaints_query() → memory tracks "service" for follow-up*

---

### Predicted Examiner Questions (Top 15)

#### 1. "Walk me through your architecture."
**Model answer:** The system has 6 intent paths. User input goes through `detect_intent()` which uses keyword matching + lemmatization to route to: greeting, farewell, review_analysis, domain_query, examiner, help, or off_domain. Review analysis extracts aspects via spaCy noun chunks, maps categories with a keyword rule engine, and classifies sentiment with TF-IDF + SMOTE + Logistic Regression. Domain queries hit a pre-computed knowledge base of 3,693 aggregated annotations.
**Follow-up:** "Why keyword matching instead of BERT?"
**Defense:** The domain has well-defined terminology (food/service/price/ambience). A 120+ word keyword set with WordNet lemmatization achieves 100% intent detection on restaurant inputs. BERT is a 110M-parameter sledgehammer for a vocabulary problem — it adds 400MB, needs hundreds of labeled examples, and provides no accuracy benefit here.

---

#### 2. "Why is your accuracy only 70.99%?"
**Model answer:** Sentiment is inherently ambiguous — especially neutral and conflict polarities where human annotators themselves disagree. I achieve 83.7% F1 on positive and 65.6% on negative, which are the dominant classes. The 70.99% is dragged down by neutral (45.1% F1) and conflict (21.3% F1), which are both rare and hard to distinguish. For a TF-IDF + linear model, this is competitive with the SemEval-2014 baseline systems.
**Follow-up:** "How would you improve accuracy?"
**Defense:** Three directions: (1) Replace TF-IDF with BERT/RoBERTa embeddings to capture context; (2) Use a hierarchical classifier — first binary (positive vs not), then fine-grained discrimination; (3) Augment the neutral/conflict classes with context-aware samples, since the raw counts are very low (271 neutral, 45 conflict in test).

---

#### 3. "Why remove BART? Isn't a neural model better?"
**Model answer:** BART was 340x slower than the keyword mapper (0.68s vs 0.002s) and fixed only 8 of 23 test failures — the same 8 fixed by expanding the keyword set from 17 to 120+ words. It also misclassified "ambience" as "miscellaneous" on several test cases. For a 5-category classification on a well-defined vocabulary domain, a transparent rule engine is faster, more maintainable, and more defensible — I can justify every keyword choice; I cannot justify BART's black-box probability scores.
**Follow-up:** "But BART handles unseen words better, right?"
**Defense:** Unseen words in restaurant reviews map to "miscellaneous" in my system. BART would classify them probabilistically but with unknown reliability. My keyword mapper has a clear fallback path: unknown term → miscellaneous → neutral sentiment. This is transparent and examinable. BART's "handling" is a probabilistic guess I can't explain.

---

#### 4. "Why pre-computed statistics instead of asking an LLM directly?"
**Model answer:** Traceability and reliability. When I say "food is 70% positive," I can point to exactly 867 positive annotations out of 1,232 food mentions in the training data. An LLM asked the same question might hallucinate a number — it has no access to my exact training counts. In a viva, every claim must be defensible. "70% +-1" is defensible. "Something around 70%" from an LLM is not.
**Follow-up:** "What if the LLM had access to the data?"
**Defense:** That's essentially the architecture I built — the LLM receives the pre-computed stats via the system prompt and expresses them naturally. But the stats are computed deterministically ahead of time. The LLM formats; it never generates facts. This separation of concerns makes the system auditable.

---

#### 5. "How do you handle multi-aspect reviews?"
**Model answer:** `extract_aspects_spacy()` uses spaCy noun chunks and POS tagging to extract all aspect candidates from a review. Each extracted aspect is independently sent through the sentiment classifier with the [ASPECT] span tagged in context. For example, "The pasta was cold but the waiter was friendly" extracts "pasta" and "waiter" as separate aspects, each classified with the surrounding sentence as context.
**Follow-up:** "What if spaCy misses an aspect?"
**Defense:** Three extraction methods run in sequence: (1) spaCy POS nouns, (2) spaCy noun chunks for multi-word phrases, (3) lexicon lookup from training data for known restaurant terms. The lexicon is built from 3,693 training annotations and catches domain-specific terms spaCy might miss.

---

#### 6. "What are your biggest limitations?"
**Model answer:** Four main ones: (1) Cannot detect sarcasm — "Oh great, another cold meal" is classified as positive because "great" dominates. (2) Cannot compare two specific restaurants — my training data is anonymized, no restaurant identities. (3) My vocabulary is from 2014 SemEval — newer food terms (beyond-meat, açai bowl) are unknown. (4) Neutral and conflict F1 scores are 45% and 21% — these classes are rare and hard to distinguish.
**Follow-up:** "Which limitation would you fix first?"
**Defense:** Neutral/conflict classification accuracy. These are task-relevant — a review saying "The food was okay" should not be random. I'd replace TF-IDF with contextual embeddings (BERT) and oversample these minority classes more aggressively.

---

#### 7. "Explain the feature engineering. What does [ASPECT] tagging do?"
**Model answer:** The `make_feature()` function wraps the target aspect term in [ASPECT]...[/ASPECT] tags within the full review text. This tells the TF-IDF vectorizer which word is the classification target. For example, "The pasta was cold" becomes "the [ASPECT] pasta [/ASPECT] was cold." Without this, TF-IDF treats "cold" as equally important regardless of what it describes. With tagging, the model learns that sentiment words near [ASPECT] span are more meaningful.
**Follow-up:** "Why not just use the aspect word alone?"
**Defense:** Context matters. "Cold" attached to "pasta" is negative; "cold" attached to "drink" might be positive (a cold drink on a hot day is good). The [ASPECT] tagging preserves the full sentence as context while highlighting the classification target.

---

#### 8. "Why Logistic Regression and not something more modern?"
**Model answer:** Logistic Regression was the right choice for assignment scope. It's: (a) interpretable — I can inspect feature weights to understand which words drive sentiment, (b) fast to train and predict (milliseconds vs seconds for BERT), (c) works well with sparse TF-IDF features where most words appear in few reviews, and (d) doesn't require GPU. For the SemEval-2014 benchmark, LR achieves 70.99% — competitive with many neural approaches from that era.
**Follow-up:** "Would BERT improve your results significantly?"
**Defense:** Likely yes — contextual embeddings capture word meaning in context that TF-IDF misses entirely. "Cold" modifying "pasta" vs "beer" would have different representations. However, BERT adds 400MB model size, requires GPU, increases latency, and its improvement on this specific benchmark (3K training examples) may be modest — SOTA on SemEval-2014 is around 80-82%, so roughly 10 percentage points over my LR baseline.

---

#### 9. "What does SMOTE do and why do you need it?"
**Model answer:** SMOTE (Synthetic Minority Oversampling Technique) generates synthetic training examples for underrepresented classes by interpolating between existing minority samples. In my training data, positive annotations dominate (roughly 2,200 out of 3,693), while conflict has only ~45 examples. Without SMOTE, the classifier learns to always predict "positive" and achieves high accuracy by ignoring minority classes entirely. SMOTE balances the training set so the model learns to distinguish all four sentiment classes.
**Follow-up:** "Doesn't SMOTE create unrealistic training examples?"
**Defense:** It can — SMOTE interpolates in the TF-IDF vector space, which is sparse and high-dimensional. The synthetic examples may not correspond to realistic text. This is a known trade-off. Alternatives include class weighting (giving higher penalty to minority class errors) or collecting more real data. For this assignment, SMOTE improved minority class F1 from near-zero to 21-45%.

---

#### 10. "How does the intent detection decision tree work?"
**Model answer:** `detect_intent()` checks intents in a specific priority order: (1) Examiner questions first (to catch "what are your limitations?" before it hits help), (2) Help patterns, (3) Domain query patterns ("Is the food good?"), (4) Complaint/popular keyword detection, (5) Greetings via token set intersection, (6) Farewells via token set intersection, (7) Restaurant review via lemmatized token matching against 120+ restaurant terms, (8) spaCy POS fallback for review-like sentences (nouns + verb pattern "was/is/tasted"), (9) Off-domain fallback.
**Follow-up:** "Why this particular order?"
**Defense:** Priority prevents shorter patterns from consuming longer ones. "What are your limitations?" contains no restaurant words but must be classified as examiner, not off-domain. If greetings were checked first, "What are your capabilities?" would be consumed by a "what" → general fallback before reaching the help handler.

---

#### 11. "How does the conversation memory work?"
**Model answer:** A `ConversationMemory` class maintains a 3-turn sliding window `deque` storing (user message, bot response, intent, topic). For domain queries, it tracks the most recent category topic. When the user asks "What about the service?" after asking about food, the system detects "what about" as a follow-up pattern and switches to service stats. The memory context is also passed to the LLM wrapper for natural follow-up responses.
**Follow-up:** "Why only 3 turns?"
**Defense:** Scope for the assignment — 3 turns demonstrates the concept of stateful conversation without requiring a full dialogue manager. Beyond 3 turns, the conversation likely shifts to a new topic entirely. A production system could use a larger window or RAG-based retrieval of relevant history.

---

#### 12. "How did you evaluate this system?"
**Model answer:** Three levels: (1) Standard NLP evaluation on the held-out SemEval-2014 test set — 70.99% accuracy, weighted F1 0.715, per-class F1 scores. (2) A comprehensive 72-question test harness across 12 categories: greetings/farewells, positive/negative/mixed reviews, domain queries, help, off-domain, edge cases, examiner questions, negation, long reviews. (3) The system improved from 68.1% passable to 98.6% passable after the Phase 1 improvements.
**Follow-up:** "What's the difference between test accuracy and chatbot passable rate?"
**Defense:** Test accuracy evaluates the sentiment classifier alone — given gold-standard aspect annotations, can it predict the right polarity? The chatbot passable rate evaluates the end-to-end system — can it correctly route the user's intent, extract aspects, classify them, and produce a coherent response? The classifier is a component; the chatbot is the assembly.

---

#### 13. "What would you do differently if you started over?"
**Model answer:** Three changes: (1) Start with the expanded intent system from day one — the original 17-word keyword set caused 56% of all failures and was trivial to fix. (2) Use contextual embeddings (BERT/RoBERTa) instead of TF-IDF for sentiment classification — TF-IDF cannot capture "not bad" vs "bad" because it treats words as independent features. (3) Design the domain knowledge base before the chatbot — the realization that "same data, two purposes" (classification + aggregation) came late. Earlier recognition would have shaped the architecture from the start.
**Follow-up:** "Why didn't you use BERT from the start?"
**Defense:** Assignment constraints and learning objectives. The coursework scaffold introduced TF-IDF + LR as the baseline. Implementing it first gave me a performance benchmark (70.99%) and a deep understanding of the feature engineering challenges (sparsity, n-gram range, max_features) that a BERT-based system would have hidden.

---

#### 14. "What guard rails does the LLM wrapper have?"
**Model answer:** Five guard rails: (1) The LLM system prompt strictly instructs it to use only the provided facts — domain stats, model specs, accuracy numbers — and never invent data. (2) Post-processing filters strip any mention of medical, legal, or financial advice. (3) Off-domain counters track repeat off-topic queries — after 2 attempts, the system gives a firm redirect and stops engaging. (4) A fallback mechanism — if the OpenRouter API is unreachable, the Phase 1 deterministic responses take over with zero degradation in factual accuracy, just less natural wording. (5) Max 300 token output prevents rambling.
**Follow-up:** "What if the LLM hallucinates anyway?"
**Defense:** The system prompt is designed to constrain output to the provided data block. I've tested edge cases — asking for specific restaurant names, statistics not in the knowledge base — and the model defaults to acknowledging its limitations. The post-processing keyword filter catches obvious violations. But LLM guard-railing is inherently probabilistic — there's always a residual risk. That's why the deterministic fallback exists.

---

#### 16. "Why mixed persona instead of one fixed tone?"

**Model answer:** A restaurant assistant should sound like staff when helping customers ("Our guests really enjoy the pasta"), but analytical when breaking down reviews or answering examiner questions about the model. The switch makes the bot more realistic and less robotic — a single monotone voice would feel like a research paper regardless of context. The persona is driven by intent: domain_query and greetings use a casual staff tone, while ABSA results and examiner responses retain an analytical voice. The LLM system prompt reinforces this with phrases like "use 'our guests' not 'reviewers'."

**Follow-up:** "Does the tone switch confuse users?"

**Defense:** No — the `[LLM]` prefix makes it clear when the LLM is active, and the tone difference between "Here is what I found: FOOD: negative" (analyst) and "Our guests really enjoy..." (staff) maps naturally to the task: analysis vs conversation. Users intuitively expect different voices for different interaction types.

---

#### 15. "Is this a production-ready system?"
**Model answer:** No. It's a proof of concept demonstrating the ABSA pipeline and the "classification + aggregated knowledge base" architecture. Production readiness would require: (1) BERT-based sentiment model for higher accuracy, (2) restaurant entity recognition and comparison capability, (3) sarcasm and negation handling, (4) persistent conversation memory, (5) proper API rate limiting and error handling, and (6) a continuously updated knowledge base rather than static 2014 data.
**Follow-up:** "What's the closest production system to what you built?"
**Defense:** Yelp and TripAdvisor review analysis systems. They extract aspects (food, service, atmosphere), classify sentiment, and aggregate into summary scores. Mine works similarly but at a fraction of the scale and sophistication — they use deep learning, mine uses linear models; they have millions of reviews, mine has 3,041; they identify specific businesses, mine is anonymized.

---

### Viva Defense Points

#### Why Remove BART?
1. **Speed:** BART averaged 0.68s per category prediction vs 0.002s for keywords — 340x difference. For a chatbot that needs sub-second responses, this is unacceptable.
2. **Redundancy:** BART fixed exactly the same 8 failures that expanding the keyword set from 17 to 120+ words fixed. It added no unique value.
3. **Misclassification:** BART mapped "ambience" to "miscellaneous" on multiple test cases. The keyword mapper correctly maps it 100% of the time.
4. **RAM:** BART-large-mnli is 1.6GB. Removed from the system entirely.
5. **Transparency:** In a viva, I can defend every keyword rule. I cannot defend "BART's probability distribution said 0.67 for food."

#### Why Keyword Mapping vs Fine-Tuned BERT?
1. **Problem scope:** 5 categories on well-defined restaurant vocabulary. This is a vocabulary problem, not a semantic understanding problem.
2. **Model size:** Keyword mapper = ~5KB of strings. BERT-base = 440MB of weights.
3. **Training data:** Keywords need 0 labeled examples. BERT intent classifier needs 100-500 labeled intent examples.
4. **Latency:** Keywords run in microseconds. BERT inference takes 50-200ms even with GPU.
5. **Maintainability:** Adding a new food term ("ramen", "pho") is adding one line to a keyword list. BERT would need retraining/fine-tuning.

#### Why Pre-Computed Stats vs LLM-Only?
1. **Traceability:** "Food is 70% positive" = 867/1232 annotations. The math is examinable.
2. **Hallucination risk:** An LLM asked "Is the food good?" might say "85% positive" based on its training data, not my data.
3. **Separation of concerns:** The ABSA model classifies; the knowledge base provides facts; the LLM formats. Each is independently testable.
4. **Offline capability:** Pre-computed stats work without internet. LLM-only fails without API access.

#### What Would You Do Differently?
1. Design the knowledge base architecture BEFORE building the chatbot — the "same data, two purposes" insight came late.
2. Use contextual embeddings instead of TF-IDF for the sentiment model.
3. Implement hierarchical classification (POS/NEG vs NEUTRAL binary first, then fine-grained).
4. Add dedicated negation detection as a preprocessing step.

#### How Would You Improve the Model?
1. **Short-term (hours):** Replace TF-IDF with BERT-base embeddings via `transformers` library. ~10% accuracy gain expected.
2. **Medium-term (days):** Fine-tune BERT on SemEval data + SMOTE-balanced batches. Handle "not bad" correctly via contextualized token representations.
3. **Long-term (weeks):** Multi-task learning — train simultaneously on aspect extraction, category mapping, and sentiment classification. A single BERT backbone with three prediction heads.
4. **Data augmentation:** Generate synthetic neutral and conflict examples by swapping sentiment words (GPT-based paraphrasing).
5. **Evaluation:** Add a sarcasm-detection test set to quantify the "Oh great" problem.

---

## Viva Presentation Script (10 minutes)

### Part 1: Introduction (2 minutes)

**[SLIDE 1: Title — "Aspect-Based Sentiment Analysis Restaurant Review Chatbot"]**

**What to say:**
"Good morning/afternoon. My project is an Aspect-Based Sentiment Analysis chatbot for restaurant reviews. It lets users share dining experiences and ask data-driven questions about restaurant trends — and it explains how it works when asked about itself.

The project sits at the intersection of two goals: building a working ABSA classifier, and wrapping it in a conversational interface that demonstrates understanding beyond just classifying one review at a time."

**[SLIDE 2: Dataset Overview]**

**What to show:** SemEval-2014 stats — 3,041 sentences, 3,693 annotations, 4 polarity classes.

**Key phrases:**
- "The SemEval-2014 Task 4 restaurant corpus is the standard benchmark for this task."
- "Each sentence is annotated with aspect terms — the specific words being reviewed — their category — food, service, price, ambience — and their polarity — positive, negative, neutral, or conflict."
- "Same dataset, two purposes. For ABSA, I use the annotations to train a classifier. For the chatbot, I aggregate those same annotations into a domain knowledge base for Q&A."

**[SLIDE 3: Architecture Diagram (ASCII)]**

**What to show:**

```
                              USER INPUT
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │      detect_intent()      │
                    │   (keyword + lemmatize    │
                    │    + WordNet + spaCy)     │
                    └──────────────────────────┘
                                  │
          ┌───────┬───────┬───────┼───────┬───────┬───────┐
          │       │       │       │       │       │       │
          ▼       ▼       ▼       ▼       ▼       ▼       ▼
     greeting farewell review  domain  examiner  help  off_domain
              _analysis  _query
          │       │       │       │       │       │       │
          ▼       ▼       ▼       ▼       ▼       ▼       ▼
     diversified  ABSA    pre-    self-   help     domain
     templates   pipeline computed knowledge templates redirect
                 │        stats   responses
                 ▼
         ┌─────────────────────────┐
         │  extract_aspects_spacy() │
         │  ┌─────────────────────┐ │
         │  │ 1. spaCy POS nouns  │ │
         │  │ 2. spaCy noun chunks│ │
         │  │ 3. Lexicon lookup   │ │
         │  └─────────────────────┘ │
         └────────────┬────────────┘
                      ▼
         ┌─────────────────────────┐
         │  predict_category_fast() │
         │  (keyword rule engine)   │
         └────────────┬────────────┘
                      ▼
         ┌─────────────────────────┐
         │  TF-IDF + SMOTE + LR    │
         │  (sentiment classifier)  │
         └────────────┬────────────┘
                      ▼
         ┌─────────────────────────┐
         │  format_absa_response()  │
         │  (5 varied templates)    │
         └────────────┬────────────┘
                      ▼
              ┌───────────────┐
              │  LLM WRAPPER  │
              │ (OpenRouter)   │
              │ z-ai/glm-     │
              │  4.5-air       │
              └───────┬───────┘
                      ▼
                  RESPONSE
```

**Key phrases:**
- "Six intent paths. Each one a self-contained handler."
- "Review analysis: spaCy extracts aspects, keyword engine maps categories, logistic regression classifies sentiment."
- "Domain Q&A: queries don't classify text — they look up pre-computed stats from the training data."
- "Examiner: the chatbot knows its own architecture, accuracy, and limitations."

---

### Part 2: Approach & Architecture (3 minutes)

**[SLIDE 4: The ABSA Pipeline in Detail]**

**What to say:**
"Let me walk through the review analysis path — it's the core of the system. When a user types a review, four things happen:

1. **Aspect Extraction:** spaCy's POS tagger finds all nouns and proper nouns in the sentence — 'pasta', 'waiter', 'atmosphere'. It also checks noun chunks for multi-word phrases like 'wait staff'. A learned lexicon from the training data catches domain-specific terms spaCy might miss. Opinion words like 'great', 'delicious', 'terrible' are explicitly excluded — we want what's being reviewed, not the review itself.

2. **Category Mapping:** Each extracted aspect is classified into food, service, price, ambience, or miscellaneous. I use a keyword rule engine. 'Pasta' → food, 'waiter' → service, 'bill' → price. This replaced a BART zero-shot classifier — the keyword engine is 340 times faster and 100% transparent.

3. **Feature Engineering:** The aspect term is wrapped in [ASPECT] tags within the full review context. This tells the classifier which word to focus on. Without this, 'cold' is just a word — with it, the model learns that 'cold' near a food [ASPECT] is negative while 'cold' near a drink [ASPECT] might be neutral or positive.

4. **Sentiment Classification:** TF-IDF converts the tagged text into a numeric vector — each word gets a weighted score based on how important it is to this sentence. SMOTE balances the training data by generating synthetic examples for minority classes — conflict has only 45 examples naturally. Logistic regression then classifies each aspect as positive, negative, neutral, or conflict."

**[SLIDE 5: Domain Knowledge Base]**

**What to say:**
"Half of what the chatbot does has nothing to do with classifying sentiment in real time. When a user asks 'Is the food good?', I pre-computed the answer. The `compute_domain_knowledge()` function runs once at startup — it reads all 3,693 training annotations and counts: how many food mentions? How many positive? These become a set of summary statistics.

Food is 70% positive across 1,232 mentions. Service is 54% positive but 37% negative — people are genuinely split on restaurant service. Price and ambience fall in between.

The key insight: these are the same annotations that trained the sentiment classifier. I'm not using a separate dataset or an LLM's general knowledge. Every percentage is traceable to a specific count in the SemEval data. That makes it defensible."

**[SLIDE 6: Intent Detection Logic]**

**What to say:**
"The intent router is a priority-ordered decision tree:
- Examiner questions checked first — 'what are your limitations?' must not fall through to help.
- Help patterns next — expanded from just 'help' and 'capabilities' to 17 patterns including 'what else can you do?' and 'what other things'.
- Domain queries use a two-stage filter: does the question contain a query pattern AND a restaurant term? 'Is the food good?' passes both. 'What is the weather?' passes the query pattern but fails the restaurant term check.
- **Restaurant about-us detection** — before the review path, checks for 'your menu', 'what meals do you serve', 'best food', 'do you have'. Prevents 'based on your menu what meals do you serve?' from being misrouted to ABSA and getting nonsense output.
- Greetings and farewells use token-set intersection — faster and more accurate than substring matching (prevents 'yo' in 'you' from triggering greeting).
- Restaurant review detection uses a lemmatized vocabulary of 120+ terms — 'prices' lemmatizes to 'price', 'waiters' to 'waiter', 'desserts' to 'dessert'.
- A spaCy fallback catches reviews with unknown food terms by looking for noun + copula patterns: 'The ramen was delicious' still triggers ABSA even though 'ramen' is not in the vocabulary."

---

### Part 3: Live Demo (3 minutes)

**[SWITCH TO TERMINAL — Live Demo]**

**What to show (6 interactions):**

**1. Greeting:**
Type: `Hello!`
Expected: Warm greeting with capability overview.
*Point out: "Notice it lists what it can do — the system is transparent about its scope."*

**2. Review Analysis:**
Type: `The steak was perfectly cooked but the service was incredibly slow and the bill was way too high.`
Expected: Three aspects extracted — FOOD positive, SERVICE negative, PRICE negative.
*Point out: "Three aspects from one sentence. Each classified independently with surrounding context."*

**3. Domain Q&A:**
Type: `Is the food here good?`
Expected: "[LLM] Our guests really enjoy the food — about 70% of mentions are positive. Pizza and pasta come up the most in feedback."
*Point out: "That 70% comes from 867 positive food annotations divided by 1,232 total. It's computed, not guessed. And the tone is casual — 'our guests' not 'reviewers' — because this is a customer question."*

**4. Examiner Q&A:**
Type: `What model are you using?`
Expected: "[LLM] I use a Logistic Regression classifier with TF-IDF features, trained on 3,693 aspect annotations from the SemEval-2014 corpus."
*Point out: "The chatbot answers questions about itself. Notice the [LLM] prefix — you can see when the LLM is active vs when keyword fallback runs."*

**5. Off-Domain:**
Type: `What's the capital of France?`
Expected: Polite, varied redirect. "I'm best at answering restaurant questions. Curious what guests tend to say about our food?"
*Point out: "Graceful degradation — five varied redirect templates, doesn't repeat itself, doesn't crash."*

**6. Follow-Up (Memory):**
Type first: `What do people complain about?`
Type second: `And the food?`
Expected: Second response switches to food stats (17% negative), maintaining conversation context.
*Point out: "The 3-turn memory tracks the previous topic. 'And the food?' has no query keywords — memory fills the gap."*

---

### Mixed Persona: Analyst vs Staff Tone

The chatbot adapts its tone based on intent — an analytical voice for reviews and examiner questions, and a casual restaurant-staff voice for domain queries and customer questions. The LLM system prompt instructs the model to use "our guests" instead of "reviewers" and "people say" instead of "sentiment was observed." Greetings and farewells skip the LLM entirely (templates are natural enough, and skipping saves ~1.5s latency). All LLM-generated responses are prefixed with `[LLM]` so you can distinguish generated from fallback output at a glance.

**LLM Decision Logic:**
| Intent | LLM called? | Reason |
|--------|-------------|--------|
| greeting, farewell | No | Templates already natural — no API cost or latency |
| restaurant_review | Yes | ABSA output needs natural summarization |
| domain_query | Yes | Stats benefit from conversational phrasing |
| examiner | Yes | Technical facts benefit from natural explanation |
| help, off_domain | Yes | Redirects need varied, contextual language |

**CLI Flags for Demo & Testing:**
| Flag | Behavior |
|------|----------|
| `python3 run_chatbot.py` | Train → evaluate → 50-question demo → interactive chat |
| `--test-only` | Train → evaluate → 50-question demo → exit |
| `--chat-only` | Train → evaluate → skip test → interactive chat directly |
| `--no-llm` | Force keyword only (even with API key) — for debugging |

**Point to make in viva:** "The LLM is a formatting layer. Every fact — the 70% positive food stat, the 70.99% accuracy, the top mentioned terms — comes from deterministic computations. If the API goes down, hit `--no-llm` and the chatbot works immediately with zero factual degradation."

---

### Part 4: Evaluation & Limitations (1 minute)

**[SLIDE 7: Performance Metrics]**

**What to show:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Positive | 0.80 | 0.88 | 0.837 |
| Negative | 0.66 | 0.65 | 0.656 |
| Neutral | 0.48 | 0.43 | 0.451 |
| Conflict | 0.22 | 0.21 | 0.213 |
| **Weighted Avg** | | | **0.715** |
| **Accuracy** | | | **70.99%** |

**What to say:**
"The model is strong where it matters most — 83.7% F1 on positive sentiment, which covers 60%+ of all restaurant reviews. The 70.99% accuracy is dragged down by two hard cases: neutral reviews, where the language is inherently ambiguous — 'the food was okay' could mean anything — and conflict cases, where the same aspect has both positive and negative words nearby. On a 12-category, 72-question comprehensive test, the chatbot achieves 98.6% passable, up from 68.1% before the improvements."

**Limitations (be honest):**
1. Sarcasm: "Oh great, another cold meal" → classified as positive
2. Cannot compare two specific restaurants — no restaurant identities in data
3. 2014 vocabulary — newer food terms are unknown
4. Neutral and conflict F1 are low

---

### Part 5: Q&A Preparation (1 minute)

**[SLIDE 8: Key Points for Defense]**

**What to say:**
"I'm ready for questions about any part of the system. Three points I want to leave you with:

1. **Same data, two purposes.** The SemEval annotations trained the classifier AND populate the knowledge base. That's the architectural insight — classification and aggregation from one dataset.

2. **Transparency over complexity.** I removed a 1.6GB BART model because a 5KB keyword list did the same job 340x faster with 100% explainability. Every design choice was about making the system defensible in a viva.

3. **The LLM is a wrapper, not the brain.** OpenRouter formats responses naturally, but all facts — the 70% positive food stat, the 70.99% accuracy, the per-class F1 scores — come from deterministic computations on my training data. If the API goes down, the Phase 1 system continues working immediately with no loss in factual accuracy.

I'm happy to take questions on any component — the intent router, the ABSA pipeline, the knowledge base, the LLM integration, or the evaluation methodology."

---

## Quick Reference: Key System Components

| Component | File Location | Function |
|-----------|--------------|----------|
| XML Parser | `run_chatbot.py:85` | `parse_restaurant_xml()` |
| Text Cleaning | `run_chatbot.py:62-73` | `clean_text()`, `normalize_text()` |
| Aspect Extraction | `run_chatbot.py:143` | `extract_aspects_spacy()` |
| Category Mapper | `run_chatbot.py:172` | `predict_category_fast()` |
| Feature Engineering | `run_chatbot.py:195` | `make_feature()` |
| Domain Knowledge | `run_chatbot.py:206` | `compute_domain_knowledge()` |
| Intent Detection | `run_chatbot.py:551` | `detect_intent()` |
| Examiner Detection | `run_chatbot.py:434` | `detect_examiner_intent()` |
| LLM Wrapper | `run_chatbot.py:853` | `call_llm()` |
| Main Chat Loop | `run_chatbot.py:965` | `chat()` |
| Evaluation | `run_chatbot.py:1130` | `evaluate_test_set()` |
