# Understanding the Restaurant Review Chatbot
## A Beginner-Friendly Code Walkthrough

---

### What is ABSA? (Explained with an Analogy)

Imagine you run a restaurant and you read a review that says: *"The pasta was cold but the waiter was incredibly friendly."* A normal sentiment analysis tool would just say "mixed" or "neutral." That's not very helpful. You want to know specifically:

- **Aspect:** What are people talking about? → "pasta" (food), "waiter" (service)
- **Sentiment:** How do they feel about each aspect? → pasta is negative (cold), waiter is positive (friendly)

That's **Aspect-Based Sentiment Analysis (ABSA)** — instead of one score for the whole review, you get a score for each specific thing mentioned. It's like a restaurant report card that says "Food: C-, Service: A" instead of just "Overall: B."

---

### The Big Picture: How the Chatbot Works

```
User types a message
        │
        ▼
┌───────────────────────────────┐
│  detect_intent()              │
│  "What does the user want?"   │
│  6 possible paths              │
└───────────────┬───────────────┘
                │
    ┌───────────┼───────────┬──────────┬──────────┬──────────┐
    ▼           ▼           ▼          ▼          ▼          ▼
 greeting   farewell   review      domain    examiner    off_domain
                      _analysis    _query
    │           │           │          │          │          │
    ▼           ▼           ▼          ▼          ▼          ▼
 template    template   ABSA runs   stats      self-     redirect
                        on review   looked up  knowledge  message
                            │
                    ┌───────┴───────┐
                    │ 1. Extract    │
                    │    aspects    │
                    │ 2. Map to     │
                    │    category   │
                    │ 3. Classify   │
                    │    sentiment  │
                    │ 4. Format     │
                    └───────┬───────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │  LLM WRAPPER    │
                  │  (optional)     │
                  │  Makes response │
                  │  sound natural  │
                  └────────┬────────┘
                           │
                           ▼
                    Response shown to user
```

In plain English: The chatbot first figures out what you want (greeting? review? question?). Then it picks the right tool for the job — the ABSA pipeline for reviews, a stats lookup for questions like "Is the food good?", or pre-written answers for questions about itself. Finally, an optional LLM layer makes the response sound like a natural conversation.

---

### Component 1: Reading the Training Data

The SemEval-2014 data comes as XML files. Each review sentence looks like this:

```xml
<sentence id="1234">
    <text>The pasta was cold but the waiter was friendly.</text>
    <aspectTerms>
        <aspectTerm term="pasta" polarity="negative" from="4" to="9"/>
        <aspectTerm term="waiter" polarity="positive" from="26" to="32"/>
    </aspectTerms>
    <aspectCategories>
        <aspectCategory category="food" polarity="negative"/>
        <aspectCategory category="service" polarity="positive"/>
    </aspectCategories>
</sentence>
```

The `parse_restaurant_xml()` function reads this and creates three neat tables:

```python
# Three DataFrames are created:
# 1. sentences — one row per sentence (text, ID, word count)
# 2. aspects   — one row per aspect term (the word "pasta", its polarity "negative")
# 3. categories — one row per category (food, service, etc.)
```

**Analogy:** Think of the XML as a messy filing cabinet. The parser opens each drawer (sentence), pulls out three index cards — one for the sentence itself, one for each underlined word (aspects), and one for each topic label (categories) — and stacks them neatly into three piles.

The result: 3,041 sentences become 3,693 aspect annotations (each one is a training example for the classifier).

---

### Component 2: Text Cleaning

Before any analysis, all text goes through a cleaning pipeline:

```python
def clean_text(text):
    text = contractions.fix(str(text)).lower()    # "I'm" → "i am"
    text = re.sub(r'http\S+|www\S+|<.*?>', '', text)  # Remove URLs, HTML tags
    return re.sub(r'\s+', ' ', text).strip()      # Normalize whitespace
```

**Why we do each step:**
- **Expand contractions:** "don't" → "do not" so the model sees "not" as a separate word.
- **Lowercase:** "Pizza" and "pizza" should be treated as the same word.
- **Remove URLs/HTML:** These are noise — they don't carry sentiment information.
- **Normalize whitespace:** Extra spaces cause problems for word matching.

**Analogy:** Text cleaning is like wiping down a whiteboard before writing on it. You're removing smudges and stray marks so you can see what's actually written.

---

### Component 3: Aspect Extraction

This is the "what are they talking about?" step. The function `extract_aspects_spacy()` uses three methods cascaded together:

```python
def extract_aspects_spacy(text, single, multi, heads):
    doc = nlp(clean_text(text))    # spaCy reads the sentence
    found = set()

    # METHOD 1: Find all nouns using spaCy's grammar engine
    for token in doc:
        if (token.pos_ in ('NOUN', 'PROPN')    # Is it a noun?
                and word not in STOPWORDS       # Not a filler word?
                and word not in NON_ASPECTS):   # Not an opinion word?
            found.add(word)                     # Example: "pizza", "waiter"

    # METHOD 2: Find multi-word noun phrases
    for chunk in doc.noun_chunks:               # spaCy groups related nouns
        found.add(chunk.text)                   # Example: "wait staff", "dining room"

    # METHOD 3: Check against learned vocabulary from training data
    for t in tokens:
        if t in single: found.add(t)            # Known restaurant words
```

**Example in action:**
- Input: *"The pizza was cold but the service was excellent."*
- Method 1 finds: `pizza` (noun), `service` (noun)
- Method 2 finds: nothing useful (no multi-word phrases)
- Method 3: confirms `pizza` and `service` are known restaurant terms

Output: `["pizza", "service"]` — two aspects to analyze.

**Why exclude opinion words?** The `NON_ASPECTS` set contains words like "great," "terrible," "delicious." These describe how someone FEELS about something — they're not the thing itself. If a review says "The food was excellent," we want "food" extracted, not "excellent."

**Analogy:** spaCy is like a grammar-aware highlighter. It knows which words are nouns (things you can review), which are adjectives (opinions about those things), and which are filler words (the, a, is). The lexicon is a backup cheat sheet — it knows that "tiramisu" is a food even if spaCy has never seen that word before.

---

### Component 4: Feature Engineering

Once we know WHAT to classify ("pasta"), we need to tell the model WHERE in the sentence to look. The `make_feature()` function wraps the aspect word in special tags:

```python
def make_feature(text, term):
    """Wraps the aspect word in [ASPECT]...[/ASPECT] tags."""
    # Find where the aspect term appears in the cleaned text
    idx = txt.find(term)
    # Insert tags around it
    return txt[:idx] + f' [ASPECT] {txt[idx:idx+len(term)]} [/ASPECT] ' + txt[idx+len(term):]
```

**Example in action:**
- Input text: *"the pasta was cold"*, term: *"pasta"*
- Output: `the [ASPECT] pasta [/ASPECT] was cold`

But here's the key insight — **the same word with different context gets different features:**

- `the [ASPECT] pasta [/ASPECT] was cold` → sent to classifier → learns "cold near pasta = negative"
- `the [ASPECT] drink [/ASPECT] was cold` → sent to classifier → might learn "cold near drink = positive" (a cold drink on a hot day is good!)

Without the [ASPECT] tags, the model just sees "cold" in both sentences and can't distinguish them. With the tags, it learns that "cold" near a food aspect is different from "cold" near a drink aspect.

**Analogy:** It's like circling the subject of a sentence with a red pen before handing it to someone. Without the circle, they might read "cold" and not know if it's the weather, the pasta, or the drink. With the circle, they know exactly what to focus on.

---

### Component 5: Training the Sentiment Model

The training pipeline has three steps:

```python
# STEP 1: TF-IDF — Turn words into numbers
tfidf = TfidfVectorizer(ngram_range=(1,2),    # Look at 1-word and 2-word sequences
                        max_features=5000,     # Keep top 5000 most important
                        sublinear_tf=True)     # Logarithmic scaling
X_vec = tfidf.fit_transform(train_features)

# STEP 2: SMOTE — Balance the classes
smote = SMOTE(random_state=42)                 # Creates synthetic examples
X_balanced, y_balanced = smote.fit_resample(X_vec, y_all)

# STEP 3: Logistic Regression — Learn to classify
clf = LogisticRegression(max_iter=1000)        # Linear classifier
clf.fit(X_balanced, y_balanced)
```

**What each step does:**

| Step | What it is | Why we need it |
|------|-----------|----------------|
| **TF-IDF** | Converts text to numbers. Each word gets a score: how often it appears in THIS sentence vs how often it appears in ALL sentences. | Computers don't understand text — they need numbers. TF-IDF gives common words (the, a, is) low scores and distinctive words (overcooked, delicious) high scores. |
| **SMOTE** | Creates synthetic (fake but realistic) training examples for rare classes. | In 3,693 training examples, only ~45 are "conflict." Without SMOTE, the model learns to ALWAYS predict "positive" and still gets 95% accuracy — but it's useless for anything else. |
| **Logistic Regression** | A simple "weighted scoring" system. Each word gets a weight for each sentiment class. The class with the highest total score wins. | It's fast, interpretable (you can see which words matter), and works well enough on this dataset. |

**Result:** 70.99% accuracy on the held-out test set. 83.7% F1 on positive (the dominant class), 45.1% F1 on neutral (the hardest class).

**Analogy — TF-IDF:** Think of it like counting word importance. In a cookbook, "the" appears everywhere (low importance). But "saffron" appears only in a few recipes (high importance — it tells you something specific about the dish). TF-IDF does this automatically: common words get low weights, rare-but-relevant words get high weights.

**Analogy — Logistic Regression:** Imagine each sentiment class has a checklist of words with point values. If a review contains "delicious" (+10 points for positive), "cold" (+3 for negative), and "pasta" (+1 for food category), the model adds up the points and says "most points go to... negative at 7, positive at 6 → negative wins."

---

### Component 6: Intent Detection

Before doing anything, the chatbot needs to know what the user wants. The `detect_intent()` function is a priority-ordered decision tree:

```python
def detect_intent(text):
    # Priority 1: Examiner questions (check FIRST)
    if "what model" in text or "how accurate" in text:
        return 'examiner'

    # Priority 2: Help requests
    if "help" in text or "what can you do" in text:
        return 'help'

    # Priority 3: Domain questions ("Is the food good?")
    if classify_domain_query(text):
        return 'domain_query'

    # Priority 4-5: Basic conversation
    if text has greeting words:
        return 'greeting'
    if text has farewell words:
        return 'farewell'

    # Priority 6: Restaurant review (lemmatized word matching)
    if tokens_lem has restaurant words:
        return 'restaurant_review'

    # Fallback: Not a restaurant conversation
    return 'off_domain'
```

**Why lemmatization matters:** Simple word matching would fail on plurals. "Waiters" doesn't match "waiter." Lemmatization reduces words to their base form: "waiters" → "waiter", "prices" → "price", "desserts" → "dessert". This single technique fixed several test failures.

**Why priority order matters:** "What are your limitations?" contains no restaurant words — if we checked for restaurant review before examiner intent, it would be rejected as off-domain. The priority ensures specific intents are caught before general ones.

**Analogy:** Intent detection is like a triage nurse in an emergency room. The most critical cases (examiner questions about the system itself) get checked first. Then domain queries. Then casual conversation. Only after ruling out everything specific does it conclude "this isn't a restaurant conversation."

---

### Component 7: Domain Knowledge Base

When a user asks "Is the food good?", the chatbot doesn't run sentiment analysis on the question — it looks up pre-computed statistics. The `compute_domain_knowledge()` function runs once at startup:

```python
def compute_domain_knowledge(train_xml_path):
    cat_polarity = defaultdict(list)    # category → list of polarities
    term_counts = Counter()             # term → how many times mentioned

    for every sentence in training data:
        for every category annotation:
            # Record: food → positive, food → negative, food → positive...
            cat_polarity[category].append(polarity)
        for every aspect term:
            # Count: pizza → 51 times, pasta → 43 times...
            term_counts[term] += 1

    # Calculate percentages
    knowledge['food'] = {
        'total': 1232,
        'positive': 867,   'positive_pct': 70,
        'negative': 209,   'negative_pct': 17,
        'neutral': 85,     'neutral_pct': 7,
        'conflict': 71,    'conflict_pct': 5
    }
    # ... same for service, price, ambience
```

**Mini example of the knowledge dict:**
```python
DOMAIN_KNOWLEDGE = {
    'food':     {'total': 1232, 'positive_pct': 70, 'negative_pct': 17},
    'service':  {'total': 597,  'positive_pct': 54, 'negative_pct': 37},
    'price':    {'total': 321,  'positive_pct': 56, 'negative_pct': 36},
    'ambience': {'total': 431,  'positive_pct': 61, 'negative_pct': 23},
    '_overall_total': 3713,
    '_top_terms': [('food', 376), ('service', 238), ('prices', 65), ...]
}
```

When the user asks about food, `answer_domain_query('food')` picks one of 5 response templates and fills in the stats:
> "Based on 3,713 reviews in my training data, food is rated positively 70% of the time and negatively 17% of the time."

**Why pre-computed and not live analysis?** Three reasons: (1) Speed — stats are computed once, not recomputed per query. (2) Accuracy — every percentage is exact (867/1232 = exactly 70%), not guessed by an LLM. (3) Traceability — in a viva, you can point to the exact annotation counts.

**Analogy:** The domain knowledge base is like a restaurant's sales report. Instead of re-counting every receipt when the owner asks "How's the pasta selling?", you pre-compute the report each morning and hand over the numbers.

---

### Component 8: Self-Knowledge Responses

The chatbot can answer questions about itself — what model it uses, how accurate it is, what its training data looks like. This is handled by `EXAMINER_KEYWORDS` (a dictionary mapping topics to trigger words) and `EXAMINER_RESPONSES` (a dictionary of pre-written answers):

```python
EXAMINER_KEYWORDS = {
    'model':      ['model', 'algorithm', 'classifier', 'logistic regression'],
    'accuracy':   ['accurate', 'accuracy', 'performance', 'f1', 'how accurate'],
    'training':   ['training', 'trained', 'dataset', 'semeval', 'what data'],
    'limitations': ['limit', 'limitation', 'weakness', 'struggle', 'fail'],
    'sarcasm':    ['sarcasm', 'irony', 'sarcastic', 'handle sarcasm'],
}

EXAMINER_RESPONSES = {
    'model': "I use a Logistic Regression classifier with TF-IDF features, "
             "trained on 3,693 aspect annotations from the SemEval-2014 corpus.",
    'accuracy': "I achieve 70.99% accuracy on the test set. Weighted F1 is 0.715. "
                "Strongest on positive (84% F1), weakest on conflict (21% F1).",
    # ... and so on for each topic
}
```

When the user types "What model are you using?", the system finds "model" in `EXAMINER_KEYWORDS['model']`, maps it to the topic key `'model'`, and returns the pre-written response from `EXAMINER_RESPONSES['model']`.

**Analogy:** This is like a FAQ page for the chatbot itself. The questions might be phrased differently ("what model?" vs "which algorithm?") but the answer is the same — so we pre-write polished answers and match incoming questions to the right one.

---

### Component 9: LLM Wrapper (Phase 2)

The optional LLM layer wraps all responses in natural conversation through OpenRouter's API:

```python
def call_llm(user_message, context, phase1_result, intent):
    messages = [
        {'role': 'system', 'content': LLM_SYSTEM_PROMPT},  # Instructions + facts
        {'role': 'user', 'content': f'User said: "{user_message}"\n'
                                     f'Phase 1 result: {phase1_result}'}
    ]

    response = requests.post(API_URL, json={
        'model': 'z-ai/glm-4.5-air',
        'messages': messages,
        'max_tokens': 300,
        'temperature': 0.7,
    })

    return apply_guard_rails(response)
```

**What the system prompt tells the LLM:**
1. **Role:** You are RestaurantXpert, a restaurant review chatbot.
2. **Facts (do not invent):** Food is 70% positive, service 54%, etc. — exact stats from the knowledge base.
3. **Model facts:** TF-IDF + SMOTE + LR, 70.99% accuracy, trained on SemEval-2014.
4. **Guard rails:** Never give medical/legal/financial advice. Never make up restaurant names. Redirect off-domain users.

**Guard rails after the LLM responds:**
```python
def apply_guard_rails(response):
    # 1. Block forbidden topics
    if any(forbidden_word in response for forbidden_word in ['medical advice', 'legal advice']):
        return redirect_to_restaurant_domain()

    # 2. Track off-domain queries — after 2 attempts, stop engaging
    if intent == 'off_domain':
        off_domain_count += 1
        if off_domain_count >= 2:
            return "I've mentioned this before — I'm a restaurant review analyst..."

    return response  # Passed all checks
```

**What happens if the API is down?** The `_llm_fallback()` function returns the Phase 1 response directly. The chatbot keeps working — the responses are just slightly less conversational (templates instead of natural language).

**Analogy:** The LLM is like a professional spokesperson who reads from carefully prepared briefing notes. The notes contain all the facts (the knowledge base stats). The spokesperson's job is to express them naturally and stay on message. If the spokesperson is unavailable, you read the notes directly — less polished, but factually identical.

---

### Component 10: Putting It All Together

Here's the complete `chat()` function — the entry point for every user message:

```python
def chat(user_input):
    # 1. Reject empty input
    if not user_input.strip():
        return 'Please type something!'

    # 2. Figure out the user's intent (which of the 6 paths?)
    intent = detect_intent(user_input)
    context = MEMORY.get_context()  # What did we talk about before?

    # 3. Route to the right handler based on intent:
    if intent == 'restaurant_review':
        absa_results = analyse(user_input)          # Extract + classify
        phase1_response = format_absa_response(absa_results)  # Format output
        MEMORY.add_exchange(user_input, phase1_response, intent)  # Remember
        return call_llm(user_input, context, phase1_response, intent)  # Wrap in LLM

    elif intent == 'domain_query':
        category = classify_domain_query(user_input)  # Which category?
        if category:
            phase1_response = answer_domain_query(category)  # Look up stats
        elif MEMORY.get_previous_topic():  # Follow-up question?
            phase1_response = answer_domain_query(MEMORY.get_previous_topic())
        else:
            phase1_response = answer_overall_query()  # General stats
        MEMORY.add_exchange(user_input, phase1_response, intent, topic=category)
        return call_llm(user_input, context, phase1_response, intent)

    elif intent == 'examiner':
        topic = detect_examiner_intent(user_input)   # Which examiner topic?
        phase1_response = answer_examiner(topic)      # Get pre-written answer
        MEMORY.add_exchange(user_input, phase1_response, intent, topic=topic)
        return call_llm(user_input, context, phase1_response, intent)

    elif intent in ('greeting', 'farewell', 'help', 'off_domain'):
        phase1_response = general_responses(intent)   # Pick a template
        MEMORY.add_exchange(user_input, phase1_response, intent)
        return call_llm(user_input, context, phase1_response, intent)

    # 4. Edge case — shouldn't reach here, but defensive return
    return general_responses('general')
```

**The full journey of a review, step by step:**

1. User types: *"The pizza was cold but the waiter was friendly"*
2. `chat()` calls `detect_intent()` → returns `'restaurant_review'` (matched "pizza" and "waiter" as restaurant terms)
3. `analyse()` runs the ABSA pipeline:
   - `extract_aspects_spacy()` → ["pizza", "waiter"]
   - For "pizza": `predict_category_fast("pizza")` → "food", `clf.predict(feature)` → "negative"
   - For "waiter": `predict_category_fast("waiter")` → "service", `clf.predict(feature)` → "positive"
4. `format_absa_response()` picks a template and formats the results:
   > "FOOD: negative (about: pizza)" / "SERVICE: positive (about: waiter)"
5. `call_llm()` sends this to OpenRouter with a system prompt
6. LLM rephrases naturally: *"The pizza got a cold reception but the waiter saved the experience!"*
7. Response shown to user

---

### Glossary

| Term | Simple Definition |
|------|-------------------|
| **TF-IDF** | "Term Frequency - Inverse Document Frequency." A scoring system that gives higher weight to words that appear often in THIS sentence but rarely in OTHER sentences. "Pizza" in a pizza review gets a high score; "the" gets a low score. |
| **SMOTE** | "Synthetic Minority Oversampling Technique." Creates artificial training examples for rare classes by blending existing examples. If you have only 5 examples of "conflict" reviews, SMOTE creates realistic new ones so the model learns to recognize them. |
| **Logistic Regression** | A classification algorithm that assigns a weight to each word for each possible outcome. It sums up the weights for a given sentence and predicts the outcome with the highest total. Think of it as a weighted voting system. |
| **spaCy** | A Python library that reads text like a grammarian — it knows which words are nouns, verbs, adjectives, and how they relate to each other. |
| **Lemmatization** | Reducing a word to its dictionary form. "Running" → "run", "waiters" → "waiter", "better" → "good". This normalizes different forms of the same word so they match. |
| **Zero-shot classification** | Asking a model (like BART) to classify something it was never specifically trained on. The model uses its general language understanding to pick the best category from a list you provide. |
| **SemEval-2014** | A standardized NLP competition dataset. Task 4 was restaurant review ABSA — given a review, find the aspects and classify their sentiment. |
| **Aspect** | The specific thing being reviewed — "food," "service," "pizza," "waiter," "atmosphere." |
| **Polarity** | The sentiment direction: positive (😊), negative (😞), neutral (😐), or conflict (🤔 — positive AND negative in the same review about the same thing). |
| **Intent** | What the user wants: analyze a review, ask a question, say hello, learn about the system. |
| **OpenRouter** | A service that provides unified API access to many LLM models. The chatbot uses model `z-ai/glm-4.5-air` through OpenRouter. |
| **Guard rails** | Safety checks applied to LLM output — blocking forbidden topics, enforcing domain scope, and counting off-topic attempts. |
| **F1 Score** | A balanced measure of a classifier's performance. It's the harmonic mean of precision (how many positive predictions were correct) and recall (how many actual positives were found). Ranges from 0 to 1, higher is better. |

---

### Why These Design Choices?

| Decision | Reason | What We Avoided |
|----------|--------|-----------------|
| **Keywords not BART** for category mapping | 340x faster (0.002s vs 0.68s), 100% transparent, same accuracy on this domain | 1.6GB model download, 340x latency, black-box decisions |
| **Pre-computed stats not LLM-only** for domain Q&A | Every answer traceable to a specific annotation count; no hallucination risk | Unverifiable LLM-generated statistics that can't be defended in a viva |
| **spaCy not NLTK** for aspect extraction | spaCy has built-in noun chunk detection and dependency parsing; NLTK requires manual rule-writing | Hundreds of lines of hand-written grammar rules |
| **Logistic Regression not BERT** for sentiment | Interpretable weights, no GPU needed, trains in seconds, competitive accuracy on this benchmark | 400MB model, GPU requirement, 50-200ms inference latency |
| **6-path intent system not 3-path** | Covers all the interaction types an examiner expects: reviews, domain Q&A, self-knowledge, off-domain redirection | A system that can only classify reviews and say "I am not sure" to everything else |
| **LLM as wrapper not as brain** | The ABSA model and knowledge base are the source of truth; the LLM only handles phrasing | An LLM hallucinating model accuracy numbers or inventing restaurant statistics |
| **3-turn memory not full dialogue manager** | Demonstrates stateful conversation within assignment scope; simple deque implementation | Unnecessary complexity for a proof-of-concept chatbot |

---

### Quick Reference: Key Functions

| Function | File | Line | What It Does |
|----------|------|------|-------------|
| `parse_restaurant_xml()` | `run_chatbot.py` | 85 | Converts XML training data into DataFrames |
| `clean_text()` | `run_chatbot.py` | 70 | Lowercases, expands contractions, removes URLs |
| `extract_aspects_spacy()` | `run_chatbot.py` | 143 | Finds aspect terms using spaCy + learned lexicon |
| `predict_category_fast()` | `run_chatbot.py` | 172 | Maps aspect terms to categories (food/service/price/ambience) |
| `make_feature()` | `run_chatbot.py` | 195 | Wraps aspect term in [ASPECT] tags for classifier context |
| `compute_domain_knowledge()` | `run_chatbot.py` | 206 | Aggregates 3,693 annotations into pre-computed stats |
| `detect_intent()` | `run_chatbot.py` | 551 | Routes user message to one of 6 intent paths |
| `detect_examiner_intent()` | `run_chatbot.py` | 434 | Identifies self-knowledge questions |
| `classify_domain_query()` | `run_chatbot.py` | 279 | Determines which category a domain question is about |
| `call_llm()` | `run_chatbot.py` | 853 | Sends structured result to OpenRouter for natural phrasing |
| `chat()` | `run_chatbot.py` | 965 | Main entry point — intent → handler → response |
| `evaluate_test_set()` | `run_chatbot.py` | 1130 | Runs the classifier on held-out test data for metrics |
