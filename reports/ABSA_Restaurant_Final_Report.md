# Aspect-Based Sentiment Analysis for Restaurant Review Intelligence

**Student:** Nikan Khadka  
**Student ID:** S388699

## 1. Project outline

This project addresses a practical restaurant review intelligence problem using the SemEval-2014 Task 4 restaurant dataset. The goal is not just to classify sentences in a benchmark setting, but to test whether restaurant review text can be converted into useful structured insight at the aspect level. In practical terms, the project asks two linked questions. First, can the system identify which restaurant area is being discussed, such as food, service, price, or ambience? Second, can it estimate whether the expressed opinion about that aspect is positive, negative, neutral, or conflicted?

This framing was chosen deliberately. Earlier feedback on the proposal showed that the project risked becoming too dataset-centred and unclear about the actual problem being solved. To fix that, the work was repositioned as a prototype for restaurant review intelligence rather than a generic Kaggle-style classification exercise. The focus is therefore on how the dataset supports the use case, what the pipeline can really do, and where the limits remain.

The work was completed in three clearly separated stages. First, the coding and analysis pipeline was rebuilt and extended. Second, beginner-friendly documentation was written so the project can be rerun and understood. Third, this final report was written using the actual outputs from the rebuilt pipeline rather than unsupported assumptions.

## 2. Dataset suitability evaluation

The SemEval-2014 restaurant dataset is **partially suitable** for the intended use case.

It is suitable because it contains human-labelled restaurant review sentences with both explicit aspect terms and coarse restaurant categories. That makes it possible to test a realistic ABSA workflow: parse review text, detect what issue is being discussed, and estimate the sentiment attached to that issue. The full restaurant split used in this project contains 3,041 training sentences and 800 test sentences, with 3,693 train aspect-term annotations and 1,134 test aspect-term annotations. The parsed project outputs match those official counts.

It is also suitable because the restaurant domain has a fixed and intuitive category schema: food, service, price, ambience, and anecdotes/miscellaneous. For a coursework project, this is useful. It allows outputs to be interpreted in business language rather than only as abstract machine learning labels.

However, the dataset is not fully suitable for real restaurant intelligence in a stronger production sense.

First, the dataset is sentence-level rather than review-level. In practice, restaurant review intelligence often requires aggregation across full reviews, users, time periods, and trends. A sentence-level benchmark is valuable for modelling, but it is only one layer of the real problem.

Second, the categories are coarse. The `food` label can cover taste, freshness, variety, portion size, or drinks. The `service` label can combine politeness, speed, delivery, and reservation handling. This is acceptable for benchmarking, but it reduces the operational precision of the output.

Third, the annotation design does not directly solve one of the project’s hardest needs: it does not explicitly align each aspect term to a category when multiple categories appear in a sentence. This matters for aspect generation work, because it limits how directly term-to-category mapping can be evaluated.

Fourth, the dataset is old, English-only, and narrow in scope. It is still useful as a standard benchmark, but it should not be treated as a direct representation of present-day, multilingual, or platform-specific restaurant feedback.

Overall, the dataset is a strong benchmark for ABSA experimentation and a reasonable fit for an academic restaurant insight prototype. It is not enough, on its own, to justify strong claims about real-world deployment.

## 3. Data preparation

At the start of this chat, the actual project package was not available in the workspace. Only a pasted project-state note was present. To move the work forward honestly, the restaurant train and gold test XML files were rebuilt into the local workspace and parsed again from raw source.

A clean data-preparation stage was then implemented. The XML files were converted into three structured table types for both train and test splits:

- sentence-level tables
- aspect-term tables
- aspect-category tables

This separation matters because later project stages use them differently. EDA mainly depends on sentence and label distributions. Aspect generation uses the term and category tables together. Sentiment classification uses the aspect-term table with sentence context.

Final claims in this report are based on the full train and official gold test splits.

## 4. Exploratory data analysis

The EDA shows that the dataset is short, dense, and clearly imbalanced.

On the training split, the average sentence length is 12.76 tokens and the median is 12. The average number of aspect terms per sentence is 1.21, while the median is 1. This suggests the dataset is compact and information-dense. It is not dominated by long review narratives. Instead, many sentences are short opinion units.

At the same time, the data is not uniformly structured around labelled targets. Around 33.5% of training sentences contain no aspect-term annotation, while about 32.8% contain more than one aspect term. This matters because the task is neither “one sentence, one label” nor “every sentence contains a clear target”. Any modelling or reporting discussion has to reflect that complexity.

The polarity distribution is strongly skewed toward positive sentiment. On the train aspect-term labels, 2,164 are positive, 805 negative, 633 neutral, and only 91 conflict. This is important for two reasons. First, raw accuracy can be misleading because a model can benefit from the dominant positive class. Second, the conflict class is so small that stable evaluation is difficult.

The category distribution is also uneven. The train aspect-category labels are distributed as 1,232 food, 597 service, 321 price, 431 ambience, and 1,132 anecdotes/miscellaneous. Food is the largest concrete operational category, which is intuitive for restaurant reviews, but the large miscellaneous class is a warning sign. It improves coverage but reduces interpretability.

The most frequent normalized aspect terms include `food`, `service`, `prices`, `place`, `staff`, `menu`, `dinner`, `pizza`, `atmosphere`, and `price`. This supports the project framing well: the benchmark really does surface common restaurant topics, and not just synthetic or arbitrary tokens.

![Train aspect-term polarity distribution](outputs/eda/train_term_polarity_distribution.png)

![Train aspect-category distribution](outputs/eda/train_category_distribution.png)

The EDA therefore supports two later modelling choices. First, macro F1 must be reported alongside accuracy because of class imbalance. Second, any aspect-generation discussion has to acknowledge that a sizeable portion of sentences are either category-broad or multi-aspect.

## 5. Methods

### 5.1 Aspect generation

The current project treats aspect generation as a practical mapping problem: given an explicit aspect term, can it be assigned to one of the restaurant categories in a usable way?

A simple rule-based baseline was kept as a starting point. It uses seed keywords and clear fallbacks. For example, terms such as `staff`, `waiter`, and `delivery` map to service; `price`, `cost`, and `bill` map to price; and terms such as `atmosphere`, `decor`, or `music` map to ambience. Unmatched terms fall back to food.

This baseline is transparent and easy to explain, but limited. Many restaurant terms are ambiguous or indirect, and hand-written rules alone do not make good use of the training data.

To improve it, a hybrid mapper was added. The improved method learns term-category tendencies from the training split by using normalized aspect terms, headword-level fallbacks, and train-set co-occurrence statistics. It then backs off to seed keywords and simple regex-style rules only when direct evidence is weak or missing.

A major dataset constraint appears here: the SemEval restaurant data does not explicitly align each aspect term to a category in multi-category sentences. Because of that, evaluation was restricted to **single-category sentences only**. In those cases, each aspect term in the sentence can reasonably inherit that single category. This is not perfect, but it is a defensible silver-standard evaluation design based on the available labels.

### 5.2 Sentiment modelling

For sentiment, a stronger classical baseline was implemented. The model uses gold aspect terms and predicts aspect-term polarity from sentence context.

The input representation marks the aspect mention inside the sentence, then combines that marked sentence with the normalized aspect term. This gives the classifier access to both the target and the nearby sentiment cues. A TF-IDF vectorizer and logistic regression classifier were used because they are lightweight, reproducible, and appropriate for a full end-to-end coursework pipeline.

A majority-class baseline was also kept for comparison. This is necessary because the positive class dominates the dataset and can otherwise make weak models look stronger than they are.

## 6. Results and discussion

### 6.1 Aspect generation results

Aspect-generation evaluation used 2,477 train rows and 709 test rows from the single-category subset.

The rule-based baseline achieved:

- accuracy: 0.7941
- macro F1: 0.5588

The hybrid mapper achieved:

- accuracy: 0.7870
- macro F1: 0.6284

The main result is not a gain in raw accuracy. In fact, accuracy drops slightly. The more useful improvement is in macro F1, which rises by about 0.07. That indicates the hybrid method is more balanced across categories and less dependent on easy majority patterns. For a restaurant intelligence use case, that is meaningful. A system that slightly sacrifices headline accuracy but improves coverage across weaker categories is often more useful than a rule set that mainly succeeds on the obvious cases.

### 6.2 Sentiment results

On the official test set, the majority baseline achieved 0.6420 accuracy and 0.1955 macro F1. The TF-IDF plus logistic regression model achieved 0.6526 accuracy and 0.4686 macro F1.

This is a modest improvement in accuracy, but a large improvement in macro F1. That is the more important result. It shows the model is doing more than repeating the majority positive label.

The conflict class remains a clear weakness. The official test set contains only 14 conflict aspect-term labels. This is too small for robust learning and makes any conflict-specific conclusion fragile.

Overall, the modelling results are strong enough to support the report’s practical framing. The project can produce structured restaurant signals from review text. At the same time, the results are not strong enough to justify calling the pipeline production-ready.

## 7. Suitable use of outputs

The most realistic use case for the current outputs is a **decision-support prototype** for restaurant review analysis.

The system can support:

- broad summary dashboards
- manual analyst review
- exploratory comparison of category and polarity trends
- coursework demonstration of aspect-level review intelligence

For example, it can help produce summaries such as “food is mostly positive, service is mixed, and price complaints are relatively concentrated”. That is a realistic and defensible use of the pipeline.

The outputs should not be used on their own for high-stakes decisions. They should not automatically rank restaurants, assess individual staff performance, or support public claims about restaurant quality without human review. The dataset and model design are not strong enough for those tasks.

## 8. Ethical considerations

The ethical discussion in this project should stay tied to the dataset and the modelling choices.

First, representativeness is limited. The dataset is old, English-only, and narrow. Any deployment-style claim would risk overgeneralising beyond the data.

Second, annotation design shapes model behaviour. Because the benchmark prefers explicit aspect terms and sentence-level context, the system may miss implicit complaints or praise that matter in real user reviews.

Third, coarse categories can hide meaningful detail. A `food` label may include very different issues such as flavour, freshness, menu breadth, or portion size. That makes the output useful for a high-level dashboard, but weak for precise operational diagnosis.

Fourth, the large `anecdotes/miscellaneous` class can absorb general opinions in a way that reduces explainability. A model may be technically correct while still being less useful for action.

Fifth, imbalance affects evaluation fairness. Since positive sentiment dominates and conflict is rare, accuracy alone would overstate system quality. Reporting macro F1 is therefore not just a technical choice but an ethical one, because it gives a fairer picture of performance across classes.

For these reasons, the project should be framed as an assistive text-analytics prototype. Human interpretation is still necessary.

## 9. Limitations and future work

The current project has three main limitations.

The first is dataset structure. Term-to-category alignment is incomplete in multi-category sentences, which weakens direct aspect-generation evaluation.

The second is modelling scope. The current sentiment model assumes gold aspect terms are known. It is therefore not yet a full joint extraction system operating from raw sentences alone.

The third is real-world gap. The benchmark is useful for structured ABSA work, but it is not a current production dataset.

Future work could improve the project by:

- adding transformer-based ABSA models
- building joint aspect extraction and sentiment prediction
- improving handling of rare conflict examples
- aggregating sentence-level outputs into review-level dashboards
- testing on newer and more diverse restaurant review data

## 10. Conclusion

This project successfully rebuilt and extended a restaurant ABSA pipeline around the SemEval-2014 restaurant dataset. The work now includes full-dataset parsing, reproducible preprocessing, EDA, aspect-generation experiments, and a stronger aspect-level sentiment baseline.

The main conclusion is balanced. The dataset is suitable enough for a solid coursework prototype in restaurant review intelligence, but not strong enough to stand in for a production environment. The modelling outputs are meaningful, especially when interpreted as structured support for human analysis. They are not a substitute for richer data, broader evaluation, or human judgement.

That is exactly where the project is strongest academically: it makes useful progress, grounds its claims in evidence, and stays honest about what the benchmark can and cannot support.

## References

Pontiki, M., Galanis, D., Pavlopoulos, J., Papageorgiou, H., Androutsopoulos, I., & Manandhar, S. (2014). *SemEval-2014 Task 4: Aspect Based Sentiment Analysis*.

Ganu, G., Elhadad, N., & Marian, A. (2009). *Beyond the stars: Improving rating predictions using review text content*.
