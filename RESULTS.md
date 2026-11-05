# RESULTS

## Dataset and training
- Training rows: 156
- Validation rows: 50
- Command that reproduces these numbers: `python train.py`
- Baseline method: Most-frequent label (`Study`).
- Trained model / features: TF-IDF + LogisticRegression; final model uses C=30 with word 1–2 grams plus `char_wb` 2–5 grams, selected by 5-fold stratified CV on `train.csv` only. CV macro-F1 was 0.972 (std 0.025).
- Library versions and random seed: Python 3.12.10, scikit-learn 1.6.1, sentence-transformers 3.4.1, random seed 42.

## Metrics (validation dataset only)

| Approach | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Most-frequent-label baseline | 0.300 | 0.115 |
| V1: default TF-IDF + LogisticRegression | 0.680 | 0.708 |
| Final model: C=30, word 1–2 grams + `char_wb` 2–5 grams | 0.900 | 0.903 |

The final model improved over V1 on both reported validation metrics.

Confusion matrices (rows are actual labels; columns are predicted labels; order: Money, Shopping, Study, Travel):

**Most-frequent-label baseline**

```text
[[ 0  0 11  0]
 [ 0  0 11  0]
 [ 0  0 15  0]
 [ 0  0 13  0]]
```

**V1**

```text
[[ 8  0  0  3]
 [ 1  9  0  1]
 [ 0  0  7  8]
 [ 0  0  3 10]]
```

**Final model**

```text
[[10  0  0  1]
 [ 0 10  0  1]
 [ 0  0 15  0]
 [ 0  0  3 10]]
```

## Error analysis (at least 3 distinct mistakes)

Notes that differ only in a day, city, or subject count as one mistake.

| Note | Expected category | Predicted category | Possible reason |
| --- | --- | --- | --- |
| I must reach the railway station by 10 AM. | Travel | Study | Time-related features dominated; the word feature `am` contributed toward Study. |
| Put snacks and cleaning items in the cart. | Shopping | Travel | The feature `in` contributed strongly toward Travel; the note had low similarity to the nearest Shopping training examples. |
| Check the credit card balance on Sunday. | Money | Travel | The features `check`, `check the`, and `sunday` contributed toward Travel, while a Travel example about a cab on Sunday was the nearest match. |

**Evidence for at least one mistake**

- For the railway-station note, top features toward Study were `word__am` (0.242), `char__st` (0.209), `word__by` (0.208), `word__10` (0.208), and `word__10 am` (0.208). Its closest training notes included “The boarding time for the plane is 10 AM.” (Travel, cosine similarity 0.320) and “Attend the chemistry tutorial at 10 AM.” (Study, cosine similarity 0.309).
- For the cart note, top features toward Travel included `word__in` (0.760), `char__in ` (0.087), `char__an` (0.083), `char__ca` (0.079), and `char__ ca` (0.079). The closest training examples were Shopping notes about buying groceries, with cosine similarities 0.195, 0.185, and 0.184.
- For the credit-card note, top features toward Travel included `word__check the` (0.343), `word__check` (0.279), and `word__sunday` (0.112). Its closest examples included “Arrange a cab to the airport on Sunday.” (Travel, cosine similarity 0.300) and “Review my bank statement on Sunday.” (Money, cosine similarity 0.288).

## Search evaluation

- Query file and command: `data/search_queries.json`; run `python eval_search.py`.
- Number of queries / number marked no-overlap: 16 / 6.

| Search approach | Hit rate @3 |
| --- | ---: |
| Keyword overlap baseline | 75.0% overall (12/16); 33.3% on no-overlap queries (2/6) |
| TF-IDF cosine | 75.0% overall (12/16); 33.3% on no-overlap queries (2/6) |
| Embeddings cosine | 93.8% overall (15/16); 83.3% on no-overlap queries (5/6) |

- Approach used in the app, and why: Embeddings cosine, using the local `all-MiniLM-L6-v2` model. It has the highest measured hit rate in the expanded query set.
- A query that still fails, and why: “where am I staying?” expects N007, but embeddings rank N006, N010, and N011 in the top three. In this example, the query phrasing does not retrieve the note that mentions the hotel.

## Notes that fit no category

- How `uncertain` is decided (method, threshold, and how you chose it): A note is uncertain when its top probability is below 0.45 **or** its vocabulary coverage is below 0.20. Coverage is the fraction of lowercased word tokens, after stopword removal, found in the fitted word TF-IDF vocabulary. The thresholds were compared using the trade-off grid printed by `train.py`.
- Did you train on `out_of_scope_examples.csv`? No. Neither that file nor `validation.csv` was used to fit or select the classifier.

| Setting | Out-of-scope notes flagged | Validation notes flagged |
| --- | ---: | ---: |
| Chosen: probability 0.45, coverage 0.20 | 58.3% (7/12) | 8.0% (4/50) |
| Alternative: probability 0.40, coverage 0.20 | 50.0% (6/12) | 2.0% (1/50) |
| Alternative: probability 0.50, coverage 0.20 | 75.0% (9/12) | 16.0% (8/50) |

- Why you chose this trade-off: The chosen setting flags more out-of-scope notes than the lower probability alternative while flagging fewer validation notes than the higher probability alternative. “Call grandma this weekend.” has top probability 0.497 and vocabulary coverage 0.000, so coverage catches it even though probability alone is above the chosen probability threshold.

## What I tried to fix, and whether it worked

- Change made, how it was measured (without fitting on validation labels), result: Used 5-fold stratified CV on `train.csv` only to select C and compare word 1–2 grams against word 1–2 grams plus `char_wb` 2–5 grams. The selected final configuration was C=30 with word and character features; its CV macro-F1 was 0.972. On the unchanged validation set, V1 scored 0.680 accuracy and 0.708 macro-F1, while the final model scored 0.900 accuracy and 0.903 macro-F1. A separate vocabulary-coverage check was added after “Call grandma this weekend.” was not flagged: the note has coverage 0.000 and top probability 0.497, and is now flagged uncertain with the saved thresholds.

## What I would improve next

- Add more varied, less templated synthetic examples and evaluate on notes written in styles not represented in training.
- Collect a larger, separately held-out dataset for choosing and checking uncertainty thresholds.
- Add search queries that reflect a wider range of phrasings and expected results.

## Limitations

- This small dataset is synthetic and cannot demonstrate real-world reliability on all kinds of notes.
- The validation set has 50 notes, which is small. Several examples use repeated templates, so related notes are not fully independent evidence of generalization.
- The search benchmark has 16 queries over the starter notes. Its hit rates describe only that small query set.
- The classifier can be confidently wrong, and probability and vocabulary coverage are imperfect uncertainty signals.
