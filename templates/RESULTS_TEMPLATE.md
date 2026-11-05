# RESULTS — replace placeholders with your own values

## Dataset and training
- Training rows: 156
- Validation rows: 50
- Command that reproduces these numbers:
- Baseline method:
- Trained model / features:
- Library versions and random seed (if relevant):

## Metrics (validation dataset only)

| Approach | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Baseline | ... | ... |
| Trained model | ... | ... |

Per-category results or a confusion matrix (optional but helpful):

## Error analysis (at least 3 distinct mistakes)

Notes that differ only in a day, city or subject count as one mistake.

| Note | Expected category | Predicted category | Possible reason |
| --- | --- | --- | --- |
| ... | ... | ... | ... |

**Evidence for at least one mistake** (for example the words or features that pushed the prediction, the most similar training notes, or a small experiment):

## Search evaluation

- Query file and command:
- Number of queries / number sharing no words with their target note:

| Search approach | Hit rate @3 |
| --- | ---: |
| Keyword baseline | ... |
| ... | ... |

- Approach used in the app, and why:
- A query that still fails, and why:

## Notes that fit no category

- How `uncertain` is decided (method, threshold, and how you chose it):
- Did you train on `out_of_scope_examples.csv`? If so, what did you hold out for evaluation?

| Setting | Out-of-scope notes flagged | Validation notes flagged |
| --- | ---: | ---: |
| Chosen: ... | ... | ... |
| Alternative: ... | ... | ... |

- Why you chose this trade-off:

## What I tried to fix, and whether it worked
- Change made, how it was measured (without fitting on validation labels), result:

## What I would improve next
- ...

## Limitations
- This small dataset is synthetic and cannot demonstrate real-world reliability on all kinds of notes.
- How much do you trust your validation score, given the size and structure of the data? Why?
- ...
