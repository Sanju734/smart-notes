# Quick glossary and label guide

## Vocabulary

- **Note:** A short personal reminder or message stored by the app.
- **Category:** One of the four groups into which a note is classified.
- **Training set:** Examples with labels used to fit your model (`data/train.csv`).
- **Validation set:** Separate labeled examples used to measure quality, not to fit the model (`data/validation.csv`).
- **Model:** A learned function that predicts categories from text, such as TF-IDF + logistic regression.
- **Baseline:** A very simple method you compare against the trained model.
- **Accuracy:** Fraction of notes assigned the correct category.
- **Macro-F1:** Average F1 across categories, giving each category equal weight; `sklearn.metrics.f1_score(..., average="macro")`.
- **API:** A program interface. Here you expose one HTTP URL, `/classify`, so the evaluator can send a note and read its predicted category.
- **Retrieval / search:** Finding previously saved notes that match a word or question; generation of a new answer is optional.
- **Hit rate @3:** Fraction of test queries where at least one expected note appears in the top 3 search results.
- **Out-of-scope note:** A note that fits none of the four categories, such as "Call grandma this weekend."
- **Uncertain:** Your model's signal that it should not be trusted for this note, for example because its highest predicted probability is low.

## Category definitions

| Label | Includes | Example |
| --- | --- | --- |
| `Study` | Exams, courses, tutorials, assignments, academic reading | Finish my chemistry assignment. |
| `Money` | Bills, budgeting, banking, payments, savings, investments, insurance payments | Transfer money to savings. |
| `Travel` | Trips, travel reservations, trains, buses, flights, hotels, itineraries | Confirm the Jaipur hotel booking. |
| `Shopping` | Buying, ordering, comparing, returning consumer goods | Return the defective headphones. |

Each example has **one primary category**. Sometimes real notes overlap: paying for a flight could involve both Money and Travel. For this task, use the **main purpose** of the note; the dataset labels are the reference for evaluation. Ambiguity is normal: describe it in your error analysis rather than claiming perfect classification.

## Notes that fit no category

Some notes belong to none of the four labels: health, family, chores, hobbies and so on. Examples are in `data/out_of_scope_examples.csv`. Your API still returns its best-guess `category`, but sets `uncertain: true`, and the app marks the note as uncertain instead of presenting the guess as fact. A note can also be uncertain when it is in scope but ambiguous.

## Example user flow

1. Enter `Practice machine learning for tomorrow's quiz`.
2. App predicts `Study` and saves a note with an ID.
3. Enter `My train to Pune leaves at 6 AM` and save it.
4. Search `Pune` and show the matching note, including ID and text.
5. Filter `Study` and show only the study notes.
6. Enter `Call grandma this weekend` and see it saved with an "Uncertain" marker.
