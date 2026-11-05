# NoteSense

NoteSense is a small local notes app that classifies saved notes into Study, Money, Travel, or Shopping, marks uncertain predictions, and lets you search or ask questions grounded in your saved notes. It uses a locally trained scikit-learn classifier and a local sentence embedding model; notes are stored in `notes.json`.

> Screenshot placeholder: add a NoteSense screenshot here.

## Requirements

- Python 3.12
- Internet access the first time the embedding model is loaded

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train and run

Train and save the classifier:

```bash
python train.py
```

Start the app:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

- UI: http://localhost:8000/
- Classification API: `POST http://localhost:8000/classify`

Example request:

```bash
curl -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"text":"Pay the electricity bill by the 28th."}'
```

## Starter notes

Click **Load starter notes** in the UI, or run:

```bash
curl -X POST http://localhost:8000/load-starter
```

Starter note IDs are kept as supplied. Loading starter notes again does not add duplicates. New notes start at `N100`, so their IDs do not reuse the starter IDs.

## Tests and evaluation

Run the automated tests:

```bash
pytest -q
```

Run the API smoke test with the app running:

```bash
python examples/test_api.py --url http://localhost:8000/classify
```

Compare search approaches:

```bash
python eval_search.py
```

## Features

- Classifies notes and displays confidence.
- Marks uncertain predictions using probability and vocabulary coverage thresholds.
- Filters by category, including **Uncertain**.
- Semantic search displays ranked results and says **No good match** when results are below the relevance cutoff.
- **Ask your notes** answers using matching saved-note text and displays evidence IDs. It reports when the notes do not contain a matching answer.
- Each note card has a **Why?** button that shows the predicted category, confidence, vocabulary coverage, and top contributing features. The details are loaded from `GET /explain/{note_id}`.
- Notes can be deleted; deleted IDs are not reused.
- Saves notes in `notes.json`.

The `all-MiniLM-L6-v2` embedding model downloads on first use and is then loaded locally. Internet is needed for that initial download; no paid services are required.

## Evaluation summary

The recorded classifier run used 156 training rows and 50 validation rows. The most-frequent-label baseline scored 0.300 accuracy and 0.115 macro-F1. The final model scored 0.900 accuracy and 0.903 macro-F1 on validation.

The recorded search evaluation used 16 queries, including 6 marked no-overlap. Hit rate @3 was 75.0% overall / 33.3% on no-overlap queries for keyword overlap, 75.0% / 33.3% for TF-IDF cosine, and 93.8% / 83.3% for embeddings cosine. Full details and known failures are in [RESULTS.md](./RESULTS.md).

## Project structure

```text
app.py                    FastAPI app and API endpoints
train.py                  Classifier training and evaluation
eval_search.py            Search approach evaluation
model.joblib              Trained classifier and uncertainty thresholds
notes.json                Locally saved notes (created by the app)
data/
  train.csv               Labeled classifier training data
  validation.csv          Classifier evaluation data
  out_of_scope_examples.csv  Uncertainty evaluation examples
  starter_notes.json      Starter notes loaded by the app
  search_queries.json     Search evaluation queries
static/
  index.html              Aurora Glass interface
tests/
  test_app.py             API and saved-note tests
examples/
  test_api.py             API smoke test
```

## Limitations

The classifier and search evaluation use a small synthetic dataset, so results may not generalize to real notes. Model confidence and vocabulary coverage are useful signals, not guarantees. The local embedding model must be downloaded once before offline use. The interface is intentionally simple and runs locally.
