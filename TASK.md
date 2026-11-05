# The Challenge: Smart Notes Assistant

A note is a short message someone saves for themselves, such as "Read two chapters tonight" or "Book train tickets for Friday." Build a small, working application that helps people organize those notes.

## Part A — Working application (required)

The user should be able to:

1. **Add and save a note** consisting of text, and see its predicted category (`Study`, `Money`, `Travel`, `Shopping`). If the model is unsure (see Part D), the note is **visibly marked as uncertain**, for example with an "Uncertain" badge.
2. **View saved notes**, including stable unique IDs and predicted categories.
3. **Filter notes by category.** A filter shows the notes whose displayed category matches. Uncertain notes appear under their best-guess category, still marked as uncertain. An extra "Uncertain" filter is welcome but optional.
4. **Search saved notes** using keywords or a short natural-language phrase (for example `hotel` or `what do I need to buy?`), with the result showing the matching note text and note ID. A helpful ranked list of notes is enough; you do not need a chatbot. If nothing matches well, say so (show no results, or mark results as low relevance) rather than presenting unrelated notes as matches.
5. **Restart the application without losing saved notes**, including their IDs, categories and uncertain markers (JSON file or SQLite is sufficient).
6. **Load the sample notes** in `data/starter_notes.json` with their original IDs (`N001`–`N012`), using a one-line command or a button. Document how in your README. Loading twice must not create duplicates, and notes added by users must never reuse the IDs `N001`–`N012`. We use these notes to test search the same way for every candidate.

**Technology choices:** Any sensible Python-based backend and a basic UI (Streamlit, Gradio, Flask, FastAPI + HTML/React) are acceptable. A locally running app is sufficient; cloud deployment is not necessary. **Streamlit and Gradio cannot serve the `POST /classify` endpoint by themselves.** If you use them, also run a small FastAPI or Flask server for `/classify`, and make both load the same saved model file. Explain in your README how to start each.

## Part B — Train an actual ML model (required)

`data/train.csv` has **156 labeled examples** and `data/validation.csv` has **50 labeled examples** with columns `text,category`. The categories are explained in `CATEGORY_GUIDE.md`.

1. Start with a simple **baseline**, such as predicting the most frequent label or using keyword rules.
2. **Train at least one machine-learning classifier** on `train.csv`; for instance TF-IDF + logistic regression or Naive Bayes. Using scikit-learn counts as real model training.
3. Training must be **reproducible with one command** (for example `python train.py`) that reads the CSV, fits the model, prints validation metrics and saves the model file your app loads.
4. Evaluate both your baseline and trained model **on the validation file without using validation labels for training**.
5. In `RESULTS.md`, report **accuracy, macro-F1** and what improved. Then analyse **at least three distinct mistakes**. Two notes that differ only in a day, city or subject count as one mistake. For at least one of them, show **evidence** for why the model got it wrong, such as the words or features that pushed the prediction, or the most similar training examples. A guess alone is not enough. If you have fewer than three distinct mistakes, analyse difficult examples to make up the three.
6. **Try one fix** for a mistake you found, such as a change to preprocessing, features or model settings, and report whether it helped. Choose the change by reasoning or cross-validation on training data, not by repeatedly checking validation scores. It's fine if the fix didn't help; say why.
7. Your **trained model must drive the predictions in the application and `/classify` endpoint**. A rule-based-only solution or prompting a hosted LLM does not satisfy the model-training requirement.

You do **not** need PyTorch, deep neural networks, fine-tuning, GPUs or reinforcement learning for the mandatory task.

## Part C — Measure your search quality (required)

A search box that "seems to work" is not enough; measure it.

1. Write a small **search test set** of **at least 10 queries** over the 12 starter notes, each with the note ID(s) a user would expect. Save it in your repository (for example `data/search_queries.json`).
   - At least **3 queries must share no words with the note they should find**, ignoring common words such as "my", "the", "is", "a", "when" and "what". For example, `money transfer` should find N006, "Move ₹2,000 into the savings account this month."
   - Example format: `[{"query": "when is my flight?", "expected": ["N003"]}]`
2. **Compare at least two search approaches** on that set. One should be a simple keyword baseline. The other could be TF-IDF similarity, fuzzy matching, or sentence embeddings. Embedding models such as `sentence-transformers/all-MiniLM-L6-v2` run free on a laptop CPU.
3. Report **hit rate @3** for each approach: the fraction of queries where an expected note appears in the top 3 results. Reproduce it with one command.
4. In `RESULTS.md`, say which approach your app uses and why, and show at least one query that still fails.

## Part D — Notes that fit no category (required)

Real notes don't always fit the four categories, for example "Call grandma this weekend." Your app should not present a confident wrong label for these.

1. Use your model's output to decide when a prediction is **uncertain**, for example when the highest predicted probability is below a threshold you choose. Return this as `uncertain` from `/classify` (see `API_CONTRACT.md`), and mark such notes in the UI. A hard-coded keyword list does not count.
2. `data/out_of_scope_examples.csv` has 12 notes that fit no category. Use them, together with the validation set, to choose your approach and threshold. If you train on these examples, keep some aside for evaluation and say so in `RESULTS.md`.
3. In `RESULTS.md`, report the trade-off at your chosen setting and at least one other: what fraction of out-of-scope notes are flagged, and what fraction of validation notes are flagged. (Flagging a few genuinely ambiguous validation notes is reasonable; flagging most of them makes the flag useless.) **We do not expect perfect detection** on such a small dataset. We are looking for a sensible method and honest numbers.

## Part E — Simple API for automated testing (required)

Implement `POST /classify` exactly as defined in `API_CONTRACT.md`. A correct response contains a single category from the four allowed labels and an `uncertain` flag. Document how to start your service.

## Part F — Engineering and AI-tool judgment (required)

Handle empty notes without crashing: the UI shows the user a clear message, and the API returns a 4xx error. Include automated tests that run with one command (for example `pytest`) and cover at least:

- `/classify` returns a valid category and `uncertain` flag for a normal note;
- `/classify` returns a 4xx error for empty or invalid input;
- either saving and reloading a note, or searching for a saved note.

Your documents should explain how your solution works, how you evaluated it, and how you checked its output.

Using AI coding tools is **encouraged**. In `AI_WORKFLOW.md`, tell us:
- Which tools you used, if any.
- One specific task or prompt you delegated, written as you actually wrote it.
- One mistake, weak assumption or bug that you caught and how you checked the correction. If you used no AI tool, document a bug you caught yourself and how you validated it.

## Optional improvements (choose at most one; only if required parts work)

**Bonus 1 — Answer questions about notes:** Return answers to questions about saved notes **with evidence** (the exact note IDs/text used). If a fact is not in the notes, say so instead of inventing it. A local deterministic method is fine. A free or local LLM is also fine, if you show the answers stay grounded in the notes and you test that on a few questions, including ones the notes can't answer.

**Bonus 2 — Improved model:** Train a second learning algorithm or tune the first model using only training data / valid cross-validation. Show a fair comparison on the unchanged validation set and explain why it did or did not help.

These are **optional**. Extra frameworks, elaborate design, paid models, or deploying to cloud are not automatically rewarded.

## What we will review

- Functionality and reproducibility: we will follow your README exactly on a clean machine.
- Proof that a classifier was trained and evaluated, not just an API call. We will re-run your validation numbers through your `/classify` endpoint and compare them with `RESULTS.md`.
- Correctness and reliability of note search and predictions, including on notes you have not seen and notes that fit no category.
- How carefully you measured your search and your model's mistakes.
- Problem-solving and ability to review AI-generated code.
- Your own explanation of engineering decisions and limitations.

Shortlisted candidates will have a **20-minute live discussion**. You will run your app, explain your model and one of its mistakes, and make **one small change to your code** while we watch. You may use the same AI tools you used for the assignment. Know your code well enough to change it.
