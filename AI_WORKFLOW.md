# AI_WORKFLOW

## 1. Tools
- VS Code with GitHub Copilot in agent mode (wrote and ran most of the code from my prompts).
- Claude (chat) to plan the work, break it into steps, write prompts and review outputs and screenshots.
- Python 3.12, scikit-learn, FastAPI, sentence-transformers, pytest, git.

I worked in small steps (model → API/app → uncertainty → search → tests → docs) and checked each step myself before moving on, instead of asking for the whole app at once.

## 2. One delegated task (prompt as I wrote it)
> Step 1 only:
> - Create a venv using python3.12 (not python3, which is the old system 3.9) and requirements.txt with pinned versions: fastapi, uvicorn, scikit-learn, pandas, joblib, pytest, httpx.
> - Create train.py that: reads data/train.csv, trains a baseline (most frequent label) and a TF-IDF + LogisticRegression model (random_state=42), evaluates both on data/validation.csv (accuracy + macro-F1 + confusion matrix), prints the misclassified validation notes with their top probability, and saves the model to model.joblib.
> - Also print, for thresholds 0.4, 0.5, 0.6, the % of data/out_of_scope_examples.csv notes and % of validation notes whose top probability is below the threshold.
> Do not train on validation or out-of-scope data.

**Kept:** the structure of train.py, the baseline and the evaluation printout.
**Changed:** the first model (default C=1.0) reached only 0.68 accuracy, and every validation note had top probability below 0.6, so no threshold could separate normal notes from out-of-scope ones. I asked for C and features to be chosen by 5-fold cross-validation on train.csv only. It picked C=30 with word + character n-grams: accuracy 0.68 → 0.90, macro-F1 0.708 → 0.903.

## 3. Problems I caught

**a) "Call grandma this weekend" was not marked uncertain.**
- Symptom: the agent said all checks passed, but when I ran the demo flow from CATEGORY_GUIDE.md by hand, the note showed "Money" with no Uncertain badge.
- How I found the cause: curl to /classify returned uncertain false, confidence 0.505, just above my 0.5 threshold. Checking train.csv showed "call", "grandma" and "weekend" never appear; only "this" does, mostly in Money notes ("Save ₹900 from this month salary").
- Fix: added vocabulary coverage (share of a note's words the trained model knows). Uncertain = probability < threshold OR coverage < threshold.
- A second mistake: the agent's first setting (coverage < 0.40) flagged 38% of normal validation notes, which makes the badge useless. I asked for a wider grid and picked prob < 0.45 OR coverage < 0.20: 58.3% of out-of-scope notes flagged vs 8% of validation notes.
- Proof: a pytest test that "Call grandma this weekend." returns uncertain true and "Pay the electricity bill by the 28th." returns false, plus the before/after table in RESULTS.md.

**b) The agent removed a feature I wanted.** A change summary showed index.html "+1 −60": the Why? button was gone. I restored it from my last git commit with `git checkout HEAD -- static/index.html` and re-ran pytest.

**c) The virtual environment was committed to git.** My first commit had no .gitignore, so .venv was included. I added .gitignore and removed it from the repo with `git rm -r --cached .venv`.

## Time spent
About 5 hours.