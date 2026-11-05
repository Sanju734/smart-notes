# Submission checklist

- [ ] My app launches locally following the README with clearly stated dependencies (a `requirements.txt` or equivalent with versions).
- [ ] I can add a note, save it, restart, and find the note again.
- [ ] I can filter notes by category and search them.
- [ ] I can load `data/starter_notes.json` with its original IDs using a documented command or button.
- [ ] I trained an ML classifier on `data/train.csv` with one documented command, and the live application uses it.
- [ ] My local `POST /classify` endpoint accepts `{"text":"..."}` and returns `{"category":"...", "uncertain": true/false}`.
- [ ] Uncertain notes are visibly marked in the app, and the `uncertain` flag comes from my model's output, not a keyword list.
- [ ] My search test set has at least 10 queries (3 or more sharing no words with their target note), and `RESULTS.md` compares hit rate @3 for at least two search approaches.
- [ ] `RESULTS.md` reports the out-of-scope trade-off: share of out-of-scope notes flagged vs share of validation notes flagged, at two or more settings.
- [ ] Loading the starter notes twice does not create duplicates, and new notes never reuse IDs `N001`–`N012`.
- [ ] If my UI is Streamlit or Gradio, my README explains how to start the separate `/classify` server too.
- [ ] `python examples/test_api.py` passes against my running server.
- [ ] My automated tests run with one command and cover classification, invalid input, and persistence or search.
- [ ] My tests and validation experiment run without a paid service or private credentials.
- [ ] `RESULTS.md` reports baseline vs trained model accuracy and macro-F1, at least three **distinct** mistakes with evidence for at least one, and one fix I tried.
- [ ] The numbers in `RESULTS.md` match what my code prints today. (We will re-run them.)
- [ ] `AI_WORKFLOW.md` explains my process, AI tool use (if any), an issue found and how it was checked.
- [ ] `README.md` explains installation, launch, tests, limitations and any external dependencies.
- [ ] I've removed API keys, tokens, private datasets, personal data and heavyweight virtual environments.
- [ ] Optional, not scored: a **2–3 minute demo video** or screenshots showing a note being added and classified, an out-of-scope note marked uncertain, a restart, then search and filter.

**Submit:** Reply to your invitation email before the deadline stated in it, with a link to a **public** GitHub repository (or a source zip attached), your demo link or screenshots if you made them, and any brief run instructions. If you prefer to keep your repository private, send a zip instead.
