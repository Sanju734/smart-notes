import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.joblib"
NOTES_PATH = ROOT / "notes.json"
ID_COUNTER_PATH = ROOT / "notes_id_counter.json"
STARTER_PATH = ROOT / "data" / "starter_notes.json"
CATEGORIES = ("Study", "Money", "Travel", "Shopping")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
NO_GOOD_MATCH_SCORE = 0.3
WORD_PATTERN = re.compile(r"(?u)\b\w\w+\b")


class NoteText(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: Annotated[StrictStr, Field(max_length=1000)]

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value):
        if not value.strip():
            raise ValueError("text must contain at least one non-whitespace character")
        return value


class AskQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: Annotated[StrictStr, Field(max_length=1000)]

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value):
        if not value.strip():
            raise ValueError("question must contain at least one non-whitespace character")
        return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    saved = joblib.load(MODEL_PATH)
    app.state.model = saved["model"]
    app.state.prob_threshold = saved["prob_threshold"]
    app.state.coverage_threshold = saved["coverage_threshold"]
    app.state.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    yield


app = FastAPI(title="Smart Notes Assistant", lifespan=lifespan)


def read_notes():
    if not NOTES_PATH.exists():
        return []
    with NOTES_PATH.open(encoding="utf-8") as file:
        notes = json.load(file)
    if not isinstance(notes, list):
        raise ValueError("notes.json must contain a JSON list")
    return notes


def write_notes(notes):
    with NOTES_PATH.open("w", encoding="utf-8") as file:
        json.dump(notes, file, ensure_ascii=False, indent=2)
        file.write("\n")


def vocabulary_coverage(model, text):
    word_vectorizer = model.named_steps["features"].transformer_list[0][1]
    vocabulary = word_vectorizer.vocabulary_
    tokens = [
        token
        for token in WORD_PATTERN.findall(text.lower())
        if token not in ENGLISH_STOP_WORDS
    ]
    if not tokens:
        return 0.0
    return sum(token in vocabulary for token in tokens) / len(tokens)


def classify(model, text, prob_threshold, coverage_threshold):
    probabilities = model.predict_proba([text])[0]
    best_index = probabilities.argmax()
    confidence = float(probabilities[best_index])
    coverage = vocabulary_coverage(model, text)
    return {
        "category": str(model.classes_[best_index]),
        "uncertain": confidence < prob_threshold or coverage < coverage_threshold,
        "confidence": confidence,
        "coverage": coverage,
    }


def make_note(request, note_id, text):
    return {
        "id": note_id,
        "text": text,
        **classify(
            request.app.state.model,
            text,
            request.app.state.prob_threshold,
            request.app.state.coverage_threshold,
        ),
    }


def next_note_id(notes):
    highest_id = 99
    for note in notes:
        match = re.fullmatch(r"N(\d+)", str(note.get("id", "")))
        if match:
            highest_id = max(highest_id, int(match.group(1)))
    if ID_COUNTER_PATH.exists():
        with ID_COUNTER_PATH.open(encoding="utf-8") as file:
            highest_id = max(highest_id, int(json.load(file)))
    next_id = highest_id + 1
    with ID_COUNTER_PATH.open("w", encoding="utf-8") as file:
        json.dump(next_id, file)
        file.write("\n")
    return f"N{next_id:03d}"


def rank_notes(embedder, query, notes, limit=3):
    if not notes:
        return []
    query_vector = embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    note_vectors = embedder.encode(
        [note["text"] for note in notes],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    scores = cosine_similarity(query_vector, note_vectors).ravel()
    indexes = scores.argsort()[::-1][:limit]
    return [(notes[index], float(scores[index])) for index in indexes]


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/classify")
def classify_text(body: NoteText, request: Request):
    return classify(
        request.app.state.model,
        body.text,
        request.app.state.prob_threshold,
        request.app.state.coverage_threshold,
    )


@app.post("/notes")
def create_note(body: NoteText, request: Request):
    notes = read_notes()
    note = make_note(request, next_note_id(notes), body.text)
    notes.append(note)
    write_notes(notes)
    return note


@app.get("/notes")
def list_notes(category: str | None = None):
    if category is not None and category not in CATEGORIES:
        raise HTTPException(status_code=422, detail="Unknown category")
    notes = read_notes()
    if category is not None:
        notes = [note for note in notes if note["category"] == category]
    return {"notes": notes}


@app.delete("/notes/{note_id}")
def delete_note(note_id: str):
    notes = read_notes()
    remaining_notes = [note for note in notes if note["id"] != note_id]
    if len(remaining_notes) == len(notes):
        raise HTTPException(status_code=404, detail="Note not found")
    write_notes(remaining_notes)
    return {"deleted": note_id}


@app.get("/explain/{note_id}")
def explain_note(note_id: str, request: Request):
    note = next((item for item in read_notes() if item["id"] == note_id), None)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")

    model = request.app.state.model
    prediction = classify(
        model,
        note["text"],
        request.app.state.prob_threshold,
        request.app.state.coverage_threshold,
    )
    feature_model = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    row = feature_model.transform([note["text"]])
    class_index = list(classifier.classes_).index(prediction["category"])
    values = row.toarray()[0]
    coefficients = classifier.coef_[class_index]
    weights = values * coefficients
    feature_names = feature_model.get_feature_names_out()
    top_indexes = [
        index
        for index in weights.argsort()[::-1]
        if weights[index] > 0
    ][:5]
    features = [
        {
            "feature": str(feature_names[index]),
            "value": float(values[index]),
            "coefficient": float(coefficients[index]),
            "weight": float(weights[index]),
        }
        for index in top_indexes
    ]
    return {
        "id": note_id,
        "category": prediction["category"],
        "confidence": prediction["confidence"],
        "coverage": prediction["coverage"],
        "features": features,
    }


@app.post("/load-starter")
def load_starter(request: Request):
    with STARTER_PATH.open(encoding="utf-8") as file:
        starter_notes = json.load(file)
    notes = read_notes()
    existing_ids = {note["id"] for note in notes}
    added = []
    for starter in starter_notes:
        if starter["id"] not in existing_ids:
            added.append(make_note(request, starter["id"], starter["text"]))
            existing_ids.add(starter["id"])
    if added:
        write_notes(notes + added)
    return {"added": len(added), "notes": read_notes()}


@app.get("/search")
def search_notes(
    request: Request,
    q: Annotated[str, Query(min_length=1, max_length=1000)],
):
    if not q.strip():
        raise HTTPException(status_code=422, detail="q must not be blank")
    notes = read_notes()
    if not notes:
        return {"results": [], "message": "No good match"}

    ranked = rank_notes(request.app.state.embedder, q, notes)
    results = [
        {
            **note,
            "score": score,
            "match": bool(score >= NO_GOOD_MATCH_SCORE),
            "match_message": (
                None
                if score >= NO_GOOD_MATCH_SCORE
                else "No good match"
            ),
        }
        for note, score in ranked
    ]
    message = (
        None
        if any(result["match"] for result in results)
        else "No good match"
    )
    return {"results": results, "message": message}


@app.post("/ask")
def ask_notes(body: AskQuestion, request: Request):
    ranked = rank_notes(request.app.state.embedder, body.question, read_notes(), limit=1)
    if not ranked or ranked[0][1] < NO_GOOD_MATCH_SCORE:
        return {"answer": "I couldn't find this in your notes.", "evidence": []}

    note, score = ranked[0]
    return {
        "answer": note["text"],
        "evidence": [{"id": note["id"], "text": note["text"], "score": score}],
    }
