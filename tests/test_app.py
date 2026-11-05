import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as notes_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("notes")
    notes_file = data_dir / "notes.json"
    counter_file = data_dir / "notes_id_counter.json"
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(notes_app, "NOTES_PATH", notes_file)
    monkeypatch.setattr(notes_app, "ID_COUNTER_PATH", counter_file)
    with TestClient(notes_app.app) as test_client:
        yield test_client
    monkeypatch.undo()


def test_classify_returns_category_and_uncertainty(client):
    response = client.post("/classify", json={"text": "Review my chemistry assignment."})

    assert response.status_code == 200
    result = response.json()
    assert result["category"] in {"Study", "Money", "Travel", "Shopping"}
    assert isinstance(result["uncertain"], bool)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["coverage"], float)


def test_out_of_scope_note_is_uncertain(client):
    response = client.post("/classify", json={"text": "Call grandma this weekend."})

    assert response.status_code == 200
    assert response.json()["uncertain"] is True


@pytest.mark.parametrize(
    "body",
    [
        {"text": ""},
        {"text": "   "},
        {},
        {"text": 42},
        {"text": None},
    ],
)
def test_invalid_classification_input_returns_4xx(client, body):
    response = client.post("/classify", json=body)

    assert 400 <= response.status_code < 500


def test_malformed_json_returns_4xx(client):
    response = client.post(
        "/classify",
        content=b'{"text":',
        headers={"Content-Type": "application/json"},
    )

    assert 400 <= response.status_code < 500


def test_note_is_saved_and_read_from_file(client):
    response = client.post("/notes", json={"text": "Review the Python lesson tonight."})

    assert response.status_code == 200
    note = response.json()
    with notes_app.NOTES_PATH.open(encoding="utf-8") as file:
        saved_notes = json.load(file)
    assert note in saved_notes

    reloaded = client.get("/notes")
    assert reloaded.status_code == 200
    assert note in reloaded.json()["notes"]


def test_delete_removes_note_and_deleted_id_is_not_reused(client):
    created = client.post("/notes", json={"text": "Book a train ticket for Monday."})
    assert created.status_code == 200
    deleted_id = created.json()["id"]

    deleted = client.delete(f"/notes/{deleted_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": deleted_id}
    assert all(
        note["id"] != deleted_id for note in client.get("/notes").json()["notes"]
    )
    assert client.delete(f"/notes/{deleted_id}").status_code == 404

    replacement = client.post(
        "/notes", json={"text": "Review my chemistry assignment."}
    )
    assert replacement.status_code == 200
    assert replacement.json()["id"] != deleted_id


def test_search_finds_saved_note(client):
    response = client.get("/search", params={"q": "Python lesson"})

    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert any("Python lesson" in note["text"] for note in results)
    assert all("score" in note for note in results)


def test_search_reports_no_good_match(client):
    response = client.get("/search", params={"q": "zxqvblorp"})

    assert response.status_code == 200
    assert response.json()["message"] == "No good match"
    results = response.json()["results"]
    assert 1 <= len(results) <= 3
    assert all(result["match"] is False for result in results)
    assert all(result["match_message"] == "No good match" for result in results)


def test_explain_returns_top_features_for_saved_note(client):
    note = client.get("/notes").json()["notes"][0]

    response = client.get(f"/explain/{note['id']}")

    assert response.status_code == 200
    result = response.json()
    assert result["id"] == note["id"]
    assert result["category"] in {"Study", "Money", "Travel", "Shopping"}
    assert isinstance(result["confidence"], float)
    assert isinstance(result["coverage"], float)
    assert len(result["features"]) <= 5
    assert all(feature["weight"] > 0 for feature in result["features"])


def test_explain_returns_404_for_missing_note(client):
    response = client.get("/explain/N999999")

    assert response.status_code == 404


def test_ask_answers_with_saved_note_evidence(client):
    loaded = client.post("/load-starter")
    assert loaded.status_code == 200

    response = client.post("/ask", json={"question": "when is my flight?"})

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == "Our flight to Delhi is on Monday morning."
    assert result["evidence"][0]["id"] == "N003"
    assert result["evidence"][0]["text"] == result["answer"]
    assert isinstance(result["evidence"][0]["score"], float)


def test_ask_does_not_invent_missing_information(client):
    response = client.post(
        "/ask", json={"question": "when is my dentist appointment?"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "I couldn't find this in your notes.",
        "evidence": [],
    }
