# Required API contract

This very small interface lets us test all submissions consistently, regardless of UI framework.

## POST `/classify`

Request body (JSON):

```json
{"text": "Revise Python loops before Friday."}
```

Response body (JSON, HTTP 200):

```json
{"category": "Study", "uncertain": false}
```

**Rules:**

- `text` must be a JSON string containing at least one non-whitespace character. Accept notes of up to 1,000 characters, including non-ASCII text such as `₹`.
- Return one `category` with exact spelling and capitalization from `Study`, `Money`, `Travel`, `Shopping`. Always return your model's best guess, even when it is unsure.
- Return `uncertain` as a JSON boolean (`true` / `false`). It is `true` when your model is not confident, or the note probably fits none of the four categories (see Part D of `TASK.md`).
- Both outputs must come from the **trained classification model**, not a hard-coded mapping, a keyword list, or an external chat completion.
- For invalid input, return a **4xx error** (400 or 422 are both fine), not an unhandled server exception. Invalid input includes: empty or whitespace-only `text`, a missing `text` field, `text` that is not a string, and a body that is not valid JSON. The server must keep working after an invalid request.
- Respond in **under 1 second** per request on a normal laptop once the server has started; a request taking more than **4 seconds** counts as failed. Load the model once at startup, not on every request.
- The same text should always return the same response.
- Additional JSON fields such as `confidence` are allowed and encouraged, but not required.
- The endpoint must be accessible on localhost; use any port, and state the exact startup command and URL in your README.
- Do not require authentication or a paid service for the local assessment endpoint.

Example command (change port if needed):

```bash
curl -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"text":"Pay my electricity bill tomorrow."}'
```

Expected response includes `"category": "Money"` and an `uncertain` flag (the model can make errors; this shows the expected format).

## Optional APIs

You can create your own routes for note CRUD, filtering and search. Only `/classify` has a fixed format. Those UI features will also be checked manually.

## Self-test

Run your backend then execute:

```bash
python examples/test_api.py --url http://localhost:8000/classify
```

The helper checks the response format, the invalid-input cases listed above, a 1,000-character note, repeatability and response time. Our private check applies the same rules to different notes. It is **not** your full ML evaluation: passing it says nothing about whether your predictions or `uncertain` flags are good.
