#!/usr/bin/env python3
"""Public smoke test for POST /classify. Standard library only.

Usage: python examples/test_api.py --url http://127.0.0.1:8000/classify
"""
import argparse
import json
import socket
import sys
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LABELS = {'Study', 'Money', 'Travel', 'Shopping'}
VALID_NOTES = [
    'Revise for my biology examination.',
    'Check my bank balance.',
    'Book a train to Pune.',
    'Purchase a pair of shoes.',
    'Move ₹2,000 into savings before the 1st.',
    'Call grandma this weekend.',            # fits no category
    'Pay the rent. ' * 70,                   # about 1,000 characters
]
INVALID_BODIES = {
    'whitespace text': b'{"text": "   "}',
    'empty text': b'{"text": ""}',
    'missing text field': b'{"note": "Pay rent"}',
    'text is a number': b'{"text": 42}',
    'text is null': b'{"text": null}',
    'malformed JSON': b'{"text": ',
}
SLOW_MS = 1000


class Unreachable(Exception):
    pass


def is_valid(status, result):
    return (status == 200 and isinstance(result, dict) and result.get('category') in LABELS
            and isinstance(result.get('uncertain'), bool))


def post(url, body, timeout):
    """Return (status, parsed_json_or_None, latency_ms). status is None on timeout."""
    req = Request(url, data=body, headers={'Content-Type': 'application/json'}, method='POST')
    start = perf_counter()
    try:
        with urlopen(req, timeout=timeout) as r:
            try:
                result = json.load(r)
            except ValueError:
                result = None
            return r.status, result, round(1000*(perf_counter()-start))
    except HTTPError as e:
        return e.code, None, round(1000*(perf_counter()-start))
    except (socket.timeout, TimeoutError):
        return None, None, round(1000*(perf_counter()-start))
    except URLError as e:
        if isinstance(e.reason, (socket.timeout, TimeoutError)):
            return None, None, round(1000*(perf_counter()-start))
        raise Unreachable(e.reason)
    except ConnectionError as e:
        raise Unreachable(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8000/classify')
    p.add_argument('--timeout', type=float, default=4.0, help='seconds; matches the API contract')
    a = p.parse_args()
    failures = 0
    responses = {}

    def check(ok, message):
        nonlocal failures
        failures += not ok
        print('PASS' if ok else 'FAIL', message)

    try:
        post(a.url, json.dumps({'text': 'Warm-up note.'}).encode('utf-8'), max(a.timeout, 15.0))
        for text in VALID_NOTES:
            status, result, ms = post(a.url, json.dumps({'text': text}).encode('utf-8'), a.timeout)
            label = text if len(text) < 60 else f'{text[:30]}... ({len(text)} chars)'
            if status is None:
                check(False, f'valid note {label!r} -> timed out after {a.timeout:g}s')
                continue
            check(is_valid(status, result), f'valid note {label!r} -> HTTP {status}, {result} ({ms} ms)')
            if ms > SLOW_MS:
                print(f'WARN slow response: {ms} ms (aim for under {SLOW_MS} ms)')
            responses[text] = result

        for name, body in INVALID_BODIES.items():
            status, _, _ = post(a.url, body, a.timeout)
            check(status is not None and 400 <= status < 500, f'{name} -> HTTP {status or "timeout"} (expected 4xx)')

        status, result, _ = post(a.url, json.dumps({'text': VALID_NOTES[0]}).encode('utf-8'), a.timeout)
        check(is_valid(status, result), 'server still answers after invalid requests')
        check(result == responses.get(VALID_NOTES[0]), 'same note gives the same response twice')
    except Unreachable as e:
        sys.exit(f'FAIL: could not reach {a.url}: {e}. Is the server running on that port?')

    flags = [r.get('uncertain') for r in responses.values() if isinstance(r, dict)]
    if flags and len(set(flags)) == 1:
        print(f'INFO: uncertain was {flags[0]} for every note, including one that fits no category. Is your flag working?')

    print(f'\n{failures} check(s) failed.' if failures else '\nAll format checks passed.')
    print('This only checks the format. Run your own model evaluation separately.')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
