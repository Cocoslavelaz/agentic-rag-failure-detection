"""Small local metadata/probe transport; never pulls a model."""
import json
import urllib.request


def request(base_url, path, payload=None):
    req = urllib.request.Request(base_url.rstrip("/") + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180 if payload is not None else 10) as response:
        return json.load(response)
