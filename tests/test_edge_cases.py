"""
Covers invalid uploads, invalid queries and concurrent uploads.

"""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

BASE_URL = "http://localhost:8000"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EDGE_CASES = DATA_DIR / "edge_cases"


@pytest.fixture(scope="module", autouse=True)
def wait_for_service():
    for _ in range(30):
        try:
            if requests.get(f"{BASE_URL}/health", timeout=3).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    pytest.fail("Service did not become healthy in time.")


def upload(path: Path, name: str | None = None) -> requests.Response:
    with open(path, "rb") as f:
        return requests.post(
            f"{BASE_URL}/ingest/", files={"input": (name or path.name, f)}, timeout=120
        )


def search(body) -> requests.Response:
    return requests.post(f"{BASE_URL}/search/", json=body, timeout=30)


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("not_a_pdf.txt", "Only PDF files are accepted."),
        ("empty.pdf", "is empty"),
        ("corrupt.pdf", "corrupt or unreadable PDF"),
        ("encrypted.pdf", "Password-protected"),
        ("no_text.pdf", "No text could be extracted"),
    ],
)
def test_invalid_file_is_rejected(filename, expected):
    response = upload(EDGE_CASES / filename)
    assert response.status_code == 400
    assert expected in response.json()["error"]


def test_bad_file_in_batch_rejects_whole_request():
    with open(DATA_DIR / "sample.pdf", "rb") as good, open(EDGE_CASES / "corrupt.pdf", "rb") as bad:
        response = requests.post(
            f"{BASE_URL}/ingest/",
            files=[("input", ("sample.pdf", good)), ("input", ("corrupt.pdf", bad))],
            timeout=120,
        )
    assert response.status_code == 400
    assert "corrupt.pdf" in response.json()["error"]


def test_missing_input_field():
    response = requests.post(f"{BASE_URL}/ingest/", data={"other": "x"}, timeout=30)
    assert response.status_code == 400
    assert response.json() == {"error": "input: Field required"}


@pytest.mark.parametrize("path", ["../etc", "does-not-exist"])
def test_invalid_directory_is_rejected(path):
    response = requests.post(f"{BASE_URL}/ingest/", data={"input": path}, timeout=30)
    assert response.status_code == 400
    assert "error" in response.json()


def test_directory_ingestion():
    response = requests.post(f"{BASE_URL}/ingest/", data={"input": "/data"}, timeout=300)
    assert response.status_code == 200, response.text
    assert "sample.pdf" in response.json()["files"]


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_is_rejected(query):
    response = search({"query": query})
    assert response.status_code == 400
    assert response.json() == {"error": "Query cannot be empty."}


@pytest.mark.parametrize(
    "body",
    [{}, {"query": 123}, {"query": "ok", "top_k": 0}, {"query": "ok", "top_k": 1000}],
)
def test_invalid_search_body_is_rejected(body):
    response = search(body)
    assert response.status_code == 400
    assert "error" in response.json()


def test_malformed_json_is_rejected():
    response = requests.post(
        f"{BASE_URL}/search/",
        data="{not json",
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    assert response.status_code == 400
    assert response.json() == {"error": "Request body is not valid JSON."}


def test_concurrent_uploads_of_different_files():
    paths = [DATA_DIR / name for name in ("sample.pdf", "football_rules.pdf", "renewable_energy.pdf", "sourdough_baking.pdf")]
    with ThreadPoolExecutor(max_workers=len(paths)) as pool:
        responses = list(pool.map(upload, paths))
    assert [r.status_code for r in responses] == [200] * len(paths)


def test_concurrent_uploads_of_same_file_store_it_once():
    path = DATA_DIR / "football_rules.pdf"
    with ThreadPoolExecutor(max_workers=5) as pool:
        responses = list(pool.map(lambda _: upload(path), range(5)))
    assert [r.status_code for r in responses] == [200] * 5

    results = search({"query": "football rules", "top_k": 50, "min_score": -1}).json()["results"]
    contents = [r["content"] for r in results if r["document"] == "football_rules.pdf"]
    assert contents, "the uploaded document should be searchable"
    assert len(contents) == len(set(contents)), "chunks were stored more than once"


def test_search_returns_relevant_document_first():
    upload(DATA_DIR / "sourdough_baking.pdf")
    results = search({"query": "how to feed a sourdough starter", "top_k": 3}).json()["results"]
    assert results and results[0]["document"] == "sourdough_baking.pdf"
