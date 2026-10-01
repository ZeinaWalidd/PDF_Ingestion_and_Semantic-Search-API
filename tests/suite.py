"""
Simple functional test for the PDF Ingestor & Semantic Search API.

Validates that:
1. The service starts correctly and accepts PDF ingestion.
2. The search endpoint returns meaningful results.
"""

import pytest
import requests
import time
import os

BASE_URL = "http://localhost:8000"

@pytest.fixture(scope="session", autouse=True)
def wait_for_service():
    """Wait for service to be ready before tests start."""
    for _ in range(20):
        try:
            r = requests.get(f"{BASE_URL}/docs")
            if r.status_code == 200:
                return
        except Exception:
            time.sleep(1)
    pytest.fail("Service did not start within expected time.")


def test_ingest_single_pdf(tmp_path):
    """Test single PDF ingestion."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("Artificial intelligence enables systems to learn from data.")

    with open(pdf_path, "rb") as f:
        response = requests.post(f"{BASE_URL}/ingest/", files={"input": f})

    assert response.status_code == 200, f"Ingest failed: {response.text}"
    body = response.json()
    assert "message" in body
    assert "files" in body
    assert pdf_path.name in body["files"]


def test_search_query():
    """Test semantic search query."""
    query = {"query": "Explain how AI learns from data"}
    response = requests.post(f"{BASE_URL}/search/", json=query)

    assert response.status_code == 200, f"Search failed: {response.text}"
    body = response.json()
    assert "results" in body
    assert isinstance(body["results"], list)
    if len(body["results"]) > 0:
        assert "document" in body["results"][0]
        assert "content" in body["results"][0]

