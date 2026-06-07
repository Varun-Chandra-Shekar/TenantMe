"""Tests for the retrieval layer."""

from dotenv import load_dotenv
load_dotenv()

from tenantmate.retrieve import search


def test_search_returns_results():
    """Basic smoke test — search returns something."""
    results = search("rent increase", k=3)
    assert len(results) == 3
    assert all("chunk_id" in r for r in results)


def test_search_finds_rent_section():
    """Rent-related query should surface section 41."""
    results = search("How much notice for a rent increase?", k=5)
    top_ids = [r["chunk_id"] for r in results]
    assert "NSW-RTA2010-s41" in top_ids, f"Expected s41 in top 5, got {top_ids}"


def test_search_finds_entry_section():
    """Landlord-entry query should surface section 55."""
    results = search("Can my landlord enter without notice?", k=5)
    top_ids = [r["chunk_id"] for r in results]
    assert "NSW-RTA2010-s55" in top_ids


def test_similarity_scores_descend():
    """Top result must score higher than later ones."""
    results = search("rent increase", k=5)
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)