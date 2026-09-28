"""Unit and integration tests for BrandBrain MCP server tools and Bedrock integration."""

import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.bedrock_client import BedrockClient, extract_json_from_response
from src.seed import chunk_brand_data, seed_database
from src.server import (
    generate_visual_prompt_spec,
    get_vector_store,
    lint_brand_compliance,
    mcp,
    retrieve_brand_guidelines,
    set_components,
)
from src.vector_store import BrandVectorStore, LocalDeterministicEmbeddingFunction


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    """Create a temporary test directory for ChromaDB to isolate test runs."""
    test_dir = tempfile.mkdtemp(prefix="brandbrain_test_chroma_")
    store = BrandVectorStore(
        persist_dir=test_dir,
        embedding_function=LocalDeterministicEmbeddingFunction(dim=128),
    )
    set_components(vector_store=store)

    # Seed the test store with the default seed data
    seed_database(vector_store=store)

    yield store

    # Cleanup temp directory
    try:
        shutil.rmtree(test_dir, ignore_errors=True)
    except Exception:
        pass


def test_fastmcp_tool_registration():
    """Verify that all three required tools are registered on the FastMCP instance."""
    tool_names = [tool.name for tool in mcp._tool_manager.list_tools()]
    assert "retrieve_brand_guidelines" in tool_names
    assert "lint_brand_compliance" in tool_names
    assert "generate_visual_prompt_spec" in tool_names


def test_chunk_brand_data():
    """Verify semantic chunking splits raw brand guidelines into 5 logical categories."""
    mock_brand = {
        "client_id": "test-brand",
        "brand_name": "Test Brand",
        "tagline": "The future of testing",
        "mission": "Deliver bug-free code",
        "voice_and_tone": {
            "primary_attributes": ["Clear", "Direct"],
            "description": "Keep it clean",
            "dos": ["Be precise"],
            "donts": ["Do not ramble"],
        },
        "forbidden_terms": ["untested", "broken"],
        "color_palette": {
            "primary": {"name": "Test Blue", "hex": "#0000FF", "usage": "Background"}
        },
        "visual_identity": {
            "style": "Minimal",
            "lighting": "Bright",
            "composition": "Centered",
            "materials_textures": "Steel",
            "avoid_in_visuals": ["Blurry"],
        },
        "audience_and_channels": {
            "target_audience": "Engineers",
            "channel_guidelines": {"social": "Short code snippets"},
        },
    }

    chunks = chunk_brand_data(mock_brand)
    assert len(chunks) == 5
    categories = {c["metadata"]["category"] for c in chunks}
    expected = {
        "voice_and_tone",
        "forbidden_terms",
        "color_palette",
        "visual_identity",
        "audience_and_channels",
    }
    assert categories == expected


def test_retrieve_brand_guidelines_solaris():
    """Test vector retrieval for Solaris Fintech returns relevant rules."""
    result = retrieve_brand_guidelines(
        client_id="solaris-fintech", query="color palette hex codes", top_k=2
    )
    assert isinstance(result, str)
    assert len(result) > 0
    assert "Solaris Fintech" in result or "COLOR_PALETTE" in result


def test_retrieve_brand_guidelines_velvet():
    """Test vector retrieval for Velvet Apothecary returns sensory guidelines."""
    result = retrieve_brand_guidelines(
        client_id="velvet-apothecary", query="poetic sensory tone", top_k=2
    )
    assert isinstance(result, str)
    assert len(result) > 0
    assert "Velvet Apothecary" in result or "VOICE_AND_TONE" in result


def test_retrieve_brand_guidelines_unknown_client():
    """Test querying a non-existent client returns an informative error message."""
    result = retrieve_brand_guidelines(client_id="unknown-crypto-startup", query="voice")
    assert "No guidelines found" in result
    assert "solaris-fintech" in result


def test_lint_brand_compliance_detects_forbidden_terms_fail():
    """Test that forbidden terms in draft copy trigger FAIL status and score penalty."""
    dirty_draft = (
        "Join Solaris Fintech today! Guaranteed easy money with 100% returns. "
        "We are going to the moon with risk-free algorithmic yield!"
    )
    res_str = lint_brand_compliance(client_id="solaris-fintech", draft_text=dirty_draft)
    data = json.loads(res_str)

    assert data["status"] == "FAIL"
    assert data["score"] < 60
    assert "guaranteed" in data["forbidden_terms_found"]
    assert "easy money" in data["forbidden_terms_found"]
    assert len(data["forbidden_terms_found"]) >= 2
    assert "suggested_rewrite" in data


def test_lint_brand_compliance_velvet_forbidden_terms():
    """Test that Velvet Apothecary banned words are flagged."""
    dirty_draft = (
        "Try our miracle cure chemical peel! It is an instant fix with synthetic extracts "
        "to erase wrinkles and achieve flawless perfection."
    )
    res_str = lint_brand_compliance(client_id="velvet-apothecary", draft_text=dirty_draft)
    data = json.loads(res_str)

    assert data["status"] == "FAIL"
    assert data["score"] < 60
    assert "miracle cure" in data["forbidden_terms_found"]
    assert "chemical peel" in data["forbidden_terms_found"]
    assert "synthetic" in data["forbidden_terms_found"]


def test_lint_brand_compliance_with_mock_bedrock():
    """Test lint_brand_compliance with Bedrock Claude/Kimi Converse API response."""
    clean_draft = (
        "Solaris provides verifiable algorithmic execution infrastructure with institutional-grade custody. "
        "Our telemetry endpoints offer real-time latency reporting."
    )

    mock_llm_json = {
        "score": 95,
        "status": "PASS",
        "forbidden_terms_found": [],
        "tone_drift_analysis": "Excellent adherence to Solaris authoritative, minimalist, and transparent voice.",
        "suggested_rewrite": clean_draft,
    }

    mock_client = MagicMock(spec=BedrockClient)
    mock_client.converse.return_value = json.dumps(mock_llm_json)

    set_components(bedrock_client=mock_client)

    res_str = lint_brand_compliance(client_id="solaris-fintech", draft_text=clean_draft)
    data = json.loads(res_str)

    assert data["score"] == 95
    assert data["status"] == "PASS"
    assert data["forbidden_terms_found"] == []
    assert "suggested_rewrite" in data
    mock_client.converse.assert_called_once()


def test_generate_visual_prompt_spec_with_mock_bedrock():
    """Test generate_visual_prompt_spec with Bedrock Claude/Kimi Converse API response."""
    mock_spec = {
        "positive_prompt": (
            "Monolithic titanium data terminal against Obsidian backdrop, High-Yield Cyan (#06B6D4) "
            "subtle telemetry illumination, sharp architectural daylight, clean brutalist symmetry, 8k"
        ),
        "negative_prompt": "clutter, green laser eyes, coins, neon blur, saturated rainbow",
        "aspect_ratio": "16:9",
        "lighting_and_palette": "Architectural daylight 5000K; Obsidian Deep (#0B0F19), High-Yield Cyan (#06B6D4)",
    }

    mock_client = MagicMock(spec=BedrockClient)
    mock_client.converse.return_value = json.dumps(mock_spec)

    set_components(bedrock_client=mock_client)

    res_str = generate_visual_prompt_spec(
        client_id="solaris-fintech",
        concept="Algorithmic telemetry architecture",
        platform="linkedin",
    )
    data = json.loads(res_str)

    assert "positive_prompt" in data
    assert "negative_prompt" in data
    assert data["aspect_ratio"] == "16:9"
    assert "High-Yield Cyan" in data["lighting_and_palette"]


def test_generate_visual_prompt_spec_fallback():
    """Test fallback visual spec generation when Bedrock converse throws an error."""
    mock_client = MagicMock(spec=BedrockClient)
    mock_client.converse.side_effect = RuntimeError("AWS Bedrock unavailable")

    set_components(bedrock_client=mock_client)

    res_str = generate_visual_prompt_spec(
        client_id="velvet-apothecary",
        concept="Evening botanical ritual serum",
        platform="instagram",
    )
    data = json.loads(res_str)

    assert "positive_prompt" in data
    assert "negative_prompt" in data
    assert data["aspect_ratio"] == "4:5"
    assert "Crushed Sage" in data["lighting_and_palette"]
    assert "amber glass" in data["positive_prompt"]


def test_extract_json_helper():
    """Test robust JSON extraction from raw strings and markdown code blocks."""
    # Plain JSON
    raw = '{"key": "value"}'
    assert extract_json_from_response(raw) == {"key": "value"}

    # Markdown JSON fence
    fenced = "Here is your response:\n```json\n{\n  \"score\": 88,\n  \"status\": \"PASS\"\n}\n```\nEnjoy!"
    assert extract_json_from_response(fenced) == {"score": 88, "status": "PASS"}

    # Embedded braces
    embedded = "Analysis complete: {\"status\": \"NEEDS_REVISION\"}. Please review."
    assert extract_json_from_response(embedded) == {"status": "NEEDS_REVISION"}
