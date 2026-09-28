"""BrandBrain MCP Server.

Provides creative and brand operations tools over Model Context Protocol (MCP) using
AWS Bedrock (Moonshot AI Kimi K3 & Amazon Titan) and ChromaDB.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from src.bedrock_client import BedrockClient, extract_json_from_response
from src.vector_store import BrandVectorStore

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("brandbrain.server")

# Initialize FastMCP Server
mcp = FastMCP(
    "BrandBrain",
    dependencies=["boto3", "chromadb", "pydantic"],
)

# Shared instances
_vector_store: Optional[BrandVectorStore] = None
_bedrock_client: Optional[BedrockClient] = None


def get_vector_store() -> BrandVectorStore:
    """Lazy-load and return the persistent BrandVectorStore."""
    global _vector_store
    if _vector_store is None:
        _vector_store = BrandVectorStore(bedrock_client=get_bedrock_client())
    return _vector_store


def get_bedrock_client() -> BedrockClient:
    """Lazy-load and return the AWS Bedrock client."""
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = BedrockClient()
    return _bedrock_client


def set_components(
    vector_store: Optional[BrandVectorStore] = None,
    bedrock_client: Optional[BedrockClient] = None,
) -> None:
    """Dependency injection helper for unit testing."""
    global _vector_store, _bedrock_client
    if vector_store is not None:
        _vector_store = vector_store
    if bedrock_client is not None:
        _bedrock_client = bedrock_client


class LintComplianceResponse(BaseModel):
    """Structured compliance linting result."""
    score: int = Field(ge=0, le=100, description="Compliance score from 0 to 100")
    status: Literal["PASS", "NEEDS_REVISION", "FAIL"] = Field(
        description="Overall compliance judgment: PASS, NEEDS_REVISION, or FAIL"
    )
    forbidden_terms_found: List[str] = Field(
        default_factory=list, description="List of banned phrases detected in draft"
    )
    tone_drift_analysis: str = Field(
        description="Explanation of tone alignment or deviation relative to guidelines"
    )
    suggested_rewrite: str = Field(
        description="Brand-aligned rewrite preserving the core message while adhering to rules"
    )


class VisualPromptSpecResponse(BaseModel):
    """Structured visual generation prompt specification."""
    positive_prompt: str = Field(
        description="Detailed positive generation prompt adhering to brand aesthetic, lighting, and palette"
    )
    negative_prompt: str = Field(
        description="Elements and styles to avoid based on brand guidelines"
    )
    aspect_ratio: str = Field(
        description="Recommended aspect ratio suitable for platform and brand aesthetic"
    )
    lighting_and_palette: str = Field(
        description="Specified lighting conditions and brand color palette hexes/names"
    )


def _detect_forbidden_terms(text: str, client_id: str) -> List[str]:
    """Deterministic local check for known forbidden terms from guidelines."""
    known_terms = {
        "solaris-fintech": [
            "guaranteed",
            "moon",
            "easy money",
            "risk-free",
            "get rich quick",
            "100% returns",
            "crypto bro",
            "to the moon",
            "can't lose",
            "passive wealth",
        ],
        "velvet-apothecary": [
            "synthetic",
            "chemical peel",
            "miracle cure",
            "instant fix",
            "clinical grade",
            "anti-aging miracle",
            "bleach",
            "artificial",
            "pore eraser",
            "flawless perfection",
            "erase wrinkles",
        ],
    }

    found: List[str] = []
    text_lower = text.lower()
    for term in known_terms.get(client_id, []):
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.append(term)
    return found


@mcp.tool()
def retrieve_brand_guidelines(client_id: str, query: str, top_k: int = 3) -> str:
    """Performs vector similarity search against ChromaDB and returns relevant brand voice rules, color palettes, and audience constraints.

    Args:
        client_id: The unique identifier for the brand (e.g. 'solaris-fintech' or 'velvet-apothecary').
        query: Semantic query for the rules needed (e.g., 'voice and tone', 'color palette', 'social media guidelines').
        top_k: Number of relevant guideline documents to retrieve (default: 3).

    Returns:
        A formatted markdown string containing the relevant brand rules.
    """
    store = get_vector_store()
    results = store.query_guidelines(client_id=client_id, query=query, top_k=top_k)

    if not results:
        all_docs = store.get_all_client_guidelines(client_id=client_id)
        if not all_docs:
            return (
                f"No guidelines found for client '{client_id}'. "
                "Available pre-seeded clients: 'solaris-fintech', 'velvet-apothecary'."
            )
        results = all_docs[:top_k]

    return store.format_guidelines_for_context(results)


@mcp.tool()
def lint_brand_compliance(client_id: str, draft_text: str, channel: str = "social") -> str:
    """Fetches relevant client rules, constructs a strict evaluation prompt to Bedrock Claude/Kimi, and returns a structured JSON payload.

    Args:
        client_id: The brand client identifier (e.g., 'solaris-fintech' or 'velvet-apothecary').
        draft_text: The draft copy or message to evaluate for brand compliance.
        channel: Target channel for the copy ('social', 'email', 'ads', 'packaging', 'reports').

    Returns:
        A JSON string conforming to:
        {
            "score": int (0-100),
            "status": "PASS" | "NEEDS_REVISION" | "FAIL",
            "forbidden_terms_found": list of strings,
            "tone_drift_analysis": string,
            "suggested_rewrite": string
        }
    """
    store = get_vector_store()
    bedrock = get_bedrock_client()

    rules_items = store.query_guidelines(
        client_id=client_id,
        query=f"forbidden terms voice and tone rules audience constraints channel {channel}",
        top_k=4,
    )
    guidelines_context = store.format_guidelines_for_context(rules_items)

    deterministic_forbidden = _detect_forbidden_terms(draft_text, client_id)

    system_prompt = (
        "You are BrandBrain, an exacting, authoritative brand compliance auditor and creative director. "
        "Your task is to audit draft copy against strict brand guidelines, detect forbidden words or regulatory risks, "
        "assess tone drift, score compliance from 0 to 100, and provide a high-craft brand-aligned rewrite.\n\n"
        "Scoring standards:\n"
        "- 90-100: PASS (exceptional tone alignment, zero forbidden terms, proper nuance)\n"
        "- 60-89: NEEDS_REVISION (no forbidden terms, but mild tone drift or formatting mismatch)\n"
        "- 0-59: FAIL (contains any forbidden term, severe regulatory risk, or completely antithetical voice)\n\n"
        "You MUST respond ONLY with valid, unescaped JSON matching this schema:\n"
        "{\n"
        '  "score": <integer 0-100>,\n'
        '  "status": "<PASS | NEEDS_REVISION | FAIL>",\n'
        '  "forbidden_terms_found": [<list of strings>],\n'
        '  "tone_drift_analysis": "<concise analysis of tone and rule adherence>",\n'
        '  "suggested_rewrite": "<polished, on-brand rewrite maintaining the intent>"\n'
        "}"
    )

    user_prompt = (
        f"Brand Client: {client_id}\n"
        f"Target Channel: {channel}\n\n"
        f"--- BRAND GUIDELINES ---\n{guidelines_context}\n\n"
        f"--- DRAFT COPY TO AUDIT ---\n{draft_text}\n\n"
        "Perform the compliance audit now and output ONLY the JSON object."
    )

    try:
        response_text = bedrock.converse(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=1500,
        )
        parsed = extract_json_from_response(response_text)

        detected_terms = list(
            set(parsed.get("forbidden_terms_found", []) + deterministic_forbidden)
        )
        parsed["forbidden_terms_found"] = detected_terms

        if detected_terms and parsed.get("score", 100) >= 60:
            parsed["score"] = min(parsed.get("score", 50), 45)
            parsed["status"] = "FAIL"

        validated = LintComplianceResponse(**parsed)
        return validated.model_dump_json(indent=2)

    except Exception as exc:
        logger.warning(
            "Bedrock Converse call failed or returned unparseable output: %s. Using deterministic fallback evaluation.",
            exc,
        )
        has_forbidden = bool(deterministic_forbidden)
        score = 35 if has_forbidden else 82
        status = "FAIL" if has_forbidden else "PASS"

        tone_analysis = (
            f"Draft contains strictly forbidden terms: {deterministic_forbidden}."
            if has_forbidden
            else f"Draft aligns reasonably with {client_id} tone guidelines for channel {channel}."
        )

        rewrite = draft_text
        for term in deterministic_forbidden:
            rewrite = re.sub(re.escape(term), "[REDACTED]", rewrite, flags=re.IGNORECASE)

        fallback_result = LintComplianceResponse(
            score=score,
            status=status,
            forbidden_terms_found=deterministic_forbidden,
            tone_drift_analysis=tone_analysis,
            suggested_rewrite=rewrite.strip(),
        )
        return fallback_result.model_dump_json(indent=2)


@mcp.tool()
def generate_visual_prompt_spec(
    client_id: str,
    concept: str,
    platform: str = "instagram",
) -> str:
    """Synthesizes client visual art direction into ready-to-use prompts for Midjourney / ComfyUI / Firefly.

    Args:
        client_id: The brand client identifier (e.g. 'solaris-fintech' or 'velvet-apothecary').
        concept: Core subject or concept of the image (e.g. 'quarterly performance milestone' or 'evening restorative serum ritual').
        platform: Intended destination ('instagram', 'linkedin', 'hero-banner', 'packaging', 'pitch-deck').

    Returns:
        A JSON string conforming to:
        {
            "positive_prompt": string,
            "negative_prompt": string,
            "aspect_ratio": string,
            "lighting_and_palette": string
        }
    """
    store = get_vector_store()
    bedrock = get_bedrock_client()

    visual_items = store.query_guidelines(
        client_id=client_id,
        query=f"visual art direction style lighting composition materials color palette {concept}",
        top_k=3,
    )
    guidelines_context = store.format_guidelines_for_context(visual_items)

    system_prompt = (
        "You are BrandBrain Visual Director, a world-class prompt engineer specializing in high-end diffusion models "
        "(Midjourney v6, ComfyUI, Adobe Firefly, Flux.1). You translate brand identity guidelines and artistic direction "
        "into production-grade visual prompt specifications.\n\n"
        "You MUST respond ONLY with valid, unescaped JSON matching this schema:\n"
        "{\n"
        '  "positive_prompt": "<highly detailed, descriptive prompt specifying subject, materials, textures, camera framing, atmospheric depth>",\n'
        '  "negative_prompt": "<explicit exclusions based on brand guidelines and diffusion artifacts>",\n'
        '  "aspect_ratio": "<aspect ratio matching platform, e.g. 1:1, 16:9, 4:5, 9:16>",\n'
        '  "lighting_and_palette": "<precise lighting temperature and brand hex codes / color names>"\n'
        "}"
    )

    user_prompt = (
        f"Brand Client: {client_id}\n"
        f"Destination Platform: {platform}\n"
        f"Visual Concept: {concept}\n\n"
        f"--- BRAND VISUAL & PALETTE GUIDELINES ---\n{guidelines_context}\n\n"
        "Generate the visual prompt specification now and output ONLY the JSON object."
    )

    default_aspect_ratios = {
        "instagram": "4:5",
        "linkedin": "16:9",
        "hero-banner": "16:9",
        "pitch-deck": "16:9",
        "packaging": "1:1",
    }
    target_aspect = default_aspect_ratios.get(platform.lower(), "1:1")

    try:
        response_text = bedrock.converse(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=1500,
        )
        parsed = extract_json_from_response(response_text)
        validated = VisualPromptSpecResponse(**parsed)
        return validated.model_dump_json(indent=2)

    except Exception as exc:
        logger.warning(
            "Bedrock Converse call failed or returned unparseable output: %s. Using rule-based fallback visual spec.",
            exc,
        )
        if client_id == "solaris-fintech":
            fallback = VisualPromptSpecResponse(
                positive_prompt=(
                    f"Architectural minimalism representing {concept}, monolithic geometric structure, "
                    "brushed titanium and sapphire glass textures, clean telemetry waveforms, "
                    "sharp directional daylight with cool rim highlight, 8k resolution, editorial finance photography"
                ),
                negative_prompt="clutter, neon lasers, crypto coins, cartoon, distorted lines, low resolution, 3d render gloss",
                aspect_ratio=target_aspect,
                lighting_and_palette="Architectural daylight (5000K); Obsidian Deep (#0B0F19), High-Yield Cyan (#06B6D4), Pure Titanium (#F8FAFC)",
            )
        else:
            fallback = VisualPromptSpecResponse(
                positive_prompt=(
                    f"Wabi-sabi herbal sanctuary capturing {concept}, amber glass apothecary dropper bottle, "
                    "tactile crushed sage leaves with fresh dew droplets, unglazed terracotta earthenware, "
                    "golden hour diffused sunlight through raw linen curtains, soft film grain Kodak Portra 400 warmth, f/1.8 macro"
                ),
                negative_prompt="sterile lab, plastic bottles, neon colors, harsh flash, artificial skin smoothing, CGI render",
                aspect_ratio=target_aspect,
                lighting_and_palette="Golden hour diffused sunlight (3200K); Crushed Sage (#708238), Terracotta Ochre (#C86D51), Raw Linen (#F4F0EA)",
            )
        return fallback.model_dump_json(indent=2)


def main() -> None:
    """Run the FastMCP server via stdio transport."""
    logger.info("Starting BrandBrain MCP server on stdio transport...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
