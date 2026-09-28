"""Seed script to index brand guidelines from seed_brands.json into ChromaDB."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.vector_store import BrandVectorStore

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("brandbrain.seed")

DEFAULT_SEED_FILE = Path(__file__).resolve().parent.parent / "data" / "seed_brands.json"


def chunk_brand_data(brand: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert raw brand profile JSON into semantic chunks for vector indexing."""
    chunks: List[Dict[str, Any]] = []
    client_id = brand["client_id"]
    brand_name = brand.get("brand_name", client_id)

    # 1. Voice, Tone, and Core Identity
    voice = brand.get("voice_and_tone", {})
    voice_text = (
        f"Brand: {brand_name} ({client_id})\n"
        f"Tagline: {brand.get('tagline', '')}\n"
        f"Mission: {brand.get('mission', '')}\n"
        f"Primary Voice Attributes: {', '.join(voice.get('primary_attributes', []))}\n"
        f"Voice & Tone Description: {voice.get('description', '')}\n"
        f"Dos: {'; '.join(voice.get('dos', []))}\n"
        f"Donts: {'; '.join(voice.get('donts', []))}"
    )
    chunks.append({
        "id": f"{client_id}_voice_tone",
        "document": voice_text,
        "metadata": {
            "client_id": client_id,
            "brand_name": brand_name,
            "category": "voice_and_tone",
            "section": "Voice, Tone & Identity",
        },
    })

    # 2. Forbidden Terms & Compliance Guardrails
    forbidden = brand.get("forbidden_terms", [])
    forbidden_text = (
        f"Brand: {brand_name} ({client_id})\n"
        f"Strictly Forbidden Words and Phrases: {', '.join(forbidden)}\n"
        "These terms are strictly prohibited from all marketing, advertising, product, "
        "and social communications. Use of any forbidden term is an immediate compliance violation."
    )
    chunks.append({
        "id": f"{client_id}_forbidden_terms",
        "document": forbidden_text,
        "metadata": {
            "client_id": client_id,
            "brand_name": brand_name,
            "category": "forbidden_terms",
            "section": "Forbidden Terms & Regulatory Guardrails",
        },
    })

    # 3. Color Palette & Aesthetics
    palette = brand.get("color_palette", {})
    palette_lines = [f"Brand: {brand_name} ({client_id}) Color System:"]
    for role, color in palette.items():
        palette_lines.append(
            f"- {role.replace('_', ' ').title()}: {color.get('name', '')} ({color.get('hex', '')}) - {color.get('usage', '')}"
        )
    chunks.append({
        "id": f"{client_id}_color_palette",
        "document": "\n".join(palette_lines),
        "metadata": {
            "client_id": client_id,
            "brand_name": brand_name,
            "category": "color_palette",
            "section": "Color Palette & Visual Tones",
        },
    })

    # 4. Visual Art Direction & Photography
    visual = brand.get("visual_identity", {})
    visual_text = (
        f"Brand: {brand_name} ({client_id}) Visual Art Direction:\n"
        f"Aesthetic Style: {visual.get('style', '')}\n"
        f"Lighting Direction: {visual.get('lighting', '')}\n"
        f"Composition & Framing: {visual.get('composition', '')}\n"
        f"Materials & Textures: {visual.get('materials_textures', '')}\n"
        f"Visual Elements to Avoid: {'; '.join(visual.get('avoid_in_visuals', []))}"
    )
    chunks.append({
        "id": f"{client_id}_visual_identity",
        "document": visual_text,
        "metadata": {
            "client_id": client_id,
            "brand_name": brand_name,
            "category": "visual_identity",
            "section": "Visual Identity & Art Direction",
        },
    })

    # 5. Audience & Channel Constraints
    audience = brand.get("audience_and_channels", {})
    target = audience.get("target_audience", "")
    channel_constraints = audience.get("channel_guidelines", {})
    channels_text = [
        f"Brand: {brand_name} ({client_id}) Audience & Channel Constraints:",
        f"Target Audience: {target}",
    ]
    for ch_name, ch_rules in channel_constraints.items():
        channels_text.append(f"- Channel [{ch_name.upper()}]: {ch_rules}")

    chunks.append({
        "id": f"{client_id}_audience_channels",
        "document": "\n".join(channels_text),
        "metadata": {
            "client_id": client_id,
            "brand_name": brand_name,
            "category": "audience_and_channels",
            "section": "Target Audience & Channel Constraints",
        },
    })

    return chunks


def seed_database(
    seed_file_path: Path = DEFAULT_SEED_FILE,
    vector_store: Optional[BrandVectorStore] = None,
) -> int:
    """Read seed_brands.json and populate ChromaDB."""
    if not seed_file_path.exists():
        logger.error("Seed file not found at: %s", seed_file_path)
        raise FileNotFoundError(f"Seed file not found: {seed_file_path}")

    with open(seed_file_path, "r", encoding="utf-8") as f:
        brands_data: List[Dict[str, Any]] = json.load(f)

    logger.info("Found %d brand profiles in seed data.", len(brands_data))

    store = vector_store or BrandVectorStore()

    all_ids: List[str] = []
    all_docs: List[str] = []
    all_metas: List[Dict[str, Any]] = []

    for brand in brands_data:
        chunks = chunk_brand_data(brand)
        for chunk in chunks:
            all_ids.append(chunk["id"])
            all_docs.append(chunk["document"])
            all_metas.append(chunk["metadata"])

    store.add_guidelines(
        ids=all_ids,
        documents=all_docs,
        metadatas=all_metas,
    )

    logger.info(
        "Successfully indexed %d guideline chunks into ChromaDB at %s",
        len(all_ids),
        store.persist_dir,
    )
    return len(all_ids)


def main() -> None:
    """CLI entrypoint for seeding."""
    try:
        count = seed_database()
        print(f"BrandBrain seeding complete! Indexed {count} guideline documents.")
    except Exception as exc:
        print(f"Seeding failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
