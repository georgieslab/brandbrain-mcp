# 🧠 BrandBrain-MCP — Portfolio Showcase & Engineering Deep Dive

> **Project Name:** BrandBrain-MCP  
> **Repository:** [https://github.com/georgieslab/brandbrain-mcp](https://github.com/georgieslab/brandbrain-mcp)  
> **Live Landing Page:** [https://georgieslab.github.io/brandbrain-mcp/](https://georgieslab.github.io/brandbrain-mcp/)  
> **Author:** Georgie (`georgieslab`)  
> **Category:** Generative AI Infrastructure • Model Context Protocol (MCP) • Enterprise Creative Ops • Cloud AI (AWS Bedrock)

---

## 1. Executive Summary

**BrandBrain-MCP** is an enterprise-grade Model Context Protocol (MCP) server that transforms static brand guidelines, regulatory constraints, and creative art direction into active, real-time AI tools inside developer IDEs and desktop LLM clients (such as **Claude Desktop**, **Antigravity**, and **Cursor**).

Instead of relying on human copywriters to manually cross-reference 50-page PDF brand manuals, or prompting general LLMs that hallucinate and drift off-tone, BrandBrain exposes three standardized MCP tools over **stdio transport**:
1. **Semantic Brand Retrieval:** Scoped vector search across brand identity, tone attributes, color palettes, and audience rules.
2. **Automated Tone & Compliance Auditing:** Real-time scoring (0–100), regulatory violation detection, tone drift analysis, and brand-aligned rewrites via **AWS Bedrock Claude Sonnet**.
3. **Diffusion Prompt Synthesis:** Generating production-ready positive/negative prompts, aspect ratios, and lighting/palette specs for **Midjourney v6**, **ComfyUI**, **Firefly**, and **Flux**.

---

## 2. Core Architecture & Data Flow

```
┌────────────────────────────────────────────────────────┐
│     Desktop LLM Clients / IDEs (Claude Desktop,        │
│          Antigravity, Cursor, Zed, Windsurf)          │
└──────────────────────────┬─────────────────────────────┘
                           │ Standard Input/Output (stdio)
                           ▼
┌────────────────────────────────────────────────────────┐
│                   BrandBrain FastMCP                   │
│                    (src/server.py)                     │
│                                                        │
│   ├── retrieve_brand_guidelines                        │
│   ├── lint_brand_compliance                            │
│   └── generate_visual_prompt_spec                      │
└──────────────┬──────────────────────────┬──────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────────────┐ ┌───────────────────────┐
│     ChromaDB Vector Store    │ │      AWS Bedrock      │
│     (data/chroma_db)         │ │   (boto3 Runtime)     │
│                              │ │                       │
│  - solaris-fintech chunks    │ │ - Claude Sonnet       │
│  - velvet-apothecary chunks  │ │   (Converse API)      │
│  - lumina-automotive chunks  │ │ - Titan Embeddings V2 │
│  - aether-gaming chunks      │ │   (eu-north-1)        │
│  - 20 Chunks (1024-dim)      │ │                       │
└──────────────────────────────┘ └───────────────────────┘
```

---

## 3. Technology Stack & Key Libraries

| Component | Technology | Rationale |
|---|---|---|
| **Protocol** | Model Context Protocol (`mcp 1.x / FastMCP`) | Standardized open protocol by Anthropic allowing any AI desktop client or IDE to call local server tools. |
| **Cloud AI Runtime** | **AWS Bedrock** (`boto3`) | Leverages enterprise AWS credits, private cloud security, and serverless foundation models in European region (`eu-north-1`). |
| **Reasoning Model** | Anthropic Claude Sonnet (`eu.anthropic.claude-sonnet-4-6`) | Frontier reasoning with JSON schema enforcement for regulatory tone audits. |
| **Vector Embeddings** | Amazon Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`) | Native 1024-dimensional normalized vector embeddings on AWS Bedrock. |
| **Vector Database** | **ChromaDB** (`chromadb 1.5+`) | Embedded persistent SQLite/HNSW vector store for sub-millisecond similarity search. |
| **Data Validation** | **Pydantic v2** (`pydantic`) | Strict validation of JSON payloads returned to MCP clients. |
| **Automated Testing** | **pytest** & `pytest-asyncio` | 11 unit and integration tests covering vector indexing, mock LLMs, and guardrails. |
| **CI/CD & Hosting** | **GitHub Actions** & **GitHub Pages** | Automated deployment pipeline serving the live web landing page from `public/`. |

---

## 4. The 3 Production MCP Tools

### Tool 1: `retrieve_brand_guidelines`
- **Signature:** `retrieve_brand_guidelines(client_id: str, query: str, top_k: int = 3) -> str`
- **What it does:** Executes semantic similarity search in ChromaDB filtered by `client_id` (e.g. `solaris-fintech` vs. `velvet-apothecary`).
- **Output:** Structured Markdown containing exact voice dos/donts, color palettes with hex codes, typography, and channel-specific audience constraints.

### Tool 2: `lint_brand_compliance`
- **Signature:** `lint_brand_compliance(client_id: str, draft_text: str, channel: str = "social") -> str`
- **What it does:**
  1. Pulls client voice rules and regulatory boundaries from ChromaDB.
  2. Runs deterministic regex pattern matching for strictly banned words (`guaranteed`, `easy money`, `synthetic`, `chemical peel`, etc.).
  3. Dispatches Bedrock Converse API request to Claude Sonnet with auditor system prompt.
  4. Returns a strictly validated JSON payload:
  ```json
  {
    "score": 91,
    "status": "PASS",
    "forbidden_terms_found": [],
    "tone_drift_analysis": "The draft is clean, direct, and free of all forbidden terms. It aligns well with Solaris's minimalist, authoritative voice and correctly targets the institutional audience.",
    "suggested_rewrite": "Solaris engineers transparent, cryptographically verifiable execution algorithms purpose-built for institutional-grade quantitative infrastructure."
  }
  ```

### Tool 3: `generate_visual_prompt_spec`
- **Signature:** `generate_visual_prompt_spec(client_id: str, concept: str, platform: str = "instagram") -> str`
- **What it does:** Synthesizes client art direction, lighting temperatures, and material textures into diffusion-ready prompts for Midjourney / ComfyUI / Firefly / Flux.
- **Output:**
  - `positive_prompt`: High-craft visual prompt specifying materials, camera framing, film stock (e.g. Kodak Portra 400), and depth-of-field.
  - `negative_prompt`: Explicit exclusion rules based on brand identity.
  - `aspect_ratio`: Dynamically formatted (e.g. `4:5` for Instagram, `16:9` for LinkedIn).
  - `lighting_and_palette`: Exact Kelvin color temperature and brand hex codes.

---

## 5. Pre-Seeded Brand Case Studies (4 Archetypes)

The system includes 4 deeply detailed, contrasting client profiles in [`data/seed_brands.json`](file:///c:/Users/georg/georgiescoding/brandbrain-mcp/data/seed_brands.json) spanning divergent industries and regulatory environments:

### A. Solaris Fintech (`solaris-fintech`)
- **Domain:** Institutional Wealth & Algorithmic Quantitative Infrastructure.
- **Philosophy:** Minimal brutalism, quantitative transparency, institutional trust.
- **Strictly Banned Terms:** `guaranteed`, `moon`, `easy money`, `risk-free`, `get rich quick`, `100% returns`, `crypto bro`, `passive wealth`.
- **Palette:** Obsidian Deep (`#0B0F19`), High-Yield Cyan (`#06B6D4`), Pure Titanium (`#F8FAFC`).
- **Visuals:** Architectural daylight (5000K), brushed titanium, sapphire glass, telemetry waveforms.

### B. Velvet Apothecary (`velvet-apothecary`)
- **Domain:** Artisanal Botanical Rituals & Clean Beauty.
- **Philosophy:** Poetic, warm, sensory, grounded, organic restorative botanicals.
- **Strictly Banned Terms:** `synthetic`, `chemical peel`, `miracle cure`, `instant fix`, `clinical grade`, `bleach`, `flawless perfection`.
- **Palette:** Crushed Sage (`#708238`), Terracotta Ochre (`#C86D51`), Raw Linen (`#F4F0EA`), Sun-Bleached Amber (`#E8A858`).
- **Visuals:** Golden hour natural light (3200K), raw linen, deckle-edge paper, unglazed earthenware, shallow depth-of-field f/2.0.

### C. Lumina Automotive (`lumina-automotive`)
- **Domain:** Scandinavian Solid-State Electric Hypercars.
- **Philosophy:** Kinetic aerodynamics, wind-tunnel downforce, lightweight carbon monocoque.
- **Strictly Banned Terms:** `gas guzzler`, `cheap`, `affordable`, `family car`, `range anxiety`, `budget friendly`, `plastic interior`.
- **Palette:** Nordic Frost (`#E2E8F0`), Electric Cobalt (`#2563EB`), Forged Carbon (`#0F172A`), Kinetic Lime (`#84CC16`).
- **Visuals:** Nordic twilight (4000K), wet asphalt reflections, laser DRL lightbars, active aero rear wing.

### D. Aether Gaming (`aether-gaming`)
- **Domain:** Neural-Link AR & Tier-1 Esports Peripherals.
- **Philosophy:** Zero latency, 8000Hz polling reflex, cast magnesium honeycomb chassis.
- **Strictly Banned Terms:** `lag`, `casual`, `childish`, `slow`, `pay-to-win`, `toy`, `clunky`, `button masher`.
- **Palette:** Matte Carbon (`#09090B`), Quantum Magenta (`#E11D48`), Neon Cyan (`#00F0FF`), Void Violet (`#581C87`).
- **Visuals:** Esports arena tournament lighting, infrared optical sensor prism, haze smoke, neon rim accents.

---

## 6. Engineering Edge Cases & Solutions

### 1. Multi-Model Resilience & Regional Fallback in Bedrock
- **Challenge:** AWS Bedrock occasionally places newer foundation models (like Sonnet 5) behind waitlists or sales forms (`AccessDeniedException`), and models in Europe (`eu-north-1`) require regional inference profiles (`eu.anthropic.claude-sonnet-4-6`).
- **Solution:** Implemented an **intelligent model fallback resolver** in [`src/bedrock_client.py`](file:///c:/Users/georg/georgiescoding/brandbrain-mcp/src/bedrock_client.py). It attempts the primary model first, and if AWS reports an account restriction, it automatically routes to active production models without throwing an error to the MCP client.

### 2. Titan V2 Embeddings with Deterministic Local Fallback
- **Challenge:** When developing locally or running offline test suites without AWS credentials, vector queries would crash if Titan V2 was unavailable.
- **Solution:** Created a dual-mode embedding function in [`src/vector_store.py`](file:///c:/Users/georg/georgiescoding/brandbrain-mcp/src/vector_store.py). When AWS credentials are active, it uses 1024-dim Titan V2. When offline, it automatically falls back to a deterministic token-hashing embedder.

### 3. Deterministic Guardrails Hybridization
- **Challenge:** LLMs can occasionally overlook prohibited terms if the prompt is complex or stylized.
- **Solution:** Hybridized audit pipeline combining deterministic regex boundary detection with Claude Sonnet's contextual reasoning. Any banned term triggers an immediate score penalty (<60) and automatic `FAIL` status regardless of LLM score.

---

## 7. How to Talk About This in Interviews (Talking Points)

### Pitch (30 seconds):
> *"I built BrandBrain-MCP, an open-standard Model Context Protocol server that bridges generative AI workflows in IDEs with enterprise brand governance. It uses AWS Bedrock's Claude Sonnet and Titan Embeddings V2 with ChromaDB to perform semantic guideline retrieval, automated regulatory tone linting, and visual diffusion prompt engineering directly within developer environments like Claude Desktop and Cursor."*

### Key Technical Achievements to Highlight:
- **Model Context Protocol Implementation:** Engineered custom FastMCP server running over `stdio` transport with type-safe Pydantic response contracts.
- **Vector Retrieval-Augmented Generation (RAG):** Designed semantic chunking for brand identities, indexing 1024-dimensional Titan V2 vectors into persistent ChromaDB storage.
- **Cloud Architecture & Resilience:** Authenticated AWS Bedrock in `eu-north-1`, handling cross-region inference profiles and automated model failovers.
- **Full Production Lifecycle:** Wrote an 11-test automated suite (`pytest`), built an interactive web dashboard in Tailwind CSS, and deployed CI/CD pipelines via GitHub Actions.

---

## 8. Quick Verification Commands

```powershell
# 1. Run the automated test suite
pytest -v

# 2. Seed / Re-index ChromaDB with Titan Embeddings
python -m src.seed

# 3. Test direct MCP tool execution
python -c "from src.server import lint_brand_compliance; print(lint_brand_compliance('solaris-fintech', 'Solaris provides transparent, verifiable execution.'))"

# 4. Run the local interactive landing page
python -m http.server 8000 --directory public
```
