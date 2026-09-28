# BrandBrain-MCP

> **Production Model Context Protocol (MCP) Server for Creative & Brand Operations**
> Powered by **AWS Bedrock** (Moonshot AI Kimi K3 `moonshotai.kimi-k3` & Amazon Titan Text Embeddings V2) and **ChromaDB**.

BrandBrain connects IDEs and desktop LLM clients (Antigravity, Claude Desktop, Cursor) to persistent brand knowledge retrieval, automated tone evaluation & regulatory compliance auditing, and visual diffusion prompt synthesis.

---

## Architecture Overview

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
│  - solaris-fintech chunks    │ │ - Moonshot AI Kimi K3 │
│  - velvet-apothecary chunks  │ │   (Converse API)      │
│  - Titan V2 / Local Embedder │ │ - Titan Embeddings V2 │
└──────────────────────────────┘ └───────────────────────┘
```

---

## Features & MCP Tools

### 1. `retrieve_brand_guidelines(client_id: str, query: str, top_k: int = 3) -> str`
- Performs vector similarity search across ChromaDB to retrieve brand identity, voice & tone rules, color palettes, audience constraints, and regulatory guardrails.
- Supports scoped filtering by `client_id` (e.g. `solaris-fintech`, `velvet-apothecary`).

### 2. `lint_brand_compliance(client_id: str, draft_text: str, channel: str = "social") -> str`
- Audits draft copy against retrieved brand guidelines.
- Calls Moonshot AI Kimi K3 (`moonshotai.kimi-k3`) via Bedrock Converse API with strict system directives and deterministic banned phrase checks.
- Returns a structured JSON payload:
  ```json
  {
    "score": 42,
    "status": "FAIL",
    "forbidden_terms_found": ["guaranteed", "easy money"],
    "tone_drift_analysis": "Draft contains aggressive financial claims and retail FOMO language directly violating Solaris institutional authority.",
    "suggested_rewrite": "Solaris algorithmic infrastructure provides institutional-grade risk controls and transparent settlement mechanisms."
  }
  ```

### 3. `generate_visual_prompt_spec(client_id: str, concept: str, platform: str = "instagram") -> str`
- Translates brand artistic direction, lighting standards, material textures, and color hexes into high-craft diffusion prompts (Midjourney v6, ComfyUI, Firefly, Flux).
- Returns a structured JSON specification:
  ```json
  {
    "positive_prompt": "Wabi-sabi herbal sanctuary capturing evening restorative serum ritual, amber glass apothecary dropper bottle, tactile crushed sage leaves with fresh dew droplets, unglazed terracotta earthenware, golden hour diffused sunlight through raw linen curtains, soft film grain Kodak Portra 400 warmth, f/1.8 macro",
    "negative_prompt": "sterile lab, plastic bottles, neon colors, harsh flash, artificial skin smoothing, CGI render",
    "aspect_ratio": "4:5",
    "lighting_and_palette": "Golden hour diffused sunlight (3200K); Crushed Sage (#708238), Terracotta Ochre (#C86D51), Raw Linen (#F4F0EA)"
  }
  ```

---

## Pre-Seeded Brand Profiles

| Client ID | Brand Name | Voice & Tone | Strictly Forbidden Words | Visual Style |
|---|---|---|---|---|
| `solaris-fintech` | Solaris Fintech | Minimalist, authoritative, transparent, institutional trust | `guaranteed`, `moon`, `easy money`, `risk-free`, `get rich quick`, `100% returns` | Minimal brutalism, brushed titanium, sapphire glass, Obsidian Deep `#0B0F19`, High-Yield Cyan `#06B6D4` |
| `velvet-apothecary` | Velvet Apothecary | Poetic, warm, sensory, grounded, reverent | `synthetic`, `chemical peel`, `miracle cure`, `instant fix`, `clinical grade`, `bleach` | Wabi-sabi herbal sanctuary, unglazed ceramic, amber glass, crushed sage `#708238`, terracotta `#C86D51` |

---

## Quickstart & Installation

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/your-org/brandbrain-mcp.git
cd brandbrain-mcp

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure AWS Bedrock Credentials

Copy `.env.example` to `.env` and provide your AWS credentials:

```bash
cp .env.example .env
```

```ini
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=us-east-1

BEDROCK_MODEL_ID=moonshotai.kimi-k3
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
CHROMA_PERSIST_DIR=./data/chroma_db
EMBEDDING_FALLBACK_LOCAL=true
```

> **Note:** If your AWS credentials lack Titan Embeddings V2 access or you are testing offline, BrandBrain automatically falls back to deterministic local embeddings so the vector store remains fully functional!

### 3. Seed ChromaDB Knowledge Base

Populate ChromaDB with the client brand guidelines from `data/seed_brands.json`:

```bash
python -m src.seed
```

Output:
```text
Indexed 10 guideline chunks into ChromaDB at ./data/chroma_db
BrandBrain seeding complete! Indexed 10 guideline documents.
```

---

## Client Integration

### Antigravity / Cursor Configuration

Add BrandBrain to your MCP settings (`mcp_config.json`):

```json
{
  "mcpServers": {
    "brandbrain-mcp": {
      "command": "python",
      "args": ["-m", "src.server"],
      "env": {
        "AWS_ACCESS_KEY_ID": "YOUR_KEY",
        "AWS_SECRET_ACCESS_KEY": "YOUR_SECRET",
        "AWS_DEFAULT_REGION": "us-east-1"
      }
    }
  }
}
```

### Claude Desktop Configuration

Add the server to your `claude_desktop_config.json` (on macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`, on Windows: `%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "brandbrain": {
      "command": "C:\\Users\\georg\\georgiescoding\\brandbrain-mcp\\.venv\\Scripts\\python.exe",
      "args": ["-m", "src.server"],
      "cwd": "C:\\Users\\georg\\georgiescoding\\brandbrain-mcp",
      "env": {
        "AWS_ACCESS_KEY_ID": "YOUR_KEY",
        "AWS_SECRET_ACCESS_KEY": "YOUR_SECRET",
        "AWS_DEFAULT_REGION": "us-east-1"
      }
    }
  }
}
```

---

## Running the Automated Test Suite

Run pytest to validate tool functionality, Bedrock integration, semantic chunking, and deterministic guardrails:

```bash
pytest -v
```

All tests run in isolated temporary vector collections with simulated Bedrock responses and local fallback tests.

---

## Interactive Studio & Landing Page

BrandBrain includes a web landing page and interactive creative operations studio located in [`public/index.html`](file:///c:/Users/georg/georgiescoding/brandbrain-mcp/public/index.html):
- **Live Tone Compliance Linter:** Test copy against brand rules with real-time scoring, forbidden term detection, and rewrites.
- **Visual Diffusion Spec Studio:** Generate ready-to-use Midjourney/ComfyUI prompts with aspect ratio switchers and brand palette swatches.
- **Vector Retrieval Inspector:** Query ChromaDB brand rules in real-time.

To open the landing page in your browser:
```bash
# Start local static server
python -m http.server 8000 --directory public
```
Then navigate to `http://localhost:8000`.

---

## Project Structure

```
brandbrain-mcp/
├── pyproject.toml              # Build & package specifications
├── requirements.txt            # Production dependencies
├── .env.example                # AWS Bedrock & ChromaDB env configuration template
├── README.md                   # Comprehensive guide and architecture documentation
├── mcp_config.json             # MCP client configuration template
├── public/
│   └── index.html              # Interactive web landing page & operations studio
├── data/
│   └── seed_brands.json        # Deep brand guidelines for Solaris and Velvet
├── src/
│   ├── __init__.py
│   ├── server.py               # FastMCP server with the 3 creative operations tools
│   ├── bedrock_client.py       # AWS Bedrock Claude Sonnet Converse & Titan client
│   ├── vector_store.py         # ChromaDB indexing, semantic search, and Titan/local fallback
│   └── seed.py                 # Semantic chunking and vector database seeding
└── tests/
    ├── __init__.py
    └── test_tools.py           # Automated unit and integration test suite
```
