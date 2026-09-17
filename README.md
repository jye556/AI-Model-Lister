# AI Model Lister

A single-page web UI to **list, test, and compare AI models** across providers — OpenAI-compatible APIs (OpenAI, OpenRouter, Azure, Ollama, DeepSeek, Mistral, Groq, Together, NVIDIA NIM), xAI Grok, Google Gemini, and Anthropic Claude.

List models per provider, run batch tests with live token streaming, measure latency / time-to-first-token / token usage / estimated cost, and compare two providers' model availability side by side.

> API keys are only used for outbound requests and are **never** stored. Other settings are remembered in the browser's `localStorage`.

---

## Features

### Listing & testing
- Live model listing per provider, with metadata (display name, context window, owner, creation date)
- Single-model and **batch** testing — sequential or concurrent, with a configurable delay between calls
- **Multimodal / Vision testing** — attach images via file picker, image URL, or `Ctrl+V` clipboard paste directly into prompts; client-side downscaling prevents payload limits
- **Native Vision adapters** — automatic payload translation for Google Gemini (`inlineData`), Anthropic Claude (`image` content blocks), xAI Grok, and OpenAI-compatible (`image_url`) endpoints
- **Vision badges** — `👁️ Vision` badge automatically tags recognized multimodal models in listings and comparisons
- **Live streaming** — tokens render in the UI as they arrive, with time-to-first-token (TTFT) measured
- **Token usage** (prompt / completion) captured per test, with **estimated cost** from an editable per-model pricing table
- **Configurable generation parameters** — system prompt, max tokens, temperature
- **Temperature auto-fallback** — reasoning models (e.g. `o1`/`o3`/`o4`) that reject non-`1` temperatures are automatically retried without the field
- **Retry Failed** button to re-run just the errored models
- Click any row to **expand** the full response, tokens, TTFT, and cost
- **Batch summary** cards — success rate, min/avg/max latency, output tokens, tokens/sec, est. cost

### Comparison & export
- Search/filter, sortable results, side-by-side comparison of selected models
- **Two-provider availability comparison** — shared, only-A, only-B by exact model ID
- **CSV / JSON export** of test results (includes token counts and cost)

### UX
- Light/dark theme toggle, toast notifications, `Ctrl+Enter` to run a test
- Checked-model selection remembered per provider in `localStorage`
- Prompt presets and settings remembered in `localStorage`
- **Settings tab** — edit and save `DEFAULT_BASE_URL` and `DEFAULT_PROVIDER` directly to `.env` from the web UI
- Extra custom headers (e.g. OpenRouter's `HTTP-Referer`) and a configurable request timeout

### Operations
- `/health` endpoint + Docker `HEALTHCHECK`
- In-app **self-update** — detect a newer version on GitHub and pull + restart with one click

---

## Quick start

### Install and Run with Docker (Recommended)

#### 1. Clone the repository
```bash
git clone https://github.com/jye556/AI-Model-Lister.git
cd AI-Model-Lister
```

#### 2. Configure environment (Optional)
```bash
cp .env.example .env
```
*(You can customize `.env` now, or update settings later directly from the in-app **Settings** tab).*

#### 3. Start with Docker Compose
```bash
# Build locally and start the container in the background:
docker compose up -d --build
```

Or to pull the pre-built image from GitHub Container Registry:
```bash
docker compose pull && docker compose up -d
```

#### Alternative: Run with standalone Docker
```bash
# Build the image
docker build -t model-lister .

# Run the container
docker run -d \
  --name model-lister \
  -p 2463:2463 \
  --restart unless-stopped \
  model-lister
```

#### 4. Access the web app
Open **http://localhost:2463** (or `http://<your-server-ip>:2463`) in your browser.

---

### Run locally with Python

1. Clone and enter the repository:
   ```bash
   git clone https://github.com/jye556/AI-Model-Lister.git
   cd AI-Model-Lister
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the server:
   ```bash
   python app.py
   ```
4. Open **http://localhost:2463**

---

## Configuration

All settings are optional environment variables (can also be configured via `.env` or copied from `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `DEFAULT_BASE_URL` | `https://api.openai.com/v1` | Default OpenAI-compatible endpoint |
| `DEFAULT_PROVIDER` | `openai` | Provider selected on first load |
| `GITHUB_REPO` | `jye556/AI-Model-Lister` | Repo checked for self-update `version.txt` |
| `UPDATE_BRANCH` | `main` | Branch to pull when updating |
| `RESTART_CMD` | *(unset)* | Shell command to restart after update (set under gunicorn/Docker/supervisor) |

Examples:

```bash
# python app.py (dev — self-restart needs no RESTART_CMD)
DEFAULT_BASE_URL=https://api.openai.com/v1 python app.py

# gunicorn / supervisor
GITHUB_REPO=jye556/AI-Model-Lister RESTART_CMD="systemctl restart model-lister" \
    gunicorn --workers 2 --bind 0.0.0.0:2463 app:app

# Docker
docker run -p 2463:2463 -e GITHUB_REPO=jye556/AI-Model-Lister model-lister
```

---

## Self-update (push to GitHub → pull in the app)

1. **Bump `version.txt`** at the repo root (single source of truth — `app.py` reads it).
2. Commit and push to the `main` branch:
   ```bash
   git add version.txt app.py templates/index.html
   git commit -m "Release v3.1"
   git push origin main
   ```
3. The running app calls `GET /check-update` on page load, comparing the local `version.txt` against `https://raw.githubusercontent.com/jye556/AI-Model-Lister/main/version.txt`. If a newer version is found, a banner appears: **New version vX available (current vY)**.
4. Press **Update** → the app automatically pulls the latest code and reloads:
   - In a **git checkout**: runs `git fetch` + `git reset --hard origin/main`.
   - In **Docker**: downloads the latest archive directly from GitHub, updates `/app`, and hot-reloads Gunicorn workers with zero downtime.
   - **Dismiss** hides the banner for that version until the next one ships.

Notes:
- Works out of the box both in **Docker** and from a **git checkout**.
- Local configuration (`.env`) is protected and never overwritten during updates.
- Set `RESTART_CMD` if a custom process supervisor manages the app.

---

## HTTP API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | The web UI |
| `GET` / `POST` | `/api/settings` | Get or update `.env` configuration |
| `POST` | `/list-models` | List models for a provider |
| `POST` | `/test-model` | Run one model test (returns response + TTFT + usage as JSON) |
| `POST` | `/test-model-stream` | Live-stream a model test as SSE (`ttft` / `delta` / `usage` / `retry` / `done`) |
| `GET` | `/check-update` | Compare local version against GitHub |
| `POST` | `/update` | Pull latest and restart (or return Docker guidance) |
| `GET` | `/health` | `{status, version}` |

---

## Provider notes

- **OpenAI-compatible** (OpenAI, OpenRouter, Azure, Ollama, DeepSeek, Mistral, Groq, Together, NVIDIA NIM) — `/models` and `/chat/completions`.
- **Azure** — set the base URL to `https://YOUR-RESOURCE.openai.azure.com/openai/v1` and use an Azure API key.
- **Ollama** — no API key; default `http://localhost:11434/v1`.
- **xAI Grok** — test calls use the `/responses` API.
- **Google Gemini** — auth via the `x-goog-api-key` header (never a `?key=` query param); context window surfaced from `inputTokenLimit`.
- **Anthropic Claude** — lists models from `/v1/models`, falling back to a static list if it is unreachable.
- **NVIDIA NIM** — `https://integrate.api.nvidia.com/v1` with a Bearer `nvapi-...` key.
- **DeepSeek, Mistral, Groq, Together** — pre-configured OpenAI-compatible providers with their own entries (each uses a Bearer API key).
- **AWS Bedrock** is not supported — it requires SigV4 signing (access key + secret + region) rather than a single API key.

---

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers provider dispatch, auth/header handling, streaming TTFT + usage, temperature auto-fallback, multimodal/vision payload adaptation (OpenAI, Claude, Gemini), the SSE endpoint, and the self-update check/guidance paths.

---

## Tech stack

- **Backend:** Python, Flask, requests, gunicorn
- **Frontend:** single-file HTML/CSS/vanilla JS (no build step)
- **Container:** Docker (Python 3.11-slim)

## License

This project is provided as-is for personal/internal use.
