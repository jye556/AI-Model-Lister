# AI Model Lister

Simple web UI to list and test AI models across providers: OpenAI-compatible APIs
(OpenAI, OpenRouter, Azure, Ollama, DeepSeek, Mistral, Groq, Together, NVIDIA NIM),
xAI Grok, Google Gemini, and Anthropic Claude.

## Features

- Live model listing per provider (with metadata such as display name, context window, owner)
- Single and batch model testing — sequential or concurrent, with a configurable delay between calls
- Optional streaming mode that measures time-to-first-token (TTFT) **and renders tokens live in the UI**
- Token usage (prompt/completion) captured per test, with **estimated cost** from an editable per-model pricing table
- Configurable generation parameters (system prompt, max tokens, temperature)
- Retry-failed button to re-run just the errored models
- Click any row to expand and view the full response, tokens, TTFT, and cost
- Batch summary stats after a run (success rate, min/avg/max latency, output tokens, tokens/sec, est. cost)
- Search/filter, sortable results, side-by-side comparison of selected models
- Two-provider model availability comparison (shared, only A, only B) by exact model ID
- CSV / JSON export of test results (includes token counts and cost)
- Light/dark theme toggle, toast notifications, `Ctrl+Enter` to run a test
- Checked-model selection remembered per provider in localStorage
- Prompt presets and settings remembered in localStorage (the API key is never stored)
- Extra custom headers (e.g. OpenRouter's `HTTP-Referer`) and a configurable request timeout
- `/health` endpoint and Docker `HEALTHCHECK`

## Run with Docker

```bash
docker build -t model-lister .
docker run -p 2463:2463 model-lister

# keep a custom default OpenAI-compatible endpoint:
docker run -p 2463:2463 -e DEFAULT_BASE_URL=https://sub2api.midah.my/v1 model-lister
```

Then open http://localhost:2463

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Deploying updates (push to GitHub / pull from GitHub)

The app can detect when a newer version is published on GitHub and self-update.

### 1. Push a new version

```bash
# bump version.txt (single source of truth — app.py reads it)
# echo "3.1" > version.txt   (or edit it)
git add version.txt app.py templates/index.html
git commit -m "Release v3.1"
git push origin main
```

### 2. Configure the running app

Set these env vars (on the host running the app):

| Variable | Required | Example | Purpose |
|---|---|---|---|
| `GITHUB_REPO` | yes | `JYENB/model-lister` | Where the app checks `version.txt` for a newer version |
| `UPDATE_BRANCH` | no (default `main`) | `main` | Branch to pull from |
| `RESTART_CMD` | no | `systemctl restart model-lister` | Shell command to restart after update (set under gunicorn/Docker/supervisor) |

```bash
# python app.py (dev / self-restart — no RESTART_CMD needed)
GITHUB_REPO=JYENB/model-lister python app.py

# gunicorn / supervisor (set RESTART_CMD)
GITHUB_REPO=JYENB/model-lister RESTART_CMD="systemctl restart model-lister" \
    gunicorn --workers 2 --bind 0.0.0.0:2463 app:app

# Docker: build & push the image, then on the host:
#   docker compose pull && docker compose up -d
# In-container `git pull` does not apply (code is COPYed in); the Update button
# instead shows the host-side docker commands to run.
```

### 3. In-app update

On page load the app calls `GET /check-update`, comparing the local `version.txt`
against `https://raw.githubusercontent.com/<GITHUB_REPO>/<UPDATE_BRANCH>/version.txt`.
If a newer version is found, a banner appears: **New version vX available (current vY)**.
Press **Update** → the app runs `git fetch` + `git reset --hard origin/<branch>`,
restarts itself, and the page auto-reloads when it is back up.
Dismiss hides the banner for that version until the next one.

Endpoints:

- `GET /check-update` → `{configured, has_update, current, remote, repo, branch}`
- `POST /update` → pulls and restarts (or returns Docker host guidance if no git checkout)
- `GET /health` → `{status: ok, version}` (used by the UI to detect the restart)

## Provider notes

- **Azure** uses the OpenAI-compatible `/openai/v1` paths: set the base URL to
  `https://YOUR-RESOURCE.openai.azure.com/openai/v1` and use an Azure API key.
- **Ollama** needs no API key; the default base URL is `http://localhost:11434/v1`.
- **xAI** test calls use the `/responses` API.
- **Claude** lists models from Anthropic's `/v1/models` endpoint, falling back to a
  static list if it is unreachable.
- **NVIDIA NIM** uses the OpenAI-compatible endpoint at `https://integrate.api.nvidia.com/v1`
  with a Bearer `nvapi-...` API key.
- **DeepSeek, Mistral, Groq, Together** are pre-configured OpenAI-compatible providers with
  their own provider entries (each uses a Bearer API key).
- **AWS Bedrock is not supported**: it requires SigV4 signing with an access key,
  secret, and region rather than a single API key.
>>>>>>> dfbb523 (Initial commit)
