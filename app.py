import json
import os
import subprocess
import sys
import threading
import time

import requests
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)


def _read_version():
    """Read the app version from version.txt (single source of truth, also used remotely)."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'version.txt')) as f:
            v = f.read().strip()
        if v:
            return v
    except OSError:
        pass
    return '3.0'


# App version shown in the UI header. Bump version.txt when the UI/API is enhanced.
VERSION = _read_version()

# Default OpenAI-compatible endpoint. Override with the DEFAULT_BASE_URL env var.
DEFAULT_BASE_URL = os.environ.get('DEFAULT_BASE_URL', 'https://sub2api.midah.my/v1')
DEFAULT_PROVIDER = os.environ.get('DEFAULT_PROVIDER', 'openai')

# Self-update configuration.
# GITHUB_REPO: "owner/repo" — where the app checks for a newer version.txt.
# UPDATE_BRANCH: branch to pull from (default "main").
# RESTART_CMD: optional shell command used to restart after an update
#   (set this under gunicorn/Docker/supervisor; if unset, the dev server self-restarts).
GITHUB_REPO = os.environ.get('GITHUB_REPO', '').strip()
UPDATE_BRANCH = (os.environ.get('UPDATE_BRANCH', 'main').strip() or 'main')
RESTART_CMD = os.environ.get('RESTART_CMD', '').strip()

PROVIDER_BASE_URLS = {
    'openai': DEFAULT_BASE_URL,
    'openrouter': 'https://openrouter.ai/api/v1',
    'azure': '',  # Azure has no shared default; a custom base URL is required
    'ollama': 'http://localhost:11434/v1',
    'deepseek': 'https://api.deepseek.com/v1',
    'mistral': 'https://api.mistral.ai/v1',
    'groq': 'https://api.groq.com/openai/v1',
    'together': 'https://api.together.xyz/v1',
    'xai': 'https://api.x.ai/v1',
    'gemini': 'https://generativelanguage.googleapis.com/v1beta',
    'claude': 'https://api.anthropic.com/v1',
    'nvidia': 'https://integrate.api.nvidia.com/v1',
}

# Fallback used when the Anthropic /v1/models endpoint cannot be reached
CLAUDE_MODELS = [
    'claude-fable-5',
    'claude-opus-5',
    'claude-sonnet-5',
    'claude-haiku-4-5-20251001',
    'claude-3-5-haiku-20241022',
]


def extract_xai_response_text(data):
    """Extract text from an xAI Responses API payload."""
    output = data.get('output', [])
    text_parts = []

    for item in output:
        for content in item.get('content', []):
            if content.get('type') == 'output_text' and content.get('text'):
                text_parts.append(content['text'])

    if text_parts:
        return "\n".join(text_parts)

    return data.get('output_text', '')


def get_base_url(provider, custom_base_url):
    """Return the correct base URL based on provider and custom input"""
    if custom_base_url:
        return custom_base_url.rstrip('/')
    return PROVIDER_BASE_URLS.get(provider, DEFAULT_BASE_URL)


def auth_headers(provider, api_key):
    """Provider-specific authentication headers"""
    if provider == 'gemini':
        return {'x-goog-api-key': api_key}
    if provider == 'claude':
        return {'x-api-key': api_key, 'anthropic-version': '2023-06-01'}
    if provider == 'azure':
        return {'api-key': api_key}
    if not api_key:
        return {}
    return {'Authorization': f'Bearer {api_key}'}


def parse_extra_headers(raw):
    """Parse extra headers supplied by the client (dict or JSON object string)"""
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except ValueError:
            pass
    return {}


def get_timeout(data):
    """Per-request timeout in seconds, clamped to [1, 600]"""
    try:
        value = float(data.get('timeout', 180))
    except (TypeError, ValueError):
        return 180
    return max(1.0, min(600.0, value))


def parse_gen_params(data):
    """Generation parameters sent from the client, clamped to safe ranges."""
    try:
        max_tokens = int(data.get('max_tokens', 300))
    except (TypeError, ValueError):
        max_tokens = 300
    max_tokens = max(1, min(100000, max_tokens))
    try:
        temperature = float(data.get('temperature', 0.7))
    except (TypeError, ValueError):
        temperature = 0.7
    temperature = max(0.0, min(2.0, temperature))
    system = (data.get('system') or '').strip()
    return {'max_tokens': max_tokens, 'temperature': temperature, 'system': system}


def normalize_usage_openai(u):
    if not isinstance(u, dict):
        return None
    return {
        'prompt': u.get('prompt_tokens', 0),
        'completion': u.get('completion_tokens', 0),
        'total': u.get('total_tokens', 0),
    }


def normalize_usage_claude(u):
    if not isinstance(u, dict):
        return None
    inp = u.get('input_tokens', 0) or 0
    out = u.get('output_tokens', 0) or 0
    return {'prompt': inp, 'completion': out, 'total': u.get('total_tokens', inp + out)}


def normalize_usage_gemini(u):
    if not isinstance(u, dict):
        return None
    return {
        'prompt': u.get('promptTokenCount', 0),
        'completion': u.get('candidatesTokenCount', 0),
        'total': u.get('totalTokenCount', 0),
    }


def normalize_usage_xai(u):
    if not isinstance(u, dict):
        return None
    return {
        'prompt': u.get('input_tokens', 0),
        'completion': u.get('output_tokens', 0),
        'total': u.get('total_tokens', 0),
    }


def build_payload(provider, model, prompt, params, include_temperature=True):
    """Build the provider-specific request payload from common params.

    When include_temperature is False the temperature field is omitted entirely
    (used as a fallback for models that reject non-1 temperatures, e.g. o-series).
    """
    max_tokens = params['max_tokens']
    temperature = params['temperature']
    system = params['system']

    if provider == 'gemini':
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        gen_cfg = {"maxOutputTokens": max_tokens}
        if include_temperature:
            gen_cfg["temperature"] = temperature
        payload["generationConfig"] = gen_cfg
        return payload

    if provider == 'claude':
        payload = {
            'model': model,
            'max_tokens': max_tokens,
            'messages': [{'role': 'user', 'content': prompt}],
        }
        if include_temperature:
            payload['temperature'] = temperature
        if system:
            payload['system'] = system
        return payload

    if provider == 'xai':
        payload = {
            'model': model,
            'reasoning': {'effort': 'low'},
            'input': prompt,
            'max_output_tokens': max_tokens,
        }
        if include_temperature:
            payload['temperature'] = temperature
        if system:
            payload['instructions'] = system
        return payload

    # OpenAI-compatible (OpenAI, OpenRouter, Azure, Ollama, DeepSeek, Mistral, Groq, Together, NVIDIA)
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': prompt})
    payload = {
        'model': model,
        'messages': messages,
        'max_tokens': max_tokens,
    }
    if include_temperature:
        payload['temperature'] = temperature
    return payload


def is_temperature_error(e):
    """True when a 400 response indicates the temperature field is invalid for this model."""
    resp = getattr(e, 'response', None)
    if resp is None or resp.status_code != 400:
        return False
    try:
        body = resp.json()
    except ValueError:
        body = None
    text = json.dumps(body).lower() if body is not None else ''
    return 'temperature' in text


def dispatch_test(provider, model, prompt, base_url, headers, timeout, params, stream,
                  include_temperature=True):
    """Run a single test attempt. Returns (content, ttft, usage)."""
    payload = build_payload(provider, model, prompt, params, include_temperature)

    if provider == 'gemini':
        if stream:
            return consume_stream(stream_gemini(
                f"{base_url}/models/{model}:streamGenerateContent?alt=sse",
                headers, payload, timeout))
        resp = requests.post(f"{base_url}/models/{model}:generateContent",
                             headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        return (body['candidates'][0]['content']['parts'][0]['text'], None,
                normalize_usage_gemini(body.get('usageMetadata')))

    if provider == 'claude':
        if stream:
            return consume_stream(stream_claude(base_url, headers, payload, timeout))
        resp = requests.post(f'{base_url}/messages', headers=headers,
                             json=payload, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        return (body['content'][0]['text'], None, normalize_usage_claude(body.get('usage')))

    if provider == 'xai':
        if stream:
            content, ttft, usage = consume_stream(stream_xai(base_url, headers, payload, timeout))
        else:
            resp = requests.post(f'{base_url}/responses', headers=headers,
                                 json=payload, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
            content, ttft, usage = extract_xai_response_text(body), None, normalize_usage_xai(body.get('usage'))
        if not content:
            raise ValueError('xAI response did not include any text output')
        return content, ttft, usage

    # OpenAI-compatible
    if stream:
        return consume_stream(stream_openai_chat(base_url, headers, payload, timeout))
    resp = requests.post(f'{base_url}/chat/completions', headers=headers,
                         json=payload, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    return (body['choices'][0]['message']['content'], None,
            normalize_usage_openai(body.get('usage')))


def run_model_test(provider, model, prompt, base_url, headers, timeout, params, stream):
    """Run a model test with auto-fallback: on a temperature 400, retry without it."""
    try:
        return dispatch_test(provider, model, prompt, base_url, headers, timeout, params, stream)
    except requests.exceptions.RequestException as e:
        if is_temperature_error(e):
            return dispatch_test(provider, model, prompt, base_url, headers, timeout, params, stream,
                                 include_temperature=False)
        raise


def error_detail(e):
    """Best-effort extraction of the provider's own error message"""
    resp = getattr(e, 'response', None)
    if resp is not None:
        try:
            body = resp.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            err = body.get('error', body)
            if isinstance(err, dict):
                msg = err.get('message') or err.get('code') or err.get('type')
                if msg:
                    return f'HTTP {resp.status_code}: {msg}'
            elif isinstance(err, str) and err:
                return f'HTTP {resp.status_code}: {err}'
        return f'HTTP {resp.status_code}: {getattr(resp, "reason", "error")}'
    return str(e)


def iter_sse(resp):
    """Yield parsed JSON payloads from an SSE stream"""
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith('data:'):
            continue
        payload = line[5:].strip()
        if payload == '[DONE]':
            break
        try:
            yield json.loads(payload)
        except ValueError:
            continue


def stream_openai_chat(base_url, headers, payload, timeout):
    """Yield events from an OpenAI-compatible streaming chat completion."""
    payload = {**payload, 'stream': True, 'stream_options': {'include_usage': True}}
    ttft = None
    start = time.monotonic()
    with requests.post(f'{base_url}/chat/completions', headers=headers, json=payload,
                       timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        for chunk in iter_sse(resp):
            choices = chunk.get('choices') or [{}]
            content = (choices[0].get('delta') or {}).get('content')
            if content:
                if ttft is None:
                    ttft = time.monotonic() - start
                    yield {'type': 'ttft', 'ttft': ttft}
                yield {'type': 'delta', 'text': content}
            usage = chunk.get('usage')
            if usage:
                yield {'type': 'usage', 'usage': normalize_usage_openai(usage)}
    yield {'type': 'done'}


def stream_claude(base_url, headers, payload, timeout):
    """Yield events from a streaming Claude messages response."""
    payload = {**payload, 'stream': True}
    ttft = None
    start = time.monotonic()
    with requests.post(f'{base_url}/messages', headers=headers, json=payload,
                       timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        for event in iter_sse(resp):
            t = event.get('type')
            if t == 'content_block_delta':
                delta = event.get('delta') or {}
                if delta.get('type') == 'text_delta' and delta.get('text'):
                    if ttft is None:
                        ttft = time.monotonic() - start
                        yield {'type': 'ttft', 'ttft': ttft}
                    yield {'type': 'delta', 'text': delta['text']}
            elif t == 'message_start':
                u = (event.get('message') or {}).get('usage')
                if u:
                    yield {'type': 'usage', 'usage': normalize_usage_claude(u)}
            elif t == 'message_delta':
                u = event.get('usage')
                if u:
                    yield {'type': 'usage', 'usage': normalize_usage_claude(u)}
    yield {'type': 'done'}


def stream_gemini(url, headers, payload, timeout):
    """Yield events from a streaming Gemini generateContent response."""
    ttft = None
    start = time.monotonic()
    with requests.post(url, headers=headers, json=payload,
                       timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        for chunk in iter_sse(resp):
            candidates = chunk.get('candidates') or [{}]
            parts = (candidates[0].get('content') or {}).get('parts') or []
            for part in parts:
                if part.get('text'):
                    if ttft is None:
                        ttft = time.monotonic() - start
                        yield {'type': 'ttft', 'ttft': ttft}
                    yield {'type': 'delta', 'text': part['text']}
            um = chunk.get('usageMetadata')
            if um:
                yield {'type': 'usage', 'usage': normalize_usage_gemini(um)}
    yield {'type': 'done'}


def stream_xai(base_url, headers, payload, timeout):
    """Yield events from a streaming xAI Responses API call."""
    payload = {**payload, 'stream': True}
    ttft = None
    start = time.monotonic()
    with requests.post(f'{base_url}/responses', headers=headers, json=payload,
                       timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        for event in iter_sse(resp):
            t = event.get('type')
            if t == 'response.output_text.delta' and event.get('delta'):
                if ttft is None:
                    ttft = time.monotonic() - start
                    yield {'type': 'ttft', 'ttft': ttft}
                yield {'type': 'delta', 'text': event['delta']}
            elif t == 'response.completed':
                r = event.get('response') or {}
                u = r.get('usage')
                if u:
                    yield {'type': 'usage', 'usage': normalize_usage_xai(u)}
                if not (r.get('output') or r.get('output_text')):
                    # No streamed deltas; fall back to the completed payload.
                    txt = extract_xai_response_text(r)
                    if txt:
                        if ttft is None:
                            ttft = time.monotonic() - start
                            yield {'type': 'ttft', 'ttft': ttft}
                        yield {'type': 'delta', 'text': txt}
    yield {'type': 'done'}


def consume_stream(gen):
    """Aggregate a stream generator into (text, ttft, usage)."""
    parts = []
    ttft = None
    usage = None
    for ev in gen:
        t = ev.get('type')
        if t == 'delta':
            parts.append(ev['text'])
        elif t == 'ttft':
            ttft = ev['ttft']
        elif t == 'usage':
            u = ev.get('usage')
            if u:
                usage = {**(usage or {}), **u}
                usage['total'] = (usage.get('prompt', 0) or 0) + (usage.get('completion', 0) or 0)
    return ''.join(parts), ttft, usage


@app.route('/')
def index():
    return render_template('index.html', default_base_url=DEFAULT_BASE_URL,
                           default_provider=DEFAULT_PROVIDER, version=VERSION)


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'version': VERSION})


# ---------------- self-update ----------------

def _repo_root():
    """Return the git work-tree root containing this app, or None if not a checkout."""
    d = os.path.dirname(os.path.abspath(__file__))
    try:
        r = subprocess.run(['git', '-c', 'safe.directory=*', 'rev-parse', '--show-toplevel'],
                           cwd=d, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def _git(args, cwd, timeout=120):
    return subprocess.run(['git', '-c', 'safe.directory=*'] + args,
                           cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _do_restart():
    """Restart the app after a successful update. Detached, then exit this process."""
    target = os.path.abspath(__file__)
    if os.name == 'nt':
        # Wait for this process to release the port, then relaunch.
        cmd = f'timeout /t 2 /nobreak >nul & "{sys.executable}" "{target}"'
        subprocess.Popen(cmd, shell=True,
                         creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        cmd = f'sleep 2 && exec "{sys.executable}" "{target}"'
        subprocess.Popen(['bash', '-c', cmd], start_new_session=True)
    os._exit(0)


@app.route('/check-update')
def check_update():
    """Compare the local VERSION against version.txt on the configured GitHub branch."""
    if not GITHUB_REPO:
        return jsonify({'configured': False})
    url = f'https://raw.githubusercontent.com/{GITHUB_REPO}/{UPDATE_BRANCH}/version.txt'
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return jsonify({'configured': True, 'has_update': False,
                            'current': VERSION, 'error': f'HTTP {resp.status_code}'})
        remote = resp.text.strip()
        has_update = bool(remote) and remote != VERSION
        return jsonify({
            'configured': True, 'has_update': has_update,
            'current': VERSION, 'remote': remote,
            'repo': GITHUB_REPO, 'branch': UPDATE_BRANCH,
        })
    except requests.exceptions.RequestException as e:
        return jsonify({'configured': True, 'has_update': False,
                        'current': VERSION, 'error': str(e)})


@app.route('/update', methods=['POST'])
def update_app():
    """Pull the latest code from GitHub and restart. Works for a git checkout;
    for a Docker image (no git) it returns host-side guidance instead."""
    if not GITHUB_REPO:
        return jsonify({'updated': False,
                        'message': 'GITHUB_REPO is not configured; cannot update.'}), 400

    root = _repo_root()
    if not root:
        # No git available (typically a Docker image with COPYed code).
        if RESTART_CMD:
            threading.Thread(target=_do_restart, daemon=True).start()
            return jsonify({'updated': True, 'message': 'Running RESTART_CMD to apply the image update...'})
        return jsonify({'updated': False, 'docker': True,
                        'message': 'Running outside a git checkout (likely a Docker image). '
                                   'Update on the host with: docker compose pull && docker compose up -d '
                                   '(or: docker pull <image> && docker restart <container>).'})

    fetch = _git(['fetch', 'origin', UPDATE_BRANCH], root)
    if fetch.returncode != 0:
        return jsonify({'updated': False,
                        'message': f'git fetch failed: {fetch.stderr.strip() or fetch.stdout.strip()}'}), 500
    reset = _git(['reset', '--hard', f'origin/{UPDATE_BRANCH}'], root)
    if reset.returncode != 0:
        return jsonify({'updated': False,
                        'message': f'git reset failed: {reset.stderr.strip() or reset.stdout.strip()}'}), 500

    # Success — relaunch. Send the response first, then restart shortly after.
    threading.Thread(target=lambda: (time.sleep(1.0), _do_restart()), daemon=True).start()
    return jsonify({'updated': True,
                    'message': f'Updated to origin/{UPDATE_BRANCH}. Restarting...'})


@app.route('/list-models', methods=['POST'])
def list_models():
    data = request.get_json(silent=True) or {}
    api_key = data.get('api_key', '').strip()
    provider = data.get('provider', DEFAULT_PROVIDER)
    custom_base_url = data.get('base_url', '').strip()
    timeout = get_timeout(data)

    if provider != 'ollama' and not api_key:
        return jsonify({'error': 'API key is required'}), 400

    base_url = get_base_url(provider, custom_base_url)
    if not base_url:
        return jsonify({'error': 'This provider requires a custom base URL'}), 400

    headers = {**auth_headers(provider, api_key), **parse_extra_headers(data.get('headers'))}

    try:
        if provider == 'gemini':
            resp = requests.get(f'{base_url}/models', headers=headers, timeout=timeout)
            resp.raise_for_status()
            out = []
            for m in resp.json().get('models', []):
                meta = {}
                if m.get('displayName'):
                    meta['display_name'] = m['displayName']
                if m.get('inputTokenLimit'):
                    meta['context'] = f"{m['inputTokenLimit']:,} ctx"
                out.append({'id': m['name'].replace('models/', ''), 'meta': meta})

        elif provider == 'claude':
            try:
                resp = requests.get(f'{base_url}/models', headers=headers, timeout=timeout)
                resp.raise_for_status()
                out = [{'id': m['id'],
                        'meta': {'display_name': m.get('display_name', ''),
                                 'created_at': (m.get('created_at') or '')[:10]}}
                       for m in resp.json().get('data', []) if 'id' in m]
            except requests.exceptions.RequestException:
                # No /v1/models access — fall back to the static list
                out = [{'id': m, 'meta': {}} for m in CLAUDE_MODELS]

        else:
            # OpenAI-compatible (OpenAI, OpenRouter, Azure, Ollama, DeepSeek, Mistral, Groq, Together, NVIDIA)
            resp = requests.get(f'{base_url}/models', headers=headers, timeout=timeout)
            resp.raise_for_status()
            out = []
            for m in resp.json().get('data', []):
                if 'id' not in m:
                    continue
                meta = {}
                if m.get('owned_by'):
                    meta['owned_by'] = m['owned_by']
                if isinstance(m.get('created'), int):
                    meta['created'] = time.strftime('%Y-%m-%d', time.gmtime(m['created']))
                out.append({'id': m['id'], 'meta': meta})

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'API request failed: {error_detail(e)}'}), 502
    except (KeyError, ValueError, AttributeError, TypeError) as e:
        return jsonify({'error': f'Unexpected response format: {e}'}), 502

    out.sort(key=lambda m: m['id'])
    return jsonify({'models': out, 'count': len(out)})


def _validate_test_request(data):
    """Shared validation for /test-model and /test-model-stream. Returns (fields, error_json)."""
    api_key = data.get('api_key', '').strip()
    provider = data.get('provider', DEFAULT_PROVIDER)
    model = data.get('model', '').strip()
    prompt = data.get('prompt', '').strip()
    custom_base_url = data.get('base_url', '').strip()
    timeout = get_timeout(data)
    params = parse_gen_params(data)

    if provider != 'ollama' and not api_key:
        return None, (jsonify({'error': 'API key is required'}), 400)
    if not model or not prompt:
        return None, (jsonify({'error': 'model and prompt are required'}), 400)

    base_url = get_base_url(provider, custom_base_url)
    if not base_url:
        return None, (jsonify({'error': 'This provider requires a custom base URL'}), 400)

    headers = {**auth_headers(provider, api_key),
               **parse_extra_headers(data.get('headers')),
               'Content-Type': 'application/json'}
    return {
        'api_key': api_key, 'provider': provider, 'model': model, 'prompt': prompt,
        'base_url': base_url, 'timeout': timeout, 'params': params, 'headers': headers,
    }, None


def _stream_for(provider, base_url, headers, model, payload, timeout):
    """Return the correct stream generator for a provider."""
    if provider == 'gemini':
        url = f"{base_url}/models/{model}:streamGenerateContent?alt=sse"
        return stream_gemini(url, headers, payload, timeout)
    if provider == 'claude':
        return stream_claude(base_url, headers, payload, timeout)
    if provider == 'xai':
        return stream_xai(base_url, headers, payload, timeout)
    return stream_openai_chat(base_url, headers, payload, timeout)


def stream_events(provider, base_url, headers, model, prompt, params, timeout):
    """Yield raw stream events, retrying once without temperature on a temperature 400."""
    try:
        payload = build_payload(provider, model, prompt, params, True)
        for ev in _stream_for(provider, base_url, headers, model, payload, timeout):
            yield ev
    except requests.exceptions.RequestException as e:
        if is_temperature_error(e):
            yield {'type': 'retry', 'field': 'temperature'}
            payload = build_payload(provider, model, prompt, params, False)
            for ev in _stream_for(provider, base_url, headers, model, payload, timeout):
                yield ev
        else:
            raise


@app.route('/test-model', methods=['POST'])
def test_model():
    data = request.get_json(silent=True) or {}
    fields, err = _validate_test_request(data)
    if err:
        return err
    stream = bool(data.get('stream'))

    try:
        content, ttft, usage = run_model_test(
            fields['provider'], fields['model'], fields['prompt'],
            fields['base_url'], fields['headers'], fields['timeout'],
            fields['params'], stream)
        return jsonify({'response': content, 'ttft': ttft, 'usage': usage})

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'API request failed: {error_detail(e)}'}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/test-model-stream', methods=['POST'])
def test_model_stream():
    """Live-stream a model test as SSE: ttft / delta / usage / retry / done | error events."""
    data = request.get_json(silent=True) or {}
    fields, err = _validate_test_request(data)
    if err:
        return err
    src = stream_events(fields['provider'], fields['base_url'], fields['headers'],
                        fields['model'], fields['prompt'], fields['params'], fields['timeout'])

    def gen():
        try:
            for ev in src:
                yield f"data: {json.dumps(ev)}\n\n"
        except requests.exceptions.RequestException as e:
            yield f"data: {json.dumps({'type': 'error', 'error': error_detail(e)})}\n\n"
        except (KeyError, ValueError, AttributeError, TypeError) as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(gen(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2463, debug=False, threaded=True)
