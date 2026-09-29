import os
import shutil
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests

import app as app_module
from app import (
    app,
    build_payload,
    consume_stream,
    error_detail,
    extract_xai_response_text,
    get_base_url,
    normalize_usage_claude,
    normalize_usage_gemini,
    normalize_usage_openai,
    normalize_usage_xai,
    parse_extra_headers,
    parse_gen_params,
    parse_image_input,
)


class FakeResponse:
    """Stand-in for requests.Response, optionally carrying an SSE line stream."""

    def __init__(self, json_data=None, status_code=200, reason='', lines=None, text=None, content=None, headers=None):
        self._json = json_data
        self.status_code = status_code
        self.reason = reason
        self._lines = lines or []
        self.text = text
        self.content = content if content is not None else (text.encode('utf-8') if text is not None else b'')
        self.headers = headers or {}

    def json(self):
        if self._json is None:
            raise ValueError('no JSON body')
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def iter_lines(self, decode_unicode=False):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class HelperTests(unittest.TestCase):
    def test_get_base_url_custom_wins(self):
        self.assertEqual(get_base_url('xai', 'https://example.com/v1/'), 'https://example.com/v1')

    def test_get_base_url_provider_default(self):
        self.assertEqual(get_base_url('gemini', ''),
                         'https://generativelanguage.googleapis.com/v1beta')
        with mock.patch('app.get_default_base_url', return_value='https://api.openai.com/v1'):
            self.assertEqual(get_base_url('openai', ''),
                             'https://api.openai.com/v1')

    def test_get_base_url_nvidia_default(self):
        self.assertEqual(get_base_url('nvidia', ''),
                         'https://integrate.api.nvidia.com/v1')

    def test_extract_xai_response_text_nested(self):
        data = {'output': [{'content': [{'type': 'output_text', 'text': 'a'}, {'type': 'other'}]}]}
        self.assertEqual(extract_xai_response_text(data), 'a')

    def test_extract_xai_response_text_fallback(self):
        self.assertEqual(extract_xai_response_text({'output_text': 'b'}), 'b')

    def test_parse_extra_headers(self):
        self.assertEqual(parse_extra_headers({'A': '1'}), {'A': '1'})
        self.assertEqual(parse_extra_headers('{"A": "1"}'), {'A': '1'})
        self.assertEqual(parse_extra_headers('not json'), {})
        self.assertEqual(parse_extra_headers(None), {})

    def test_error_detail_json_body(self):
        resp = FakeResponse(json_data={'error': {'message': 'bad key'}}, status_code=401)
        err = requests.exceptions.HTTPError(response=resp)
        self.assertEqual(error_detail(err), 'HTTP 401: bad key')

    def test_error_detail_without_response(self):
        self.assertEqual(error_detail(requests.exceptions.Timeout('t')), 't')


class ListModelsTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def post(self, payload):
        return self.client.post('/list-models', json=payload)

    def test_requires_api_key(self):
        resp = self.post({'provider': 'openai', 'api_key': ''})
        self.assertEqual(resp.status_code, 400)

    def test_ollama_allows_missing_key(self):
        fake = FakeResponse(json_data={'data': [{'id': 'llama3'}]})
        with mock.patch.object(app_module.requests, 'get', return_value=fake):
            resp = self.post({'provider': 'ollama', 'api_key': ''})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()['models'][0]['id'], 'llama3')

    def test_openai_models_parsed_with_meta(self):
        fake = FakeResponse(json_data={'data': [{'id': 'b', 'owned_by': 'org'}, {'id': 'a'}]})
        with mock.patch.object(app_module.requests, 'get', return_value=fake) as mg:
            resp = self.post({'provider': 'openai', 'api_key': 'sk-x'})
        body = resp.get_json()
        self.assertEqual([m['id'] for m in body['models']], ['a', 'b'])
        self.assertEqual(body['models'][1]['meta']['owned_by'], 'org')
        self.assertEqual(mg.call_args.kwargs['headers']['Authorization'], 'Bearer sk-x')

    def test_gemini_uses_header_auth_not_query_param(self):
        fake = FakeResponse(json_data={'models': [
            {'name': 'models/gemini-1.5-pro', 'displayName': 'Gemini 1.5 Pro', 'inputTokenLimit': 2000000}
        ]})
        with mock.patch.object(app_module.requests, 'get', return_value=fake) as mg:
            resp = self.post({'provider': 'gemini', 'api_key': 'AIza-x'})
        body = resp.get_json()
        self.assertEqual(body['models'][0]['id'], 'gemini-1.5-pro')
        self.assertIn('context', body['models'][0]['meta'])
        headers = mg.call_args.kwargs['headers']
        self.assertEqual(headers['x-goog-api-key'], 'AIza-x')
        self.assertNotIn('key=', mg.call_args.args[0])

    def test_claude_live_models(self):
        fake = FakeResponse(json_data={'data': [{'id': 'claude-x', 'display_name': 'Claude X'}]})
        with mock.patch.object(app_module.requests, 'get', return_value=fake):
            resp = self.post({'provider': 'claude', 'api_key': 'sk-ant-x'})
        body = resp.get_json()
        self.assertEqual(body['models'][0]['id'], 'claude-x')
        self.assertEqual(body['models'][0]['meta']['display_name'], 'Claude X')

    def test_claude_falls_back_to_static_list(self):
        with mock.patch.object(app_module.requests, 'get',
                               side_effect=requests.exceptions.ConnectionError('boom')):
            resp = self.post({'provider': 'claude', 'api_key': 'sk-ant-x'})
        body = resp.get_json()
        self.assertEqual(body['count'], len(app_module.CLAUDE_MODELS))

    def test_provider_error_body_surfaced(self):
        fake = FakeResponse(json_data={'error': {'message': 'invalid api key'}}, status_code=401)
        with mock.patch.object(app_module.requests, 'get', return_value=fake):
            resp = self.post({'provider': 'openai', 'api_key': 'sk-bad'})
        self.assertEqual(resp.status_code, 502)
        self.assertIn('invalid api key', resp.get_json()['error'])

    def test_extra_headers_merged(self):
        fake = FakeResponse(json_data={'data': []})
        with mock.patch.object(app_module.requests, 'get', return_value=fake) as mg:
            self.post({'provider': 'openai', 'api_key': 'sk-x',
                       'headers': {'HTTP-Referer': 'https://ex.com'}})
        self.assertEqual(mg.call_args.kwargs['headers']['HTTP-Referer'], 'https://ex.com')


class TestModelTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def post(self, payload):
        return self.client.post('/test-model', json=payload)

    def test_requires_model_and_prompt(self):
        resp = self.post({'provider': 'openai', 'api_key': 'sk-x', 'model': '', 'prompt': ''})
        self.assertEqual(resp.status_code, 400)

    def test_openai_chat(self):
        fake = FakeResponse(json_data={'choices': [{'message': {'content': 'hi'}}]})
        with mock.patch.object(app_module.requests, 'post', return_value=fake) as mp:
            resp = self.post({'provider': 'openai', 'api_key': 'sk-x',
                              'model': 'gpt-x', 'prompt': 'p'})
        self.assertEqual(resp.get_json()['response'], 'hi')
        self.assertIsNone(resp.get_json()['ttft'])
        body = mp.call_args.kwargs['json']
        self.assertEqual(body['model'], 'gpt-x')
        self.assertEqual(mp.call_args.args[0], f'{app_module.DEFAULT_BASE_URL}/chat/completions')

    def test_claude_messages(self):
        fake = FakeResponse(json_data={'content': [{'type': 'text', 'text': 'bonjour'}]})
        with mock.patch.object(app_module.requests, 'post', return_value=fake) as mp:
            resp = self.post({'provider': 'claude', 'api_key': 'sk-ant',
                              'model': 'claude-x', 'prompt': 'p'})
        self.assertEqual(resp.get_json()['response'], 'bonjour')
        headers = mp.call_args.kwargs['headers']
        self.assertEqual(headers['x-api-key'], 'sk-ant')
        self.assertEqual(headers['anthropic-version'], '2023-06-01')

    def test_gemini_no_key_in_url(self):
        fake = FakeResponse(json_data={'candidates': [{'content': {'parts': [{'text': 'yo'}]}}]})
        with mock.patch.object(app_module.requests, 'post', return_value=fake) as mp:
            resp = self.post({'provider': 'gemini', 'api_key': 'AIza',
                              'model': 'gemini-x', 'prompt': 'p'})
        self.assertEqual(resp.get_json()['response'], 'yo')
        url = mp.call_args.args[0]
        self.assertNotIn('key=', url)
        self.assertTrue(url.endswith(':generateContent'))
        self.assertEqual(mp.call_args.kwargs['headers']['x-goog-api-key'], 'AIza')

    def test_xai_responses(self):
        fake = FakeResponse(json_data={'output': [
            {'content': [{'type': 'output_text', 'text': 'grok says hi'}]}
        ]})
        with mock.patch.object(app_module.requests, 'post', return_value=fake) as mp:
            resp = self.post({'provider': 'xai', 'api_key': 'xai-1',
                              'model': 'grok-x', 'prompt': 'p'})
        self.assertEqual(resp.get_json()['response'], 'grok says hi')
        self.assertTrue(mp.call_args.args[0].endswith('/responses'))

    def test_openai_stream_returns_ttft(self):
        lines = [
            'data: {"choices": [{"delta": {"content": "Hel"}}]}',
            'data: {"choices": [{"delta": {"content": "lo"}}]}',
            'data: [DONE]',
        ]
        fake = FakeResponse(lines=lines)
        with mock.patch.object(app_module.requests, 'post', return_value=fake) as mp:
            resp = self.post({'provider': 'openai', 'api_key': 'sk-x',
                              'model': 'gpt-x', 'prompt': 'p', 'stream': True})
        body = resp.get_json()
        self.assertEqual(body['response'], 'Hello')
        self.assertIsNotNone(body['ttft'])
        self.assertTrue(mp.call_args.kwargs['stream'])
        self.assertTrue(mp.call_args.kwargs['json']['stream'])

    def test_claude_stream_returns_ttft(self):
        lines = [
            'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "sa"}}',
            'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "lut"}}',
        ]
        fake = FakeResponse(lines=lines)
        with mock.patch.object(app_module.requests, 'post', return_value=fake):
            resp = self.post({'provider': 'claude', 'api_key': 'sk-ant',
                              'model': 'claude-x', 'prompt': 'p', 'stream': True})
        body = resp.get_json()
        self.assertEqual(body['response'], 'salut')
        self.assertIsNotNone(body['ttft'])

    def test_extra_headers_merged(self):
        fake = FakeResponse(json_data={'choices': [{'message': {'content': 'hi'}}]})
        with mock.patch.object(app_module.requests, 'post', return_value=fake) as mp:
            self.post({'provider': 'openai', 'api_key': 'sk-x', 'model': 'gpt-x',
                       'prompt': 'p', 'headers': {'X-Custom': 'v'}})
        self.assertEqual(mp.call_args.kwargs['headers']['X-Custom'], 'v')

    def test_provider_error_body_surfaced(self):
        fake = FakeResponse(json_data={'error': {'message': 'model not found'}}, status_code=404)
        with mock.patch.object(app_module.requests, 'post', return_value=fake):
            resp = self.post({'provider': 'openai', 'api_key': 'sk-x',
                              'model': 'nope', 'prompt': 'p'})
        self.assertEqual(resp.status_code, 502)
        self.assertIn('model not found', resp.get_json()['error'])


class EnhancementTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        resp = self.client.get('/health')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()['status'], 'ok')
        self.assertEqual(resp.get_json()['version'], app_module.VERSION)

    def test_parse_gen_params_defaults_and_clamp(self):
        p = parse_gen_params({})
        self.assertEqual(p['max_tokens'], 300)
        self.assertAlmostEqual(p['temperature'], 0.7)
        self.assertEqual(p['system'], '')
        self.assertEqual(p['image'], '')
        p2 = parse_gen_params({'max_tokens': 0, 'temperature': 9, 'system': '  hi ', 'image': 'data:image/png;base64,123'})
        self.assertEqual(p2['max_tokens'], 1)          # clamped up to 1
        self.assertAlmostEqual(p2['temperature'], 2.0)  # clamped to 2.0
        self.assertEqual(p2['system'], 'hi')
        self.assertEqual(p2['image'], 'data:image/png;base64,123')

    def test_parse_image_input(self):
        self.assertIsNone(parse_image_input(None))
        self.assertIsNone(parse_image_input(''))
        # Data URL
        d = parse_image_input('data:image/png;base64,iVBORw0KGgo=')
        self.assertEqual(d['type'], 'data_url')
        self.assertEqual(d['mime'], 'image/png')
        self.assertEqual(d['base64'], 'iVBORw0KGgo=')
        # HTTP URL
        u = parse_image_input('https://example.com/pic.jpg')
        self.assertEqual(u['type'], 'url')
        self.assertEqual(u['url'], 'https://example.com/pic.jpg')

    def test_vision_payload_openai(self):
        params = parse_gen_params({'image': 'https://example.com/pic.jpg'})
        payload = build_payload('openai', 'gpt-4o', 'describe this', params)
        msg = payload['messages'][0]
        self.assertEqual(msg['role'], 'user')
        self.assertEqual(msg['content'][0], {'type': 'text', 'text': 'describe this'})
        self.assertEqual(msg['content'][1], {'type': 'image_url', 'image_url': {'url': 'https://example.com/pic.jpg'}})

    def test_vision_payload_claude(self):
        params = parse_gen_params({'image': 'data:image/jpeg;base64,abc123=='})
        payload = build_payload('claude', 'claude-3-5-sonnet', 'what is this', params)
        msg = payload['messages'][0]
        self.assertEqual(msg['role'], 'user')
        self.assertEqual(msg['content'][0]['type'], 'image')
        self.assertEqual(msg['content'][0]['source']['data'], 'abc123==')
        self.assertEqual(msg['content'][0]['source']['media_type'], 'image/jpeg')
        self.assertEqual(msg['content'][1], {'type': 'text', 'text': 'what is this'})

    def test_vision_payload_gemini(self):
        params = parse_gen_params({'image': 'data:image/png;base64,xyz999=='})
        payload = build_payload('gemini', 'gemini-1.5-pro', 'look at this', params)
        parts = payload['contents'][0]['parts']
        self.assertEqual(parts[0]['inlineData']['data'], 'xyz999==')
        self.assertEqual(parts[0]['inlineData']['mimeType'], 'image/png')
        self.assertEqual(parts[1]['text'], 'look at this')

    def test_usage_normalizers(self):
        self.assertEqual(normalize_usage_openai(
            {'prompt_tokens': 4, 'completion_tokens': 6, 'total_tokens': 10}),
            {'prompt': 4, 'completion': 6, 'total': 10})
        self.assertEqual(normalize_usage_claude(
            {'input_tokens': 4, 'output_tokens': 6}),
            {'prompt': 4, 'completion': 6, 'total': 10})
        self.assertEqual(normalize_usage_gemini(
            {'promptTokenCount': 4, 'candidatesTokenCount': 6, 'totalTokenCount': 10}),
            {'prompt': 4, 'completion': 6, 'total': 10})
        self.assertEqual(normalize_usage_xai(
            {'input_tokens': 4, 'output_tokens': 6, 'total_tokens': 10}),
            {'prompt': 4, 'completion': 6, 'total': 10})
        self.assertIsNone(normalize_usage_openai(None))

    def test_consume_stream_merges_usage(self):
        events = [
            {'type': 'ttft', 'ttft': 0.5},
            {'type': 'delta', 'text': 'He'},
            {'type': 'delta', 'text': 'llo'},
            {'type': 'usage', 'usage': {'prompt': 4}},
            {'type': 'usage', 'usage': {'completion': 6}},
            {'type': 'done'},
        ]
        text, ttft, usage = consume_stream(iter(events))
        self.assertEqual(text, 'Hello')
        self.assertEqual(ttft, 0.5)
        self.assertEqual(usage['prompt'], 4)
        self.assertEqual(usage['completion'], 6)
        self.assertEqual(usage['total'], 10)  # recomputed after merge

    def test_new_providers_have_defaults(self):
        self.assertEqual(app_module.get_base_url('deepseek', ''),
                         'https://api.deepseek.com/v1')
        self.assertEqual(app_module.get_base_url('mistral', ''),
                         'https://api.mistral.ai/v1')
        self.assertEqual(app_module.get_base_url('groq', ''),
                         'https://api.groq.com/openai/v1')
        self.assertEqual(app_module.get_base_url('together', ''),
                         'https://api.together.xyz/v1')

    def test_openai_nonstream_returns_usage(self):
        fake = FakeResponse(json_data={
            'choices': [{'message': {'content': 'hi'}}],
            'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5},
        })
        with mock.patch.object(app_module.requests, 'post', return_value=fake):
            resp = self.client.post('/test-model', json={
                'provider': 'openai', 'api_key': 'sk-x',
                'model': 'gpt-x', 'prompt': 'p'
            })
        body = resp.get_json()
        self.assertEqual(body['usage']['prompt'], 3)
        self.assertEqual(body['usage']['completion'], 2)
        self.assertIsNone(body['ttft'])

    def test_openai_stream_options_requested(self):
        lines = [
            'data: {"choices": [{"delta": {"content": "Hi"}}]}',
            'data: {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}',
            'data: [DONE]',
        ]
        fake = FakeResponse(lines=lines)
        with mock.patch.object(app_module.requests, 'post', return_value=fake) as mp:
            resp = self.client.post('/test-model', json={
                'provider': 'openai', 'api_key': 'sk-x',
                'model': 'gpt-x', 'prompt': 'p', 'stream': True
            })
        body = resp.get_json()
        self.assertEqual(body['response'], 'Hi')
        self.assertIsNotNone(body['ttft'])
        self.assertEqual(body['usage']['total'], 2)
        self.assertTrue(mp.call_args.kwargs['json']['stream'])
        self.assertTrue(mp.call_args.kwargs['json']['stream_options']['include_usage'])

    def test_stream_endpoint_is_sse(self):
        lines = [
            'data: {"choices": [{"delta": {"content": "ab"}}]}',
            'data: {"choices": [{"delta": {"content": "cd"}}]}',
            'data: [DONE]',
        ]
        fake = FakeResponse(lines=lines)
        with mock.patch.object(app_module.requests, 'post', return_value=fake):
            resp = self.client.post('/test-model-stream', json={
                'provider': 'openai', 'api_key': 'sk-x',
                'model': 'gpt-x', 'prompt': 'p'
            })
        self.assertEqual(resp.status_code, 200)
        self.assertIn('text/event-stream', resp.content_type)
        body = resp.get_data(as_text=True)
        self.assertIn('"type": "delta"', body)
        self.assertIn('[DONE]', body)

    def test_stream_endpoint_missing_model(self):
        resp = self.client.post('/test-model-stream', json={
            'provider': 'openai', 'api_key': 'sk-x', 'model': '', 'prompt': ''
        })
        self.assertEqual(resp.status_code, 400)

    def test_system_prompt_included_in_payload(self):
        fake = FakeResponse(json_data={'choices': [{'message': {'content': 'hi'}}]})
        with mock.patch.object(app_module.requests, 'post', return_value=fake) as mp:
            self.client.post('/test-model', json={
                'provider': 'openai', 'api_key': 'sk-x',
                'model': 'gpt-x', 'prompt': 'p', 'system': 'be brief'
            })
        messages = mp.call_args.kwargs['json']['messages']
        self.assertEqual(messages[0], {'role': 'system', 'content': 'be brief'})

    def test_temperature_auto_fallback_retries_without_temperature(self):
        bad = FakeResponse(json_data={'error': {'message':
            'field Temperature invalid, only 1 is allowed for this model'}}, status_code=400)
        good = FakeResponse(json_data={'choices': [{'message': {'content': 'ok'}}]})
        with mock.patch.object(app_module.requests, 'post',
                               side_effect=[bad, good]) as mp:
            resp = self.client.post('/test-model', json={
                'provider': 'openai', 'api_key': 'sk-x',
                'model': 'o3-mini', 'prompt': 'p', 'temperature': 0.5
            })
        body = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(body['response'], 'ok')
        self.assertEqual(len(mp.call_args_list), 2)
        # First attempt sent temperature; the retry omitted it.
        self.assertIn('temperature', mp.call_args_list[0].kwargs['json'])
        self.assertNotIn('temperature', mp.call_args_list[1].kwargs['json'])

    def test_non_temperature_400_is_not_retried(self):
        bad = FakeResponse(json_data={'error': {'message': 'model not found'}}, status_code=404)
        with mock.patch.object(app_module.requests, 'post', return_value=bad) as mp:
            resp = self.client.post('/test-model', json={
                'provider': 'openai', 'api_key': 'sk-x',
                'model': 'nope', 'prompt': 'p'
            })
        self.assertEqual(resp.status_code, 502)
        self.assertEqual(len(mp.call_args_list), 1)


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_check_update_not_configured(self):
        with mock.patch.object(app_module, 'GITHUB_REPO', ''):
            resp = self.client.get('/check-update')
            self.assertEqual(resp.status_code, 200)
            self.assertFalse(resp.get_json()['configured'])

    def test_check_update_detects_new_version(self):
        with mock.patch.object(app_module, 'GITHUB_REPO', 'me/repo'), \
             mock.patch.object(app_module, 'UPDATE_BRANCH', 'main'), \
             mock.patch.object(app_module, 'VERSION', '3.0'):
            fake = FakeResponse(text='3.1\n', status_code=200)
            with mock.patch.object(app_module.requests, 'get', return_value=fake) as mg:
                resp = self.client.get('/check-update')
            body = resp.get_json()
            self.assertTrue(body['configured'])
            self.assertTrue(body['has_update'])
            self.assertEqual(body['remote'], '3.1')
            self.assertEqual(body['current'], '3.0')
            self.assertIn('me/repo/main/version.txt', mg.call_args.args[0])

    def test_check_update_same_version(self):
        with mock.patch.object(app_module, 'GITHUB_REPO', 'me/repo'), \
             mock.patch.object(app_module, 'VERSION', '3.0'):
            fake = FakeResponse(text='3.0\n', status_code=200)
            with mock.patch.object(app_module.requests, 'get', return_value=fake):
                resp = self.client.get('/check-update')
            self.assertFalse(resp.get_json()['has_update'])

    def test_check_update_git_fallback_when_http_404(self):
        with mock.patch.object(app_module, 'GITHUB_REPO', 'me/repo'), \
             mock.patch.object(app_module, 'UPDATE_BRANCH', 'main'), \
             mock.patch.object(app_module, 'VERSION', '3.0'), \
             mock.patch.object(app_module, '_repo_root', return_value='/fake/root'), \
             mock.patch.object(app_module, '_get_remote_version_git', return_value='3.1'):
            fake = FakeResponse(text='404: Not Found', status_code=404)
            with mock.patch.object(app_module.requests, 'get', return_value=fake):
                resp = self.client.get('/check-update')
            body = resp.get_json()
            self.assertTrue(body['configured'])
            self.assertTrue(body['has_update'])
            self.assertEqual(body['remote'], '3.1')
            self.assertEqual(body['current'], '3.0')

    def test_update_requires_repo_config(self):
        with mock.patch.object(app_module, 'GITHUB_REPO', ''):
            resp = self.client.post('/update')
        self.assertEqual(resp.status_code, 400)

    def test_update_archive_when_no_git(self):
        with mock.patch.object(app_module, 'GITHUB_REPO', 'me/repo'), \
             mock.patch.object(app_module, '_repo_root', return_value=None), \
             mock.patch.object(app_module, '_update_from_archive') as mock_update, \
             mock.patch.object(app_module, '_do_restart'):
            resp = self.client.post('/update')
        body = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(body['updated'])
        self.assertIn('version', body)
        self.assertIn('Reloading', body['message'])
        mock_update.assert_called_once_with(app_module.APP_DIR)

    def test_update_archive_failure_surfaced(self):
        with mock.patch.object(app_module, 'GITHUB_REPO', 'me/repo'), \
             mock.patch.object(app_module, '_repo_root', return_value=None), \
             mock.patch.object(app_module, '_update_from_archive', side_effect=RuntimeError('archive error')):
            resp = self.client.post('/update')
        body = resp.get_json()
        self.assertEqual(resp.status_code, 500)
        self.assertFalse(body['updated'])
        self.assertIn('archive error', body['message'])

    def test_update_from_archive_extracts_files(self):
        import io
        import shutil
        import tarfile
        import tempfile
        temp_dir = tempfile.mkdtemp()
        try:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode='w:gz') as tf:
                vdata = b"3.3"
                ti = tarfile.TarInfo(name="repo-main/version.txt")
                ti.size = len(vdata)
                tf.addfile(ti, io.BytesIO(vdata))

                app_data = b"# new code"
                ti2 = tarfile.TarInfo(name="repo-main/app.py")
                ti2.size = len(app_data)
                tf.addfile(ti2, io.BytesIO(app_data))

                env_data = b"SECRET_KEY=overwrite"
                ti3 = tarfile.TarInfo(name="repo-main/.env")
                ti3.size = len(env_data)
                tf.addfile(ti3, io.BytesIO(env_data))
            buf.seek(0)
            fake = FakeResponse(status_code=200)
            fake.content = buf.getvalue()
            with mock.patch.object(app_module.requests, 'get', return_value=fake):
                app_module._update_from_archive(temp_dir)

            # version.txt should be updated, while .env is protected
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'version.txt')))
            with open(os.path.join(temp_dir, 'version.txt'), 'r') as f:
                self.assertEqual(f.read().strip(), "3.3")
            with open(os.path.join(temp_dir, 'app.py'), 'r') as f:
                self.assertEqual(f.read().strip(), "# new code")
            self.assertFalse(os.path.exists(os.path.join(temp_dir, '.env')))
        finally:
            shutil.rmtree(temp_dir)


class SettingsApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_get_settings(self):
        resp = self.client.get('/api/settings')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('default_base_url', data)
        self.assertIn('default_provider', data)

    def test_save_settings_writes_env(self):
        import tempfile
        temp_dir = tempfile.mkdtemp()
        fake_env = os.path.join(temp_dir, '.env')
        orig_base = app_module.DEFAULT_BASE_URL
        orig_prov = app_module.DEFAULT_PROVIDER
        orig_env_base = os.environ.get('DEFAULT_BASE_URL')
        orig_env_prov = os.environ.get('DEFAULT_PROVIDER')
        try:
            with open(fake_env, 'w', encoding='utf-8') as f:
                f.write("DEFAULT_BASE_URL=old\nDEFAULT_PROVIDER=openai\n")
            with mock.patch.object(app_module, 'APP_DIR', temp_dir):
                resp = self.client.post('/api/settings', json={
                    'default_base_url': 'https://custom.api/v1',
                    'default_provider': 'claude'
                })
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertEqual(data['status'], 'ok')
            self.assertEqual(data['default_base_url'], 'https://custom.api/v1')
            self.assertEqual(data['default_provider'], 'claude')

            with open(fake_env, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertIn('DEFAULT_BASE_URL=https://custom.api/v1', content)
            self.assertIn('DEFAULT_PROVIDER=claude', content)
        finally:
            if orig_env_base is None:
                os.environ.pop('DEFAULT_BASE_URL', None)
            else:
                os.environ['DEFAULT_BASE_URL'] = orig_env_base
            if orig_env_prov is None:
                os.environ.pop('DEFAULT_PROVIDER', None)
            else:
                os.environ['DEFAULT_PROVIDER'] = orig_env_prov
            app_module.DEFAULT_BASE_URL = orig_base
            app_module.DEFAULT_PROVIDER = orig_prov
            app_module.PROVIDER_BASE_URLS['openai'] = orig_base
            shutil.rmtree(temp_dir)


class VersionTests(unittest.TestCase):
    def test_clean_version(self):
        self.assertEqual(app_module._clean_version('3.5.7\n'), '3.5.7')
        self.assertEqual(app_module._clean_version(b'3.5.7\n'), '3.5.7')
        self.assertEqual(app_module._clean_version(b'\xef\xbb\xbf3.5.7\n'), '3.5.7')
        self.assertEqual(app_module._clean_version('3.5.7\r\n'.encode('utf-16-le')), '3.5.7')
        self.assertEqual(app_module._clean_version('3.5.7\r\n'.encode('utf-16-be')), '3.5.7')
        # Test UTF-16 with BOM
        self.assertEqual(app_module._clean_version(b'\xff\xfe3\x00.\x005\x00.\x007\x00'), '3.5.7')
        self.assertEqual(app_module._clean_version(b'\xfe\xff\x003\x00.\x005\x00.\x007'), '3.5.7')
        # Test string containing \ufeff or \ufffe BOM
        self.assertEqual(app_module._clean_version('\ufeff3.5.7'), '3.5.7')
        self.assertEqual(app_module._clean_version('\ufffe3.5.7'), '3.5.7')
        # Test leading 'v' or 'V'
        self.assertEqual(app_module._clean_version('v3.5.8'), '3.5.8')
        self.assertEqual(app_module._clean_version('V3.5.9'), '3.5.9')

    def test_parse_version_tuple(self):
        self.assertEqual(app_module._parse_version_tuple('3.5.7'), (3, 5, 7))
        self.assertEqual(app_module._parse_version_tuple('\ufeff3.5.7'), (3, 5, 7))
        self.assertEqual(app_module._parse_version_tuple('v3.5.8'), (3, 5, 8))
        self.assertEqual(app_module._parse_version_tuple(''), ())
        self.assertGreater(app_module._parse_version_tuple('3.5.10'), app_module._parse_version_tuple('3.5.9'))
        self.assertGreater(app_module._parse_version_tuple('3.8.0'), app_module._parse_version_tuple('3.5.9'))

    def test_check_update_with_utf16_remote(self):
        with app.test_client() as client:
            with mock.patch.object(app_module, 'GITHUB_REPO', 'me/repo'), \
                 mock.patch.object(app_module, 'UPDATE_BRANCH', 'main'), \
                 mock.patch.object(app_module, 'VERSION', '3.5.6'):
                utf16_content = '3.5.7\r\n'.encode('utf-16-le')
                fake = FakeResponse(content=utf16_content, status_code=200)
                with mock.patch.object(app_module.requests, 'get', return_value=fake):
                    resp = client.get('/check-update')
                body = resp.get_json()
                self.assertTrue(body['has_update'])
                self.assertEqual(body['remote'], '3.5.7')

    def test_check_update_identical_or_older_version_no_update(self):
        with app.test_client() as client:
            with mock.patch.object(app_module, 'GITHUB_REPO', 'me/repo'), \
                 mock.patch.object(app_module, 'UPDATE_BRANCH', 'main'), \
                 mock.patch.object(app_module, 'VERSION', '3.5.7'):
                # Identical version even with BOM should NOT trigger update
                fake_bom = FakeResponse(text='\ufeff3.5.7\r\n', status_code=200)
                with mock.patch.object(app_module.requests, 'get', return_value=fake_bom):
                    resp = client.get('/check-update')
                body = resp.get_json()
                self.assertFalse(body['has_update'])
                self.assertEqual(body['remote'], '3.5.7')

                # Older remote version should NOT trigger update
                fake_older = FakeResponse(text='3.5.6\n', status_code=200)
                with mock.patch.object(app_module.requests, 'get', return_value=fake_older):
                    resp = client.get('/check-update')
                body = resp.get_json()
                self.assertFalse(body['has_update'])
                self.assertEqual(body['remote'], '3.5.6')


class NewFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_gen_params_with_penalties(self):
        params = parse_gen_params({
            'top_p': 0.85,
            'frequency_penalty': 0.5,
            'presence_penalty': -0.2
        })
        self.assertEqual(params['top_p'], 0.85)
        self.assertEqual(params['frequency_penalty'], 0.5)
        self.assertEqual(params['presence_penalty'], -0.2)

    def test_build_payload_openai_penalties(self):
        params = parse_gen_params({
            'top_p': 0.9,
            'frequency_penalty': 1.2,
            'presence_penalty': 0.4
        })
        payload = build_payload('openai', 'gpt-4o', 'test', params)
        self.assertEqual(payload['top_p'], 0.9)
        self.assertEqual(payload['frequency_penalty'], 1.2)
        self.assertEqual(payload['presence_penalty'], 0.4)

    def test_test_chat_openai(self):
        fake = FakeResponse(json_data={'choices': [{'message': {'content': 'I am fine, thanks!'}}]})
        with mock.patch.object(app_module.requests, 'post', return_value=fake) as mp:
            resp = self.client.post('/test-chat', json={
                'provider': 'openai',
                'api_key': 'sk-test',
                'model': 'gpt-4o',
                'messages': [
                    {'role': 'user', 'content': 'Hello'},
                    {'role': 'assistant', 'content': 'Hi'},
                    {'role': 'user', 'content': 'How are you?'}
                ]
            })
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body['response'], 'I am fine, thanks!')
        json_sent = mp.call_args.kwargs['json']
        self.assertEqual(len(json_sent['messages']), 3)

    def test_test_suite(self):
        fake = FakeResponse(json_data={'choices': [{'message': {'content': 'def reverse(): pass'}}]})
        with mock.patch.object(app_module.requests, 'post', return_value=fake):
            resp = self.client.post('/test-suite', json={
                'provider': 'openai',
                'api_key': 'sk-test',
                'model': 'gpt-4o',
                'suite': 'coding'
            })
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body['suite'], 'coding')
        self.assertTrue(len(body['results']) >= 3)
        self.assertEqual(body['results'][0]['status'], 'success')

    def test_ollama_tags(self):
        fake = FakeResponse(json_data={'models': [{'name': 'llama3:latest', 'size': 4000000000}]})
        with mock.patch.object(app_module.requests, 'get', return_value=fake):
            resp = self.client.post('/ollama/tags', json={'base_url': 'http://localhost:11434'})
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body['models'][0]['name'], 'llama3:latest')

    def test_ollama_pull(self):
        fake = FakeResponse(json_data={'status': 'success'})
        with mock.patch.object(app_module.requests, 'post', return_value=fake):
            resp = self.client.post('/ollama/pull', json={'model': 'llama3:latest'})
        self.assertEqual(resp.status_code, 200)

    def test_local_health(self):
        fake = FakeResponse(json_data={'models': [{'name': 'qwen2.5:7b'}]}, status_code=200)
        with mock.patch.object(app_module.requests, 'get', return_value=fake):
            resp = self.client.get('/api/local-health')
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn('services', body)
        self.assertTrue(len(body['services']) >= 4)
        ollama_svc = next((s for s in body['services'] if s['name'] == 'Ollama'), None)
        self.assertIsNotNone(ollama_svc)
        self.assertTrue(ollama_svc['online'])

    def test_judge_responses(self):
        judge_output = '{"evaluations": [{"id": "model_1", "name": "gpt-4o", "score": 9.2, "strengths": "accurate", "weaknesses": "none", "rationale": "great"}], "winner_id": "model_1", "summary": "Model 1 is superior"}'
        fake = FakeResponse(json_data={'choices': [{'message': {'content': judge_output}}]})
        with mock.patch.object(app_module.requests, 'post', return_value=fake):
            resp = self.client.post('/api/judge-responses', json={
                'judge_provider': 'openai',
                'judge_model': 'gpt-4o',
                'judge_api_key': 'sk-test',
                'prompt': 'Write a quick sort algorithm in python',
                'rubric': 'Code correctness and explanation',
                'candidates': [
                    {'id': 'model_1', 'name': 'gpt-4o', 'response': 'def quicksort(arr): pass'}
                ]
            })
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn('evaluation', body)
        self.assertEqual(body['evaluation']['winner_id'], 'model_1')

    def test_json_mode_payload(self):
        schema = {"type": "object", "properties": {"score": {"type": "number"}}, "required": ["score"]}
        params = {'json_mode': True, 'json_schema': schema, 'max_tokens': 200, 'temperature': 0.5}
        
        # OpenAI
        payload_openai = app_module.build_payload('openai', 'gpt-4o', 'Give me score', params)
        self.assertIn('response_format', payload_openai)
        self.assertEqual(payload_openai['response_format']['type'], 'json_schema')
        
        # Gemini
        payload_gemini = app_module.build_payload('gemini', 'gemini-1.5-pro', 'Give me score', params)
        self.assertEqual(payload_gemini['generationConfig']['responseMimeType'], 'application/json')
        self.assertEqual(payload_gemini['generationConfig']['responseSchema'], schema)
        
        # Claude
        payload_claude = app_module.build_payload('claude', 'claude-3-5-sonnet', 'Give me score', params)
        self.assertIn('JSON Schema', payload_claude['system'])

    def test_validate_output_json_schema(self):
        schema = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}
        
        # Valid JSON matching schema
        res1 = app_module.validate_output_json_schema('{"count": 42}', schema)
        self.assertTrue(res1['is_json'])
        self.assertTrue(res1['schema_valid'])
        
        # Valid JSON violating schema
        res2 = app_module.validate_output_json_schema('{"count": "not an int"}', schema)
        self.assertTrue(res2['is_json'])
        self.assertFalse(res2['schema_valid'])
        
        # Invalid JSON text
        res3 = app_module.validate_output_json_schema('Not valid json', schema)
        self.assertFalse(res3['is_json'])

    def test_local_providers_allow_missing_key(self):
        for prov in ('ollama', 'lmstudio', 'vllm', 'localai', 'jan'):
            fields, err = app_module._validate_test_request({
                'provider': prov,
                'model': 'local-model',
                'prompt': 'Hello',
                'api_key': ''
            })
            self.assertIsNone(err, f"Provider {prov} should allow missing API key")
            self.assertEqual(fields['provider'], prov)

    def test_test_chat_stream_openai(self):
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
            'data: [DONE]\n\n'
        ]
        fake = FakeResponse(lines=sse_lines, status_code=200)
        with mock.patch.object(app_module.requests, 'post', return_value=fake):
            resp = self.client.post('/test-chat-stream', json={
                'provider': 'openai',
                'api_key': 'sk-test',
                'model': 'gpt-4o',
                'messages': [{'role': 'user', 'content': 'Hi'}]
            })
        self.assertEqual(resp.status_code, 200)
        self.assertIn('text/event-stream', resp.headers.get('Content-Type', ''))
        data_text = resp.get_data(as_text=True)
        self.assertIn('delta', data_text)
        self.assertIn('Hello', data_text)
        self.assertIn('[DONE]', data_text)

    def test_test_suite_matrix(self):
        fake = FakeResponse(json_data={'choices': [{'message': {'content': 'The quick brown fox jumps'}}]})
        with mock.patch.object(app_module.requests, 'post', return_value=fake):
            resp = self.client.post('/test-suite-matrix', json={
                'provider': 'openai',
                'api_key': 'sk-test',
                'models': ['gpt-4o', 'gpt-4o-mini'],
                'custom_prompts': [
                    {'name': 'Fox Test', 'prompt': 'Say quick brown fox', 'expected_keywords': ['quick', 'fox']}
                ]
            })
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn('matrix', body)
        self.assertEqual(len(body['matrix']), 1)
        self.assertEqual(len(body['matrix'][0]['results']), 2)
        self.assertEqual(body['matrix'][0]['results']['gpt-4o']['assertion_passed'], True)
        self.assertEqual(body['matrix'][0]['results']['gpt-4o-mini']['assertion_passed'], True)

    def test_api_pricing_sync(self):
        fake = FakeResponse(json_data={
            'data': [
                {
                    'id': 'openai/gpt-4o',
                    'pricing': {'prompt': '0.000005', 'completion': '0.000015'}
                },
                {
                    'id': 'anthropic/claude-3-5-sonnet',
                    'pricing': {'prompt': '0.000003', 'completion': '0.000015'}
                }
            ]
        }, status_code=200)
        with mock.patch.object(app_module.requests, 'get', return_value=fake):
            resp = self.client.get('/api/pricing/sync')
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertTrue(body['success'])
        self.assertIn('openai/gpt-4o', body['pricing'])
        self.assertEqual(body['pricing']['openai/gpt-4o']['prompt'], 5.0)
        self.assertEqual(body['pricing']['openai/gpt-4o']['completion'], 15.0)

    def test_generate_code_snippets(self):
        resp = self.client.post('/api/generate-code', json={
            'provider': 'openai',
            'model': 'gpt-4o',
            'prompt': 'Write hello world in python',
            'api_key': 'sk-test123'
        })
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn('curl', body)
        self.assertIn('python_requests', body)
        self.assertIn('python_sdk', body)
        self.assertIn('javascript', body)
        self.assertIn('curl -X POST', body['curl'])
        self.assertIn('from openai import OpenAI', body['python_sdk'])
        self.assertIn('fetch(', body['javascript'])

    def test_evaluate_assertion(self):
        # Keyword assertion
        resp1 = self.client.post('/api/evaluate-assertion', json={
            'response': 'The capital of France is Paris.',
            'type': 'keyword',
            'expected': ['Paris']
        })
        self.assertEqual(resp1.status_code, 200)
        self.assertTrue(resp1.get_json()['passed'])

        # Regex assertion
        resp2 = self.client.post('/api/evaluate-assertion', json={
            'response': 'Result status: SUCCESS [code: 200]',
            'type': 'regex',
            'expected': r'SUCCESS\s*\[code:\s*\d+\]'
        })
        self.assertEqual(resp2.status_code, 200)
        self.assertTrue(resp2.get_json()['passed'])

        # Forbidden not_contains assertion
        resp3 = self.client.post('/api/evaluate-assertion', json={
            'response': 'Safe and filtered output',
            'type': 'not_contains',
            'expected': ['forbidden_token', 'unsafe']
        })
        self.assertEqual(resp3.status_code, 200)
        self.assertTrue(resp3.get_json()['passed'])

        # Length assertion
        resp4 = self.client.post('/api/evaluate-assertion', json={
            'response': '12345',
            'type': 'length',
            'min': 3,
            'max': 10
        })
        self.assertEqual(resp4.status_code, 200)
        self.assertTrue(resp4.get_json()['passed'])

    def test_prompt_templates(self):
        resp = self.client.get('/api/prompt-templates')
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn('templates', body)
        self.assertTrue(len(body['templates']) >= 5)
        self.assertTrue(any(t['id'] == 'senior-architect' for t in body['templates']))

    def test_test_suite_matrix_regex(self):
        fake = FakeResponse(json_data={'choices': [{'message': {'content': 'The answer is 42 units.'}}]})
        with mock.patch.object(app_module.requests, 'post', return_value=fake):
            resp = self.client.post('/test-suite-matrix', json={
                'provider': 'openai',
                'api_key': 'sk-test',
                'models': ['gpt-4o'],
                'custom_prompts': [
                    {'name': 'Regex Num Test', 'prompt': 'What is answer?', 'expected': ['regex:\\d+\\s+units']}
                ]
            })
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body['matrix'][0]['results']['gpt-4o']['assertion_passed'], True)


if __name__ == '__main__':
    unittest.main()



