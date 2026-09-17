import unittest
from unittest import mock

import requests

import app as app_module
from app import (
    app,
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
)


class FakeResponse:
    """Stand-in for requests.Response, optionally carrying an SSE line stream."""

    def __init__(self, json_data=None, status_code=200, reason='', lines=None, text=None):
        self._json = json_data
        self.status_code = status_code
        self.reason = reason
        self._lines = lines or []
        self.text = text

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
        self.assertEqual(mp.call_args.args[0], 'https://api.openai.com/v1/chat/completions')

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
        p2 = parse_gen_params({'max_tokens': 0, 'temperature': 9, 'system': '  hi '})
        self.assertEqual(p2['max_tokens'], 1)          # clamped up to 1
        self.assertAlmostEqual(p2['temperature'], 2.0)  # clamped to 2.0
        self.assertEqual(p2['system'], 'hi')

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

    def test_update_requires_repo_config(self):
        with mock.patch.object(app_module, 'GITHUB_REPO', ''):
            resp = self.client.post('/update')
        self.assertEqual(resp.status_code, 400)

    def test_update_docker_guidance_when_no_git(self):
        with mock.patch.object(app_module, 'GITHUB_REPO', 'me/repo'), \
             mock.patch.object(app_module, 'RESTART_CMD', ''), \
             mock.patch.object(app_module, '_repo_root', return_value=None):
            resp = self.client.post('/update')
        body = resp.get_json()
        self.assertFalse(body['updated'])
        self.assertTrue(body.get('docker'))
        self.assertIn('docker', body['message'])


if __name__ == '__main__':
    unittest.main()
