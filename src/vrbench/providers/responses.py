"""Direct, zero-tool Responses API transport for every provider."""
import base64
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit
import httpx
from ..io import read, write


def endpoint(raw):
    if not isinstance(raw, str):
        raise ValueError('Responses base_url is required')
    value = raw.strip().rstrip('/')
    u = urlsplit(value)
    if (u.scheme != 'https' or not u.hostname or u.username or u.password
            or u.query or u.fragment or value.endswith('/chat/completions')):
        raise ValueError('Expected HTTPS Responses base URL, not a Chat endpoint')
    return value if value.endswith('/responses') else value + '/responses'


def load_config(path):
    from ..schema import validate
    config = read(path)
    validate('model', config)
    if config.get('credentials_file'):
        p = Path(config['credentials_file']).expanduser()
        config['credentials_file'] = str(p if p.is_absolute() else Path(path).resolve().parent / p)
    return config


def build_request(config, inputs):
    content = [{'type': 'input_text', 'text': inputs['prompt']}]
    for _, path in inputs['images']:
        content.append({'type': 'input_image', 'detail': 'auto',
                        'image_url': 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode('ascii')})
    body = {'model': config['model'], 'input': [{'role': 'user', 'content': content}],
            'tools': [], 'tool_choice': 'none', 'stream': False, 'store': False}
    if config['reasoning_effort'] is not None:
        body['reasoning'] = {'effort': config['reasoning_effort']}
    if config.get('max_output_tokens') is not None:
        body['max_output_tokens'] = config['max_output_tokens']
    return body


def parse_response(data):
    if data.get('status') != 'completed' or data.get('error'):
        raise ValueError('Response not completed; inspect response.json')
    texts = []
    for item in data.get('output', []):
        kind = item.get('type')
        if kind == 'reasoning':
            continue
        if kind != 'message' or item.get('role') != 'assistant':
            raise ValueError('Unexpected or tool output item: ' + str(kind))
        for part in item.get('content', []):
            if part.get('type') != 'output_text' or not isinstance(part.get('text'), str):
                raise ValueError('Refusal or unsupported response content')
            texts.append(part['text'])
    text = ''.join(texts)
    if not text.strip():
        raise ValueError('Missing response text')
    return text


class ProviderBlocked(RuntimeError):
    """Service or account failure, rather than an invalid model prediction."""


def call(config, inputs, output):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    credentials = read(config['credentials_file']) if config.get('credentials_file') else {}
    environment_key = os.environ.get(config.get('api_key_env', 'VRBENCH_API_KEY'))
    secret = config.get('api_key') or environment_key or credentials.get('api_key')
    if not isinstance(secret, str) or not secret.strip():
        raise ValueError('Missing API key')
    base = config.get('base_url') or credentials.get('base_url') or credentials.get('responses_base_url')
    chat = config['transport'] == 'chat_completions'
    body = build_request(config, inputs)
    if chat:
        u = urlsplit(base)
        if u.scheme!='https' or not u.hostname or u.username or u.password or u.query or u.fragment:
            raise ValueError('Expected HTTPS Chat Completions base URL')
        url = base.rstrip('/')
        if not url.endswith('/chat/completions'):
            url += '/chat/completions'
        parts = body['input'][0]['content']
        message = [{'type':'text','text':p['text']} if p['type']=='input_text' else
                   {'type':'image_url','image_url':{'url':p['image_url']}} for p in parts]
        body = dict(model=config['model'],messages=[dict(role='user',content=message)],
                    tools=[],tool_choice='none',stream=False)
        if config['reasoning_effort'] is not None:
            body['reasoning_effort'] = config['reasoning_effort']
        if config.get('max_output_tokens') is not None:
            body['max_completion_tokens'] = config['max_output_tokens']
    else:
        url = endpoint(base)
    def safe(text):
        return text.replace(secret, '[REDACTED]')
    def save(name, value):
        (out / name).write_text(safe(json.dumps(value, ensure_ascii=False, indent=2)), encoding='utf8')
    save('request.json', body)
    metadata = {'transport': config['transport'], 'requested_model': config['model'], 'endpoint': url,
                'tool_count': 0, 'automatic_retries': 0, 'model_context_equivalence_claimed': False}
    start = time.monotonic()
    try:
        with httpx.Client(timeout=config.get('timeout_seconds', 900), follow_redirects=False) as client:
            response = client.post(url, headers={'Authorization': 'Bearer ' + secret}, json=body)
        metadata['http_status'] = response.status_code
        (out / 'response.txt').write_text(safe(response.text), encoding='utf8')
        if response.status_code in (401,402,403,404,429):
            raise ProviderBlocked(f'Provider HTTP {response.status_code}; inspect transport/response.txt')
        if response.status_code != 200:
            raise RuntimeError('Provider HTTP ' + str(response.status_code) + '; inspect transport/response.txt')
        data = response.json()
        save('response.json', data)
        metadata.update(resolved_model=data.get('model'), usage=data.get('usage'), response_id=data.get('id'))
        if chat:
            choice, = data['choices']
            msg = choice['message']
            if choice['finish_reason']!='stop' or msg.get('tool_calls') or msg.get('function_call'):
                raise ValueError('Incomplete response or tool output')
            if not isinstance(msg.get('content'),str) or not msg['content'].strip():
                raise ValueError('Missing response text')
            return safe(msg['content'])
        return safe(parse_response(data))
    except httpx.RequestError:
        raise RuntimeError('Provider network/timeout error; request was not retried') from None
    finally:
        metadata['elapsed_seconds'] = time.monotonic() - start
        save('metadata.json', metadata)
