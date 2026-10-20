"""Register the adapter in a running FCC Admin API; do not change global routing."""
import json
import urllib.request
from pathlib import Path

base = 'http://127.0.0.1:8082'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=60) as r:
        return json.load(r)


config = call('/admin/api/config')
existing = next((p for p in config['custom_providers'] if p['display_name'] == 'Lemon Juice Filter'), None)
values = {'display_name': 'Lemon Juice Filter', 'base_url': 'http://127.0.0.1:13306/v1',
    'api_format': 'openai_chat', 'reasoning_format': 'provider_default',
    'reasoning_history_format': 'disabled', 'model_ids': ['lemon-hybrid']}
mutation = {'action': 'update' if existing else 'create', 'values': values}
if existing:
    mutation['provider_id'] = existing['provider_id']
result = call('/admin/api/config/apply', {'values': {}, 'custom_provider': mutation})
if result.get('errors') or result.get('ok') is False:
    raise RuntimeError('FCC rejected the provider: ' + json.dumps({k: result[k] for k in ('ok', 'errors', 'message') if k in result}))
config = call('/admin/api/config')
provider = next((p for p in config['custom_providers'] if p['display_name'] == 'Lemon Juice Filter'), None)
if provider is None:
    raise RuntimeError('FCC did not save the provider: ' + json.dumps({k: result[k] for k in ('ok', 'errors', 'message') if k in result}))
ref = provider['provider_id'] + '/lemon-hybrid'
runtime = Path(__file__).resolve().parents[1] / 'runtime'
runtime.mkdir(exist_ok=True)
(runtime / 'fcc-provider.json').write_text(json.dumps({'model': ref}, indent=2) + '\n')
print('FCC model registered:', ref)
print('Global default model was not changed.')
