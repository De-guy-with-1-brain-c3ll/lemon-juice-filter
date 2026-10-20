"""Register the adapter in a running FCC Admin API; do not change global routing."""
import json
import argparse
import urllib.request
from pathlib import Path

base = 'http://127.0.0.1:8082'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=60) as r:
        return json.load(r)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native', action='store_true', help='Preserve native thinking and structured tool calls')
args = parser.parse_args()
config = call('/admin/api/config')
display_name = 'Lemon Juice Filter Native' if args.native else 'Lemon Juice Filter'
model_id = 'lemon-hybrid'
if args.native:
    with opener.open('http://127.0.0.1:13306/health', timeout=5) as response:
        model_id = json.load(response)['upstream_model']
existing = next((p for p in config['custom_providers'] if p['display_name'] == display_name), None)
values = {'display_name': display_name, 'base_url': 'http://127.0.0.1:13306/' + ('native/v1' if args.native else 'v1'),
    'api_format': 'openai_chat', 'reasoning_format': 'chat_template' if args.native else 'provider_default',
    'reasoning_history_format': 'disabled', 'model_ids': [model_id]}
mutation = {'action': 'update' if existing else 'create', 'values': values}
if existing:
    mutation['provider_id'] = existing['provider_id']
result = call('/admin/api/config/apply', {'values': {}, 'custom_provider': mutation})
if result.get('errors') or result.get('ok') is False:
    raise RuntimeError('FCC rejected the provider: ' + json.dumps({k: result[k] for k in ('ok', 'errors', 'message') if k in result}))
config = call('/admin/api/config')
provider = next((p for p in config['custom_providers'] if p['display_name'] == display_name), None)
if provider is None:
    raise RuntimeError('FCC did not save the provider: ' + json.dumps({k: result[k] for k in ('ok', 'errors', 'message') if k in result}))
ref = provider['provider_id'] + '/' + model_id
runtime = Path(__file__).resolve().parents[1] / 'runtime'
runtime.mkdir(exist_ok=True)
(runtime / ('fcc-native-provider.json' if args.native else 'fcc-provider.json')).write_text(json.dumps({'model': ref}, indent=2) + '\n')
print('FCC model registered:', ref)
print('Global default model was not changed.')
