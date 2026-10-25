"""Add a standalone local Pi provider; keep existing defaults and credentials intact."""
import datetime
import json
import os
import shutil
import urllib.request
from pathlib import Path

base = os.getenv('LEMON_FILTER_URL', 'http://127.0.0.1:13306').rstrip('/')
name, context, max_tokens = 'Hybrid model (adapted)', int(os.getenv('LEMON_CONTEXT_WINDOW', '32768')), 4096
try:
    # Advertise the verified window instead of a stale constant, so Pi does not
    # send prompts the backend silently truncates.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base + '/v1/models', timeout=5) as response:
        model = json.load(response)['data'][0]
    name = model.get('display_name') or name
    context = model.get('context_length') or context
    max_tokens = model.get('maxCompletionTokens') or max_tokens
except (OSError, ValueError, KeyError, IndexError, TypeError):
    pass

path = Path.home() / '.pi' / 'agent' / 'models.json'
config = json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {'providers': {}}
if path.exists():
    backup = path.with_name('models.json.before-lemon-juice-filter-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    shutil.copy2(path, backup)
config.setdefault('providers', {})['lemon-juice-filter'] = {
    'name': 'Lemon Juice Filter', 'api': 'anthropic-messages', 'baseUrl': base,
    'apiKey': 'local-only', 'models': [{'id': 'lemon-hybrid', 'name': name,
        'reasoning': False, 'input': ['text'], 'contextWindow': context, 'maxTokens': max_tokens,
        'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0}}]}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
print('Added Pi provider lemon-juice-filter / lemon-hybrid. Existing defaults preserved.')
