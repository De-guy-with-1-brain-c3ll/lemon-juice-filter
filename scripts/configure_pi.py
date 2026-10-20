"""Add a standalone local Pi provider; keep existing defaults and credentials intact."""
import datetime
import json
import shutil
from pathlib import Path

path = Path.home() / '.pi' / 'agent' / 'models.json'
config = json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {'providers': {}}
if path.exists():
    backup = path.with_name('models.json.before-lemon-juice-filter-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    shutil.copy2(path, backup)
config.setdefault('providers', {})['lemon-juice-filter'] = {
    'name': 'Lemon Juice Filter', 'api': 'anthropic-messages', 'baseUrl': 'http://127.0.0.1:13306',
    'apiKey': 'local-only', 'models': [{'id': 'lemon-hybrid', 'name': 'Qwen2.5 Coder Hybrid (adapted)',
        'reasoning': False, 'input': ['text'], 'contextWindow': 32768, 'maxTokens': 4096,
        'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0}}]}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
print('Added Pi provider lemon-juice-filter / lemon-hybrid. Existing defaults preserved.')
