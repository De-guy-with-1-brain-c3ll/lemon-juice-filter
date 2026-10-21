"""Enable chunked prefill and an explicit prompt limit for a local hybrid model."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model-dir', type=Path, required=True)
parser.add_argument('--prompt-limit', type=int, default=16384)
parser.add_argument('--ryzenai-version', default='1.7.0')
args = parser.parse_args()
root = args.model_dir.resolve(strict=True)
genai = root / 'genai_config.json'
config = json.loads(genai.read_text(encoding='utf-8'))
providers = config['model']['decoder']['session_options'].get('provider_options', [])
provider = next((p['RyzenAI'] for p in providers if 'RyzenAI' in p), None)
if provider is None or provider.get('hybrid_opt_token_backend') == 'npu':
    parser.error('This utility supports hybrid models only.')
if not 4096 <= args.prompt_limit <= 16384 or args.prompt_limit + 4096 > config['model']['context_length']:
    parser.error('Choose a prompt limit from 4096 to 16384, reserving 4096 output tokens within model context_length.')
rai_path = root / 'rai_config.json'
rai = json.loads(rai_path.read_text(encoding='utf-8')) if rai_path.exists() else {}
backup = Path(__file__).resolve().parents[1] / 'runtime' / f'context-backup-{time.time_ns()}'
backup.mkdir(parents=True)
for path in (genai, rai_path):
    if path.exists():
        shutil.copyfile(path, backup / path.name)
provider['hybrid_opt_chunk_context'] = '1'
config.setdefault('search', {})['chunk_size'] = 2048
rai.setdefault('max_prompt_length', {})[args.ryzenai_version] = args.prompt_limit
for path, value in ((genai, config), (rai_path, rai)):
    # Replace the snapshot entry rather than modifying a shared HF cache blob.
    temporary = path.with_name(path.name + '.lemon-tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)
print(f'Configured {args.prompt_limit} prompt tokens with chunked prefill. Reload the model and verify backend /health.')
print(f'Original configuration backed up to {backup}')
