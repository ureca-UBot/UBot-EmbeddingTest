"""Check public worktree candidates or staged bytes. Does not modify Git."""
import argparse
import ast
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
PRIVATE_PARTS = {'faq', 'data', 'datasets', 'source', 'sources', 'authoring', 'review', 'exports', 'cache', 'runtime', '_verification', 'qa', 'node_modules', '__pycache__', '.private'}
CODE_SUFFIXES = {'.py', '.ps1', '.java', '.mjs', '.md', '.yaml', '.yml'}
CONFIG_NAMES = {'.gitignore', '.gitattributes', '.dockerignore', '.env.example'}
FORBIDDEN_COLUMNS = {'question', 'answer', 'query', 'model_query', 'winning_window_text', 'text', 'context', 'user_question', 'faq_question', 'faq_answer'}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def numeric(value):
    if value is None or type(value) in (int, float, bool):
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not re.fullmatch(r'[A-Za-z0-9_@:+./<>=-]+', k):
                raise ValueError('Unexpected metric field')
            numeric(v)
        return
    if isinstance(value, list):
        for v in value:
            numeric(v)
        return
    raise ValueError('Non-numeric payload in public metrics')


def validate_source_rows(rows):
    seen = set()
    for row in rows:
        if set(row) != {'faq_id', 'source_urls'} or not re.fullmatch(r'FAQ-(KT|SKT|LGU)-[a-f0-9]{12}', row['faq_id']):
            raise ValueError('Source index contains unexpected fields/ID')
        if row['faq_id'] in seen:
            raise ValueError('Duplicate source identifier')
        seen.add(row['faq_id'])
        for url in row['source_urls']:
            parsed = urlsplit(url)
            if parsed.scheme != 'https' or parsed.hostname not in {'ermsweb.kt.com', 'globalshop.kt.com', 'www.tworld.co.kr', 'www.lguplus.com'}:
                raise ValueError('Unexpected source URL')


def local_needles():
    """Scan long verbatim passages when the private corpus is available locally."""
    base = ROOT / 'experiments/readme_benchmark/readme_serving_retest/fair_eval_v4/datasets'
    needles = set()
    for carrier in ('kt', 'skt', 'lgu'):
        path = base / carrier / 'faq_pairs.jsonl'
        if not path.exists():
            return None
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            for key in ('question', 'answer'):
                text = re.sub(r'\s+', '', row[key])
                if len(text) >= 32:
                    needles.update(text[i:i + 32] for i in range(0, len(text) - 31, 8))
    return needles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracked', action='store_true', help='Inspect index blobs, including staged deletions')
    args = parser.parse_args()
    policy_path = 'experiments/publication/policy.json'
    def read(path):
        return git('show', ':' + path) if args.tracked else (ROOT / path).read_bytes()
    staged_paths = set(git('ls-files', '--cached', '-z').decode('utf-8').split('\0'))
    # Initial rollout can validate tracked files before the policy itself is staged.
    policy_bytes = git('show', ':' + policy_path) if args.tracked and policy_path in staged_paths else (ROOT / policy_path).read_bytes()
    approved = json.loads(policy_bytes)['approved_data_files']
    command = ['ls-files', '--cached', '-z']
    if not args.tracked:
        command += ['--others', '--exclude-standard']
    paths = sorted(set(s for s in git(*command).decode('utf-8').split('\0') if s))
    needles = local_needles()
    errors = []
    for name in paths:
        try:
            p = Path(name)
            if set(p.parts) & PRIVATE_PARTS:
                raise ValueError('Private/local path is included in Git')
            data = read(name)
            if name in approved:
                reviewed_bytes = data.replace(b'\r\n', b'\n') if approved[name].get('line_endings') == 'LF' else data
                if hashlib.sha256(reviewed_bytes).hexdigest() != approved[name]['sha256']:
                    raise ValueError('Reviewed data artifact changed; review before updating policy')
            elif name != policy_path and not (p.suffix in CODE_SUFFIXES or p.name in CONFIG_NAMES or p.name.startswith(('Dockerfile', 'requirements'))):
                raise ValueError('Unreviewed data/binary artifact')
            if p.suffix == '.csv':
                header = next(csv.reader(io.StringIO(data.decode('utf-8-sig'))))
                if set(map(str.lower, header)) & FORBIDDEN_COLUMNS:
                    raise ValueError('Body column in result CSV')
            if name.endswith('/faq_sources.json'):
                validate_source_rows(json.loads(data))
            if name.endswith(('/aggregate_metrics.json', '/load_metrics.json')):
                numeric(json.loads(data))
            if p.suffix == '.zip':
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    for member in archive.namelist():
                        mp = Path(member)
                        if set(mp.parts) & PRIVATE_PARTS or mp.suffix in {'.jsonl', '.xlsx', '.csv', '.html', '.log', '.zip'}:
                            raise ValueError('Data payload inside code archive')
                    manifest = json.loads(archive.read('ubot-v4-runtime/PACKAGE_CONTENTS.json'))
                    if set(archive.namelist()) != {'ubot-v4-runtime/' + row['path'] for row in manifest['files']} | {'ubot-v4-runtime/PACKAGE_CONTENTS.json'}:
                        raise ValueError('Unexpected archive member')
                    for row in manifest['files']:
                        if hashlib.sha256(archive.read('ubot-v4-runtime/' + row['path'])).hexdigest() != row['sha256']:
                            raise ValueError('Code archive checksum mismatch')
            # Source URLs are deliberately public, and may also occur in FAQ answers.
            if needles and p.suffix in CODE_SUFFIXES | {'.json', '.txt'} and not name.endswith('/faq_sources.json'):
                text = re.sub(r'\s+', '', data.decode('utf-8-sig'))
                if any(text[i:i + 32] in needles for i in range(max(0, len(text) - 31))):
                    raise ValueError('Verbatim passage from private carrier FAQ')
            if p.suffix == '.py':
                ast.parse(data.decode('utf-8-sig'))
        except (ValueError, KeyError, UnicodeError, SyntaxError, zipfile.BadZipFile) as error:
            errors.append({'file': name, 'reason': str(error)})
    print(json.dumps({'checked_files': len(paths), 'private_corpus_text_scan': needles is not None, 'errors': errors}, ensure_ascii=True))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
