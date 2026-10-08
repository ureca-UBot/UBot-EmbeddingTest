import argparse,importlib.metadata as m,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['benchmark','jina'],required=True);a=p.parse_args()
expected=json.loads((ROOT/'environment'/f'{a.kind}.json').read_text())
actual={d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}
different={name:{'expected':version,'actual':actual.get(name)} for name,version in expected['packages'].items() if actual.get(name)!=version}
if sys.version.split()[0]!=expected['python'].split()[0]:different['python']={'expected':expected['python'],'actual':sys.version}
print(json.dumps({'kind':a.kind,'checked_packages':len(expected['packages']),'differences':different}))
if different:raise SystemExit(1)
