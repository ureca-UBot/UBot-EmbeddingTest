"""Scope-safe inputs for serving adapters. No model or gold-dependent routing."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BRANDS={'existing':None,'kt':'KT','skt':'SKT','lgu':'LG U+'}
SPLITS={'development','calibration','holdout'}

def read(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Dataset:
    def __init__(self,dataset_id,root=ROOT):
        if dataset_id not in BRANDS:raise ValueError('Choose exactly one: existing, kt, skt, lgu')
        self.dataset_id=dataset_id;self.root=Path(root)
        m=json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        self.metadata=m['datasets'][dataset_id]
        self.path=self.root/'datasets'/dataset_id
        corpus_path=self.path/'faq_pairs.jsonl'
        if digest(corpus_path)!=self.metadata['corpus_sha256']:raise ValueError('Corpus hash mismatch; rebuild and review')
        self.corpus=read(corpus_path);self.ids={d['faq_id'] for d in self.corpus}
        if len(self.ids)!=len(self.corpus):raise ValueError('Duplicate FAQ IDs')
        self._validate_scope()

    def _validate_scope(self):
        if len(self.corpus)!=len(self.ids) or {d['faq_id'] for d in self.corpus}!=self.ids:
            raise ValueError('Corpus membership changed after scope selection')
        for d in self.corpus:
            if d['dataset_id']!=self.dataset_id:raise ValueError('Cross-dataset embedding input')
            if BRANDS[self.dataset_id] is not None and d['brand']!=BRANDS[self.dataset_id]:raise ValueError('Cross-carrier embedding input')

    def labels(self,split,*,allow_unreviewed=False):
        if split not in SPLITS:raise ValueError('Unknown split')
        questions=self.path/'user_questions.jsonl'
        if digest(questions)!=self.metadata['questions_sha256']:raise ValueError('Question hash mismatch')
        rows=[q for q in read(questions) if q['split']==split]
        for q in rows:
            if not allow_unreviewed and not q.get('eligible_for_final_benchmark',False):
                raise ValueError('Independent label review pending; diagnostic loading requires allow_unreviewed=True')
            if q['dataset_id']!=self.dataset_id:raise ValueError('Cross-dataset query')
            ids=set(q['answer_faq_ids'])
            if not ids<=self.ids:raise ValueError('Cross-dataset answer IDs')
            if ids!={x for f in q['required_facts'] for x in f['acceptable_faq_ids']}:raise ValueError('Answer/fact ID mismatch')
        return rows

    def inputs(self,split,*,allow_unreviewed=False):
        return [{'question_id':q['question_id'],'query':q['model_query']}
                for q in self.labels(split,allow_unreviewed=allow_unreviewed)]

    def documents(self,view):
        self._validate_scope()
        if view not in {'question','question_answer'}:raise ValueError('Unknown document view')
        return [{'faq_id':d['faq_id'],'text':d['question'] if view=='question' else d['question']+'\n\n'+d['answer']} for d in self.corpus]

    def namespace(self,engine,model,representation,view,settings):
        self._validate_scope()
        for field in (engine,representation,view):
            if not re.fullmatch(r'[A-Za-z0-9_.-]+',field):raise ValueError('Invalid namespace component')
        if view not in {'question','question_answer'}:raise ValueError('Unknown document view')
        if not model or not isinstance(settings,dict):raise ValueError('Explicit model and encoder settings required')
        config={'dataset_id':self.dataset_id,'corpus_sha256':self.metadata['corpus_sha256'],
            'engine':engine,'model':model,'representation':representation,'view':view,'settings':settings,
            'serialization':'question-v1' if view=='question' else 'question-blank-line-answer-v1',
            'faq_ids':[d['faq_id'] for d in self.corpus]}
        identity=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        # Full corpus/config hashes avoid accidental truncated namespace reuse.
        path=self.root/'runtime'/self.dataset_id/self.metadata['corpus_sha256']/engine/representation/identity
        return path,config

    @staticmethod
    def validate_cache(saved,expected):
        if saved!=expected:raise ValueError('Index/cache belongs to a different dataset, model, view or encoder configuration')

    def embed_corpus(self,embed,*,engine,model,representation,view,settings,batch_size=32):
        if batch_size<1:raise ValueError('Positive batch size required')
        path,metadata=self.namespace(engine,model,representation,view,settings)
        docs=self.documents(view);vectors=[]
        for offset in range(0,len(docs),batch_size):
            batch=docs[offset:offset+batch_size]
            result=list(embed([d['text'] for d in batch]))
            if len(result)!=len(batch):raise ValueError('Encoder returned wrong number of vectors')
            vectors.extend(result)
        return {'namespace':path,'metadata':metadata,'vectors':vectors}

    def run_namespace(self,*,split,engine,model,representation,view,settings):
        if split not in SPLITS:raise ValueError('Unknown split')
        cache,config=self.namespace(engine,model,representation,view,settings)
        config=dict(config,questions_sha256=self.metadata['questions_sha256'],split=split)
        identity=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return self.root/'outputs'/self.dataset_id/engine/identity,config

    def score(self,question,ranked_ids,k=10):
        if question['dataset_id']!=self.dataset_id:raise ValueError('Cross-dataset scoring label')
        ranked_ids=list(ranked_ids)
        if not set(ranked_ids)<=self.ids:raise ValueError('Ranked IDs outside selected corpus')
        if not set(question['answer_faq_ids'])<=self.ids:raise ValueError('Gold IDs outside selected corpus')
        return score(question,ranked_ids,k)

def score(question,ranked_ids,k=10):
    if k<1:raise ValueError('k must be positive')
    ranked=list(dict.fromkeys(ranked_ids))[:k]
    hits=[bool(set(ranked)&set(f['acceptable_faq_ids'])) for f in question['required_facts']]
    relevant=set(question['answer_faq_ids'])
    return {'hit':any(x in relevant for x in ranked),
        'fact_recall':sum(hits)/len(hits) if hits else None,
        'all_facts_hit':all(hits) if hits else None,
        'mrr':next((1/(i+1) for i,fid in enumerate(ranked) if fid in relevant),0)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=BRANDS)
    p.add_argument('--split',default='development',choices=sorted(SPLITS))
    p.add_argument('--allow-unreviewed',action='store_true');a=p.parse_args()
    d=Dataset(a.dataset)
    print(json.dumps({'dataset_id':d.dataset_id,'faq_count':len(d.corpus),'input_count':len(d.inputs(a.split,allow_unreviewed=a.allow_unreviewed)),
        'review_status':'independent_review_pending','serving_executed':False},ensure_ascii=False))
