"""Integrity and failure-mode checks. These do not certify semantic labels."""
import copy
import json
import tempfile
import unittest
from collections import Counter,defaultdict
from pathlib import Path
import build_dataset as b
from dataset import Dataset,BRANDS,SPLITS,score

ORIGINALS={
 '../data/corpus.jsonl':'c361681bb3911fa25d2ba9d4c339637bd4e13a77bc827874242697731444913d',
 '../data/cases.jsonl':'64ed82884092d5eaa57159e90f756f5843ef7bcd00572abb6fa167fd05a85b95',
 '../data/evaluation_cases.jsonl':'c54ee3358199e1fbef268c22488fdb14f3cfcfa0026b2d74dab826c47a261a0a',
 '../fair_eval_v2/data/cases.jsonl':'3b38eddad3e8cf5fe47a9d3688b66d5a4d1b8794a0184297a28c37790ffd9cf8',
}

class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets={k:Dataset(k) for k in BRANDS}
        cls.questions={k:b.read(d.path/'user_questions.jsonl') for k,d in cls.datasets.items()}
        cls.manifest=json.loads((b.ROOT/'manifest.json').read_text(encoding='utf-8'))

    def test_01_originals_unchanged(self):
        for path,expected in ORIGINALS.items():self.assertEqual(b.sha(b.ROOT/path),expected,path)
        self.assertEqual(b.sha(b.V3/'manifest.json'),'ca8ea2c3dfc361a901116e3ed1b7564cb8705002fb51ecbf38101576e9256bca')
        self.assertEqual(b.sha(b.V3/'source/faqs.jsonl'),'76d867e4b4e183e6b52c8edfb31ec7da4e74f8495b4a2a2ce10fb2472ecf8a2d')

    def test_02_exact_four_scoped_corpora(self):
        self.assertEqual(set(self.manifest['dataset_ids']),set(BRANDS))
        self.assertEqual({p.name for p in (b.ROOT/'datasets').iterdir() if p.is_dir()},set(BRANDS))
        counts={'existing':1024,'kt':1099,'skt':1305,'lgu':131}
        all_ids=[]
        for k,d in self.datasets.items():
            self.assertEqual(len(d.corpus),counts[k]);all_ids+=list(d.ids)
            if BRANDS[k]:self.assertEqual({r['brand'] for r in d.corpus},{BRANDS[k]})
        self.assertEqual(len(all_ids),len(set(all_ids)))
        for invalid in ('all','pooled','LG U+','kt,skt','../kt'):
            with self.assertRaises(ValueError):Dataset(invalid)

    def test_03_existing_copy_preserves_original_content(self):
        for old,new in zip(b.read(b.ROOT/'../data/corpus.jsonl'),self.datasets['existing'].corpus):
            for key,value in old.items():self.assertEqual(new[key],value)
        for key,d in self.datasets.items():
            self.assertEqual(b.sha(d.path/'faq_pairs.jsonl'),b.sha(b.V3/'datasets'/key/'faq_pairs.jsonl'))
            before=b.read(b.V3/'datasets'/key/'user_questions.jsonl')
            current={q['question_id']:q for q in self.questions[key]}
            for q in before:
                for field in ('query','user_question','answer_faq_ids'):
                    self.assertEqual(q[field],current[q['question_id']][field])

    def test_04_labels_and_evidence_refer_to_own_corpus(self):
        for key,d in self.datasets.items():
            docs={x['faq_id']:x for x in d.corpus}
            facts=b.read(d.path/'facts.jsonl');fm={f['fact_id']:f for f in facts}
            self.assertEqual(len(facts),len(fm))
            for f in facts:
                self.assertTrue(set(f['acceptable_faq_ids'])<=d.ids)
                for s in f['support']:
                    self.assertIn(s['faq_id'],f['acceptable_faq_ids'])
                    for quote in s.get('quotes',[s.get('quote')]):
                        self.assertTrue(quote);self.assertIn(quote,docs[s['faq_id']]['answer'])
            for split in SPLITS:
                for q in d.labels(split,allow_unreviewed=True):
                    self.assertEqual(q['answer_faq_ids'],q['expected_faq_ids'])
                    for f in q['required_facts']:
                        if f['fact_id'] in fm:self.assertEqual(f['acceptable_faq_ids'],fm[f['fact_id']]['acceptable_faq_ids'])
                    for near in q.get('near_miss_candidates',[]):
                        self.assertIn(near['faq_id'],d.ids);self.assertNotIn(near['faq_id'],q['answer_faq_ids'])

    def test_05_unique_fresh_questions(self):
        total=0
        for key,qs in self.questions.items():
            self.assertEqual(len(qs),len({q['question_id'] for q in qs}))
            if key=='existing':continue
            total+=len(qs)
            self.assertEqual(len(qs),len({b.norm(q['user_question']) for q in qs}))
            source_questions={b.norm(d['question']) for d in self.datasets[key].corpus}
            self.assertFalse(source_questions&{b.norm(q['user_question']) for q in qs})
        self.assertGreaterEqual(total,1500);self.assertLessEqual(total,3000)

    def test_06_model_inputs_and_document_views_exclude_gold(self):
        for key,d in self.datasets.items():
            for split in SPLITS:
                exported=b.read(d.path/'inputs'/f'{split}.jsonl')
                self.assertEqual(exported,d.inputs(split,allow_unreviewed=True))
                self.assertTrue(all(set(x)=={'question_id','query'} for x in exported))
                self.assertEqual(b.read(d.path/'splits'/f'{split}.jsonl'),d.labels(split,allow_unreviewed=True))
            for view in ('question','question_answer'):
                docs=d.documents(view)
                for source,doc in zip(d.corpus,docs):
                    self.assertEqual(set(doc),{'faq_id','text'})
                    expected=source['question'] if view=='question' else source['question']+'\n\n'+source['answer']
                    self.assertEqual(doc['text'],expected)

    def test_07_embedding_is_scoped_before_encoder_call(self):
        for key,d in self.datasets.items():
            seen=[]
            def encoder(texts):
                seen.extend(texts)
                return [[float(i),1.0] for i in range(len(texts))]
            result=d.embed_corpus(encoder,engine='fake',model='test-model',representation='dense',
                view='question_answer',settings={'revision':'test','normalize':True},batch_size=37)
            self.assertEqual(seen,[r['text'] for r in d.documents('question_answer')])
            self.assertEqual(len(result['vectors']),len(d.corpus))
            self.assertEqual(result['metadata']['dataset_id'],key)
            self.assertIn(key,result['namespace'].parts)
        with self.assertRaises(ValueError):
            d.embed_corpus(lambda x:[],engine='fake',model='test',representation='dense',view='question',settings={})

    def test_08_cache_namespace_and_mismatch_rejection(self):
        configs=[];paths=[]
        for d in self.datasets.values():
            for engine in ('ollama','vllm','tei'):
                for rep in ('dense','sparse','multi_vector'):
                    for view in ('question','question_answer'):
                        path,cfg=d.namespace(engine,'BAAI/bge-m3',rep,view,{'normalize':True,'revision':'snapshot-A'})
                        paths.append(str(path));configs.append(cfg)
        self.assertEqual(len(paths),len(set(paths)))
        Dataset.validate_cache(configs[0],copy.deepcopy(configs[0]))
        for field,value in [('dataset_id','wrong'),('model','different'),('settings',{'normalize':False}),('faq_ids',[])]:
            bad=copy.deepcopy(configs[0]);bad[field]=value
            with self.assertRaises(ValueError):Dataset.validate_cache(bad,configs[0])

    def test_09_split_leakage_known_connections(self):
        for key,qs in self.questions.items():
            seen=defaultdict(set)
            for q in qs:
                identities=[q['leakage_group_id']]+q['answer_faq_ids']+q.get('leakage_anchor_ids',[])+q.get('pair_ids',[])
                identities += [f['fact_id'] for f in q['required_facts']]
                identities += [u['missing_fact_id'] for u in q.get('unsupported_facts',[])]
                identities += q.get('component_question_ids',[])
                for identity in identities:seen[identity].add(q['split'])
            self.assertFalse({x:s for x,s in seen.items() if len(s)>1},key)

    def test_10_controls_keep_ground_truth_and_split(self):
        for key,qs in self.questions.items():
            if key=='existing':continue
            qm={q['question_id']:q for q in qs}
            for q in qs:
                if 'control_of' not in q:continue
                original=qm[q['control_of']]
                for field in ('answer_faq_ids','required_facts','split','corpus_status'):
                    self.assertEqual(q[field],original[field])
                self.assertFalse(q['binary_fit_eligible'])
                if q['primary_purpose']=='CONTEXT_RETRIEVAL':
                    self.assertTrue(q['dialogue_history'])
                    expected='이전 대화:\n'+'\n'.join(x['role']+': '+x['content'] for x in q['dialogue_history'])+'\n현재 질문:\n'+q['query']
                    self.assertEqual(q['model_query'],expected)
                    self.assertEqual(q['rewrite_method'],'context_concat_v1')
                    self.assertNotEqual(q['query'],q['model_query'])

    def test_11_all_facts_versus_any_alternative(self):
        q={'answer_faq_ids':['a','b','c'],'required_facts':[
            {'acceptable_faq_ids':['a','b']},{'acceptable_faq_ids':['c']}]}
        partial=score(q,['b','z'],2)
        self.assertTrue(partial['hit']);self.assertFalse(partial['all_facts_hit'])
        self.assertEqual(partial['fact_recall'],0.5)
        self.assertTrue(score(q,['b','c'],2)['all_facts_hit'])
        self.assertFalse(score(q,['b','c'],1)['all_facts_hit'])
        q['required_facts'].append({'acceptable_faq_ids':[]})
        with self.assertRaises(ValueError):score(q,['a','b','c'],3)

    def test_12_cross_corpus_scoring_rejected(self):
        kt=self.datasets['kt'];q=self.questions['kt'][0]
        with self.assertRaises(ValueError):kt.score(q,[next(iter(self.datasets['skt'].ids))])
        with self.assertRaises(ValueError):kt.score(self.questions['skt'][0],[])

    def test_13_review_and_absence_fit_guard(self):
        self.assertFalse(self.manifest['threshold_fitting_ready'])
        self.assertFalse(self.manifest['serving_runs_performed'])
        for key,d in self.datasets.items():
            with self.assertRaises(ValueError):d.inputs('holdout')
            for q in self.questions[key]:
                self.assertFalse(q['eligible_for_final_benchmark'])
                if key!='existing' and q['corpus_status']!='FAQ_EXISTS':self.assertFalse(q['binary_fit_eligible'])

    def test_14_wrong_brand_and_gold_rejected_even_with_updated_hash(self):
        for corrupt in ('brand','gold'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);out=root/'datasets/kt'
                corpus=copy.deepcopy(self.datasets['kt'].corpus)
                qs=copy.deepcopy(self.questions['kt'])
                m=copy.deepcopy(self.manifest)
                if corrupt=='brand':corpus[0]['brand']='SKT'
                else:qs[0]['answer_faq_ids']=[next(iter(self.datasets['skt'].ids))]
                b.write(out/'faq_pairs.jsonl',corpus);b.write(out/'user_questions.jsonl',qs)
                m['datasets']['kt']['corpus_sha256']=b.sha(out/'faq_pairs.jsonl')
                m['datasets']['kt']['questions_sha256']=b.sha(out/'user_questions.jsonl')
                (root/'manifest.json').write_text(json.dumps(m),encoding='utf-8')
                with self.assertRaises(ValueError):
                    d=Dataset('kt',root)
                    d.labels(qs[0]['split'],allow_unreviewed=True)

    def test_15_output_namespace_contains_question_and_corpus_identity(self):
        paths=[]
        for d in self.datasets.values():
            for split in SPLITS:
                path,meta=d.run_namespace(split=split,engine='ollama',model='bge-m3',representation='dense',view='question',settings={})
                self.assertEqual(meta['questions_sha256'],d.metadata['questions_sha256'])
                self.assertEqual(meta['corpus_sha256'],d.metadata['corpus_sha256'])
                paths.append(str(path))
        self.assertEqual(len(paths),len(set(paths)))

    def test_16_accidental_pooling_blocked_before_encoder(self):
        d=Dataset('kt');d.corpus.append(copy.deepcopy(self.datasets['skt'].corpus[0]))
        calls=[]
        with self.assertRaises(ValueError):
            d.embed_corpus(lambda texts:calls.append(texts),engine='fake',model='test',representation='dense',view='question',settings={})
        self.assertEqual(calls,[])

    def test_17_every_dataset_passes_every_core_purpose(self):
        coverage=json.loads((b.ROOT/'data/coverage_matrix.json').read_text(encoding='utf-8'))
        diagnostics=json.loads((b.ROOT/'data/diagnostic_coverage.json').read_text(encoding='utf-8'))
        for key in BRANDS:
            self.assertEqual(set(coverage[key]),set(b.PURPOSES))
            for purpose,v in coverage[key].items():
                self.assertGreaterEqual(v['distinct_scenarios'],10,(key,purpose))
                self.assertGreaterEqual(v['evidence_units'],10,(key,purpose))
                self.assertTrue(v['meets_authoring_minimum'])
            for metric in ('question_cue_rows','answer_only_source_cue_rows','distributed_document_rows','authored_hard_negative_rows'):
                self.assertGreaterEqual(diagnostics[key][metric],10,(key,metric))

    def test_18_partial_and_tool_requests_not_mis_scored(self):
        tools=0
        for key,qs in self.questions.items():
            d=self.datasets[key]
            for q in qs:
                self.assertTrue(all(f['acceptable_faq_ids'] for f in q['required_facts']))
                result=d.score(q,q['answer_faq_ids'],max(1,len(q['answer_faq_ids'])))
                if q['corpus_status']=='PARTIAL_FACTS_EXIST':
                    self.assertTrue(q['unsupported_facts']);self.assertTrue(q['required_facts'])
                    self.assertTrue(result['all_facts_hit']);self.assertEqual(result['fact_recall'],1)
                    self.assertFalse(result['whole_request_evidence_hit'])
                    self.assertIsNone(q['binary_answerability_target'])
                if q['tool_requirements']:
                    tools+=1;self.assertNotEqual(q['corpus_status'],'NO_FAQ')
                    self.assertFalse(result['whole_request_evidence_hit'])
                    self.assertIsNone(q['binary_answerability_target'])
                if not q['required_facts']:
                    for name in ('hit','fact_recall','all_facts_hit','mrr'):self.assertIsNone(result[name])
        self.assertEqual(tools,13)

    def test_19_document_attack_and_repeats_are_isolated(self):
        for key,d in self.datasets.items():
            qs={q['question_id']:q for q in self.questions[key]}
            repeat=b.read(d.path/'fixtures/repeat.jsonl');attack=b.read(d.path/'fixtures/document_attack.jsonl')
            self.assertEqual(len(repeat),10);self.assertEqual(len(attack),10)
            for x in repeat:
                self.assertEqual(x['runs'],10);self.assertFalse(x['executed'])
                self.assertEqual(x['model_query'],qs[x['question_id']]['model_query'])
            for x in attack:
                self.assertIn(x['faq_id'],d.ids);self.assertFalse(x['executed'])
                self.assertFalse(x['source_corpus_mutation_allowed'])
                self.assertTrue(set(x['expected_faq_ids'])<=d.ids)
                self.assertFalse(any(x['forbidden_output_marker'] in doc['answer'] for doc in d.corpus))

    def test_20_context_queries_use_no_gold(self):
        for qs in self.questions.values():
            for q in qs:
                if q['primary_purpose']!='CONTEXT_RETRIEVAL':continue
                expected='이전 대화:\n'+'\n'.join(x['role']+': '+x['content'] for x in q['dialogue_history'])+'\n현재 질문:\n'+q['query']
                self.assertEqual(q['model_query'],expected)

    def test_21_source_cues_have_literal_evidence(self):
        for panel in b.read(b.ROOT/'authoring/cue_panels.jsonl'):
            d=next(d for d in self.datasets[panel['dataset_id']].corpus if d['faq_id']==panel['source_faq_id'])
            if panel['panel']=='answer_only_source_cue':
                self.assertIn(b.norm(panel['cue']),b.norm(d['answer']))
                self.assertNotIn(b.norm(panel['cue']),b.norm(d['question']))

    def test_22_deterministic_build(self):
        tracked=[b.ROOT/'manifest.json',b.ROOT/'data/coverage_matrix.json']
        tracked += [d.path/'user_questions.jsonl' for d in self.datasets.values()]
        before={str(p):b.sha(p) for p in tracked};b.build()
        self.assertEqual(before,{str(p):b.sha(p) for p in tracked})

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(DatasetTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'passed':result.wasSuccessful(),'semantic_review_certified':False,'real_encoder_called':False,
        'manifest_sha256':b.sha(b.ROOT/'manifest.json')}
    (b.ROOT/'data/validation_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
