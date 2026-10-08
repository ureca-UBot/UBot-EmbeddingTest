import copy,json,unittest,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from retrieval_common import fact_metrics,semantic_contract_errors
from validate_dataset import catalog,lines,DATA

class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cases=lines(DATA/'cases.jsonl');cls.catalog=catalog()
    def test_fresh_provenance(self):
        self.assertTrue(all(c['source']['origin']=='authored' and c['source']['row'] is None and c['source']['original_question_id'] is None for c in self.cases))
    def test_dataset_semantic_contract(self):self.assertEqual([],semantic_contract_errors(self.cases,self.catalog))
    def test_fact_logic_is_and_of_or(self):
        groups=[{'fact_id':'a','primary_ids':['a'],'acceptable_ids':['a2']},{'fact_id':'b','primary_ids':['b'],'acceptable_ids':[]}]
        one=fact_metrics(['a2','x'],groups,2);both=fact_metrics(['a2','b'],groups,2)
        self.assertEqual(.5,one['fact_recall']);self.assertFalse(one['all_facts_hit']);self.assertTrue(both['all_facts_hit'])
    def test_changed_pair_gt_is_rejected(self):
        pair=next(c['semantic_pair_id'] for c in self.cases if c.get('semantic_pair_id'))
        group=copy.deepcopy([c for c in self.cases if c.get('semantic_pair_id')==pair]);group[1]['fact_groups'][0]['primary_ids']=['FAQ-001']
        self.assertTrue(any('language pair changed fact_groups' in e for e in semantic_contract_errors(group,self.catalog)))
    def test_changed_pair_split_is_rejected(self):
        pair=next(c['semantic_pair_id'] for c in self.cases if c.get('semantic_pair_id'))
        group=copy.deepcopy([c for c in self.cases if c.get('semantic_pair_id')==pair]);group[1]['split']='holdout' if group[0]['split']=='calibration' else 'calibration'
        self.assertTrue(any('split leakage' in e for e in semantic_contract_errors(group,self.catalog)))
    def test_no_faq_is_not_personal_data_absence(self):
        self.assertTrue(all(c['track']!='no_faq' and c['retrieval_status']=='PARTIAL_FACTS_EXIST' for c in self.cases if c['primary_type']=='MULTI_INTENT'))
    def test_attack_preserves_fact_and_replaces_one_doc(self):
        base=self.catalog['readme-faq-1024-v1']
        for d in lines(DATA/'attack_fixtures.jsonl'):
            self.assertTrue(d['answer'].startswith(base[d['base_faq_id']]['answer']))
            self.assertNotIn(d['base_faq_id'],self.catalog[d['corpus_version']])
            self.assertEqual(1024,len(self.catalog[d['corpus_version']]))
    def test_repeat_not_counted_as_unique_cases(self):
        manifest=json.loads((DATA/'split_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(40,sum(c['track']=='repeat' for c in self.cases));self.assertFalse(manifest['repeat_executions_are_independent_cases'])

if __name__=='__main__':unittest.main()
