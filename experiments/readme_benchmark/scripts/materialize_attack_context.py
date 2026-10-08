"""Build actual contexts from saved attack rankings, without model reexecution."""
import hashlib,json
from pathlib import Path
from validate_dataset import catalog,lines,DATA
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/run-v1'
cases={c['case_id']:c for c in lines(DATA/'cases.jsonl')}
catalogs=catalog();contexts=[]
for r in lines(OUT/'diagnostics/attacks/per_query.jsonl'):
    c=cases[r['case_id']];docs=catalogs[c['corpus_version']]
    parts=[f"[{i}]\n질문: {docs[i]['question']}\n답변: {docs[i]['answer']}" for i in r['top_ids'][:3]]
    context='\n\n'.join(parts);fixture=c['attack_context']['fixture_document_ids'][0]
    exposed=docs[fixture]['attack_text'] in context
    assert exposed==(fixture in r['top_ids'][:3]),'Context source/exposure mismatch'
    contexts.append({'case_id':c['case_id'],'format_version':'faq_content_source_ids_v1','source_ids':r['top_ids'][:3],
                     'context':context,'context_hash':hashlib.sha256(context.encode()).hexdigest(),
                     'attack_text_present_in_constructed_context':exposed,'llm_input_exposure':'NOT_EXECUTED',
                     'generation_attack_success':'NOT_EXECUTED'})
p=OUT/'diagnostics/attacks/constructed_contexts.jsonl'
p.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in contexts),encoding='utf-8')
print(json.dumps({'actual_contexts_built':len(contexts),'contexts_containing_attack':sum(c['attack_text_present_in_constructed_context'] for c in contexts),
                  'model_queries_reexecuted':0,'llm_requests':0}))
