"""Posthoc meaning-pair check on the frozen architecture; never a tuning set."""
from collections import defaultdict
import numpy as np
from run_retrieval_benchmark import OUT,DATA,lines,jsread,save,save_lines,rank,setup
from run_selected_regression import FrozenRetriever

def main():
    setup();name=jsread(OUT/'selection.json')['selected_with_top1_gate'];model=FrozenRetriever(name)
    model.set_corpus(lines(DATA/'corpus.jsonl'))
    pairs=[r for r in lines(OUT/'diagnostics/language-pairs.jsonl') if r['view']=='question']
    scores=model.score([r['query'] for r in pairs]);results=[]
    for r,s in zip(pairs,scores):
        ordered=[model.ids[i] for i in rank(s,model.ids) if s[i]>-1e19]
        rank_value=ordered.index(r['faq_id'])+1 if r['faq_id'] in ordered else None
        results.append({**{k:v for k,v in r.items() if k not in ['rank','hit_at_3','top_10','view']},
                        'architecture':name,'rank':rank_value,'hit_at_3':int(r['faq_id'] in ordered[:3]),'top_10':ordered[:10]})
    save_lines(OUT/'diagnostics/selected-language-pairs.jsonl',results)
    groups=defaultdict(dict)
    for r in results:groups[r['pair_id']][r['variant']]=r
    save(OUT/'diagnostics/selected-language-summary.json',{'architecture':name,'pairs':len(groups),
         'pair_hit_at_3':float(np.mean([g['ko_anchor']['hit_at_3'] and g['en_term']['hit_at_3'] for g in groups.values()])),
         'anchor_hit_at_3':float(np.mean([g['ko_anchor']['hit_at_3'] for g in groups.values()])),
         'en_term_hit_at_3':float(np.mean([g['en_term']['hit_at_3'] for g in groups.values()])),
         'pair_top10_overlap':float(np.mean([len(set(g['ko_anchor']['top_10'])&set(g['en_term']['top_10']))/10 for g in groups.values()])),
         'stage1_misses':sum(r['rank'] is None for r in results),'status':'posthoc_grounded_terms_pending_review',
         'used_to_change_selection_or_threshold':False,'full_sentence_translation_coverage':False})
    print((OUT/'diagnostics/selected-language-summary.json').read_text(encoding='utf-8'),flush=True)

if __name__=='__main__':main()
