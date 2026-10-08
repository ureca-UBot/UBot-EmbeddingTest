"""Record frozen-test failures without changing the selected architecture."""
from run_retrieval_benchmark import OUT,jsread,save

def main():
    frozen=jsread(OUT/'selection.json');name=frozen['selected_with_top1_gate']
    holdout=jsread(OUT/'holdout/policy_results.json')
    selected=next(r for r in holdout['results'] if r['experiment']==name)
    legacy=jsread(OUT/'regression/legacy_235/summary.json')['cohorts']['legacy_regression']['family_mean']
    new=jsread(OUT/'regression/selected-legacy_235/summary.json')['cohorts']['legacy_regression']['family_mean']
    delta=new['semantic_hit@1']-legacy['semantic_hit@1']
    checks=[{'gate':'NO_FAQ unsafe acceptance <= 1%','passed':selected['no_faq']['unsafe_accept_rate']<=.01,
             'observed':selected['no_faq']['unsafe_accept_rate'],'counts':{'accepted_no_faq':selected['no_faq']['fn'],'no_faq_rows':selected['no_faq']['tp']+selected['no_faq']['fn']}},
            {'gate':'legacy_235 Top1 loss <= 1 percentage point','passed':delta>=-.01,'observed_delta':delta},
            {'gate':'independent domain review complete','passed':False,'status':'SOURCE_LABELS_PENDING'},
            {'gate':'reviewed EXACT_ENTITY >= 95%','passed':None,'status':'NOT_ASSESSABLE_DESIGN_EXAMPLES_ONLY'},
            {'gate':'REAL_FAILURE service regressions','passed':None,'status':'NO_SERVICE_LOGS_SUPPLIED'}]
    save(OUT/'quality-gates.json',{'calibration_selected_architecture':name,'production_adoption':False,
         'verdict':'DO_NOT_DEPLOY_FROZEN_POLICY','checks':checks,'holdout_retuned':False,
         'next_data_version_required_for_further_tuning':True})
    print((OUT/'quality-gates.json').read_text(encoding='utf-8'),flush=True)

if __name__=='__main__':main()
