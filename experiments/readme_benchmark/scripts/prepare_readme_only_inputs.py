"""Extract only the requested README and FAQ knowledge, never old test rows."""
import hashlib
import json
from pathlib import Path
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data'
README = Path(r'C:\Users\eongp\.codex\attachments\5a4cf486-832a-47cb-84e7-0ba4de6b422d\붙여넣은 텍스트.txt')
FAQ = Path(r'C:\Users\eongp\Downloads\FAQ_RAG_3000건_사전Context제거_피드백수정본(1).xlsx')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'input_provenance.json').exists():
        raise SystemExit('Fresh inputs already frozen; do not overwrite')
    text = README.read_text(encoding='utf-8-sig')
    (OUT / 'requested_readme.md').write_text(text, encoding='utf-8')
    wb = openpyxl.load_workbook(FAQ, read_only=True, data_only=True)
    # This is deliberately the only sheet queried by this extraction.
    rows = list(wb['FAQ 원문'].iter_rows(min_row=2, values_only=True))
    corpus = [{'faq_id': r[0], 'category': r[1], 'question': r[2], 'answer': r[3], 'intent': r[4]}
              for r in rows if r[0]]
    wb.close()
    assert len(corpus) == 1024
    for d in corpus:
        d['body_hash'] = hashlib.sha256((d['question'] + '\n' + d['answer']).encode()).hexdigest()
    (OUT / 'corpus.jsonl').write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in corpus), encoding='utf-8')
    provenance = {'authoring_standard': 'requested README only',
                  'README_path': str(README), 'README_sha256': sha(README),
                  'knowledge_workbook': str(FAQ), 'knowledge_workbook_sha256': sha(FAQ),
                  'knowledge_sheet_read': 'FAQ 원문', 'corpus_count': len(corpus),
                  'old_test_questions_imported': 0, 'old_labels_imported': 0,
                  'old_category_quotas_imported': False, 'old_selection_or_thresholds_imported': False,
                  'old_test_sheets_are_not_authoring_inputs': True,
                  'prior_results_are_reference_only': True}
    (OUT / 'input_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(provenance, ensure_ascii=False))

if __name__ == '__main__':
    main()
