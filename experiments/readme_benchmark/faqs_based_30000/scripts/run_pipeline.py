"""Sequential offline data pipeline, with no serving or calibration invocation."""
import build_dataset
import audit_and_prepare
import validate_dataset
import build_review

build_dataset.main()
rows = audit_and_prepare.read('variations.jsonl')
corpus = audit_and_prepare.read('corpus.jsonl')
selected = audit_and_prepare.read('source_pairs.jsonl')
audit_and_prepare.token_audit(rows, corpus)
audit_and_prepare.mine_competitors(selected, corpus)
validate_dataset.main()
build_review.main()
