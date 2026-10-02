"""
BGE-M3 dense+sparse(hybrid) + (선택) reranker로 FAQ 의도 분류 threshold 테스트를
실행하는 "한 번 돌리고 끝나는" 스크립트.

기존 IntentClassificationAnalysis.java(Ollama dense-only, pgvector)와 같은 방법론을
그대로 따르되, 임베딩 방식만 BGE-M3 dense+sparse hybrid(+reranker)로 바꾼 버전이다.
서버(app.py/uvicorn)를 띄우지 않고, 이 스크립트 하나를 그냥 실행하면 corpus/test set을
읽어서 검색 → threshold 스캔 → 4종 리포트 파일 저장까지 한 번에 끝낸다.

=== 측정 방법론 (Java 버전과 동일) ===
- 코퍼스: (질문, 권장 처리 의도[FAQ_RAG/MAP_API/USER_INFO]) 2컬럼 CSV
- 테스트셋: (질문, 의도) 2컬럼 CSV. 의도는 일반/매장검색관련/사용자검색관련 중 하나 이상을
  "+"로 이어 적거나(예: "일반+매장검색관련"), 정답이 없으면 "무관"으로 적는다.
- 각 테스트 질문마다 Top-K(K=3, 기본값) 후보를 뽑고, 그중 threshold 이상인 후보들의
  의도를 모아 "검색 결과 집합"으로 삼는다.
- 의도별 재현율 = 정답 집합에 그 의도가 포함된 질문들 중, 검색 결과에도 포함된 비율
- 무관 정확도 = 정답이 무관(빈 집합)인 질문 중, 검색 결과도 정확히 빈 집합인 비율
- 여러 의도가 동시에 살아남는 것 자체는 실패가 아니다(각자 독립적으로 자기 역할만
  수행하면 됨) → "혼입"이나 "정답 집합과 완전히 일치하는지"는 측정하지 않는다.

=== 검색 파이프라인 (3단계, 각 단계는 옵션으로 켜고 끌 수 있음) ===
1. Dense 1차 후보 추출: 코퍼스 전체를 dense 벡터 dot product로 정렬해 상위
   --prefilter-n개만 추린다 (전체 코퍼스에 대해 매번 sparse를 계산하면 느리므로,
   빠른 dense 연산으로 먼저 후보 폭을 좁히는 실전형 2-stage 구조).
2. Hybrid 재정렬: 그 --prefilter-n개에 대해서만 sparse(lexical) 점수를 계산해서
   hybrid = alpha*dense + (1-alpha)*sparse 로 재정렬한다. (--no-hybrid 로 끄면
   dense 점수만 그대로 사용 - 순수 dense-only 비교용)
3. 재정렬 (둘 중 하나만 선택, 동시 사용 불가):
   - --rerank: hybrid 상위 --rerank-window개를 cross-encoder reranker에 넣어
     재정렬한다. 이때 threshold 비교에 쓰는 "최종 점수"는 reranker 점수로
     바뀐다 (0~1로 정규화됨).
   - --colbert: hybrid 상위 --colbert-window개에 대해 BGE-M3의 multi-vector
     (ColBERT 스타일 토큰 벡터)로 MaxSim 재정렬한다. 문장 전체를 벡터 1개로
     뭉치는 dense와 달리, 질문의 각 토큰이 후보 문장의 토큰들 중 가장 비슷한
     것과 얼마나 가까운지(=MaxSim)를 토큰별로 구해 평균낸 값을 최종 점수로
     쓴다. 코퍼스 전체에 대해 매번 계산하면 무거우므로(문장당 벡터가 토큰
     수만큼 늘어남), 반드시 hybrid로 좁힌 후보 폭 안에서만 재정렬하는 구조다.
     주의: 이 점수의 스케일은 dense/hybrid/reranker와 전혀 다르므로, threshold
     스캔 범위를 그대로 재사용하지 말고 넓게(예: -0.2:1.0:0.05) 먼저 훑어서
     점수 분포를 확인한 뒤 좁혀야 한다.

최종적으로 각 단계 이후 상위 K개만 남겨서, Java 버전의 topKHits와 동일한 역할을 한다.

=== 사전 준비 ===
    pip install FlagEmbedding torch

=== 사용법 예시 ===
    # 1) dense-only (순수 임베딩 비교용, Ollama 버전과 가장 유사한 조건)
    python intent_classification_bge.py --corpus faq_intent_corpus.csv \\
        --test-set intent_test_set.csv --no-hybrid --method-name dense-only

    # 2) dense+sparse hybrid (기본값)
    python intent_classification_bge.py --corpus faq_intent_corpus.csv \\
        --test-set intent_test_set.csv --method-name hybrid

    # 3) hybrid로 후보를 넓게 뽑은 뒤 reranker로 최종 재정렬
    python intent_classification_bge.py --corpus faq_intent_corpus.csv \\
        --test-set intent_test_set.csv --rerank --rerank-window 10 --method-name hybrid+rerank

    # 4) hybrid로 후보를 넓게 뽑은 뒤 multi-vector(ColBERT MaxSim)로 최종 재정렬
    #    처음엔 넓은 범위로 점수 분포부터 확인할 것을 권장
    python intent_classification_bge.py --corpus faq_intent_corpus.csv \\
        --test-set intent_test_set.csv --colbert --colbert-window 10 \\
        --method-name hybrid+colbert --thresholds -0.2:1.0:0.05

    # threshold 스캔 범위를 직접 지정하고 싶으면 (쉼표 구분 또는 start:stop:step)
    python intent_classification_bge.py ... --thresholds 0.3:0.9:0.02
    python intent_classification_bge.py ... --thresholds 0.5,0.6,0.7,0.8,0.9

결과는 --output-dir(기본 ./intent-analysis-output) 아래에 다음 4개 파일로 저장된다
(파일명에 method-name과 실행 시각이 들어가서 여러 방법을 돌려도 서로 덮어쓰지 않는다):
    candidate-details-<method>-<timestamp>.txt   질문별 Top-K 매칭 상세
    summary-report-<method>-<timestamp>.txt      threshold별 의도별 재현율 / 무관 정확도
    recall-failures-<method>-<timestamp>.txt     threshold별 재현 실패 케이스
    irrelevant-failures-<method>-<timestamp>.txt threshold별 무관 판정 실패 케이스
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

# ────────────────────────────── 상수 (Java 버전과 동일한 의도 매핑) ──────────────────────────────

ROUTABLE_GROUPS = ["일반", "매장검색관련", "사용자검색관련"]
INTENT_TO_GROUP = {
    "FAQ_RAG": "일반",
    "MAP_API": "매장검색관련",
    "USER_INFO": "사용자검색관련",
}


# ────────────────────────────── 데이터 구조 ──────────────────────────────

@dataclass
class TestCase:
    question: str
    true_intents: frozenset[str]  # 빈 집합이면 "무관"


@dataclass
class CandidateHit:
    group: str
    matched_question: str
    score: float


# ────────────────────────────── CSV 로딩 ──────────────────────────────

def read_csv_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"빈 CSV 파일입니다: {path}")
    return rows[1:]  # 헤더 제외


def load_corpus(path: Path) -> tuple[list[str], list[str]]:
    """반환: (질문 리스트, group 리스트). group은 ROUTABLE_GROUPS 중 하나로 변환됨."""
    questions, groups = [], []
    for row in read_csv_rows(path):
        if len(row) < 2 or not row[0].strip():
            continue
        question, raw_intent = row[0].strip(), row[1].strip()
        group = INTENT_TO_GROUP.get(raw_intent)
        if group is None:
            raise ValueError(
                f"알 수 없는 의도값: '{raw_intent}' (질문: '{question}'). "
                f"코퍼스의 의도값은 FAQ_RAG/MAP_API/USER_INFO 중 하나여야 합니다."
            )
        questions.append(question)
        groups.append(group)
    return questions, groups


def parse_test_intents(raw: str) -> frozenset[str]:
    raw = raw.strip()
    if not raw or raw == "무관":
        return frozenset()
    return frozenset(part.strip() for part in raw.split("+") if part.strip())


def load_test_set(path: Path) -> list[TestCase]:
    cases = []
    for row in read_csv_rows(path):
        if len(row) < 2 or not row[0].strip():
            continue
        question, raw_intent = row[0].strip(), row[1].strip()
        intents = parse_test_intents(raw_intent)
        for intent in intents:
            if intent not in ROUTABLE_GROUPS:
                raise ValueError(
                    f"알 수 없는 의도값: '{intent}' (질문: '{question}'). "
                    f"테스트셋의 의도값은 일반/매장검색관련/사용자검색관련 조합이거나 '무관'이어야 합니다."
                )
        cases.append(TestCase(question=question, true_intents=intents))
    return cases


# ────────────────────────────── threshold 스펙 파싱 ──────────────────────────────

def parse_thresholds(spec: str) -> list[float]:
    if ":" in spec:
        start, stop, step = (float(x) for x in spec.split(":"))
        n = int(round((stop - start) / step)) + 1
        return [round(start + i * step, 6) for i in range(n)]
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


# ────────────────────────────── 모델 로딩 ──────────────────────────────

def load_embed_model(model_name: str):
    from FlagEmbedding import BGEM3FlagModel

    print(f"[모델 로딩] {model_name} (use_fp16=True) ...", file=sys.stderr)
    model = BGEM3FlagModel(model_name, use_fp16=True)
    print(f"[모델 로딩 완료] {model_name}", file=sys.stderr)
    return model


def load_reranker_model(model_name: str):
    from FlagEmbedding import FlagReranker

    print(f"[모델 로딩] {model_name} (use_fp16=True) ...", file=sys.stderr)
    model = FlagReranker(model_name, use_fp16=True)
    print(f"[모델 로딩 완료] {model_name}", file=sys.stderr)
    return model


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    """토큰 벡터 행렬(T, D)의 각 행을 단위 벡터로 정규화한다. ColBERT MaxSim을
    코사인 유사도 기준으로 계산하기 위함 — BGE-M3의 colbert_vecs는 dense_vecs와
    달리 단위 노름이 보장되어 있지 않다."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def maxsim_score(query_token_vecs: np.ndarray, doc_token_vecs: np.ndarray) -> float:
    """ColBERT 스타일 late-interaction 점수(MaxSim). 질문의 각 토큰 벡터에 대해
    문서 토큰 벡터 중 가장 비슷한 것과의 코사인 유사도를 구하고, 그 값들을
    질문 토큰 개수로 평균낸다. (원 논문은 합산을 쓰지만, 질문 길이가 저마다
    달라 threshold를 일관되게 잡기 어려우므로 여기서는 평균을 쓴다.)"""
    q = normalize_rows(np.asarray(query_token_vecs, dtype=np.float32))
    d = normalize_rows(np.asarray(doc_token_vecs, dtype=np.float32))
    sim = q @ d.T  # (Q, T)
    return float(sim.max(axis=1).mean())


# ────────────────────────────── 검색 파이프라인 ──────────────────────────────

def encode_corpus(embed_model, questions: list[str], batch_size: int, return_colbert: bool = False):
    print(f"[코퍼스 임베딩] {len(questions)}개 질문 인코딩 중...", file=sys.stderr)
    t0 = time.perf_counter()
    output = embed_model.encode(
        questions,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=return_colbert,
    )
    print(f"[코퍼스 임베딩 완료] {time.perf_counter() - t0:.1f}초", file=sys.stderr)
    dense = np.asarray(output["dense_vecs"], dtype=np.float32)  # (N, D)
    sparse = output["lexical_weights"]  # list[dict[str, float]]
    colbert = output["colbert_vecs"] if return_colbert else None  # list[(T_i, D) ndarray] | None
    return dense, sparse, colbert


def search_one_query(
    embed_model,
    query: str,
    corpus_questions: list[str],
    corpus_groups: list[str],
    corpus_dense: np.ndarray,
    corpus_sparse: list[dict],
    k: int,
    prefilter_n: int,
    use_hybrid: bool,
    hybrid_alpha: float,
    reranker=None,
    rerank_window: int = 0,
    corpus_colbert: list | None = None,
    colbert_window: int = 0,
) -> list[CandidateHit]:
    use_colbert = corpus_colbert is not None
    out = embed_model.encode(
        [query], return_dense=True, return_sparse=True, return_colbert_vecs=use_colbert
    )
    query_dense = np.asarray(out["dense_vecs"][0], dtype=np.float32)
    query_sparse = out["lexical_weights"][0]

    # 1단계: dense 전체 정렬 후 상위 prefilter_n개만 추림 (빠른 벡터 연산)
    dense_scores = corpus_dense @ query_dense  # (N,)
    prefilter_n = min(prefilter_n, len(corpus_questions))
    top_idx = np.argpartition(-dense_scores, prefilter_n - 1)[:prefilter_n]
    top_idx = top_idx[np.argsort(-dense_scores[top_idx])]

    # 2단계: hybrid 재정렬 (옵션)
    if use_hybrid:
        scored = []
        for idx in top_idx:
            sparse_score = embed_model.compute_lexical_matching_score(query_sparse, corpus_sparse[idx])
            hybrid_score = hybrid_alpha * float(dense_scores[idx]) + (1 - hybrid_alpha) * float(sparse_score)
            scored.append((idx, hybrid_score))
        scored.sort(key=lambda x: x[1], reverse=True)
    else:
        scored = [(idx, float(dense_scores[idx])) for idx in top_idx]

    # 3단계: 재정렬 (reranker 또는 colbert 중 하나만 적용, 둘 다 None이면 2단계 결과 그대로)
    if reranker is not None:
        window = scored[: max(rerank_window, k)]
        pairs = [[query, corpus_questions[idx]] for idx, _ in window]
        rerank_scores = reranker.compute_score(pairs, normalize=True)
        if isinstance(rerank_scores, float):
            rerank_scores = [rerank_scores]
        final = sorted(zip([idx for idx, _ in window], rerank_scores), key=lambda x: x[1], reverse=True)
    elif use_colbert:
        query_colbert = np.asarray(out["colbert_vecs"][0], dtype=np.float32)
        window = scored[: max(colbert_window, k)]
        colbert_scores = [maxsim_score(query_colbert, corpus_colbert[idx]) for idx, _ in window]
        final = sorted(zip([idx for idx, _ in window], colbert_scores), key=lambda x: x[1], reverse=True)
    else:
        final = scored

    top_k = final[:k]
    return [
        CandidateHit(group=corpus_groups[idx], matched_question=corpus_questions[idx], score=float(score))
        for idx, score in top_k
    ]


# ────────────────────────────── 평가 / 리포트 (Java 버전과 동일 로직) ──────────────────────────────

def surviving_intents(hits: list[CandidateHit], threshold: float) -> set[str]:
    return {h.group for h in hits if h.score >= threshold}


def describe_missed_intent(missed_group: str, hits: list[CandidateHit], threshold: float) -> str:
    scores = [h.score for h in hits if h.group == missed_group]
    if not scores:
        return f"{missed_group}(Top-K 후보 없음)"
    best = max(scores)
    return f"{missed_group}(threshold 미달, 최고 점수={best:.6f} < {threshold:.6f})"


def write_candidate_details(path: Path, cases: list[TestCase], all_hits: list[list[CandidateHit]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"=== 질문별 Top-K 매칭 상세 ===\n")
        for case, hits in zip(cases, all_hits):
            true_label = "+".join(sorted(case.true_intents)) if case.true_intents else "무관"
            f.write(f'[{true_label}] "{case.question}"\n')
            for h in hits:
                f.write(f"   -> [{h.group}] {h.score:.6f} | {h.matched_question}\n")


def write_summary_report(
    path: Path, cases: list[TestCase], all_hits: list[list[CandidateHit]], thresholds: list[float]
) -> None:
    with path.open("w", encoding="utf-8") as f:
        for threshold in thresholds:
            true_count = {g: 0 for g in ROUTABLE_GROUPS}
            recalled_count = {g: 0 for g in ROUTABLE_GROUPS}
            muhwan_total = 0
            muhwan_correct = 0

            for case, hits in zip(cases, all_hits):
                predicted = surviving_intents(hits, threshold)
                for g in case.true_intents:
                    true_count[g] += 1
                    if g in predicted:
                        recalled_count[g] += 1
                if not case.true_intents:
                    muhwan_total += 1
                    if not predicted:
                        muhwan_correct += 1

            f.write(f"\n=== threshold={threshold:.6f} ===\n")
            for g in ROUTABLE_GROUPS:
                total = true_count[g]
                recall = recalled_count[g] / total if total else 0.0
                f.write(f"  [{g}] 재현율={recall:.6f} (정답에 포함된 질문 {total}개 중 {recalled_count[g]}개 회수)\n")
            muhwan_acc = muhwan_correct / muhwan_total if muhwan_total else 0.0
            f.write(f"  [무관 정확도] {muhwan_acc:.6f} ({muhwan_correct}/{muhwan_total}건이 정확히 빈 결과로 판정됨)\n")


def write_recall_failures(
    path: Path, cases: list[TestCase], all_hits: list[list[CandidateHit]], thresholds: list[float]
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("=== threshold별 재현 실패 케이스 (정답 의도인데 검색 결과에서 빠짐) ===\n")
        for threshold in thresholds:
            f.write(f"\n=== threshold={threshold:.6f} ===\n")
            any_failure = False
            for case, hits in zip(cases, all_hits):
                if not case.true_intents:
                    continue
                predicted = surviving_intents(hits, threshold)
                missed = case.true_intents - predicted
                if missed:
                    any_failure = True
                    missed_label = " + ".join(describe_missed_intent(g, hits, threshold) for g in sorted(missed))
                    f.write(f'[놓친 의도: {missed_label}] "{case.question}"\n')
                    for h in hits:
                        f.write(f"   -> [{h.group}] {h.score:.6f} | {h.matched_question}\n")
            if not any_failure:
                f.write("  (재현 실패 없음)\n")


def write_irrelevant_failures(
    path: Path, cases: list[TestCase], all_hits: list[list[CandidateHit]], thresholds: list[float]
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("=== threshold별 무관 판정 실패 케이스 (정답은 무관인데 결과가 비지 않음) ===\n")
        for threshold in thresholds:
            f.write(f"\n=== threshold={threshold:.6f} ===\n")
            any_failure = False
            for case, hits in zip(cases, all_hits):
                if case.true_intents:
                    continue
                predicted = surviving_intents(hits, threshold)
                if predicted:
                    any_failure = True
                    f.write(f'[정답: 무관 / 오탐으로 나온 의도: {"+".join(sorted(predicted))}] "{case.question}"\n')
                    for h in hits:
                        f.write(f"   -> [{h.group}] {h.score:.6f} | {h.matched_question}\n")
            if not any_failure:
                f.write("  (무관 판정 실패 없음)\n")


# ────────────────────────────── 메인 ──────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BGE-M3 hybrid(+reranker)로 FAQ 의도 분류 threshold 테스트를 실행합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpus", type=Path, default=Path("faq_intent_corpus.csv"), help="코퍼스 CSV (질문, 권장 처리 의도)")
    parser.add_argument("--test-set", type=Path, default=Path("intent_test_set.csv"), help="테스트셋 CSV (질문, 의도)")
    parser.add_argument("--output-dir", type=Path, default=Path("intent-analysis-output"), help="결과 저장 디렉터리")
    parser.add_argument("--method-name", default=None, help="결과 파일명에 붙일 방법 이름 (예: hybrid, hybrid+rerank). 생략 시 자동 결정")

    parser.add_argument("--k", type=int, default=3, help="최종 Top-K (기본 3, Java 버전과 동일)")
    parser.add_argument("--prefilter-n", type=int, default=50, help="1단계 dense 정렬로 추릴 후보 수 (기본 50)")
    parser.add_argument("--embed-model", default="BAAI/bge-m3", help="dense+sparse 임베딩 모델")
    parser.add_argument("--embed-batch-size", type=int, default=32, help="코퍼스 임베딩 배치 크기")

    parser.add_argument("--no-hybrid", action="store_true", help="sparse 점수를 쓰지 않고 dense-only로만 평가 (비교용)")
    parser.add_argument("--hybrid-alpha", type=float, default=0.6, help="hybrid = alpha*dense + (1-alpha)*sparse (기본 0.6)")

    parser.add_argument("--rerank", action="store_true", help="hybrid 상위 후보를 cross-encoder reranker로 재정렬 (--colbert와 동시 사용 불가)")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3", help="reranker 모델")
    parser.add_argument("--rerank-window", type=int, default=10, help="reranker에게 보여줄 후보 폭 (final k보다 넓게 줄수록 reranker가 더 많이 볼 수 있음)")

    parser.add_argument("--colbert", action="store_true", help="hybrid 상위 후보를 BGE-M3 multi-vector(ColBERT MaxSim)로 재정렬 (--rerank와 동시 사용 불가)")
    parser.add_argument("--colbert-window", type=int, default=10, help="colbert 재정렬에 보여줄 후보 폭")

    parser.add_argument(
        "--thresholds",
        default="0.30:0.90:0.02",
        help="threshold 스캔 범위. 'start:stop:step' 또는 쉼표로 구분된 값 목록. "
        "reranker를 쓰면 점수가 0/1 근처로 극단적으로 갈리므로 예: 0.05:0.95:0.05 를 권장",
    )
    args = parser.parse_args()

    if not args.corpus.exists():
        sys.exit(f"[오류] 코퍼스 파일을 찾을 수 없습니다: {args.corpus}")
    if not args.test_set.exists():
        sys.exit(f"[오류] 테스트셋 파일을 찾을 수 없습니다: {args.test_set}")
    if args.rerank and args.colbert:
        sys.exit("[오류] --rerank와 --colbert는 동시에 쓸 수 없습니다. 한 번에 하나씩 실행해서 비교하세요.")

    use_hybrid = not args.no_hybrid
    if args.rerank:
        method_name = args.method_name or "hybrid+rerank"
    elif args.colbert:
        method_name = args.method_name or "hybrid+colbert"
    else:
        method_name = args.method_name or ("hybrid" if use_hybrid else "dense-only")

    print(f"코퍼스: {args.corpus}")
    print(f"테스트셋: {args.test_set}")
    print(f"방법: {method_name} (K={args.k}, prefilter_n={args.prefilter_n}, hybrid_alpha={args.hybrid_alpha if use_hybrid else 'N/A'})")

    corpus_questions, corpus_groups = load_corpus(args.corpus)
    cases = load_test_set(args.test_set)
    print(f"코퍼스 {len(corpus_questions)}개, 테스트 케이스 {len(cases)}개 로딩 완료.")

    thresholds = parse_thresholds(args.thresholds)

    embed_model = load_embed_model(args.embed_model)
    corpus_dense, corpus_sparse, corpus_colbert = encode_corpus(
        embed_model, corpus_questions, args.embed_batch_size, return_colbert=args.colbert
    )

    reranker = load_reranker_model(args.reranker_model) if args.rerank else None

    print(f"Top-{args.k} 통합 검색 시작: 총 {len(cases)}개")
    all_hits: list[list[CandidateHit]] = []
    t0 = time.perf_counter()
    for i, case in enumerate(cases, start=1):
        hits = search_one_query(
            embed_model=embed_model,
            query=case.question,
            corpus_questions=corpus_questions,
            corpus_groups=corpus_groups,
            corpus_dense=corpus_dense,
            corpus_sparse=corpus_sparse,
            k=args.k,
            prefilter_n=args.prefilter_n,
            use_hybrid=use_hybrid,
            hybrid_alpha=args.hybrid_alpha,
            reranker=reranker,
            rerank_window=args.rerank_window,
            corpus_colbert=corpus_colbert,
            colbert_window=args.colbert_window,
        )
        all_hits.append(hits)
        if i % 50 == 0 or i == len(cases):
            print(f"  평가 진행: {i}/{len(cases)}  ({time.perf_counter() - t0:.1f}초 경과)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_method = method_name.replace("+", "_").replace(" ", "_")

    detail_path = args.output_dir / f"candidate-details-{safe_method}-{timestamp}.txt"
    summary_path = args.output_dir / f"summary-report-{safe_method}-{timestamp}.txt"
    recall_failure_path = args.output_dir / f"recall-failures-{safe_method}-{timestamp}.txt"
    irrelevant_failure_path = args.output_dir / f"irrelevant-failures-{safe_method}-{timestamp}.txt"

    write_candidate_details(detail_path, cases, all_hits)
    write_summary_report(summary_path, cases, all_hits, thresholds)
    write_recall_failures(recall_failure_path, cases, all_hits, thresholds)
    write_irrelevant_failures(irrelevant_failure_path, cases, all_hits, thresholds)

    print("\n=== 완료 ===")
    print(f"상세 매칭 내역: {detail_path}")
    print(f"요약 리포트: {summary_path}")
    print(f"재현 실패 목록: {recall_failure_path}")
    print(f"무관 판정 실패 목록: {irrelevant_failure_path}")


if __name__ == "__main__":
    main()
