#!/usr/bin/env python3
"""threshold_lab.py가 올바르게 계산하는지 합성 데이터로 확인합니다. 실제 데이터와 무관합니다.

실행: python selftest.py
통과하면 마지막 줄에 'ALL OK'가 나옵니다.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import threshold_lab as lab  # noqa: E402


def check(cond, message):
    if not cond:
        raise AssertionError(message)
    print("ok  -", message)


# ---------------------------------------------------------------------------
# 1. 손으로 만든 사례로 분류 규칙 확인
# ---------------------------------------------------------------------------
def unit_cases():
    inf = np.inf
    # 열: 후보 3개. kind: 1 답있음, 0 답없음, -1 경계
    sim = np.array([
        [0.80, 0.50, -inf],   # 0: 정답 1위, 문장 0.75, 필요문장 0.75
        [0.80, 0.50, -inf],   # 1: 정답 1위, 문장 0.75, 필요문장 중 0.60짜리가 있음
        [0.85, 0.75, -inf],   # 2: 1위는 오답(문장 0.90), 정답은 2위(문장 0.80)
        [0.85, 0.75, -inf],   # 3: 1위 오답이 통과하고 정답은 문장이 0.60이라 탈락
        [0.72, 0.50, -inf],   # 4: 답없음, 1위 0.72 / 문장 0.70
        [0.80, 0.50, -inf],   # 5: 경계
    ])
    max_s = np.array([
        [0.75, 0.40, -inf],
        [0.75, 0.40, -inf],
        [0.90, 0.80, -inf],
        [0.90, 0.60, -inf],
        [0.70, 0.40, -inf],
        [0.75, 0.40, -inf],
    ])
    req_min = np.full((6, 3), inf)
    req_min[0, 0] = 0.75
    req_min[1, 0] = 0.60
    req_min[2, 1] = 0.80
    req_min[3, 1] = 0.60
    is_gold = np.zeros((6, 3), dtype=bool)
    is_gold[0, 0] = is_gold[1, 0] = is_gold[2, 1] = is_gold[3, 1] = True
    kind = np.array([1, 1, 1, 1, 0, -1])
    ds = lab.Dataset(np.array([f"Q{i}" for i in range(6)]), kind, np.array(["ANS"] * 4 + ["OFF", "BORDER"]),
                     np.array(["dev"] * 6), sim, max_s, req_min, is_gold,
                     {"gold_rank": np.zeros(6), "top1_sim": sim[:, 0],
                      "sent_rows": pd.DataFrame(columns=["qid", "faq_id", "needed", "similarity"]),
                      "warn_no_needed": 0, "has_sentence_labels": True})
    cat, _ = lab.simulate(ds, np.array([0.70, 0.75]), np.array([0.70, 0.80]))
    # cat[q, T1 idx, T2 idx]
    check(cat[0, 0, 0] == lab.CORRECT, "정답 1위 + 문장 통과 → 정답 전달")
    check(cat[0, 0, 1] == lab.NOANS, "문장 기준을 0.80으로 올리면 → 미응답")
    check(cat[1, 0, 0] == lab.INCOMPLETE, "필요 문장 중 0.60짜리가 빠지면 → 불완전 전달")
    check(cat[2, 0, 0] == lab.CORRECT, "정답이 2위여도 통과하면 → 정답 전달")
    check(cat[3, 0, 0] == lab.WRONG, "정답이 문장에서 탈락하고 오답 FAQ가 통과 → 엉뚱 전달")
    check(cat[4, 0, 0] == lab.FALSEPASS, "답없음 질문이 T1·T2 모두 통과 → 잘못 통과")
    check(cat[4, 1, 0] == lab.REFUSE, "T1을 0.75로 올리면 → 올바른 거절")
    check(cat[4, 0, 1] == lab.REFUSE, "T2를 0.80으로 올리면 → 올바른 거절")
    check(cat[5, 0, 0] == -1, "경계 질문은 분류에서 제외")
    check(cat[2, 1, 1] == lab.CORRECT, "T1=0.75, T2=0.80: 정답 2위(0.75, 문장 0.80) 통과")


# ---------------------------------------------------------------------------
# 2. 합성 데이터로 전체 흐름 실행
# ---------------------------------------------------------------------------
def synthetic(tmp: Path, seed: int = 7, sharp: bool = False):
    """sharp=True면 정답과 오답 점수가 뚜렷이 갈라진 쉬운 데이터(최적 구간을 알고 있음)."""
    rng = np.random.default_rng(seed)
    P = dict(gold=(0.80, 0.04), other=(0.58, 0.05), neg={"NOFAQ": 0.58, "OFF": 0.50, "ATTACK": 0.52}, neg_sd=0.05,
             need=(0.78, 0.04), sent_other=(0.50, 0.06), sent_neg=(0.48, 0.06)) if sharp else dict(
        gold=(0.78, 0.07), other=(0.62, 0.08), neg={"NOFAQ": 0.66, "OFF": 0.52, "ATTACK": 0.55}, neg_sd=0.07,
        need=(0.76, 0.08), sent_other=(0.58, 0.10), sent_neg=(0.55, 0.09))
    labels, cands, sents, sentlab = [], [], [], []
    faq_counter = 1000

    def new_faq():
        nonlocal faq_counter
        faq_counter += 1
        return str(faq_counter)

    specs = [("ANS", "ANSWER", 350), ("NOFAQ", "NO_ANSWER", 80), ("OFF", "NO_ANSWER", 40),
             ("ATTACK", "NO_ANSWER", 30), ("BORDER", "BORDER", 50)]
    qn = 0
    for group, label, count in specs:
        for _ in range(count):
            qn += 1
            qid = f"Q{qn:04d}"
            gold_pos = None
            if label == "ANSWER":
                gold_pos = 1 if rng.random() < 0.8 else int(rng.integers(2, 6))
            faqs = [new_faq() for _ in range(10)]
            gold = faqs[gold_pos - 1] if gold_pos else None
            for rank in range(1, 11):
                fid = faqs[rank - 1]
                if fid == gold:
                    s = float(np.clip(rng.normal(*P["gold"]), 0, 1))
                elif label == "NO_ANSWER":
                    s = float(np.clip(rng.normal(P["neg"][group], P["neg_sd"]), 0, 1))
                else:
                    s = float(np.clip(rng.normal(*P["other"]), 0, 1))
                cands.append((qid, rank, fid, "1", s, "GENERAL", f"FAQ {fid}", f"variant {fid}"))
            # 점수가 순위 순서가 되도록 정렬
            block = [c for c in cands if c[0] == qid]
            block_sorted = sorted(block, key=lambda c: -c[4])
            cands[:] = [c for c in cands if c[0] != qid]
            cands += [(c[0], i + 1) + c[2:] for i, c in enumerate(block_sorted)]
            # 문장
            for rank, c in enumerate(block_sorted, start=1):
                fid = c[2]
                n_sent = int(rng.integers(3, 8))
                for si in range(n_sent):
                    is_gold_needed = (fid == gold and si < 2)
                    if is_gold_needed:
                        s = float(np.clip(rng.normal(*P["need"]), 0, 1))
                    elif label == "NO_ANSWER":
                        s = float(np.clip(rng.normal(*P["sent_neg"]), 0, 1))
                    else:
                        s = float(np.clip(rng.normal(*P["sent_other"]), 0, 1))
                    sents.append((qid, fid, si, s, f"문장 {fid}-{si}"))
                    if fid == gold:
                        sentlab.append((qid, "Q", fid, si, f"문장 {fid}-{si}", "Y" if is_gold_needed else "N"))
            labels.append({"id": qid, "question": f"질문 {qn}", "intended_group": group,
                           "expression_type": "NORMAL", "final_label": label,
                           "final_gold_faq_ids": gold or ""})
    data = tmp / "data"
    data.mkdir()
    pd.DataFrame(cands, columns=["qid", "rank", "faq_id", "variant_id", "similarity", "intent", "faq_question",
                                 "variant_question"]).to_csv(data / "candidates.csv", index=False)
    pd.DataFrame(sents, columns=["qid", "faq_id", "sentence_idx", "similarity", "sentence"]).to_csv(
        data / "sentences.csv", index=False)
    pd.DataFrame(labels).to_csv(tmp / "labels_final.csv", index=False)
    pd.DataFrame(sentlab, columns=["qid", "question", "faq_id", "sentence_idx", "sentence", "needed"]).to_csv(
        tmp / "sentence_labels.csv", index=False)
    shutil.copy(Path(__file__).parent.parent / "templates" / "weights.json", tmp / "weights.json")
    return data


def merge_questions_case():
    """양식(빈 id 열 포함)을 그대로 합쳐도 되는지, 중복이 제거되는지 확인."""
    tmp = Path(tempfile.mkdtemp(prefix="threshold-lab-mq-"))
    try:
        template = Path(__file__).parent.parent / "templates" / "questions_template.csv"
        extra = tmp / "extra.csv"
        extra.write_text("\ufeffid,group,expression_type,question,writer\n,ANS,NORMAL,요금제를  바꾸고 싶은데 어떻게 하나요?,x\n,OFF,NORMAL,새 질문,x\n",
                         encoding="utf-8")
        lab.main(["mergequestions", str(template), str(extra), "--out", str(tmp / "q.csv")])
        q = lab.read_csv(tmp / "q.csv")
        check(len(q) == 7 + 1, f"양식 7건 + 새 질문 1건, 공백만 다른 중복 1건 제거 → {len(q)}건")
        check(q["id"].tolist()[:2] == ["Q0001", "Q0002"], "id가 새로 붙음")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def sharp_case():
    """뚜렷이 갈라지는 데이터에서는 알려진 최적 구간(T1 0.62~0.76, T2 0.55~0.76)을 찾아야 한다."""
    tmp = Path(tempfile.mkdtemp(prefix="threshold-lab-sharp-"))
    try:
        data = synthetic(tmp, seed=11, sharp=True)
        out = tmp / "out"
        lab.main(["dev", "--labels", str(tmp / "labels_final.csv"), "--data", str(data),
                  "--sentence-labels", str(tmp / "sentence_labels.csv"), "--weights", str(tmp / "weights.json"),
                  "--draws", "300", "--out", str(out)])
        cand = json.load(open(out / "dev_candidates.json", encoding="utf-8"))
        a, b = cand["균형형"]["T1"], cand["균형형"]["T2"]
        check(0.62 <= a <= 0.76, f"뚜렷한 데이터에서 균형형 T1이 알려진 구간 안: {a}")
        check(0.55 <= b <= 0.76, f"뚜렷한 데이터에서 균형형 T2가 알려진 구간 안: {b}")
        grid = pd.read_csv(out / "grid_dev.csv")
        row = grid[(grid["T1"].round(2) == round(a, 2)) & (grid["T2"].round(2) == round(b, 2))].iloc[0]
        check(row["정답전달"] > 0.85 and row["잘못통과"] < 0.05,
              f"균형형 성능: 정답 전달 {row['정답전달']:.1%}, 잘못 통과 {row['잘못통과']:.1%}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def end_to_end():
    tmp = Path(tempfile.mkdtemp(prefix="threshold-lab-selftest-"))
    try:
        data = synthetic(tmp)
        out = tmp / "out"
        base = ["--labels", str(tmp / "labels_final.csv"), "--data", str(data),
                "--sentence-labels", str(tmp / "sentence_labels.csv"), "--weights", str(tmp / "weights.json"),
                "--prior-answerable", "0.75", "--draws", "300", "--out", str(out)]
        lab.main(["dev"] + base)
        for name in ["dev_report.md", "grid_dev.csv", "t1_view.csv", "t2_view.csv", "dev_candidates.json"]:
            check((out / name).exists(), f"dev 산출물 생성: {name}")
        cand = json.load(open(out / "dev_candidates.json", encoding="utf-8"))
        check(set(cand) == {"균형형", "보수형", "응답형"}, "후보 3종 생성")
        c = {k: (v["T1"], v["T2"]) for k, v in cand.items()}
        check(0.55 <= c["균형형"][0] <= 0.85, f"균형형 T1이 합리적 범위: {c['균형형']}")
        check(c["보수형"][0] >= c["응답형"][0] - 0.0001 or c["보수형"][1] >= c["응답형"][1] - 0.0001,
              "보수형이 응답형보다 엄격하거나 같음")

        grid = pd.read_csv(out / "grid_dev.csv")
        low = grid[(grid["T1"] == 0.5) & (grid["T2"] == 0.4)].iloc[0]
        high = grid[(grid["T1"] == 0.9) & (grid["T2"] == 0.9)].iloc[0]
        check(low["잘못통과"] > 0.5, "기준을 가장 낮추면 잘못 통과가 많음")
        check(high["미응답"] > 0.9, "기준을 가장 높이면 미응답이 대부분")

        # dev 분석에 test 질문이 섞이지 않았는지
        labels = lab.read_csv(tmp / "labels_final.csv")
        split = lab.assign_split(labels)
        n_dev = sum(v == "dev" for v in split.values())
        n_test = len(split) - n_dev
        check(abs(n_dev / len(split) - lab.DEV_RATIO) < 0.02, f"dev/test 분할 비율 {n_dev}/{n_test}")
        check(split == lab.assign_split(labels), "분할이 매번 같음 (결정적)")

        # final 은 한 번만
        lab.main(["final"] + base + ["--candidates-json", str(out / "dev_candidates.json")])
        check((out / "final_report.md").exists(), "final 보고서 생성")
        try:
            lab.main(["final"] + base + ["--candidates-json", str(out / "dev_candidates.json")])
            check(False, "두 번째 final이 막혀야 함")
        except SystemExit:
            check(True, "두 번째 final 실행이 막힘")

        # 라벨 합치기 / 문장 시트
        sheet_dir = tmp / "sheets"
        sheet_dir.mkdir()
        q = labels[["id", "intended_group", "expression_type", "question"]].rename(
            columns={"intended_group": "group"})
        q["writer"] = "x"
        lab.write_csv(q, sheet_dir / "questions.csv")
        # labelsheet는 faq_catalog가 필요하므로 최소 파일을 만든다
        cands = lab.read_csv(data / "candidates.csv")
        catalog = cands[["faq_id", "intent", "faq_question"]].drop_duplicates().rename(
            columns={"faq_question": "question"})
        catalog["answer"] = "답변"
        lab.write_csv(catalog, data / "faq_catalog.csv")
        lab.main(["labelsheet", "--questions", str(sheet_dir / "questions.csv"), "--data", str(data),
                  "--out", str(sheet_dir / "label_sheet.csv")])
        sheet = lab.read_csv(sheet_dir / "label_sheet.csv")
        check(len(sheet) == len(labels) and "candidate_10" in sheet.columns, "라벨 시트 생성")
        # 후보 순서가 섞였는지: 1번 칸이 항상 점수 1위는 아니어야 함
        first_ids = sheet["candidate_1"].str.extract(r"\[(\d+)\]")[0]
        top1_ids = cands[cands["rank"] == "1"].set_index("qid")["faq_id"]
        same = sum(first_ids.iloc[i] == top1_ids[sheet["id"].iloc[i]] for i in range(len(sheet))) / len(sheet)
        check(same < 0.3, f"후보가 점수 순서대로 놓여 있지 않음 (1위가 1번 칸일 확률 {same:.1%})")

        a, b = sheet.copy(), sheet.copy()
        truth = labels.set_index("id")
        for df in (a, b):
            df["label"] = df["id"].map(truth["final_label"])
            df["gold_faq_ids"] = df["id"].map(truth["final_gold_faq_ids"])
        flip = b.sample(frac=0.05, random_state=1).index
        b.loc[flip, "label"] = "NO_ANSWER"
        lab.write_csv(a, sheet_dir / "a.csv")
        lab.write_csv(b, sheet_dir / "b.csv")
        lab.main(["merge", "--a", str(sheet_dir / "a.csv"), "--b", str(sheet_dir / "b.csv"),
                  "--out", str(sheet_dir / "merged.csv")])
        merged = lab.read_csv(sheet_dir / "merged.csv")
        check((merged["status"] == "CONFLICT").sum() > 0, "일부러 바꾼 라벨이 CONFLICT로 잡힘")
        merged.loc[merged["status"] == "CONFLICT", "final_label"] = "BORDER"
        lab.write_csv(merged, sheet_dir / "merged_resolved.csv")
        lab.main(["sentencesheet", "--labels", str(sheet_dir / "merged_resolved.csv"), "--data", str(data),
                  "--out", str(sheet_dir / "sentence_sheet.csv")])
        check((sheet_dir / "sentence_sheet.csv").exists(), "문장 시트 생성")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unit_cases()
    merge_questions_case()
    sharp_case()
    end_to_end()
    print("ALL OK")
