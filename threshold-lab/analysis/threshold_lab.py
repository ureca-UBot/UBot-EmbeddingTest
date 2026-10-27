#!/usr/bin/env python3
"""임계값 결정 실험 분석 도구.

하위 명령
  mergequestions  작성자별 질문 CSV를 합쳐 questions.csv 생성 (id 부여, 중복 제거)
  labelsheet      라벨링용 시트 생성 (후보 FAQ를 섞어서, 점수 없이)
  merge           두 사람의 라벨을 합쳐 labels_final.csv 생성, 일치율·kappa 출력
  sentencesheet   정답 FAQ의 문장 목록(필요한 문장 라벨링용) 생성
  dev             개발 세트로 T1×T2 격자 탐색 + 부트스트랩 → 후보 3쌍
  final           후보 3쌍만 최종 세트로 채점 (한 번만 실행 가능)

필요한 패키지: numpy, pandas (그림은 matplotlib가 있을 때만)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
TOP_K = 3  # 운영 CHAT_TOP_K 기본값. 운영 모의는 상위 3개 후보만 사용한다.
CORRECT, INCOMPLETE, WRONG, NOANS, REFUSE, FALSEPASS = range(6)
CAT_NAMES = ["정답 전달", "불완전 전달", "엉뚱 전달", "미응답", "올바른 거절", "잘못 통과"]
DEV_RATIO = 0.6
VALID_LABELS = {"ANSWER", "NO_ANSWER", "BORDER"}
VALID_GROUPS = {"ANS", "NOFAQ", "OFF", "ATTACK", "BORDER"}
VALID_TYPES = {"NORMAL", "KEYWORD", "CASUAL", "SITUATION", "INDIRECT", "MULTI", "NEGATIVE", "ABBREV"}


def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def write_csv(df: pd.DataFrame, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")  # 엑셀에서 바로 열리도록 BOM 포함


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def norm_text(text: str) -> str:
    return "".join(str(text).split()).lower()


# ---------------------------------------------------------------------------
# 1. 질문 합치기
# ---------------------------------------------------------------------------
def cmd_mergequestions(args):
    frames = []
    for path in args.inputs:
        df = read_csv(path)
        df["_source"] = Path(path).name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    required = {"group", "expression_type", "question", "writer"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"필수 열이 없습니다: {sorted(missing)}")

    df["question"] = df["question"].str.strip()
    df = df[df["question"] != ""].copy()
    problems = []
    bad_group = df[~df["group"].isin(VALID_GROUPS)]
    bad_type = df[~df["expression_type"].isin(VALID_TYPES)]
    for _, row in bad_group.iterrows():
        problems.append(f"group 값 오류({row['group']!r}): {row['_source']} / {row['question'][:30]}")
    for _, row in bad_type.iterrows():
        problems.append(f"expression_type 값 오류({row['expression_type']!r}): {row['_source']} / {row['question'][:30]}")
    if problems:
        print("\n".join(problems[:50]))
        sys.exit(f"{len(problems)}건의 값 오류가 있습니다. 해당 파일을 고치고 다시 실행하세요.")

    df["_key"] = df["question"].map(norm_text)
    dup = df.duplicated("_key", keep="first")
    if dup.any():
        print(f"완전히 같은 질문 {int(dup.sum())}건을 제거했습니다.")
    df = df[~dup].reset_index(drop=True)
    df = df.drop(columns=["id"], errors="ignore")  # 양식의 빈 id 열은 버리고 새로 붙인다
    df.insert(0, "id",[f"Q{i + 1:04d}" for i in range(len(df))])
    out = df[["id", "group", "expression_type", "question", "writer"]]
    write_csv(out, args.out)
    print(f"{len(out)}건 저장: {args.out}")
    print(out.groupby("group").size().to_string())
    print(out.groupby("expression_type").size().to_string())


# ---------------------------------------------------------------------------
# 2. 라벨 시트
# ---------------------------------------------------------------------------
def cmd_labelsheet(args):
    questions = read_csv(args.questions)
    data = Path(args.data)
    cands = read_csv(data / "candidates.csv")
    catalog = read_csv(data / "faq_catalog.csv").set_index("faq_id")
    rows = []
    for _, q in questions.iterrows():
        c = cands[cands["qid"] == q["id"]].copy()
        items = []
        for _, r in c.iterrows():
            answer = " ".join(catalog.loc[r["faq_id"], "answer"].split()) if r["faq_id"] in catalog.index else ""
            items.append(f"[{r['faq_id']}] {r['faq_question']} / 답: {answer[:100]}")
        random.Random(f"{args.seed}-{q['id']}").shuffle(items)  # 점수 순서를 감추려고 섞는다
        items += [""] * (10 - len(items))
        row = {"id": q["id"], "question": q["question"], "intended_group": q["group"],
               "expression_type": q["expression_type"]}
        row.update({f"candidate_{i + 1}": items[i] for i in range(10)})
        row.update({"label": "", "gold_faq_ids": "", "gold_outside": "", "note": ""})
        rows.append(row)
    write_csv(pd.DataFrame(rows), args.out)
    print(f"{len(rows)}건 저장: {args.out}  (faq_catalog.csv도 함께 나눠 주세요)")


# ---------------------------------------------------------------------------
# 3. 라벨 합치기
# ---------------------------------------------------------------------------
def parse_ids(text: str) -> set[str]:
    return {t.strip() for t in str(text).replace(",", ";").split(";") if t.strip()}


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def cmd_merge(args):
    a = read_csv(args.a).set_index("id")
    b = read_csv(args.b).set_index("id")
    ids = [i for i in a.index if i in b.index]
    rows, la, lb = [], [], []
    for qid in ids:
        ra, rb = a.loc[qid], b.loc[qid]
        la_, lb_ = ra["label"].strip().upper(), rb["label"].strip().upper()
        if la_ not in VALID_LABELS or lb_ not in VALID_LABELS:
            sys.exit(f"{qid}: label이 비었거나 잘못되었습니다 (A={la_!r}, B={lb_!r})")
        la.append(la_)
        lb.append(lb_)
        ga, gb = parse_ids(ra["gold_faq_ids"]), parse_ids(rb["gold_faq_ids"])
        status, final_label, final_gold = "CONFLICT", "", ""
        if la_ == lb_:
            if la_ == "ANSWER":
                if ga & gb:
                    status, final_label, final_gold = "AGREED", "ANSWER", ";".join(sorted(ga | gb))
            else:
                status, final_label = "AGREED", la_
        rows.append({"id": qid, "question": ra["question"], "intended_group": ra["intended_group"],
                     "expression_type": ra["expression_type"],
                     "label_a": la_, "gold_a": ra["gold_faq_ids"], "label_b": lb_, "gold_b": rb["gold_faq_ids"],
                     "status": status, "final_label": final_label, "final_gold_faq_ids": final_gold,
                     "note": (ra["note"] + " " + rb["note"]).strip()})
    out = pd.DataFrame(rows)
    write_csv(out, args.out)
    agree = sum(x == y for x, y in zip(la, lb)) / len(la)
    bin_a = ["ANSWER" if x == "ANSWER" else "OTHER" for x in la]
    bin_b = ["ANSWER" if x == "ANSWER" else "OTHER" for x in lb]
    print(f"질문 {len(ids)}건, 라벨 일치율 {agree:.1%}")
    print(f"Cohen's kappa: 3분류 {cohen_kappa(la, lb):.3f} / ANSWER 여부 {cohen_kappa(bin_a, bin_b):.3f}")
    print(f"합의 필요(CONFLICT): {(out['status'] == 'CONFLICT').sum()}건 → {args.out} 의 final_label, final_gold_faq_ids를 채우세요")
    if cohen_kappa(bin_a, bin_b) < 0.6:
        print("주의: kappa가 0.6 미만입니다. 판정 기준을 다시 맞춘 뒤 갈린 유형을 재라벨링하세요.")


def cmd_sentencesheet(args):
    labels = read_csv(args.labels)
    sentences = read_csv(Path(args.data) / "sentences.csv")
    rows, missing = [], []
    for _, r in labels[labels["final_label"] == "ANSWER"].iterrows():
        for faq_id in sorted(parse_ids(r["final_gold_faq_ids"])):
            s = sentences[(sentences["qid"] == r["id"]) & (sentences["faq_id"] == faq_id)]
            if s.empty:
                missing.append((r["id"], faq_id))
            for _, srow in s.iterrows():
                rows.append({"qid": r["id"], "question": r["question"], "faq_id": faq_id,
                             "sentence_idx": srow["sentence_idx"], "sentence": srow["sentence"], "needed": ""})
    write_csv(pd.DataFrame(rows), args.out)
    print(f"{len(rows)}문장 저장: {args.out}  (needed 열에 Y 또는 N)")
    if missing:
        print(f"참고: 정답 FAQ가 후보 10개 밖이라 문장이 없는 건 {len(missing)}건 (검색 실패로 집계됩니다)")


# ---------------------------------------------------------------------------
# 4. 데이터셋 구성
# ---------------------------------------------------------------------------
class Dataset:
    """질문별 상위 3개 후보의 점수를 배열로 들고 있다."""

    def __init__(self, ids, kind, group, split, sim, max_s, req_min, is_gold, extra):
        self.ids, self.kind, self.group, self.split = ids, kind, group, split
        self.sim, self.max_s, self.req_min, self.is_gold = sim, max_s, req_min, is_gold
        self.extra = extra  # T1/T2 단독 시각 계산용 부가 정보

    def subset(self, mask) -> "Dataset":
        """부분집합. 부가 정보도 같은 질문만 남긴다 (dev 분석에 test 문장이 섞이지 않게)."""
        extra = dict(self.extra)
        extra["gold_rank"] = self.extra["gold_rank"][mask]
        extra["top1_sim"] = self.extra["top1_sim"][mask]
        keep = set(self.ids[mask])
        sr = self.extra["sent_rows"]
        extra["sent_rows"] = sr[sr["qid"].isin(keep)].reset_index(drop=True)
        return Dataset(self.ids[mask], self.kind[mask], self.group[mask], self.split[mask], self.sim[mask],
                       self.max_s[mask], self.req_min[mask], self.is_gold[mask], extra)


def assign_split(labels: pd.DataFrame, seed: str = "threshold-lab") -> dict[str, str]:
    """id의 해시로 (의도 그룹, 최종 라벨)마다 60% dev / 40% test로 나눈다. 같은 입력이면 항상 같은 결과."""
    result = {}
    for _, grp in labels.groupby(["intended_group", "final_label"]):
        ordered = sorted(grp["id"], key=lambda i: hashlib.sha1(f"{seed}:{i}".encode()).hexdigest())
        cut = int(round(len(ordered) * DEV_RATIO))
        for pos, qid in enumerate(ordered):
            result[qid] = "dev" if pos < cut else "test"
    return result


def build_dataset(labels: pd.DataFrame, cands: pd.DataFrame, sents: pd.DataFrame,
                  senlab: pd.DataFrame | None) -> Dataset:
    labels = labels.copy()
    split_map = assign_split(labels)
    cands = cands.copy()
    cands["rank"] = cands["rank"].astype(int)
    cands["similarity"] = pd.to_numeric(cands["similarity"], errors="coerce")
    sents = sents.copy()
    sents["similarity"] = pd.to_numeric(sents["similarity"], errors="coerce")
    sents["sentence_idx"] = sents["sentence_idx"].astype(int)

    needed = {}
    if senlab is not None:
        for _, r in senlab.iterrows():
            if r["needed"].strip().upper() == "Y":
                needed.setdefault((r["qid"], r["faq_id"]), set()).add(int(r["sentence_idx"]))

    sent_groups = {k: g for k, g in sents.groupby(["qid", "faq_id"])}
    cand_groups = {k: g.sort_values("rank") for k, g in cands.groupby("qid")}

    n = len(labels)
    sim = np.full((n, TOP_K), -np.inf)
    max_s = np.full((n, TOP_K), -np.inf)
    req_min = np.full((n, TOP_K), np.inf)
    is_gold = np.zeros((n, TOP_K), dtype=bool)
    kind = np.zeros(n, dtype=int)
    gold_rank = np.full(n, 99)  # 정답 FAQ의 후보 순위(상위 10개 중), 없으면 99
    top1_sim = np.full(n, np.nan)
    sent_rows = []  # T2 단독 시각용: (qid, 정답FAQ 여부, 필요 여부, 유사도)
    warn_no_needed = 0

    for i, (_, r) in enumerate(labels.iterrows()):
        label = r["final_label"]
        kind[i] = {"ANSWER": 1, "NO_ANSWER": 0, "BORDER": -1}[label]
        gold = parse_ids(r["final_gold_faq_ids"]) if label == "ANSWER" else set()
        c = cand_groups.get(r["id"])
        if c is None or c.empty:
            continue
        top1_sim[i] = c.iloc[0]["similarity"]
        for _, row in c.iterrows():
            if row["faq_id"] in gold:
                gold_rank[i] = min(gold_rank[i], int(row["rank"]))
        for _, row in c[c["rank"] <= TOP_K].iterrows():
            k = int(row["rank"]) - 1
            s = row["similarity"]
            sim[i, k] = s if np.isfinite(s) else -np.inf
            fid = row["faq_id"]
            g = sent_groups.get((r["id"], fid))
            if g is not None and len(g):
                vals = g["similarity"].to_numpy(dtype=float)
                finite = vals[np.isfinite(vals)]
                max_s[i, k] = finite.max() if len(finite) else -np.inf
            is_gold[i, k] = fid in gold
            if fid in gold and senlab is not None and g is not None:
                idxs = needed.get((r["id"], fid), set())
                if idxs:
                    nv = g[g["sentence_idx"].isin(idxs)]["similarity"].to_numpy(dtype=float)
                    req_min[i, k] = np.nanmin(nv) if len(nv) and np.isfinite(nv).any() else np.inf
                else:
                    warn_no_needed += 1
        # T2 단독 시각: 정답 FAQ의 모든 문장 (필요 여부 포함)
        for fid in gold:
            g = sent_groups.get((r["id"], fid))
            if g is None:
                continue
            idxs = needed.get((r["id"], fid), set())
            for _, srow in g.iterrows():
                if np.isfinite(srow["similarity"]):
                    sent_rows.append((r["id"], fid, int(srow["sentence_idx"]) in idxs, float(srow["similarity"])))

    extra = {"gold_rank": gold_rank, "top1_sim": top1_sim, "sent_rows": pd.DataFrame(
        sent_rows, columns=["qid", "faq_id", "needed", "similarity"]), "warn_no_needed": warn_no_needed,
        "has_sentence_labels": senlab is not None}
    groups = labels["intended_group"].to_numpy()
    split = np.array([split_map[i] for i in labels["id"]])
    return Dataset(labels["id"].to_numpy(), kind, groups, split, sim, max_s, req_min, is_gold, extra)


# ---------------------------------------------------------------------------
# 5. 운영 모의: (T1, T2) 조합마다 결과 분류
# ---------------------------------------------------------------------------
def simulate(ds: Dataset, t1: np.ndarray, t2: np.ndarray):
    """반환: (cat[n,A,B], delivered_faq_count[n,A,B]). BORDER 질문은 정답/거절 구분이 없어 cat을 -1로 둔다."""
    sim = ds.sim[:, None, None, :]
    cand_ok = sim >= t1[None, :, None, None]
    sent_ok = ds.max_s[:, None, None, :] >= t2[None, None, :, None]
    passed = cand_ok & sent_ok
    any_pass = passed.any(-1)
    gold_pass = passed & ds.is_gold[:, None, None, :]
    any_gold = gold_pass.any(-1)
    complete = (gold_pass & (ds.req_min[:, None, None, :] >= t2[None, None, :, None])).any(-1)
    ans = (ds.kind == 1)[:, None, None]
    cat = np.where(
        ans,
        np.where(~any_pass, NOANS, np.where(~any_gold, WRONG, np.where(complete, CORRECT, INCOMPLETE))),
        np.where(any_pass, FALSEPASS, REFUSE))
    cat = np.where((ds.kind == -1)[:, None, None], -1, cat)
    return cat.astype(np.int8), passed.sum(-1)


def load_weights(path) -> np.ndarray:
    w = json.load(open(path, encoding="utf-8"))
    table = np.zeros(6)
    table[CORRECT] = w["answerable"]["correct"]
    table[INCOMPLETE] = w["answerable"]["incomplete"]
    table[WRONG] = w["answerable"]["wrong_delivery"]
    table[NOANS] = w["answerable"]["no_answer"]
    table[REFUSE] = w["unanswerable"]["correct_refusal"]
    table[FALSEPASS] = w["unanswerable"]["false_pass"]
    return table


def bootstrap_means(matrix: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
    """matrix[n, cells] → 복원추출 평균 [draws, cells]. 표본 인덱스를 실제로 뽑지 않고 횟수 행렬로 계산한다."""
    n = matrix.shape[0]
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=draws).astype(np.float32)
    return counts @ matrix.astype(np.float32) / n


def smooth(grid: np.ndarray) -> np.ndarray:
    """격자에서 상하좌우 한 칸(±0.01)까지 평균. 한 칸에서만 좋은 우연한 최저점을 피하려는 것."""
    padded = np.pad(grid, 1, mode="edge")
    out = np.zeros_like(grid, dtype=float)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            out += padded[1 + di:1 + di + grid.shape[0], 1 + dj:1 + dj + grid.shape[1]]
    return out / 9.0


# ---------------------------------------------------------------------------
# 6. dev: 격자 탐색 → 후보
# ---------------------------------------------------------------------------
def grid_values(spec):
    lo, hi, step = spec
    return np.round(np.arange(lo, hi + step / 2, step), 4)


def analyze_grid(ds: Dataset, weights: np.ndarray, t1, t2, prior: float, draws: int, seed: int):
    cat, count = simulate(ds, t1, t2)
    a_idx = np.where(ds.kind == 1)[0]
    u_idx = np.where(ds.kind == 0)[0]
    if len(a_idx) == 0 or len(u_idx) == 0:
        sys.exit("답이 있는 질문과 없는 질문이 모두 필요합니다.")
    cells = (len(t1), len(t2))
    flat = lambda c: c.reshape(c.shape[0], -1)
    rng = np.random.default_rng(seed)
    out = {}
    ca, cu = flat(cat[a_idx]), flat(cat[u_idx])

    def rate(c, code):
        return (c == code).astype(np.float32)

    loss_a, loss_u = weights[ca], weights[cu]
    metrics_a = {"loss_a": loss_a, "correct": rate(ca, CORRECT), "incomplete": rate(ca, INCOMPLETE),
                 "wrong": rate(ca, WRONG), "noans": rate(ca, NOANS)}
    metrics_u = {"loss_u": loss_u, "false_pass": rate(cu, FALSEPASS)}
    boots = {}
    for name, m in metrics_a.items():
        boots[name] = bootstrap_means(m, draws, rng)
        out[name] = m.mean(0).reshape(cells)
    for name, m in metrics_u.items():
        boots[name] = bootstrap_means(m, draws, rng)
        out[name] = m.mean(0).reshape(cells)
    out["loss"] = prior * out["loss_a"] + (1 - prior) * out["loss_u"]
    boot_loss = prior * boots["loss_a"] + (1 - prior) * boots["loss_u"]
    out["loss_lo"] = np.percentile(boot_loss, 5, axis=0).reshape(cells)
    out["loss_hi"] = np.percentile(boot_loss, 95, axis=0).reshape(cells)
    out["loss_p90"] = np.percentile(boot_loss, 90, axis=0).reshape(cells)
    out["fp_lo"] = np.percentile(boots["false_pass"], 5, axis=0).reshape(cells)
    out["fp_hi"] = np.percentile(boots["false_pass"], 95, axis=0).reshape(cells)
    out["boot_loss"] = boot_loss
    out["cat"], out["count"] = cat, count
    return out


def pick_candidates(res: dict, t1, t2, fp_cap_conservative: float, fp_cap_responsive: float):
    sm_loss_p90 = smooth(res["loss_p90"])
    sm_loss = smooth(res["loss"])
    sm_fp = smooth(res["false_pass"])
    sm_noans = smooth(res["noans"])
    picks = {}

    i, j = np.unravel_index(np.argmin(sm_loss_p90), sm_loss_p90.shape)
    picks["균형형"] = (i, j, "이웃 칸 평균 손해(90% 상한)가 가장 낮은 조합")

    ok = sm_fp <= fp_cap_conservative
    if ok.any():
        masked = np.where(ok, sm_loss, np.inf)
        i, j = np.unravel_index(np.argmin(masked), masked.shape)
        picks["보수형"] = (i, j, f"잘못 통과율(이웃 평균) {fp_cap_conservative:.0%} 이하 중 손해 최소")
    else:
        i, j = np.unravel_index(np.argmin(sm_fp), sm_fp.shape)
        picks["보수형"] = (i, j, f"잘못 통과율 {fp_cap_conservative:.0%} 이하 조합이 없어 잘못 통과율이 가장 낮은 조합")

    ok = sm_fp <= fp_cap_responsive
    if ok.any():
        masked = np.where(ok, sm_noans + 1e-6 * sm_loss, np.inf)
        i, j = np.unravel_index(np.argmin(masked), masked.shape)
        picks["응답형"] = (i, j, f"잘못 통과율(이웃 평균) {fp_cap_responsive:.0%} 이하 중 미응답 최소")
    else:
        i, j = np.unravel_index(np.argmin(sm_fp), sm_fp.shape)
        picks["응답형"] = (i, j, f"잘못 통과율 {fp_cap_responsive:.0%} 이하 조합이 없어 잘못 통과율이 가장 낮은 조합")
    return {k: (float(t1[i]), float(t2[j]), why, (int(i), int(j))) for k, (i, j, why) in picks.items()}


def t1_view(ds: Dataset, t1: np.ndarray) -> pd.DataFrame:
    """T1 단독 시각: 문장 필터 없이 질문 점수만으로 본 FAQ 선택 능력."""
    rows = []
    a = ds.kind == 1
    u = ds.kind == 0
    all_q = ds.kind != -1
    top1 = ds.sim[:, 0]
    for v in t1:
        gold_top3 = (ds.is_gold & (ds.sim >= v)).any(1)
        gold_top1 = ds.is_gold[:, 0] & (top1 >= v)
        wrong_top1 = a & (top1 >= v) & ~ds.is_gold[:, 0]
        rows.append({
            "T1": v,
            "정답FAQ_상위3개_통과율(답있음)": gold_top3[a].mean() if a.any() else np.nan,
            "정답FAQ_1위_통과율(답있음)": gold_top1[a].mean() if a.any() else np.nan,
            "엉뚱FAQ가_1위로_통과(답있음)": wrong_top1[a].mean() if a.any() else np.nan,
            "후보가_하나라도_통과(답없음)=문장필터_전_잘못통과": (top1[u] >= v).mean() if u.any() else np.nan,
            "판정호출_예상비율(1위<T1, 전체)": (top1[all_q] < v).mean() if all_q.any() else np.nan,
        })
    return pd.DataFrame(rows)


def t2_view(ds: Dataset, t2: np.ndarray) -> pd.DataFrame:
    """T2 단독 시각: 정답 FAQ가 주어졌을 때 필요한 문장이 남는지."""
    s = ds.extra["sent_rows"]
    rows = []
    if s.empty:
        return pd.DataFrame()
    needed = s[s["needed"]]
    not_needed = s[~s["needed"]]
    per_q = needed.groupby(["qid", "faq_id"])["similarity"].min() if not needed.empty else pd.Series(dtype=float)
    for v in t2:
        kept_needed = (needed["similarity"] >= v).mean() if len(needed) else np.nan
        kept_other = (not_needed["similarity"] >= v).mean() if len(not_needed) else np.nan
        kept_all = (per_q >= v).mean() if len(per_q) else np.nan
        passed = s[s["similarity"] >= v]
        precision = passed["needed"].mean() if len(passed) else np.nan
        rows.append({"T2": v, "필요문장_재현율": kept_needed, "불필요문장_통과율": kept_other,
                     "정밀도(통과문장 중 필요)": precision, "필요문장_전부_남는_비율": kept_all})
    return pd.DataFrame(rows)


def border_report(ds: Dataset, pairs: list[tuple[float, float]]) -> list[float]:
    b = ds.kind == -1
    if not b.any():
        return []
    sub = ds.subset(b)
    out = []
    for t1v, t2v in pairs:
        passed = ((sub.sim >= t1v) & (sub.max_s >= t2v)).any(1)
        out.append(float(passed.mean()))
    return out


def group_false_pass(ds: Dataset, t1v: float, t2v: float) -> dict[str, tuple[int, int]]:
    u = ds.kind == 0
    sub = ds.subset(u)
    passed = ((sub.sim >= t1v) & (sub.max_s >= t2v)).any(1)
    out = {}
    for g in sorted(set(sub.group)):
        m = sub.group == g
        out[g] = (int(passed[m].sum()), int(m.sum()))
    return out


def try_plots(out_dir: Path, ds: Dataset, res: dict, t1, t2, picks, t1v: pd.DataFrame, t2v: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib가 없어 그림은 건너뜁니다.")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(res["loss"], origin="lower", aspect="auto", cmap="viridis_r",
                    extent=[t2[0], t2[-1], t1[0], t1[-1]])
    fig.colorbar(img, label="expected loss (lower is better)")
    for name, (a, b, _, _) in picks.items():
        ax.plot(b, a, "o", ms=9, mfc="none", mec="white", mew=2)
        ax.annotate({"균형형": "balanced", "보수형": "conservative", "응답형": "responsive"}[name],
                    (b, a), color="white", xytext=(4, 6), textcoords="offset points")
    ax.set_xlabel("T2 (sentence threshold)")
    ax.set_ylabel("T1 (question threshold)")
    ax.set_title("Expected loss by (T1, T2) - dev set")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_heatmap.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    a = ds.kind == 1
    u = ds.kind == 0
    gold_sims = np.where(ds.is_gold, ds.sim, -np.inf).max(1)
    axes[0].hist(gold_sims[a & np.isfinite(gold_sims)], bins=40, alpha=0.6, label="answerable: gold FAQ score")
    axes[0].hist(ds.sim[u, 0][np.isfinite(ds.sim[u, 0])], bins=40, alpha=0.6, label="unanswerable: top1 score")
    axes[0].set_xlabel("question-vector similarity (T1 axis)")
    axes[0].legend()
    s = ds.extra["sent_rows"]
    if not s.empty:
        axes[1].hist(s[s["needed"]]["similarity"], bins=40, alpha=0.6, label="needed sentences")
        axes[1].hist(s[~s["needed"]]["similarity"], bins=40, alpha=0.6, label="other sentences")
        axes[1].set_xlabel("sentence similarity (T2 axis)")
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "score_distributions.png", dpi=130)
    plt.close(fig)


def load_inputs(args):
    labels = read_csv(args.labels)
    labels = labels[labels["final_label"] != ""]
    unresolved = read_csv(args.labels)
    unresolved = unresolved[(unresolved["final_label"] == "") | (
        (unresolved["final_label"] == "ANSWER") & (unresolved["final_gold_faq_ids"] == ""))]
    if len(unresolved):
        sys.exit(f"final_label/final_gold_faq_ids가 비어 있는 질문이 {len(unresolved)}건 있습니다: "
                 f"{', '.join(unresolved['id'].head(10))}")
    data = Path(args.data)
    cands = read_csv(data / "candidates.csv")
    sents = read_csv(data / "sentences.csv")
    senlab = None
    if args.sentence_labels:
        senlab = read_csv(args.sentence_labels)
    else:
        print("주의: --sentence-labels가 없어 '불완전 전달'은 판정하지 않습니다 (정답 FAQ가 통과하면 정답 전달로 봅니다).")
    return build_dataset(labels, cands, sents, senlab)


def common_header(args, weights_path) -> list[str]:
    return [f"- 손해 가중치 파일: `{weights_path}` (해시 {sha256_of(weights_path)})",
            f"- 실제 질문 중 답이 있는 질문 비율(prior): {args.prior_answerable:.0%}"]


def cmd_dev(args):
    ds_all = load_inputs(args)
    ds = ds_all.subset(ds_all.split == "dev")
    weights = load_weights(args.weights)
    t1, t2 = grid_values(args.t1_range), grid_values(args.t2_range)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_a, n_u, n_b = int((ds.kind == 1).sum()), int((ds.kind == 0).sum()), int((ds.kind == -1).sum())
    res = analyze_grid(ds, weights, t1, t2, args.prior_answerable, args.draws, args.seed)
    picks = pick_candidates(res, t1, t2, args.fp_cap_conservative, args.fp_cap_responsive)

    # 격자 전체 표
    rows = []
    for i, a in enumerate(t1):
        for j, b in enumerate(t2):
            rows.append({"T1": a, "T2": b, "손해": res["loss"][i, j], "손해_5%": res["loss_lo"][i, j],
                         "손해_95%": res["loss_hi"][i, j], "정답전달": res["correct"][i, j],
                         "불완전": res["incomplete"][i, j], "엉뚱전달": res["wrong"][i, j],
                         "미응답": res["noans"][i, j], "잘못통과": res["false_pass"][i, j],
                         "잘못통과_5%": res["fp_lo"][i, j], "잘못통과_95%": res["fp_hi"][i, j]})
    write_csv(pd.DataFrame(rows), out_dir / "grid_dev.csv")
    v1, v2 = t1_view(ds, t1), t2_view(ds, t2)
    write_csv(v1, out_dir / "t1_view.csv")
    if len(v2):
        write_csv(v2, out_dir / "t2_view.csv")
    try_plots(out_dir, ds, res, t1, t2, picks, v1, v2)

    pairs = [(p[0], p[1]) for p in picks.values()]
    border = border_report(ds, pairs)
    lines = ["# 임계값 실험 — 개발(dev) 세트 결과", "",
             f"- 개발 세트: 답이 있음 {n_a} / 답이 없음 {n_u} / 경계(BORDER, 계산에서 제외) {n_b}",
             *common_header(args, args.weights),
             f"- 격자: T1 {t1[0]}~{t1[-1]}, T2 {t2[0]}~{t2[-1]}, 부트스트랩 {args.draws}회, 씨앗 {args.seed}"]
    if not ds.extra["has_sentence_labels"]:
        lines.append("- **필요 문장 라벨이 없어 '불완전 전달'은 판정하지 않았습니다.**")
    elif ds.extra["warn_no_needed"]:
        lines.append(f"- 정답 FAQ인데 필요한 문장이 하나도 `Y`가 아닌 경우가 {ds.extra['warn_no_needed']}건 있습니다. 라벨을 확인하세요.")
    gr = ds.extra["gold_rank"][ds.kind == 1]
    lines += ["", "## 검색 자체의 한계 (임계값으로 해결할 수 없는 부분)", "",
              f"- 답이 있는 질문 중 정답 FAQ가 상위 3개 안에 없는 비율: **{(gr > TOP_K).mean():.1%}** "
              f"(상위 10개에도 없음 {(gr > 10).mean():.1%})",
              "  이 비율은 T1·T2를 어떻게 정해도 '엉뚱 전달' 또는 '미응답'으로 남습니다. FAQ 추가나 질문 변형 보강으로 풀어야 합니다.",
              "", "## 후보 3쌍", "",
              "| 유형 | T1 | T2 | 손해(90% 구간) | 정답 전달 | 불완전 | 엉뚱 전달 | 미응답 | 잘못 통과(90% 구간) | 경계 질문 중 통과 | 선택 기준 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for idx, (name, (a, b, why, (i, j))) in enumerate(picks.items()):
        lines.append(
            f"| {name} | {a:.2f} | {b:.2f} | {res['loss'][i, j]:.3f} ({res['loss_lo'][i, j]:.3f}~{res['loss_hi'][i, j]:.3f}) "
            f"| {res['correct'][i, j]:.1%} | {res['incomplete'][i, j]:.1%} | {res['wrong'][i, j]:.1%} | {res['noans'][i, j]:.1%} "
            f"| {res['false_pass'][i, j]:.1%} ({res['fp_lo'][i, j]:.1%}~{res['fp_hi'][i, j]:.1%}) "
            f"| {border[idx]:.1%} | {why} |" if border else
            f"| {name} | {a:.2f} | {b:.2f} | {res['loss'][i, j]:.3f} | {res['correct'][i, j]:.1%} | {res['incomplete'][i, j]:.1%} "
            f"| {res['wrong'][i, j]:.1%} | {res['noans'][i, j]:.1%} | {res['false_pass'][i, j]:.1%} | - | {why} |")
    lines += ["", "## 후보별 '잘못 통과' 분해 (의도 그룹별)", ""]
    for name, (a, b, _, _) in picks.items():
        parts = [f"{g} {k}/{n}" for g, (k, n) in group_false_pass(ds, a, b).items()]
        lines.append(f"- {name} (T1 {a:.2f}, T2 {b:.2f}): " + ", ".join(parts))
    lines += ["", "## 읽는 법", "",
              "- 후보 3쌍은 개발 세트로만 골랐습니다. 이 숫자는 낙관적이며, 최종 세트(`final`)에서 한 번만 확인합니다.",
              "- 손해는 `weights.json` 가중치와 prior 비율로 합친 값입니다. 가중치가 바뀌면 후보도 바뀝니다.",
              "- 주변 칸과 비슷한 값이 넓게 퍼져 있는 곳이 안전합니다. `loss_heatmap.png`에서 후보가 평평한 구간 안에 있는지 확인하세요.",
              "- `t1_view.csv`: 질문 점수만 본 FAQ 선택 능력과 판정 호출 예상 비율(PR #228 머지 시 1위 점수<T1인 질문 비율).",
              "- `t2_view.csv`: 정답 FAQ가 주어졌을 때 필요한 문장이 남는 비율(재현율)과 정밀도."]
    (out_dir / "dev_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump({name: {"T1": a, "T2": b, "why": why} for name, (a, b, why, _) in picks.items()},
              open(out_dir / "dev_candidates.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n".join(lines))
    print(f"\n저장 위치: {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# 7. final: 최종 세트 채점 (한 번만)
# ---------------------------------------------------------------------------
def cmd_final(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / "final_lock.txt"
    if lock.exists() and not args.force_rerun:
        sys.exit("이미 최종 세트를 채점했습니다 (final_lock.txt). 후보를 바꿔 다시 보면 최종 세트가 개발 세트가 됩니다.\n"
                 "정말 필요하면 --force-rerun 으로 실행하되, 보고서에 재실행 사실을 적으세요.")
    if args.pairs:
        pairs = {}
        for k, token in enumerate(args.pairs.split(",")):
            a, b = token.split(":")
            pairs[f"후보{k + 1}"] = (float(a), float(b))
    else:
        cand = json.load(open(args.candidates_json, encoding="utf-8"))
        pairs = {name: (v["T1"], v["T2"]) for name, v in cand.items()}
    if len(pairs) > 4:
        sys.exit("후보는 4쌍 이하로 제한합니다. 최종 세트는 선택된 소수의 후보만 확인하는 용도입니다.")

    ds_all = load_inputs(args)
    ds_test = ds_all.subset(ds_all.split == "test")
    ds_dev = ds_all.subset(ds_all.split == "dev")
    weights = load_weights(args.weights)
    rng_seed = args.seed
    lines = ["# 임계값 실험 — 최종(test) 세트 결과", "",
             f"- 최종 세트: 답이 있음 {(ds_test.kind == 1).sum()} / 답이 없음 {(ds_test.kind == 0).sum()} / 경계 {(ds_test.kind == -1).sum()}",
             *common_header(args, args.weights), f"- 부트스트랩 {args.draws}회", "",
             "| 후보 | T1 | T2 | 손해 test (90% 구간) | 손해 dev | 정답 전달 | 불완전 | 엉뚱 전달 | 미응답 | 잘못 통과 (90% 구간) | 경계 질문 중 통과 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, (a, b) in pairs.items():
        t1, t2 = np.array([a]), np.array([b])
        rt = analyze_grid(ds_test, weights, t1, t2, args.prior_answerable, args.draws, rng_seed)
        rd = analyze_grid(ds_dev, weights, t1, t2, args.prior_answerable, args.draws, rng_seed)
        border = border_report(ds_test, [(a, b)])
        lines.append(
            f"| {name} | {a:.2f} | {b:.2f} | {rt['loss'][0, 0]:.3f} ({rt['loss_lo'][0, 0]:.3f}~{rt['loss_hi'][0, 0]:.3f}) "
            f"| {rd['loss'][0, 0]:.3f} | {rt['correct'][0, 0]:.1%} | {rt['incomplete'][0, 0]:.1%} | {rt['wrong'][0, 0]:.1%} "
            f"| {rt['noans'][0, 0]:.1%} | {rt['false_pass'][0, 0]:.1%} ({rt['fp_lo'][0, 0]:.1%}~{rt['fp_hi'][0, 0]:.1%}) "
            f"| {border[0] if border else float('nan'):.1%} |")
    lines += ["", "## 읽는 법", "",
              "- 최종 세트의 손해가 개발 세트보다 눈에 띄게 크면, 개발 세트에서 고른 값이 우연히 좋았을 가능성이 있습니다. "
              "그 경우 더 보수적인 후보를 택하세요.",
              "- 후보들의 90% 구간이 크게 겹치면 통계적으로 구분되지 않습니다. 이때는 운영 사정(판정 호출 비용, 오답의 위험)으로 정하세요.",
              "- 이 결과를 본 뒤 후보를 바꿔 다시 실행하지 마세요."]
    (out_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lock.write_text("final 실행됨\n" + json.dumps(pairs), encoding="utf-8")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("mergequestions")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--out", default="questions.csv")
    s.set_defaults(func=cmd_mergequestions)

    s = sub.add_parser("labelsheet")
    s.add_argument("--questions", required=True)
    s.add_argument("--data", required=True, help="수집 결과 폴더(candidates.csv 등)")
    s.add_argument("--out", default="label_sheet.csv")
    s.add_argument("--seed", default="threshold-lab")
    s.set_defaults(func=cmd_labelsheet)

    s = sub.add_parser("merge")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--out", default="labels_final.csv")
    s.set_defaults(func=cmd_merge)

    s = sub.add_parser("sentencesheet")
    s.add_argument("--labels", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--out", default="sentence_sheet.csv")
    s.set_defaults(func=cmd_sentencesheet)

    def add_common(sp):
        sp.add_argument("--labels", required=True, help="labels_final.csv")
        sp.add_argument("--data", required=True, help="수집 결과 폴더")
        sp.add_argument("--sentence-labels", help="필요 문장 라벨(sentence_sheet.csv를 채운 것)")
        sp.add_argument("--weights", required=True)
        sp.add_argument("--prior-answerable", type=float, default=0.75,
                        help="실제 사용자 질문 중 FAQ로 답할 수 있는 질문의 비율 (question_log 등으로 추정)")
        sp.add_argument("--draws", type=int, default=1000)
        sp.add_argument("--seed", type=int, default=20261019)
        sp.add_argument("--out", default="out")

    s = sub.add_parser("dev")
    add_common(s)
    s.add_argument("--t1-range", nargs=3, type=float, default=[0.50, 0.90, 0.01])
    s.add_argument("--t2-range", nargs=3, type=float, default=[0.40, 0.90, 0.01])
    s.add_argument("--fp-cap-conservative", type=float, default=0.05)
    s.add_argument("--fp-cap-responsive", type=float, default=0.15)
    s.set_defaults(func=cmd_dev)

    s = sub.add_parser("final")
    add_common(s)
    s.add_argument("--pairs", help='예: "0.72:0.65,0.75:0.70" (T1:T2 쉼표 구분)')
    s.add_argument("--candidates-json", help="dev가 만든 dev_candidates.json")
    s.add_argument("--force-rerun", action="store_true")
    s.set_defaults(func=cmd_final)

    args = p.parse_args(argv)
    if args.cmd == "final" and not (args.pairs or args.candidates_json):
        p.error("final에는 --pairs 또는 --candidates-json이 필요합니다.")
    args.func(args)


if __name__ == "__main__":
    main()
