# -*- coding: utf-8 -*-
"""
Conformal Calibration for the Scope Gate (Layer 1) -- Train Ticket
========================================================================
Ban tong quat hoa cua experiments/parser/scope_gate_conformal_calibration.py
(ban goc chi chay tren 61/81 prompt SockShop) -- dung lai DUNG phuong phap do
(5-fold CV logistic regression lam nonconformity score, roi split-conformal
hai nguong TAU_PASS/TAU_REFUSE), nhung tren
data/processed/scm_results/trainticket_scope_gate_layerA_validation.csv
(46 prompt Train Ticket: 30 hop le + 16 doi khang, xem
trainticket_scope_gate_feature_extraction.py va
data/benchmark/trainticket_scope_gate_prompts_DRAFT.csv).

HAI NGUONG (giong dung thiet ke dang trien khai trong parser_agent.py):
  score <  TAU_PASS   -> confidently in-scope, auto-pass.
    Hieu chinh tren cac diem OUT-OF-SCOPE: split-conformal dam bao
    Pr[mot prompt out-of-scope MOI co score duoi nguong nay] <= alpha.
  score >= TAU_REFUSE -> confidently out-of-scope, auto-refuse.
    Hieu chinh tren cac diem LEGITIMATE (anh guong cua cong thuc tren):
    Pr[mot prompt legitimate MOI co score o hoac tren nguong nay] <= beta.
  TAU_PASS <= score < TAU_REFUSE -> NEEDS_HUMAN_REVIEW.

CANH BAO CO BAN, PHAI NEU RO (da thong bao truoc voi nguoi dung khi chay):
n=46 (16 out-of-scope, 30 legitimate) NHO HON ca bo SockShop goc (61, 11
out-of-scope) chu chua noi den ban mo rong 81-diem cua SockShop. Voi n_pos=16
va alpha=0.01, k=ceil(17*0.01)=1 -- TAU_PASS se la gia tri NHO NHAT trong 16
diem out-of-scope, tuc mot bao dam rat long leo (thuc chat chi dua tren 1
diem cuc tri). Day la PILOT de kiem chung co che tong quat hoa duoc, KHONG
phai so lieu du manh de cong bo thay cho bang RQ3/Scope-Gate chinh cua
SockShop -- phai noi ro dieu nay o moi noi trich dan ket qua nay.

Khong goi LLM nao trong script nay -- chi phan tich lai CSV da thu o buoc
truoc.
"""

import math
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
while PROJECT_ROOT != os.path.dirname(PROJECT_ROOT) and not os.path.isdir(os.path.join(PROJECT_ROOT, 'src')):
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
assert os.path.isdir(os.path.join(PROJECT_ROOT, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results',
                          'trainticket_scope_gate_layerA_validation.csv')


def conformal_threshold_low(cal_scores_pos: np.ndarray, alpha: float) -> float:
    """TAU_PASS: k-th SMALLEST score among TRUE OUT-OF-SCOPE calibration
    points, k = ceil((n+1)*alpha). Guarantees Pr[new true-positive scores
    BELOW this] <= alpha (split-conformal, finite-sample correction).
    Giong HET conformal_threshold() trong scope_gate_conformal_calibration.py
    goc -- giu lai ten rieng (_low) o day de doi xung voi _high ben duoi va
    tranh nham voi huong nguoc (xem canh bao lich su trong docstring goc ve
    mot lan tinh sai chieu)."""
    n = len(cal_scores_pos)
    k = max(1, math.ceil((n + 1) * alpha))
    k = min(k, n)
    return np.sort(cal_scores_pos)[k - 1]


def conformal_threshold_high(cal_scores_neg: np.ndarray, beta: float) -> float:
    """TAU_REFUSE: anh guong cua ham tren, tren cac diem LEGITIMATE.
    j-th SMALLEST score among TRUE LEGITIMATE calibration points,
    j = ceil((n+1)*(1-beta)), tuc gan DINH tren cua phan phoi legitimate.
    Guarantees Pr[new true-legitimate scores O HOAC TREN this] <= beta."""
    n = len(cal_scores_neg)
    j = max(1, math.ceil((n + 1) * (1.0 - beta)))
    j = min(j, n)
    return np.sort(cal_scores_neg)[j - 1]


def main():
    print("=" * 80)
    print("  CONFORMAL CALIBRATION FOR THE SCOPE GATE -- TRAIN TICKET (PILOT, n=46)")
    print(f"  Data: {DATA_PATH}")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    y = df['true_should_refuse'].astype(bool).to_numpy()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    print(f"\n[Data] n={len(df)} ({n_pos} true out-of-scope, {n_neg} legitimate)")
    print("[CANH BAO] n nho hon ca 61-prompt goc cua SockShop -- xem docstring dau file.")

    # ---- Step 0: raw signal discrimination (giong buoc dau cua ban goc) ----
    print("\n[Step 0] Raw signal discrimination check:")
    ctab = pd.crosstab(df['llm_is_customer_facing'], df['true_should_refuse'])
    print(ctab.to_string())
    recall_bool = ((df['llm_is_customer_facing'] == False) & y).sum() / n_pos
    fpr_bool = ((df['llm_is_customer_facing'] == False) & (~y)).sum() / n_neg
    print(f"  'is_customer_facing==False' alone: recall={recall_bool:.1%}, false-positive-rate={fpr_bool:.1%}")
    means = df.groupby('true_should_refuse')['keyword_similarity_score'].mean()
    direction = ('WRONG direction' if means[True] < means[False]
                 else 'right direction, weak' if means[True] - means[False] < 0.3
                 else 'informative')
    print(f"  keyword_similarity_score class means: legitimate={means[False]:.3f}, "
          f"out-of-scope={means[True]:.3f} ({direction})")

    # ---- Fit a combined score via 5-fold CV (tranh danh gia tren dong da fit) ----
    features = ['keyword_similarity_score', 'llm_pick_similarity', 'rb_similarity']
    X = df[features].fillna(0.0).to_numpy()
    X = np.hstack([X, df[['llm_is_customer_facing']].astype(float).to_numpy()])
    y_int = y.astype(int)

    n_splits = min(5, n_pos, n_neg)  # n_pos=16 >=5 nen van la 5-fold nhu ban goc
    cv_scores = np.zeros(len(df))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(X, y_int):
        if y_int[train_idx].sum() == 0 or y_int[train_idx].sum() == len(train_idx):
            cv_scores[test_idx] = y_int[train_idx].mean()
            continue
        clf = LogisticRegression(class_weight='balanced', max_iter=1000)
        clf.fit(X[train_idx], y_int[train_idx])
        cv_scores[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    df['nonconformity_score'] = cv_scores

    print(f"\n[Fitted score] {n_splits}-fold CV, class-conditional distribution (out-of-fold):")
    print(df.groupby('true_should_refuse')['nonconformity_score'].describe()[['mean', 'std', 'min', 'max']].to_string())

    # Fit MOT lan tren TOAN BO du lieu (de co he so cuoi dung trong production,
    # giong dung vai tro cua _SCOPE_GATE_LR_COEF/_INTERCEPT trong parser_agent.py --
    # phien ban "deployed scorer nen dung moi nhan co", khac ban CV chi de
    # UOC LUONG hieu nang ngoai-mau truoc khi quyet dinh nguong).
    full_clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    full_clf.fit(X, y_int)
    coef = dict(zip(['keyword_similarity_score', 'llm_pick_similarity',
                     'rb_similarity', 'llm_is_customer_facing'], full_clf.coef_[0]))
    intercept = float(full_clf.intercept_[0])
    print("\n[Full-data fit] He so dung lam scope_gate_lr_coef cho ParserAgent (Train Ticket):")
    for k, v in coef.items():
        print(f"    {k:28s} = {v:+.6f}")
    print(f"    intercept{' '*20} = {intercept:+.6f}")

    pos_scores = df.loc[y, 'nonconformity_score'].to_numpy()
    neg_scores = df.loc[~y, 'nonconformity_score'].to_numpy()

    ALPHA, BETA = 0.10, 0.10  # LONG HON ban goc (0.01/0.05) vi n nho hon han --
    # dung 0.01/0.05 tren n_pos=16 se cho TAU_PASS gan nhu = min tuyet doi cua
    # 16 diem (k=1), mot bao dam qua long leo de coi la co nghia. 0.10/0.10 la
    # lua chon RO RANG, khong giau: doi lay bao dam long hon (10% thay vi 1%/5%)
    # de nguong con phan anh duoc hon 1 diem du lieu don le. PHAI neu lai dung
    # alpha/beta nay moi khi trich dan, khong duoc ngam coi la "giong SockShop".
    tau_pass = conformal_threshold_low(pos_scores, ALPHA)
    tau_refuse = conformal_threshold_high(neg_scores, BETA)

    print(f"\n[Conformal thresholds] alpha={ALPHA} (TAU_PASS, tren {n_pos} diem out-of-scope), "
          f"beta={BETA} (TAU_REFUSE, tren {n_neg} diem legitimate):")
    print(f"    TAU_PASS   = {tau_pass:.6f}")
    print(f"    TAU_REFUSE = {tau_refuse:.6f}")

    # Bao ve dung chieu -- giong dung kiem tra da ghi lai trong lich su cua
    # parser_agent.py (mot lan tinh nguoc chieu tung bi phat hien nho in ra
    # dung bang sanity-check nay truoc khi hardcode).
    thresholds_valid = tau_pass <= tau_refuse
    if not thresholds_valid:
        print("\n[CANH BAO NGHIEM TRONG] TAU_PASS > TAU_REFUSE -- vung review RONG hoac dao "
              "nguoc logic refuse/pass. KHONG dung 2 gia tri nay. Giam ALPHA/BETA hoac thu them du lieu.")
    else:
        all_scores = df['nonconformity_score'].to_numpy()
        auto_pass = all_scores < tau_pass
        auto_refuse = all_scores >= tau_refuse
        review = (~auto_pass) & (~auto_refuse)
        missed = (auto_pass & y).sum()
        wrongly_refused = (auto_refuse & ~y).sum()
        print(f"\n[Sanity breakdown tren CHINH {len(df)} diem dung de dat nguong -- "
              f"danh gia trong-mau, chua co tap test doc lap]:")
        print(f"    Out-of-scope lot qua vung auto-pass:  {missed}/{n_pos}")
        print(f"    Legitimate bi tu-choi oan:              {wrongly_refused}/{n_neg}")
        print(f"    Day sang NEEDS_HUMAN_REVIEW:            {review.sum()}/{len(df)} "
              f"({100*review.sum()/len(df):.1f}%)")

    out_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results',
                            'trainticket_scope_gate_conformal_calibration.csv')
    df[['prompt_id', 'source', 'true_should_refuse', 'keyword_similarity_score',
        'llm_pick_similarity', 'rb_similarity', 'llm_is_customer_facing',
        'nonconformity_score']].to_csv(out_csv, index=False)
    print(f"\n[OK] Saved per-prompt scores: {out_csv}")

    coef_suffix = '_coefficients.csv' if thresholds_valid else '_coefficients_INVALID_TAU_INVERTED.csv'
    coef_path = out_csv.replace('.csv', coef_suffix)
    note = ('' if thresholds_valid else
            f'INVALID: tau_pass > tau_refuse (nghich chieu) do n qua nho + tach lop qua sach. '
            f'KHONG dung 2 nguong nay lam scope_gate_tau_pass/tau_refuse trong ParserAgent.')
    pd.DataFrame([{**coef, 'intercept': intercept, 'tau_pass': tau_pass,
                   'tau_refuse': tau_refuse, 'alpha': ALPHA, 'beta': BETA,
                   'n_pos': n_pos, 'n_neg': n_neg, 'thresholds_valid': thresholds_valid,
                   'note': note}]).to_csv(coef_path, index=False)
    print(f"[OK] Saved coefficients/thresholds: {coef_path}")
    print("\n[NHAC LAI] n=46 la PILOT -- so sanh ngang hang voi SockShop (61-81 diem)")
    print("la khong hop ly. Dung ket qua nay de BAO CAO 'co che tong quat hoa duoc va cho")
    print("ket qua tot tren mau nho', KHONG thay the cho bang RQ3/Scope-Gate chinh.")
    print("=" * 80)
    return df, coef, intercept, tau_pass, tau_refuse


if __name__ == '__main__':
    main()
