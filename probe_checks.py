"""
PROBE ECG case study -- sensitivity checks (probe_checks.py).

Three questions a careful reader can ask about probe_reproduce.py, answered with
the same measurements, labels and metrics (nothing here changes the paper's
main protocol; it only re-runs it under stricter conditions).

  A  Calibration set.  In the paper the conformal quantile is calibrated on DS1,
     the same records that fixed the thresholds.  Strict split-conformal
     prediction wants a calibration set that played no part in fitting.  Here
     DS1 is split by RECORD into a fitting half (thresholds) and a calibration
     half (quantile), many times at random; every split is then scored on DS2
     and on INCART exactly as in the paper.  If the abstention behaviour were an
     artefact of reusing the fitting data, it would move here.
  B  Subject overlap.  Records 201 (DS1) and 202 (DS2) come from the same
     subject.  (i) the paper's DS1 fit is tested on DS2 without record 202;
     (ii) thresholds and logistic regression are refitted on DS1 without record
     201 and tested on all of DS2; (iii) both records removed.
  C  INCART independence.  The 75 INCART recordings come from 32 patients.  The
     header comments name the patient ("patient 1" ...), so the bootstrap can
     resample PATIENTS.  It is also run per recording and with blocks of 2 and
     3 consecutive recordings for comparison.  With --cnn-csv (the
     cnn_predictions.csv written by probe_cnn.py) the CNN-versus-rule
     differences and the CNN-confidence-versus-disagreement abstention
     comparison on INCART get the same patient-level intervals.

Run (Google Colab or any Python 3.9+; keep probe_reproduce.py beside this file):
    pip install wfdb scipy scikit-learn numpy awscli
    aws s3 sync --no-sign-request s3://physionet-open/mitdb/1.0.0/ mitdb
    aws s3 sync --no-sign-request s3://physionet-open/incartdb/1.0.0/ incartdb
    python probe_checks.py [--cnn-csv cnn_predictions.csv]

Outputs: checks_results.json and a printed block "== CHECKS RESULTS ==".
Takes a few minutes (extracting the beats dominates).  Deterministic.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time

import numpy as np

import probe_reproduce as pr

NAMES3 = ("prem", "pause", "width")


# ------------------------------------------------------------ small helpers
def thresholds(rows):
    return {k: pr.midpoint(rows, k, "V", "N")[0] for k in NAMES3}


def votes(rows, th):
    return ((pr.col(rows, "prem") < th["prem"]).astype(int)
            + (pr.col(rows, "pause") > th["pause"]).astype(int)
            + (pr.col(rows, "width") > th["width"]).astype(int))


def labels(rows):
    return np.array([r["label"] == "V" for r in rows])


def feats(rows):
    return np.c_[pr.col(rows, "prem"), pr.col(rows, "pause"), pr.col(rows, "width")]


def fit_lr(rows):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression().fit(feats(rows), labels(rows))


def conformal_one(cal_true_scores, v_test, y_test, target):
    """Same procedure as probe_reproduce.main(): vote score, one target."""
    n = len(cal_true_scores)
    k = min(n, math.ceil((n + 1) * target))
    q = float(np.sort(cal_true_scores)[k - 1])
    sN, sP = v_test / 3.0, 1 - v_test / 3.0
    inN, inP = sN <= q + 1e-12, sP <= q + 1e-12
    size = inN.astype(int) + inP.astype(int)
    single = size == 1
    cover = np.where(y_test, inP, inN)
    pred = inP[single]
    risk = 100 * float(np.mean(pred != y_test[single])) if single.any() else float("nan")
    return dict(qhat=round(q, 3), coverage=100 * float(cover.mean()),
                abstain=100 * float((~single).mean()), risk=risk,
                pvc_abstain=100 * float((~single)[y_test].mean()) if y_test.any() else float("nan"))


def cal_scores(rows, th):
    v = votes(rows, th)
    y = labels(rows)
    return np.where(y, 1 - v / 3.0, v / 3.0)


def unanimity(v, y):
    keep = (v == 0) | (v == 3)
    wrong = ((v == 3) != y)[keep]
    return dict(coverage_pct=round(100 * float(keep.mean()), 2),
                risk_pct=round(100 * float(wrong.mean()), 3) if keep.any() else float("nan"),
                n=int(len(y)))


def lr_risk_at(conf, wrong, k):
    if k <= 0:
        return float("nan")
    top = np.argpartition(-conf, k - 1)[:k]
    return float(np.mean(wrong[top]))


def summ(x):
    x = np.asarray(x, float)
    return dict(mean=round(float(np.mean(x)), 3), min=round(float(np.min(x)), 3),
                max=round(float(np.max(x)), 3))


def pick_split(recs, y, rng, min_pvc, tries=1000):
    """Random record-level split of DS1 into (fit, calibration) halves."""
    ids = sorted(set(recs))
    n_fit = len(ids) // 2
    for _ in range(tries):
        fit = set(rng.choice(ids, n_fit, replace=False).tolist())
        m = np.array([r in fit for r in recs])
        if y[m].sum() >= min_pvc and y[~m].sum() >= min_pvc:
            return m
    raise SystemExit("could not find a split with enough PVCs in both halves")


# ------------------------------------------------------ A: calibration set
def check_A(d1, tests, n_splits=100, targets=(0.97, 0.99, 0.995), min_pvc=300):
    y1 = labels(d1)
    recs1 = np.array([r["rec"] for r in d1])
    out = {"targets": list(targets), "n_splits": n_splits, "protocol_in_paper": {}, "splits": {},
           "calibration_diagnostics": {}}
    # reference: the paper's protocol (fit and calibrate on all of DS1)
    th_all = thresholds(d1)
    sc_all = cal_scores(d1, th_all)
    out["calibration_diagnostics"]["paper_protocol"] = dict(
        share_score_le_1_3=round(100 * float(np.mean(sc_all <= 1 / 3 + 1e-9)), 3),
        share_score_le_2_3=round(100 * float(np.mean(sc_all <= 2 / 3 + 1e-9)), 3))
    for tgt in targets:
        out["protocol_in_paper"][str(tgt)] = {}
        for name, rows in tests.items():
            v, y = votes(rows, th_all), labels(rows)
            out["protocol_in_paper"][str(tgt)][name] = dict(
                conformal_one(sc_all, v, y, tgt), rule_accuracy=100 * float(np.mean((v >= 2) == y)))
    per = {(t, name): [] for t in targets for name in tests}
    diag = {"share_score_le_1_3": [], "share_score_le_2_3": []}
    for s_ in range(n_splits):
        rng = np.random.default_rng(1000 + s_)
        m = pick_split(recs1, y1, rng, min_pvc)
        fit_rows = [r for r, k in zip(d1, m) if k]
        cal_rows = [r for r, k in zip(d1, m) if not k]
        th = thresholds(fit_rows)
        sc = cal_scores(cal_rows, th)
        diag["share_score_le_1_3"].append(100 * float(np.mean(sc <= 1 / 3 + 1e-9)))
        diag["share_score_le_2_3"].append(100 * float(np.mean(sc <= 2 / 3 + 1e-9)))
        for name, rows in tests.items():
            v, y = votes(rows, th), labels(rows)
            for tgt in targets:
                c = conformal_one(sc, v, y, tgt)
                c["rule_accuracy"] = 100 * float(np.mean((v >= 2) == y))
                per[(tgt, name)].append(c)
    out["calibration_diagnostics"]["splits_share_score_le_1_3"] = summ(diag["share_score_le_1_3"])
    out["calibration_diagnostics"]["splits_share_score_le_2_3"] = summ(diag["share_score_le_2_3"])
    for tgt in targets:
        out["splits"][str(tgt)] = {}
        for name in tests:
            L = per[(tgt, name)]
            qc = {}
            for c in L:
                qc[str(c["qhat"])] = qc.get(str(c["qhat"]), 0) + 1
            out["splits"][str(tgt)][name] = dict(
                qhat_counts=qc,
                abstain_pct=summ([c["abstain"] for c in L]),
                selective_risk_pct=summ([c["risk"] for c in L]),
                empirical_coverage_pct=summ([c["coverage"] for c in L]),
                pvc_abstained_pct=summ([c["pvc_abstain"] for c in L]),
                rule_accuracy_pct=summ([c["rule_accuracy"] for c in L]),
                share_of_splits_meeting_target=round(
                    float(np.mean([c["coverage"] >= 100 * tgt for c in L])), 3))
    return out


# ------------------------------------------------------- B: subject overlap
def one_setup(th, lr, rows):
    y = labels(rows)
    v = votes(rows, th)
    p = lr.predict_proba(feats(rows))[:, 1]
    return dict(n=int(len(y)), pvc=int(y.sum()),
                rule=pr.metrics(y, v >= 2), logreg=pr.metrics(y, p > 0.5),
                unanimity=unanimity(v, y),
                logreg_risk_same_coverage_pct=round(100 * lr_risk_at(
                    np.abs(p - 0.5), (p > 0.5) != y,
                    int(((v == 0) | (v == 3)).sum())), 3))


def check_B(d1, d2):
    out = {}
    th, lr = thresholds(d1), fit_lr(d1)
    out["paper_protocol_full_DS2"] = one_setup(th, lr, d2)
    out["fit_DS1_test_DS2_without_202"] = one_setup(th, lr, [r for r in d2 if r["rec"] != "202"])
    d1b = [r for r in d1 if r["rec"] != "201"]
    thb, lrb = thresholds(d1b), fit_lr(d1b)
    out["fit_DS1_without_201_thresholds"] = {k: round(v, 3) for k, v in thb.items()}
    out["fit_DS1_without_201_test_full_DS2"] = one_setup(thb, lrb, d2)
    out["fit_DS1_without_201_test_DS2_without_202"] = one_setup(thb, lrb, [r for r in d2 if r["rec"] != "202"])
    return out


# ------------------------------------------------ C: INCART resampling units
def m4(y, p):
    tp = np.sum(y & p); tn = np.sum(~y & ~p); fp = np.sum(~y & p); fn = np.sum(y & ~p)
    return np.array([100 * (tp + tn) / len(y), 100 * tp / max(tp + fn, 1),
                     100 * tn / max(tn + fp, 1), 100 * tp / max(tp + fp, 1)])


M4 = ("accuracy", "sensitivity", "specificity", "precision")


def patient_map(comments):
    """recording -> patient id parsed from header comments ("patient 12")."""
    m, bad = {}, []
    for rec, com in comments.items():
        hit = None
        for line in com:
            g = re.search(r"patient\s*(\d+)", line, re.I)
            if g:
                hit = int(g.group(1))
                break
        if hit is None:
            bad.append(rec)
        m[rec] = hit
    return m, bad


def read_cnn_csv(path, inc):
    """Ensemble probabilities from cnn_predictions.csv, aligned to the INCART rows."""
    P = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["set"] == "incart":
                P[(r["rec"], int(r["idx"]))] = (r["label"], float(r["wave_p"]), float(r["wave_rr_p"]))
    miss = sum((r["rec"], r["idx"]) not in P for r in inc)
    if miss:
        print(f"WARNING: {miss} INCART beats not found in {path}; CNN comparison skipped")
        return None
    same = all(P[(r["rec"], r["idx"])][0] == r["label"] for r in inc)
    if not same:
        print("WARNING: labels in the CNN csv disagree with ours; CNN comparison skipped")
        return None
    return {"wave": np.array([P[(r["rec"], r["idx"])][1] for r in inc]),
            "wave_rr": np.array([P[(r["rec"], r["idx"])][2] for r in inc])}


def check_C(inc, d1, comments, cnn=None, reps=1000, seed=0):
    th, lr = thresholds(d1), fit_lr(d1)
    y = labels(inc)
    v = votes(inc, th)
    rule = v >= 2
    p = lr.predict_proba(feats(inc))[:, 1]
    keep = (v == 0) | (v == 3)
    wrong_v = (v == 3) != y
    conf_l, wrong_l = np.abs(p - 0.5), (p > 0.5) != y
    recs = []
    for r in inc:
        if r["rec"] not in recs:
            recs.append(r["rec"])
    rec_arr = np.array([r["rec"] for r in inc])
    by = {r: np.where(rec_arr == r)[0] for r in recs}
    pmap, bad = patient_map(comments)
    res = {"n_recordings": len(recs)}
    units = {}
    for b in (1, 2, 3):
        units["recording" if b == 1 else f"block_{b}"] = [
            np.concatenate([by[r] for r in recs[i:i + b]]) for i in range(0, len(recs), b)]
    if not bad:
        pats = {}
        for r in recs:
            pats.setdefault(pmap[r], []).append(r)
        units["patient"] = [np.concatenate([by[r] for r in rs]) for rs in pats.values()]
        res["n_patients"] = len(pats)
        res["recordings_per_patient"] = {str(k): len(v_) for k, v_ in sorted(pats.items())}
        res["patient_of_recording"] = {r: pmap[r] for r in recs}
    else:
        res["patient_parse_failed_for"] = bad

    cnnc = {}
    if cnn is not None:
        for name, pe in cnn.items():
            cnnc[name] = dict(p=pe, conf=np.abs(pe - 0.5), wrong=(pe > 0.5) != y)

    def stats(ii):
        out = {}
        yy = y[ii]
        out["rule"] = m4(yy, rule[ii]); out["logreg"] = m4(yy, (p[ii] > 0.5))
        out["rule_minus_logreg"] = out["rule"] - out["logreg"]
        kk = keep[ii]
        k = int(kk.sum())
        rv = float(np.mean(wrong_v[ii][kk])) if k else float("nan")
        rl = lr_risk_at(conf_l[ii], wrong_l[ii], k)
        out["risk"] = np.array([100 * rv, 100 * rl, 100 * (rv - rl)])
        for name, c in cnnc.items():
            m = m4(yy, c["p"][ii] > 0.5)
            out[name] = m
            out[name + "_minus_rule"] = m - out["rule"]
            rc = lr_risk_at(c["conf"][ii], c["wrong"][ii], k)
            out[name + "_risk"] = np.array([100 * rc, 100 * (rv - rc)])
        return out

    allidx = np.arange(len(y))
    point = stats(allidx)
    res["point"] = {k: [round(float(x), 3) for x in v_] for k, v_ in point.items()}
    for uname, groups in units.items():
        rng = np.random.default_rng(seed)
        acc = {k: [] for k in point}
        for _ in range(reps):
            ii = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
            st = stats(ii)
            for k, v_ in st.items():
                acc[k].append(v_)
        ci = {}
        for k, L in acc.items():
            A = np.array(L, float)
            lo, hi = np.nanpercentile(A, 2.5, axis=0), np.nanpercentile(A, 97.5, axis=0)
            ci[k] = [[round(float(a), 2), round(float(b), 2)] for a, b in zip(lo, hi)]
        res[uname] = dict(n_units=len(groups), ci_95=ci)
    res["order_of_numbers"] = dict(
        rule_logreg_cnn="accuracy, sensitivity, specificity, precision",
        risk="[unanimity risk %, logreg risk at same coverage %, unanimity minus logreg]",
        cnn_risk="[cnn-confidence risk %, unanimity minus cnn-confidence]")
    return res


def incart_header_comments():
    import wfdb
    out = {}
    for rec in pr.list_incart_records():
        try:
            out[rec] = list(wfdb.rdheader(os.path.join(pr.INCART_DIR, rec)).comments or [])
        except Exception as e:  # pragma: no cover
            out[rec] = ["<unreadable: %s>" % e]
    return out


# --------------------------------------------------------------------- main
def pr_line(tag, d):
    print(f"  {tag}: {json.dumps(d)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--no-incart", action="store_true")
    ap.add_argument("--cnn-csv", default=None, help="cnn_predictions.csv from probe_cnn.py (optional)")
    ap.add_argument("--only", choices=["A", "B", "C"], default=None,
                    help="run just one check (B needs no INCART and takes about a minute)")
    ap.add_argument("--out", default="checks_results.json")
    args = ap.parse_args()
    want = lambda k: args.only in (None, k)
    t0 = time.time()
    pr.download()
    d1, _ = pr.extract(pr.DS1, {"N", "V"})
    d2, _ = pr.extract(pr.DS2, {"N", "V"})
    print(f"DS1 {len(d1)} beats, DS2 {len(d2)} beats (paper: 41,730 and 39,612)")
    inc = None
    if not args.no_incart and (want("A") or want("C")):
        pr.download_incart()
        inc, drop = [], 0
        for rec in pr.list_incart_records():
            sig, t, sym = pr.load_incart_record(rec)
            r, d = pr.measure_sequence(sig, t, sym, {"N", "V"}, rec, db="incartdb")
            inc += r
            drop += d
        print(f"INCART {len(inc)} beats (paper: 170,206)")

    R = {}
    tests = {"DS2": d2}
    if inc is not None:
        tests["INCART"] = inc
    if want("A"):
        R["A_disjoint_calibration"] = check_A(d1, tests, n_splits=args.splits)
    if want("B"):
        R["B_subject_overlap"] = check_B(d1, d2)
    if inc is not None and want("C"):
        com = incart_header_comments()
        cnn = read_cnn_csv(args.cnn_csv, inc) if args.cnn_csv else None
        R["C_incart_units"] = check_C(inc, d1, com, cnn=cnn)
        R["C_incart_header_comments"] = com
    R["runtime_s"] = round(time.time() - t0, 1)
    with open(args.out, "w") as f:
        json.dump(R, f, indent=1)

    print("\n================ CHECKS RESULTS ================")
    if "A_disjoint_calibration" in R:
        A = R["A_disjoint_calibration"]
        print("A. calibration on a record-disjoint half of DS1, several targets")
        pr_line("calibration scores (paper protocol)", A["calibration_diagnostics"]["paper_protocol"])
        pr_line("calibration scores (splits)", {k: v for k, v in A["calibration_diagnostics"].items() if k.startswith("splits")})
        for tgt, d in A["protocol_in_paper"].items():
            pr_line(f"paper protocol, target {tgt}", d)
        for tgt, d in A["splits"].items():
            for name, dd in d.items():
                pr_line(f"splits, target {tgt}, {name}", dd)
    if "B_subject_overlap" in R:
        print("B. subject overlap (records 201 / 202)")
        for k, d in R["B_subject_overlap"].items():
            if isinstance(d, dict) and "rule" in d:
                print(f"  {k}: n={d['n']}  pvc={d['pvc']}")
                print(f"     rule    {d['rule']}")
                print(f"     logreg  {d['logreg']}")
                print(f"     unanimity {d['unanimity']}  | LR risk at same coverage {d['logreg_risk_same_coverage_pct']}")
            else:
                print(f"  {k}: {d}")
    if inc is not None and "C_incart_units" in R:
        C = R["C_incart_units"]
        print("C. INCART resampling units (patient / recording / blocks)")
        for k in ("n_recordings", "n_patients", "patient_parse_failed_for"):
            if k in C:
                print(f"  {k}: {C[k]}")
        if "recordings_per_patient" in C:
            print("  recordings_per_patient:", json.dumps(C["recordings_per_patient"]))
            print("  patient_of_recording:", json.dumps(C["patient_of_recording"]))
        pr_line("point estimates", C["point"])
        for u in ("recording", "block_2", "block_3", "patient"):
            if u in C:
                pr_line(f"95% interval, unit={u} (n_units={C[u]['n_units']})", C[u]["ci_95"])
        pr_line("order of numbers", C["order_of_numbers"])
        nonempty = {k: v for k, v in com.items() if v}
        print(f"  header comments: {len(nonempty)} of {len(com)} recordings have any")
    print("runtime_s", R["runtime_s"])
    print("================================================")
    print("Wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
