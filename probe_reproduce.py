"""
PROBE ECG case study -- full reproduction script.

Reproduces every deterministic result in the paper from the public
MIT-BIH Arrhythmia Database (PhysioNet), using the inter-patient
DS1/DS2 split of de Chazal et al. (2004).

How to run (Google Colab or any Python 3.9+):
    !pip install wfdb scipy scikit-learn matplotlib numpy
    !python probe_reproduce.py

Outputs:
    results.json   every number reported in the paper
    fig1.pdf       prematurity vs. peak-width scatter (Fig. 1)
    console log    the same numbers, human readable

Parameters marked RECONSTRUCTED were not fully specified in the original
manuscript; the values here are documented choices and are reported in
the paper as such.
"""

import json
import math
import os
import time

import numpy as np
from scipy.signal import find_peaks, peak_widths, resample

# ---------------------------------------------------------------- config
FS = 360                      # MIT-BIH sampling rate (Hz)
WIN_S = 0.30                  # measurement window, centred on the beat
CHANNEL = 0                   # MLII in almost all records
SEED = 0
DATA_DIR = "mitdb"

PACED = {"102", "104", "107", "217"}
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
       "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
       "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
       "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
       "233", "234"]

# All MIT-BIH beat annotation symbols (non-beat annotations are ignored
# when finding the previous/next beat).
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")

# RECONSTRUCTED: evidence-grounding perturbation (time-compression factor
# applied to a 0.6 s segment around the beat before re-measuring width).
EGS_COMPRESS = 0.5
EGS_SEG_S = 0.60

# RECONSTRUCTED: coarse pre-scan used by the selective-measurement policy.
COARSE_EARLY = 0.85           # RR / mean of previous 8 RR below this = early
COARSE_LATE = 1.15            # ... above this = late (atypical timing)

CONFORMAL_TARGETS = [0.90, 0.97, 0.995]


# ------------------------------------------------------------ data access
def load_record(rec):
    """Return (signal_channel0, beat_samples, beat_symbols) for one record."""
    import wfdb
    path = os.path.join(DATA_DIR, rec)
    sig, _ = wfdb.rdsamp(path, channels=[CHANNEL])
    ann = wfdb.rdann(path, "atr")
    keep = [i for i, s in enumerate(ann.symbol) if s in BEAT_SYMBOLS]
    samples = np.asarray(ann.sample)[keep]
    symbols = [ann.symbol[i] for i in keep]
    return sig[:, 0].astype(float), samples, symbols


def download():
    import wfdb
    if not os.path.exists(os.path.join(DATA_DIR, "100.hea")):
        print("Downloading MIT-BIH from PhysioNet ...")
        wfdb.dl_database("mitdb", DATA_DIR)


# ----------------------------------------------------------- measurements
def window(sig, centre, half):
    lo, hi = centre - half, centre + half
    if lo < 0 or hi > len(sig):
        return None
    return sig[lo:hi]


def peak_width_ms(seg):
    """Width at half prominence of the most prominent peak (ms).

    The segment is median-centred and rectified so that negative-going
    (inverted) complexes are measured the same way as positive ones.
    """
    x = np.abs(seg - np.median(seg))
    peaks, props = find_peaks(x, prominence=0)
    if len(peaks) == 0:
        return None
    k = int(np.argmax(props["prominences"]))
    w = peak_widths(x, peaks[[k]], rel_height=0.5)[0][0]
    return 1000.0 * w / FS


def crude_width_ms(seg):
    """Crude proxy: samples deviating from the median by > half the max."""
    d = np.abs(seg - np.median(seg))
    if d.max() == 0:
        return None
    return 1000.0 * np.sum(d > 0.5 * d.max()) / FS


def extract(records, classes):
    """Measurements for every beat whose symbol is in `classes`."""
    half = int(round(WIN_S * FS / 2))
    rows, dropped = [], 0
    for rec in records:
        sig, t, sym = load_record(rec)
        for i in range(len(t)):
            if sym[i] not in classes:
                continue
            if i < 2 or i + 1 >= len(t):
                dropped += 1
                continue
            rr_prev = t[i - 1] - t[i - 2]
            rr_cur = t[i] - t[i - 1]
            rr_next = t[i + 1] - t[i]
            seg = window(sig, t[i], half)
            if seg is None or rr_prev <= 0 or rr_cur <= 0:
                dropped += 1
                continue
            pw = peak_width_ms(seg)
            cw = crude_width_ms(seg)
            if pw is None or cw is None:
                dropped += 1
                continue
            past = [t[j] - t[j - 1] for j in range(max(1, i - 8), i)]
            rows.append(dict(
                rec=rec, idx=int(t[i]), label=sym[i],
                prem=rr_cur / rr_prev,
                pause=rr_next / rr_cur,
                width=pw,
                crude=cw,
                rel_rr=rr_cur / np.mean(past),
            ))
    return rows, dropped


# ---------------------------------------------------------------- metrics
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def metrics(y, yhat):
    y, yhat = np.asarray(y, bool), np.asarray(yhat, bool)
    tp = int(np.sum(y & yhat)); tn = int(np.sum(~y & ~yhat))
    fp = int(np.sum(~y & yhat)); fn = int(np.sum(y & ~yhat))
    n = len(y)
    out = {}
    for name, k, m in [("accuracy", tp + tn, n), ("sensitivity", tp, tp + fn),
                       ("specificity", tn, tn + fp), ("precision", tp, tp + fp)]:
        out[name] = round(100 * k / m, 2) if m else float("nan")
        out[name + "_ci"] = [round(v, 1) for v in wilson(k, m)]
    out.update(n=n, tp=tp, tn=tn, fp=fp, fn=fn)
    return out


def show(title, m):
    print(f"  {title:<34} acc {m['accuracy']:6.2f} {m['accuracy_ci']}  "
          f"sens {m['sensitivity']:6.2f}  spec {m['specificity']:6.2f}  "
          f"prec {m['precision']:6.2f}  (n={m['n']})")


def col(rows, k):
    return np.array([r[k] for r in rows], float)


def midpoint(rows, key, pos_label, neg_label):
    a = np.median([r[key] for r in rows if r["label"] == pos_label])
    b = np.median([r[key] for r in rows if r["label"] == neg_label])
    return float((a + b) / 2), float(a), float(b)


# ------------------------------------------------------------------- main
def main():
    t0 = time.time()
    download()
    R = {}

    # ---- PVC task: data
    d1, drop1 = extract(DS1, {"N", "V"})
    d2, drop2 = extract(DS2, {"N", "V"})
    y1 = np.array([r["label"] == "V" for r in d1])
    y2 = np.array([r["label"] == "V" for r in d2])
    R["data"] = dict(ds1_beats=len(d1), ds2_beats=len(d2),
                     ds1_dropped=drop1, ds2_dropped=drop2,
                     ds1_pvc=int(y1.sum()), ds2_pvc=int(y2.sum()))
    print("DATA", R["data"])

    # ---- thresholds (DS1 only)
    th = {}
    for k in ["prem", "pause", "width", "crude"]:
        th[k], mv, mn = midpoint(d1, k, "V", "N")
        th[k + "_median_V"], th[k + "_median_N"] = mv, mn
    R["thresholds"] = {k: round(v, 3) for k, v in th.items()}
    print("THRESHOLDS", R["thresholds"])

    def votes(rows, width_key="width", widths=None):
        p = col(rows, "prem") < th["prem"]
        q = col(rows, "pause") > th["pause"]
        w = (col(rows, width_key) if widths is None else widths) > th[width_key]
        return p.astype(int) + q.astype(int) + w.astype(int), p, q, w

    v1, *_ = votes(d1)
    v2, p2, q2, w2 = votes(d2)
    rule2 = v2 >= 2

    print("\nPVC TASK")
    R["rule_ds2"] = metrics(y2, rule2); show("Fixed rule 2-of-3, DS2", R["rule_ds2"])
    R["rule_ds1"] = metrics(y1, v1 >= 2); show("Fixed rule 2-of-3, DS1 (in-sample)", R["rule_ds1"])

    # ---- logistic regression
    from sklearn.linear_model import LogisticRegression
    X1 = np.c_[col(d1, "prem"), col(d1, "pause"), col(d1, "width")]
    X2 = np.c_[col(d2, "prem"), col(d2, "pause"), col(d2, "width")]
    lr = LogisticRegression().fit(X1, y1)
    R["logreg_ds2"] = metrics(y2, lr.predict(X2)); show("Logistic regression", R["logreg_ds2"])

    # ---- ablation
    R["ablation"] = {
        "prematurity_only": metrics(y2, p2),
        "peak_width_only": metrics(y2, w2),
        "pause_only": metrics(y2, q2),
        "three_of_three": metrics(y2, v2 == 3),
    }
    for k, m in R["ablation"].items():
        show(k, m)

    # ---- crude raw-signal proxy (H2)
    vc, *_ = votes(d2, width_key="crude")
    R["crude_proxy_ds2"] = metrics(y2, vc >= 2); show("Crude width proxy 2-of-3", R["crude_proxy_ds2"])

    # ---- selective measurement policy (RECONSTRUCTED coarse cues)
    # Coarse timing: RR relative to the mean of the previous 8 RRs.
    # Coarse shape: crude width proxy above its DS1 threshold.
    # Policy: always request pause. If timing and shape cues agree and the
    # pause agrees with them, decide from the cues. Otherwise (cues disagree,
    # timing late/atypical, or pause contradicts) request all three exact
    # measurements and apply the 2-of-3 rule.
    rel = col(d2, "rel_rr")
    early = rel < COARSE_EARLY
    late = rel > COARSE_LATE
    broad = col(d2, "crude") > th["crude"]
    cue_pvc = early & broad
    cue_n = (~early) & (~late) & (~broad)
    agree = cue_pvc | cue_n
    pause_ok = np.where(cue_pvc, q2, ~q2)
    pause_only = agree & pause_ok
    sel = np.where(pause_only, cue_pvc, rule2)
    R["selective_ds2"] = metrics(y2, sel)
    R["selective_ds2"]["pause_only_pct"] = round(100 * pause_only.mean(), 2)
    R["selective_ds2"]["escalated_pct"] = round(100 * (1 - pause_only.mean()), 2)
    show("Selective policy", R["selective_ds2"])
    print(f"    resolved with pause only: {R['selective_ds2']['pause_only_pct']}%  "
          f"escalated: {R['selective_ds2']['escalated_pct']}%")

    # ---- evidence-grounding intervention
    half = int(round(WIN_S * FS / 2))
    seg_half = int(round(EGS_SEG_S * FS / 2))
    eligible = np.where(y2 & rule2 & w2)[0]
    falsified = flipped = 0
    sigs = {}
    for j in eligible:
        r = d2[j]
        if r["rec"] not in sigs:
            sigs = {r["rec"]: load_record(r["rec"])[0]}
        sig = sigs[r["rec"]]
        lo, hi = r["idx"] - seg_half, r["idx"] + seg_half
        seg = sig[max(lo, 0):min(hi, len(sig))].copy()
        n = len(seg)
        comp = resample(seg, max(8, int(round(n * EGS_COMPRESS))))
        pad_l = (n - len(comp)) // 2
        new = np.r_[np.full(pad_l, comp[0]), comp,
                    np.full(n - len(comp) - pad_l, comp[-1])]
        c = len(new) // 2
        new_w = peak_width_ms(new[c - half:c + half])
        if new_w is None or new_w <= th["width"]:
            falsified += 1
            if int(p2[j]) + int(q2[j]) < 2:
                flipped += 1
    R["evidence_grounding"] = dict(
        eligible=int(len(eligible)), falsified=falsified,
        falsified_pct=round(100 * falsified / max(len(eligible), 1), 2),
        decisions_flipped=flipped,
        flipped_pct=round(100 * flipped / max(len(eligible), 1), 2),
        compress_factor=EGS_COMPRESS)
    print("EVIDENCE GROUNDING", R["evidence_grounding"])

    # ---- split-conformal abstention
    def scores(v):           # returns (s_N, s_PVC)
        return v / 3.0, 1 - v / 3.0

    def conformal(cal_true_scores, sN, sP, y):
        out = []
        n = len(cal_true_scores)
        for tgt in CONFORMAL_TARGETS:
            k = min(n, math.ceil((n + 1) * tgt))
            q = float(np.sort(cal_true_scores)[k - 1])
            inN, inP = sN <= q + 1e-12, sP <= q + 1e-12
            size = inN.astype(int) + inP.astype(int)
            cover = np.where(y, inP, inN)
            single = size == 1
            pred = inP[single]
            sel_acc = 100 * np.mean(pred == y[single]) if single.any() else float("nan")
            out.append(dict(target=tgt, qhat=round(q, 3),
                            empirical_coverage=round(float(100 * cover.mean()), 2),
                            avg_set_size=round(float(size.mean()), 3),
                            abstain_pct=round(float(100 * (~single).mean()), 2),
                            empty_pct=round(float(100 * (size == 0).mean()), 2),
                            selective_accuracy=round(float(sel_acc), 2)))
        return out

    sN1, sP1 = scores(v1); sN2, sP2 = scores(v2)
    cal = np.where(y1, sP1, sN1)
    R["conformal"] = conformal(cal, sN2, sP2, y2)
    print("CONFORMAL (vote score)")
    for c in R["conformal"]:
        print("   ", c)

    # Richer score: vote fraction + agreement of fixed rule and selective policy
    # (both computed identically on DS1 for calibration).
    rel1 = col(d1, "rel_rr"); e1 = rel1 < COARSE_EARLY; l1 = rel1 > COARSE_LATE
    b1 = col(d1, "crude") > th["crude"]; q1 = col(d1, "pause") > th["pause"]
    cp1 = e1 & b1; cn1 = (~e1) & (~l1) & (~b1)
    po1 = (cp1 | cn1) & np.where(cp1, q1, ~q1)
    sel1 = np.where(po1, cp1, v1 >= 2)

    def rich(v, rule, sel):
        aP = (rule.astype(float) + sel.astype(float)) / 2
        return 1 - (0.5 * (1 - aP) + 0.5 * (1 - v / 3)), 1 - (0.5 * aP + 0.5 * v / 3)

    rN1, rP1 = rich(v1, v1 >= 2, sel1)
    rN2, rP2 = rich(v2, rule2, sel)
    R["conformal_rich"] = conformal(np.where(y1, rP1, rN1), rN2, rP2, y2)
    R["rule_selective_disagreements"] = int(np.sum(rule2 != sel))
    print("CONFORMAL (rich score); rule/selective disagreements:",
          R["rule_selective_disagreements"])
    for c in R["conformal_rich"]:
        print("   ", c)

    # ---- APB boundary case (Normal vs. A)
    a1, _ = extract(DS1, {"N", "A"})
    a2, _ = extract(DS2, {"N", "A"})
    ya2 = np.array([r["label"] == "A" for r in a2])
    tp_, mvp, mnp = midpoint(a1, "prem", "A", "N")
    tq_, mvq, mnq = midpoint(a1, "pause", "A", "N")
    _, mvw, mnw = midpoint(a1, "width", "A", "N")
    apb_pred = (col(a2, "prem") < tp_) & (col(a2, "pause") > tq_)
    R["apb"] = dict(ds1_apb=sum(r["label"] == "A" for r in a1),
                    ds2_apb=int(ya2.sum()), ds2_beats=len(a2),
                    thresholds=dict(prem=round(tp_, 3), pause=round(tq_, 3)),
                    ds1_width_median_A=round(mvw, 2), ds1_width_median_N=round(mnw, 2),
                    always_normal_accuracy=round(float(100 * (1 - ya2.mean())), 2),
                    metrics=metrics(ya2, apb_pred))
    print("\nAPB TASK", {k: v for k, v in R["apb"].items() if k != "metrics"})
    show("APB rule (both conditions)", R["apb"]["metrics"])

    # ---- Fig. 1: balanced random DS2 sample (30 N, 30 V)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(SEED)
    iN = rng.choice(np.where(~y2)[0], 30, replace=False)
    iV = rng.choice(np.where(y2)[0], 30, replace=False)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.scatter(col(d2, "prem")[iN], col(d2, "width")[iN], s=14, facecolors="none",
               edgecolors="tab:blue", label="Normal (N)")
    ax.scatter(col(d2, "prem")[iV], col(d2, "width")[iV], s=14, marker="^",
               color="tab:red", label="PVC (V)")
    ax.axvline(th["prem"], ls="--", lw=0.8, color="gray")
    ax.axhline(th["width"], ls="--", lw=0.8, color="gray")
    ax.set_xlabel("Prematurity ratio"); ax.set_ylabel("Peak width (ms)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig("fig1.pdf")

    R["runtime_s"] = round(time.time() - t0, 1)
    R["config"] = dict(window_s=WIN_S, channel=CHANNEL, seed=SEED,
                       egs_compress=EGS_COMPRESS, egs_segment_s=EGS_SEG_S,
                       coarse_early=COARSE_EARLY, coarse_late=COARSE_LATE)
    with open("results.json", "w") as f:
        json.dump(R, f, indent=1, default=float)
    print(f"\nDone in {R['runtime_s']} s. Wrote results.json and fig1.pdf.")


if __name__ == "__main__":
    main()
