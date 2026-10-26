"""
PROBE ECG case study -- full reproduction script.

Reproduces every deterministic result in the paper from the public
MIT-BIH Arrhythmia Database (PhysioNet), using the inter-patient
DS1/DS2 split of de Chazal et al. (2004).

Also runs an external validation on the INCART database and an evaluation
with automatic beat detection (XQRS) instead of reference annotations.

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


def measure_sequence(sig, t, sym, classes, rec, db="mitdb"):
    """Measurements for every beat in (t, sym) whose symbol is in `classes`.

    `t` is the full beat sequence (all beat types, or all detections), so
    neighbouring-beat timing uses whatever beats are present in it.
    """
    half = int(round(WIN_S * FS / 2))
    rows, dropped = [], 0
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
            rec=rec, db=db, idx=int(t[i]), label=sym[i],
            prem=rr_cur / rr_prev,
            pause=rr_next / rr_cur,
            width=pw,
            crude=cw,
            rel_rr=rr_cur / np.mean(past),
        ))
    return rows, dropped


def extract(records, classes):
    """Measurements for every MIT-BIH beat whose symbol is in `classes`."""
    rows, dropped = [], 0
    for rec in records:
        sig, t, sym = load_record(rec)
        r, d = measure_sequence(sig, t, sym, classes, rec)
        rows += r
        dropped += d
    return rows, dropped


# ------------------------------------------------- external database (INCART)
INCART_DIR = "incartdb"
INCART_FS = 257


def download_incart():
    """Download INCART, resuming: only records whose files are missing."""
    import wfdb
    recs = ["I%02d" % i for i in range(1, 76)]
    missing = [r for r in recs if not all(
        os.path.exists(os.path.join(INCART_DIR, r + ext)) and
        os.path.getsize(os.path.join(INCART_DIR, r + ext)) > 0
        for ext in (".hea", ".dat", ".atr"))]
    if missing:
        print(f"Downloading {len(missing)} INCART records from PhysioNet ...")
        for k, r in enumerate(missing, 1):
            wfdb.dl_database("incartdb", INCART_DIR, records=[r], overwrite=True)
            print(f"  {k}/{len(missing)} {r}", flush=True)
    if not os.path.exists(os.path.join(INCART_DIR, "RECORDS")):
        with open(os.path.join(INCART_DIR, "RECORDS"), "w") as f:
            f.write("\n".join(recs) + "\n")


def list_incart_records():
    with open(os.path.join(INCART_DIR, "RECORDS")) as f:
        return [x.strip() for x in f if x.strip()]


def load_incart_record(rec):
    """Lead II resampled 257 -> 360 Hz; annotation times rescaled to match."""
    import wfdb
    from scipy.signal import resample_poly
    path = os.path.join(INCART_DIR, rec)
    hdr = wfdb.rdheader(path)
    lead = [n.upper() for n in hdr.sig_name].index("II")
    sig, _ = wfdb.rdsamp(path, channels=[lead])
    x = resample_poly(sig[:, 0].astype(float), FS, INCART_FS)
    ann = wfdb.rdann(path, "atr")
    keep = [i for i, s in enumerate(ann.symbol) if s in BEAT_SYMBOLS]
    samples = np.round(np.asarray(ann.sample)[keep] * FS / INCART_FS).astype(int)
    return x, samples, [ann.symbol[i] for i in keep]


def load_db_record(db, rec):
    return load_record(rec) if db == "mitdb" else load_incart_record(rec)


def detect_beats(sig):
    """Automatic QRS detection (wfdb XQRS) at 360 Hz."""
    from wfdb import processing
    return np.asarray(processing.xqrs_detect(sig=sig, fs=FS, verbose=False), int)


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

    # ---- evidence-grounding intervention, with controls
    half = int(round(WIN_S * FS / 2))
    seg_half = int(round(EGS_SEG_S * FS / 2))
    sig_cache = {}

    def get_sig(rec, db="mitdb"):
        key = (db, rec)
        if key not in sig_cache:
            sig_cache.clear()
            sig_cache[key] = load_db_record(db, rec)[0]
        return sig_cache[key]

    def perturbed_width(r, factor, amp=1.0):
        """Re-measure peak width after time-scaling a 0.6 s segment.

        factor < 1 compresses (edge-padded back to length), factor > 1
        stretches (centre-cropped back to length), factor == 1 passes the
        segment through the same resampling pipeline unchanged (control).
        """
        sig = get_sig(r["rec"], r.get("db", "mitdb"))
        lo, hi = r["idx"] - seg_half, r["idx"] + seg_half
        if lo < 0 or hi > len(sig):
            return None
        seg = amp * sig[lo:hi].copy()
        n = len(seg)
        m = max(8, int(round(n * factor)))
        out = resample(seg, m)
        if m < n:
            pl = (n - m) // 2
            out = np.r_[np.full(pl, out[0]), out, np.full(n - m - pl, out[-1])]
        elif m > n:
            cl = (m - n) // 2
            out = out[cl:cl + n]
        c = n // 2
        return peak_width_ms(out[c - half:c + half])

    def run_test(idx_list, factor, amp, claim_wide, decision_votes,
                 rows=None, p_arr=None, q_arr=None):
        """claim_wide=True: cited claim is 'width > thr'; else 'width <= thr'.

        Returns how often the cited claim stops holding, and how often the
        decision changes, after the perturbation.
        """
        rows = d2 if rows is None else rows
        p_arr = p2 if p_arr is None else p_arr
        q_arr = q2 if q_arr is None else q_arr
        stop = flip = 0
        fails = []
        for j in idx_list:
            r = rows[j]
            w = perturbed_width(r, factor, amp)
            wide_now = (w is not None) and (w > th["width"])
            if wide_now != claim_wide:
                stop += 1
                other = int(p_arr[j]) + int(q_arr[j])
                if (other + int(wide_now) >= 2) != decision_votes:
                    flip += 1
            else:
                fails.append(dict(rec=r["rec"], sample=r["idx"],
                                  width_before=round(r["width"], 1),
                                  width_after=None if w is None else round(w, 1)))
        n = max(len(idx_list), 1)
        return dict(n=int(len(idx_list)), claim_stopped=stop,
                    claim_stopped_pct=round(100 * stop / n, 2),
                    decisions_flipped=flip,
                    flipped_pct=round(100 * flip / n, 2)), fails

    elig_wide = np.where(y2 & rule2 & w2)[0]       # correct PVCs citing 'wide'
    elig_narrow = np.where(~y2 & ~rule2 & ~w2)[0]  # correct normals citing 'narrow'

    main_res, main_fails = run_test(elig_wide, EGS_COMPRESS, 1.0, True, True)
    ctrl_id, _ = run_test(elig_wide, 1.0, 1.0, True, True)
    ctrl_amp, _ = run_test(elig_wide, 1.0, 0.5, True, True)
    rev_res, _ = run_test(elig_narrow, 2.0, 1.0, False, False)

    # failures: grouped by record, plus how close they were to the threshold
    fail_recs = {}
    for f in main_fails:
        fail_recs[f["rec"]] = fail_recs.get(f["rec"], 0) + 1
    R["evidence_grounding"] = dict(
        eligible=main_res["n"], falsified=main_res["claim_stopped"],
        falsified_pct=main_res["claim_stopped_pct"],
        decisions_flipped=main_res["decisions_flipped"],
        flipped_pct=main_res["flipped_pct"],
        compress_factor=EGS_COMPRESS,
        control_identity_resample=ctrl_id,
        control_amplitude_half=ctrl_amp,
        reverse_stretch_normals=rev_res,
        failures_by_record=fail_recs,
        failures=main_fails)
    print("EVIDENCE GROUNDING (compress 0.5, wide-claim PVCs)", main_res)
    print("  control: identity resample (claim should HOLD)", ctrl_id)
    print("  control: amplitude x0.5 (claim should HOLD)  ", ctrl_amp)
    print("  reverse: stretch x2 on narrow-claim normals   ", rev_res)
    print("  failures by record:", fail_recs)

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

    # ---- abstention detail: risk-coverage, class-wise, per record
    recs2 = np.array([r["rec"] for r in d2])
    rc = []
    for q in [0.0, 1 / 3, 2 / 3, 1.0]:
        inN, inP = sN2 <= q + 1e-12, sP2 <= q + 1e-12
        single = (inN.astype(int) + inP.astype(int)) == 1
        cov = single.mean()
        risk = np.mean(inP[single] != y2[single]) if single.any() else float("nan")
        rc.append(dict(qhat=round(q, 3), coverage_pct=round(float(100 * cov), 2),
                       selective_risk_pct=round(float(100 * risk), 3)))
    q97 = R["conformal"][1]["qhat"]
    inN, inP = sN2 <= q97 + 1e-12, sP2 <= q97 + 1e-12
    single = (inN.astype(int) + inP.astype(int)) == 1
    kept = metrics(y2[single], inP[single])
    per_rec = []
    for rec in DS2:
        m = recs2 == rec
        if not m.any():
            continue
        k = single & m
        per_rec.append(dict(rec=rec, beats=int(m.sum()),
                            coverage_pct=round(float(100 * k.sum() / m.sum()), 1),
                            kept_accuracy_pct=round(float(100 * np.mean(inP[k] == y2[k])), 2)
                            if k.any() else None))
    accs = [x["kept_accuracy_pct"] for x in per_rec if x["kept_accuracy_pct"] is not None]
    R["abstention_detail"] = dict(
        risk_coverage=rc,
        target97_kept_metrics=kept,
        pvc_abstained_pct=round(float(100 * np.mean(~single[y2])), 2),
        normal_abstained_pct=round(float(100 * np.mean(~single[~y2])), 2),
        per_record=per_rec,
        per_record_kept_accuracy_min=min(accs), per_record_kept_accuracy_median=float(np.median(accs)),
        worst_records=sorted(per_rec, key=lambda x: x["kept_accuracy_pct"] or 100)[:3])
    print("ABSTENTION DETAIL: risk-coverage", rc)
    show("Kept beats at 97% target", kept)
    print("  PVCs abstained %:", R["abstention_detail"]["pvc_abstained_pct"],
          " normals abstained %:", R["abstention_detail"]["normal_abstained_pct"])
    print("  per-record kept accuracy: min", min(accs), " median", float(np.median(accs)))
    print("  worst records:", R["abstention_detail"]["worst_records"])

    # ---- per-record error concentration (fixed rule)
    fn_by = {rec: int(np.sum((recs2 == rec) & y2 & ~rule2)) for rec in DS2}
    fp_by = {rec: int(np.sum((recs2 == rec) & ~y2 & rule2)) for rec in DS2}
    top_fn = sorted(fn_by.items(), key=lambda x: -x[1])[:3]
    top_fp = sorted(fp_by.items(), key=lambda x: -x[1])[:3]
    R["error_by_record"] = dict(fn_total=int(sum(fn_by.values())), fp_total=int(sum(fp_by.values())),
                                top_fn_records=top_fn, top_fp_records=top_fp)
    print("ERRORS BY RECORD", R["error_by_record"])

    # ---- patient-level (record) bootstrap, 1000 resamples of the 22 DS2 records
    rng_b = np.random.default_rng(SEED)
    lr_pred2 = lr.predict(X2)
    rec_ids = [rec for rec in DS2 if (recs2 == rec).any()]
    idx_by = {rec: np.where(recs2 == rec)[0] for rec in rec_ids}
    boot = {"rule": [], "logreg": []}
    for _ in range(1000):
        pick = rng_b.choice(rec_ids, len(rec_ids), replace=True)
        ii = np.concatenate([idx_by[r] for r in pick])
        for name, pr in [("rule", rule2), ("logreg", lr_pred2)]:
            yy, pp = y2[ii], pr[ii]
            tp = np.sum(yy & pp); fn = np.sum(yy & ~pp); fp = np.sum(~yy & pp)
            boot[name].append((100 * np.mean(yy == pp),
                               100 * tp / max(tp + fn, 1), 100 * tp / max(tp + fp, 1)))
    R["record_bootstrap_95ci"] = {}
    for name, vals in boot.items():
        a = np.array(vals)
        R["record_bootstrap_95ci"][name] = {
            k: [round(float(np.percentile(a[:, i], 2.5)), 1), round(float(np.percentile(a[:, i], 97.5)), 1)]
            for i, k in enumerate(["accuracy", "sensitivity", "precision"])}
    print("RECORD-LEVEL BOOTSTRAP 95% CI", R["record_bootstrap_95ci"])

    # ---- swapped split: thresholds / model fit on DS2, tested on DS1
    th_sw = {k: midpoint(d2, k, "V", "N")[0] for k in ["prem", "pause", "width"]}
    v_sw = ((col(d1, "prem") < th_sw["prem"]).astype(int) + (col(d1, "pause") > th_sw["pause"]).astype(int)
            + (col(d1, "width") > th_sw["width"]).astype(int))
    lr_sw = LogisticRegression().fit(X2, y2)
    R["swapped_split"] = dict(thresholds={k: round(v, 3) for k, v in th_sw.items()},
                              rule_on_ds1=metrics(y1, v_sw >= 2),
                              logreg_on_ds1=metrics(y1, lr_sw.predict(X1)))
    print("SWAPPED SPLIT (fit DS2, test DS1) thresholds", R["swapped_split"]["thresholds"])
    show("Fixed rule, swapped", R["swapped_split"]["rule_on_ds1"])
    show("Logistic regression, swapped", R["swapped_split"]["logreg_on_ds1"])

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
    matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3 fonts (IEEE requirement)
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

    # =================================================================
    # EXTENSIONS: evidence margin, external validation, beat detection
    # =================================================================
    # ---- unanimity abstention vs. logistic-regression confidence ----------
    # The DS1-calibrated vote-score conformal at the 97% target keeps exactly
    # the beats on which all three measurements agree (v = 0 or 3). Compare
    # its selective risk with logistic-regression confidence at the SAME
    # coverage, with a paired record-level bootstrap of the difference.
    def lr_risk_at(conf, wrong, k):
        if k <= 0:
            return float("nan")
        top = np.argpartition(-conf, k - 1)[:k]
        return float(np.mean(wrong[top]))

    def unanimity_vs_lr(v, y, p_lr, recs, seed=SEED, reps=1000):
        keep = (v == 0) | (v == 3)
        pred = v == 3
        wrong_v = pred != y
        conf = np.abs(p_lr - 0.5)
        wrong_lr = (p_lr > 0.5) != y
        k = int(keep.sum())
        risk_v = float(np.mean(wrong_v[keep]))
        risk_lr = lr_risk_at(conf, wrong_lr, k)
        rng_u = np.random.default_rng(seed)
        ids = sorted(set(recs))
        by = {r: np.where(recs == r)[0] for r in ids}
        diffs = []
        for _ in range(reps):
            ii = np.concatenate([by[r] for r in rng_u.choice(ids, len(ids), replace=True)])
            kk = keep[ii]
            if kk.sum() == 0:
                continue
            rv = np.mean(wrong_v[ii][kk])
            rl = lr_risk_at(conf[ii], wrong_lr[ii], int(kk.sum()))
            diffs.append(100 * (rv - rl))
        diffs = np.array(diffs)
        kept = metrics(y[keep], pred[keep])
        return dict(
            coverage_pct=round(float(100 * keep.mean()), 2),
            empirical_coverage_pct=round(float(100 * (1 - wrong_v[keep].sum() / len(y))), 2),
            unanimity_risk_pct=round(100 * risk_v, 3),
            logreg_risk_same_coverage_pct=round(100 * risk_lr, 3),
            diff_pct_points=round(100 * (risk_v - risk_lr), 3),
            diff_record_bootstrap_95ci=[round(float(np.percentile(diffs, 2.5)), 3),
                                        round(float(np.percentile(diffs, 97.5)), 3)],
            frac_bootstrap_unanimity_lower=round(float(np.mean(diffs < 0)), 3),
            kept_sensitivity=kept["sensitivity"], kept_precision=kept["precision"],
            kept_specificity=kept["specificity"],
            pvc_abstained_pct=round(float(100 * np.mean(~keep[y])), 2),
            normal_abstained_pct=round(float(100 * np.mean(~keep[~y])), 2),
            errors_kept=int(wrong_v[keep].sum()), n=int(len(y)))

    # ---- evidence margin: signed distance of each measurement from its
    # threshold, in robust (MAD) units fitted on DS1, oriented so that
    # positive = favours PVC. The 2-of-3 vote is positive exactly when the
    # median of the three margins is positive; |median| is how far the
    # evidence is from flipping the decision.
    def mad(x):
        return 1.4826 * float(np.median(np.abs(x - np.median(x))))
    sc = {k: mad(col(d1, k)) for k in ["prem", "pause", "width"]}

    def margin(rows):
        mp = (th["prem"] - col(rows, "prem")) / sc["prem"]
        mq = (col(rows, "pause") - th["pause"]) / sc["pause"]
        mw = (col(rows, "width") - th["width"]) / sc["width"]
        return np.median(np.c_[mp, mq, mw], axis=1)

    m1, m2 = margin(d1), margin(d2)
    mismatch = int(np.sum((m2 > 0) != rule2))
    print(f"\nEVIDENCE MARGIN: scales {dict((k, round(v, 4)) for k, v in sc.items())}, "
          f"mismatches with 2-of-3 rule on DS2: {mismatch}")

    MARGIN_TARGETS = [0.90, 0.95, 0.97, 0.98, 0.99, 0.995]

    def margin_conformal(m_cal, y_cal, m_test, y_test):
        cal = np.where(y_cal, -m_cal, m_cal)
        n = len(cal)
        out = []
        for tgt in MARGIN_TARGETS:
            k = min(n, math.ceil((n + 1) * tgt))
            q = float(np.sort(cal)[k - 1])
            inN, inP = m_test <= q, -m_test <= q
            size = inN.astype(int) + inP.astype(int)
            single = size == 1
            cover = np.where(y_test, inP, inN)
            kept = metrics(y_test[single], inP[single]) if single.any() else None
            out.append(dict(
                target=tgt, qhat=round(q, 4),
                empirical_coverage=round(float(100 * cover.mean()), 2),
                abstain_pct=round(float(100 * (~single).mean()), 2),
                selective_accuracy=None if kept is None else kept["accuracy"],
                kept_sensitivity=None if kept is None else kept["sensitivity"],
                kept_precision=None if kept is None else kept["precision"],
                pvc_abstained_pct=round(float(100 * np.mean(~single[y_test])), 2),
                normal_abstained_pct=round(float(100 * np.mean(~single[~y_test])), 2)))
        return out

    R["margin"] = dict(scales=sc, rule_mismatches=mismatch,
                       conformal_ds2=margin_conformal(m1, y1, m2, y2))
    print("MARGIN CONFORMAL (DS1-calibrated, DS2)")
    for c in R["margin"]["conformal_ds2"]:
        print("   ", c)

    # ---- risk-coverage: margin vs. logistic-regression confidence
    def risk_coverage(conf, err):
        order = np.argsort(-conf, kind="stable")
        e = err[order].astype(float)
        risk = np.cumsum(e) / np.arange(1, len(e) + 1)
        return risk

    def at_cov(risk, cov):
        k = max(1, int(round(cov * len(risk))))
        return round(float(100 * risk[k - 1]), 3)

    lr_p2 = lr.predict_proba(X2)[:, 1]
    rc_margin = risk_coverage(np.abs(m2), (m2 > 0) != y2)
    rc_lr = risk_coverage(np.abs(lr_p2 - 0.5), (lr_p2 > 0.5) != y2)
    cov_vote = 1 - R["conformal"][1]["abstain_pct"] / 100
    R["risk_coverage"] = dict(
        aurc_margin_pct=round(float(100 * rc_margin.mean()), 3),
        aurc_logreg_pct=round(float(100 * rc_lr.mean()), 3),
        at_vote_coverage=dict(coverage_pct=round(100 * cov_vote, 2),
                              margin_risk_pct=at_cov(rc_margin, cov_vote),
                              logreg_risk_pct=at_cov(rc_lr, cov_vote),
                              vote_risk_pct=R["abstention_detail"]["risk_coverage"][0]["selective_risk_pct"]),
        curve=[dict(coverage_pct=c, margin_risk_pct=at_cov(rc_margin, c / 100),
                    logreg_risk_pct=at_cov(rc_lr, c / 100)) for c in [50, 70, 80, 90, 95, 100]])
    print("RISK-COVERAGE", {k: v for k, v in R["risk_coverage"].items() if k != "curve"})
    for c in R["risk_coverage"]["curve"]:
        print("   ", c)
    R["unanimity_vs_logreg_ds2"] = unanimity_vs_lr(v2, y2, lr_p2, recs2)
    print("UNANIMITY vs LOGREG (DS2)", R["unanimity_vs_logreg_ds2"])

    # ---- external validation on INCART (no refitting)
    download_incart()
    inc, inc_drop = [], 0
    for rec in list_incart_records():
        sig, t, sym = load_incart_record(rec)
        r, d = measure_sequence(sig, t, sym, {"N", "V"}, rec, db="incartdb")
        inc += r
        inc_drop += d
    yi = np.array([r["label"] == "V" for r in inc])
    vi, pi, qi, wi = votes(inc)
    rule_i = vi >= 2
    Xi = np.c_[col(inc, "prem"), col(inc, "pause"), col(inc, "width")]
    mi = margin(inc)
    elig_i = np.where(yi & rule_i & wi)[0]
    gi_main, _ = run_test(elig_i, EGS_COMPRESS, 1.0, True, True, rows=inc, p_arr=pi, q_arr=qi)
    gi_ctrl, _ = run_test(elig_i, 1.0, 1.0, True, True, rows=inc, p_arr=pi, q_arr=qi)
    inc_recs = np.array([r["rec"] for r in inc])
    rec_ids_i = sorted(set(inc_recs))
    idx_by_i = {rec: np.where(inc_recs == rec)[0] for rec in rec_ids_i}
    bi = []
    rng_i = np.random.default_rng(SEED)
    for _ in range(1000):
        pick = rng_i.choice(rec_ids_i, len(rec_ids_i), replace=True)
        ii = np.concatenate([idx_by_i[r] for r in pick])
        yy, pp = yi[ii], rule_i[ii]
        tp = np.sum(yy & pp); fn = np.sum(yy & ~pp); fp = np.sum(~yy & pp)
        bi.append((100 * np.mean(yy == pp), 100 * tp / max(tp + fn, 1), 100 * tp / max(tp + fp, 1)))
    bi = np.array(bi)
    R["external_incart"] = dict(
        records=len(rec_ids_i), beats=len(inc), pvc=int(yi.sum()), dropped=inc_drop,
        always_normal_accuracy=round(float(100 * (1 - yi.mean())), 2),
        rule=metrics(yi, rule_i), logreg=metrics(yi, lr.predict(Xi)),
        rule_record_bootstrap_95ci={k: [round(float(np.percentile(bi[:, i], 2.5)), 1),
                                        round(float(np.percentile(bi[:, i], 97.5)), 1)]
                                    for i, k in enumerate(["accuracy", "sensitivity", "precision"])},
        margin_conformal=margin_conformal(m1, y1, mi, yi),
        grounding_compress=gi_main, grounding_identity_control=gi_ctrl)
    R["external_incart"]["unanimity_vs_logreg"] = unanimity_vs_lr(vi, yi, lr.predict_proba(Xi)[:, 1], inc_recs)
    print("\nEXTERNAL: INCART", {k: R["external_incart"][k] for k in ["records", "beats", "pvc", "dropped", "always_normal_accuracy"]})
    show("Fixed rule (DS1 thresholds)", R["external_incart"]["rule"])
    show("Logistic regression (DS1 fit)", R["external_incart"]["logreg"])
    print("  rule record-bootstrap 95% CI", R["external_incart"]["rule_record_bootstrap_95ci"])
    print("  grounding compress", gi_main, " identity control", gi_ctrl)
    print("  UNANIMITY vs LOGREG (INCART)", R["external_incart"]["unanimity_vs_logreg"])
    for c in R["external_incart"]["margin_conformal"]:
        print("   ", c)

    # ---- automatic beat detection on DS2 (no reference beat positions)
    TOL = int(round(0.15 * FS))
    det_rows, det_drop = [], 0
    n_ref = n_det = n_match = 0
    for rec in DS2:
        sig, t_ref, sym_ref = load_record(rec)
        det = np.sort(detect_beats(sig))
        used = np.zeros(len(t_ref), bool)
        lab = []
        for dpos in det:
            j = int(np.searchsorted(t_ref, dpos))
            best = None
            for jj in (j - 1, j):
                if 0 <= jj < len(t_ref) and not used[jj] and abs(int(t_ref[jj]) - int(dpos)) <= TOL:
                    if best is None or abs(int(t_ref[jj]) - int(dpos)) < abs(int(t_ref[best]) - int(dpos)):
                        best = jj
            if best is None:
                lab.append("X")
            else:
                used[best] = True
                lab.append(sym_ref[best])
        n_ref += len(t_ref); n_det += len(det); n_match += int(used.sum())
        r, d = measure_sequence(sig, det, lab, {"N", "V"}, rec)
        det_rows += r
        det_drop += d
    yd = np.array([r["label"] == "V" for r in det_rows])
    vd, *_ = votes(det_rows)
    Xd = np.c_[col(det_rows, "prem"), col(det_rows, "pause"), col(det_rows, "width")]
    R["detected_beats"] = dict(
        detector="wfdb XQRS", match_tolerance_s=0.15,
        reference_beats=n_ref, detections=n_det, matched=n_match,
        detection_sensitivity=round(100 * n_match / max(n_ref, 1), 2),
        detection_ppv=round(100 * n_match / max(n_det, 1), 2),
        beats=len(det_rows), pvc=int(yd.sum()), dropped=det_drop,
        rule=metrics(yd, vd >= 2), logreg=metrics(yd, lr.predict(Xd)),
        margin_conformal=margin_conformal(m1, y1, margin(det_rows), yd))
    det_recs = np.array([r["rec"] for r in det_rows])
    R["detected_beats"]["unanimity_vs_logreg"] = unanimity_vs_lr(vd, yd, lr.predict_proba(Xd)[:, 1], det_recs)
    print("\nDETECTED BEATS (XQRS) on DS2", {k: R["detected_beats"][k] for k in
          ["reference_beats", "detections", "matched", "detection_sensitivity", "detection_ppv", "beats", "pvc"]})
    show("Fixed rule, detected beats", R["detected_beats"]["rule"])
    show("Logistic regression, detected", R["detected_beats"]["logreg"])
    print("  UNANIMITY vs LOGREG (detected)", R["detected_beats"]["unanimity_vs_logreg"])
    for c in R["detected_beats"]["margin_conformal"]:
        print("   ", c)

    R["runtime_s"] = round(time.time() - t0, 1)
    R["config"] = dict(window_s=WIN_S, channel=CHANNEL, seed=SEED,
                       egs_compress=EGS_COMPRESS, egs_segment_s=EGS_SEG_S,
                       coarse_early=COARSE_EARLY, coarse_late=COARSE_LATE)
    with open("results.json", "w") as f:
        json.dump(R, f, indent=1, default=float)
    print(f"\nDone in {R['runtime_s']} s. Wrote results.json and fig1.pdf.")


if __name__ == "__main__":
    main()
