# probe-ecg

Reproduction code for the paper *Executable, Falsifiable Evidence for ECG Beat Classification: PVC Detection with External Validation and Calibrated Abstention*.

A single script (`probe_reproduce.py`) reproduces every deterministic number, table and figure in the paper from two public PhysioNet databases: the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/), using the inter-patient DS1/DS2 split of de Chazal et al. (2004), and, for external validation, the [St Petersburg INCART 12-lead Arrhythmia Database](https://physionet.org/content/incartdb/1.0.0/).

## Run

Python 3.9+ (tested on Google Colab):

```
pip install wfdb scipy scikit-learn matplotlib numpy awscli
aws s3 sync --no-sign-request s3://physionet-open/incartdb/1.0.0/ incartdb
python probe_reproduce.py
```

The script downloads MIT-BIH automatically. INCART can also be downloaded by the script, but PhysioNet's own server is often very slow for it; the `aws s3 sync` line above fetches PhysioNet's official open-data mirror in a few minutes. Once the data are present, a full run takes about four minutes on a free Colab instance.

## Outputs

- `results.json`: every reported number (included here from the run used in the paper)
- `fig1.pdf`: the scatter plot (Fig. 2 in the paper; Fig. 1 is a TikZ flowchart inside the LaTeX source)
- console log: the same numbers in readable form

## What the script computes

- Three executable measurements per beat: prematurity ratio, pause ratio, peak width
- The fixed two-of-three rule, with thresholds set from DS1 medians
- Logistic regression, crude raw-signal proxy and ablation controls
- The selective-measurement policy
- The evidence-grounding perturbation test
- Conformal-style abstention, with the four-level and six-level scores
- The Normal vs. APB boundary case
- External validation on INCART, with no refitting
- Automatic beat detection (WFDB XQRS) in place of reference annotations
- Abstention on measurement disagreement vs. logistic-regression confidence at equal coverage, with a record-level bootstrap
- The continuous evidence-margin score (reported as a negative result)

`probe_reproduce.py` is deterministic and calls no language model.

## Sensitivity checks (`probe_checks.py`, optional)

Three stricter re-runs of the paper's protocol, using the same beats, measurements and metrics (it reuses `probe_reproduce.py`; keep both files together):

- **A. Calibration set.** The paper calibrates the conformal quantile on DS1, the records that also fixed the thresholds. Here DS1 is split by record into a fitting half and a calibration half, 100 random times, and each split is scored on DS2 and INCART. The 97% target is repeated at 99% and 99.5%.
- **B. Subject overlap.** Records 201 (DS1) and 202 (DS2) come from the same subject. The checks test the paper's fit without record 202, and refit thresholds and logistic regression without record 201.
- **C. INCART independence.** The INCART header files name the patient ("patient 12"), so the 75 recordings can be grouped into 32 patients. The bootstrap is run per patient, per recording, and with blocks of 2 and 3 consecutive recordings.

```
pip install wfdb scipy scikit-learn numpy awscli
aws s3 sync --no-sign-request s3://physionet-open/mitdb/1.0.0/ mitdb
aws s3 sync --no-sign-request s3://physionet-open/incartdb/1.0.0/ incartdb
python probe_checks.py    # about 2 minutes
```

Outputs `checks_results.json` and a printed `CHECKS RESULTS` block. Deterministic; a few minutes.
