# probe-ecg

Reproduction code for the paper *Executable, Falsifiable Evidence for ECG Beat Classification: PVC Detection with External Validation and Calibrated Abstention*.

A single script reproduces every number, table and figure in the paper from two public PhysioNet databases: the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/), using the inter-patient DS1/DS2 split of de Chazal et al. (2004), and, for external validation, the [St Petersburg INCART 12-lead Arrhythmia Database](https://physionet.org/content/incartdb/1.0.0/).

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
- `fig1.pdf`: Fig. 1
- console log: the same numbers in readable form

## What the script computes

- Three executable measurements per beat: prematurity ratio, pause ratio, peak width
- The fixed two-of-three rule, with thresholds set from DS1 medians
- Logistic regression, crude raw-signal proxy and ablation controls
- The selective-measurement policy
- The evidence-grounding perturbation test
- Split-conformal abstention, with the four-level and six-level scores
- The Normal vs. APB boundary case
- External validation on INCART, with no refitting
- Automatic beat detection (WFDB XQRS) in place of reference annotations
- Abstention on measurement disagreement vs. logistic-regression confidence at equal coverage, with a record-level bootstrap
- The continuous evidence-margin score (reported as a negative result)

All results are deterministic; no language model is called.
