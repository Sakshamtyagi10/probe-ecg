# probe-ecg

Reproduction code for the paper *Executable Evidence for Sensor Reasoning: An ECG Case Study in Auditable Measurement and Calibrated Abstention*.

A single script reproduces every number, table and figure in the paper from the public [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) (PhysioNet), using the inter-patient DS1/DS2 split of de Chazal et al. (2004).

## Run

Python 3.9+ (tested on Google Colab):

```
pip install wfdb scipy scikit-learn matplotlib numpy
python probe_reproduce.py
```

The script downloads MIT-BIH automatically. A full run takes under 20 minutes on a free Colab instance, mostly for the download.

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

All results are deterministic; no language model is called.
