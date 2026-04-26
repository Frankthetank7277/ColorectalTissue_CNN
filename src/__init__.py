"""BIOE 486 Final Project — Colorectal cancer tissue classification via transfer learning.

Source package containing all reusable modules for the tissue classification pipeline:
configuration, dataset construction, model definition, training, and evaluation.

Datasets used (Kather et al., 2019):
  - NCT-CRC-HE-100K — 100,000 H&E patches across 9 tissue classes (train + val)
  - CRC-VAL-HE-7K  — 7,180 H&E patches, held-out test set

Author: Frank Lato
"""