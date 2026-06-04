# Alpha Defect Detection in Hot Rolling

Binary classification task to identify steel coils with **alpha defects** from hot-rolling process sensor data.

---

## Problem

- **Train:** 1,352 coils · 49 sensor features · 66 defects (4.88%)
- **Test:** 339 coils
- **Challenge:** Severe class imbalance (19.5:1), missing values, no feature names provided

The goal is to flag defective coils with high recall while minimising false alarms.

---

## Repository Structure

```
.
├── alpha_defect_detection.ipynb   ← Main notebook (run this)
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
└── submission.csv                 ← Generated output
```

---

## Approach

### 1. Feature Engineering (49 → 122 features)
- **Temperature progression** (X1–X9): mean, std, range, stage-to-stage drops, coefficient of variation
- **Top-feature transforms** (X10, X13, X30–X36): log, squared, pairwise products and differences
- **Group aggregates**: rolling reduction (X23–X33), speed (X17–X22), vibration (X41–X49)
- **Cross-group interactions**: temp × force, speed × reduction, etc.
- **Percentile ranks**: scale-free position in distribution for top discriminating features
- **Global stats**: per-row mean, std, range, skewness

### 2. Anomaly Detection Features
`IsolationForest` and `LocalOutlierFactor` scores appended as 2 extra features — fully unsupervised, no label leakage.

### 3. Hyperparameter Tuning
60-trial **Optuna** search maximising **OOF Average Precision** (5-fold stratified CV). This ensures no data leakage during tuning.

### 4. Four-Model Ensemble
| Model | Weight |
|---|---|
| XGBoost | 35% |
| LightGBM | 35% |
| Random Forest | 15% |
| Extra Trees | 15% |

Blended via **rank normalisation** (more robust than raw probability averaging for imbalanced data).

### 5. SMOTE Inside Each Fold
1:1 oversampling applied only on the training portion of each fold — prevents synthetic samples from contaminating the validation set.

### 6. Top-K Selection
Instead of a probability threshold (which breaks with poorly calibrated minority-class scores), the **top-K ranked** test coils are flagged, where K ≈ expected defect count.

---

## OOF Results (no data leakage)

| Model | AUC | Average Precision |
|---|---|---|
| XGBoost | 0.8693 | 0.3117 |
| LightGBM | 0.8703 | 0.2918 |
| Random Forest | 0.8645 | 0.2920 |
| Extra Trees | 0.8585 | 0.2995 |
| **Blend** | **0.8765** | **0.3159** |

---

## How to Run

### Requirements
```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn optuna
```

### Execute
Open and run **`alpha_defect_detection.ipynb`** cell by cell (or Run All).

Output: `submission.csv` with columns `CoilID, Y` (1 = defect, 0 = no defect).

> **Note:** The Optuna search (60 trials × 5-fold CV) takes ~8–10 minutes on a standard laptop.

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| OOF for threshold/tuning | Avoids data leakage; training AUC=1.0 in naive eval is a red flag |
| Rank-based blend | Raw probabilities poorly calibrated with SMOTE + heavy class imbalance |
| Top-K over threshold | Threshold selection is unreliable when positive class is <5% of data |
| `scale_pos_weight` ≈ 5–10 | Optuna found lower values work better than the naive ratio of 19.5 |
| Anomaly scores as features | Adds unsupervised signal without any risk of label leakage |
