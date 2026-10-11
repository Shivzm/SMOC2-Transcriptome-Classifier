# Model Tuning and Validation Log

This document records how the classifier in `notebooks/01_eda_and_fibrosis_classification.ipynb` was tuned and checked: what went wrong at first, why, what was changed, and how much the final result can be trusted. The README holds the final pipeline and a short summary. This file holds the reasoning behind it.

## Contents

1. [Setup](#1-setup)
2. [Experiment 1: C = 0.1](#2-experiment-1-c--01)
3. [Why the first model selected nothing](#3-why-the-first-model-selected-nothing)
4. [Why the accuracy was exactly 42.9%](#4-why-the-accuracy-was-exactly-429)
5. [Experiment 2: C = 1.0](#5-experiment-2-c--10)
6. [Why 100% should not be taken at face value](#6-why-100-should-not-be-taken-at-face-value)
7. [Validation checks](#7-validation-checks)
8. [Conclusions](#8-conclusions)
9. [Next steps](#9-next-steps)

## 1. Setup

- **Data:** `df_ml`, a matrix of 7 samples by 29,556 genes (log2 CPM + 1), with labels `y` (4 fibrosis, 3 normal).
- **Model:** logistic regression with an L1 penalty, `solver='liblinear'`.
- **Evaluation:** leave-one-out cross-validation, with `StandardScaler` fitted inside each fold.

The only setting changed between the two experiments is `C`, the inverse regularization strength.

## 2. Experiment 1: C = 0.1

The first run used a strong penalty, on the reasoning that a small `C` would force the model to pick only a handful of genes.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# A small C value (like 0.1) forces strong regularization to select fewer genes.
model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)

loo = LeaveOneOut()
y_true, y_pred = [], []

for train_idx, test_idx in loo.split(df_ml):
    X_train, X_test = df_ml.iloc[train_idx], df_ml.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scale within cross-validation to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    y_pred.append(model.predict(X_test_scaled)[0])
    y_true.append(y_test[0])

cv_accuracy = accuracy_score(y_true, y_pred)
print(f"Leave-One-Out Cross-Validation Accuracy: {cv_accuracy * 100:.1f}%")

# Fit on the full dataset to see which genes the model selected globally
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(df_ml)
model.fit(X_full_scaled, y)

coefficients = model.coef_[0]
selected_indices = np.where(coefficients != 0)[0]
selected_genes = df_ml.columns[selected_indices]
selected_weights = coefficients[selected_indices]

signature_df = pd.DataFrame({
    'Gene_Ensembl_ID': selected_genes,
    'Model_Weight': selected_weights
}).sort_values(by='Model_Weight', key=abs, ascending=False)

print(f"\nThe model selected {len(signature_df)} top predictive genes:")
print(signature_df.to_string(index=False))
```

Output:

```text
Leave-One-Out Cross-Validation Accuracy: 42.9%

The model selected 0 top predictive genes:
Empty DataFrame
Columns: [Gene_Ensembl_ID, Model_Weight]
Index: []
```

The model chose no genes, and the accuracy was 42.9%, below what a coin flip would give. Two questions follow: why zero genes, and why exactly 42.9%.

## 3. Why the first model selected nothing

This was not an unlucky result or a problem with the data. With these settings, no gene could ever have been selected.

For `liblinear`, the quantity being minimized is

```text
(L1 penalty on the weights)  +  C x (logistic loss over the samples)
```

The L1 penalty pulls every weight toward zero with a constant strength of 1. The loss term pulls a weight away from zero with a strength that depends on the data and is multiplied by `C`. A weight only becomes non-zero if the pull from the data beats the pull from the penalty.

At the starting point, where all weights are zero, every sample has a predicted probability of 0.5. The pull on gene *j* is then

```text
C x | sum over samples of (y_i - 0.5) * x_ij |
```

and the gene can enter the model only if this exceeds 1, that is, if

```text
| sum of (y_i - 0.5) * x_ij |  >  1 / C
```

The left-hand side has a hard ceiling. After standardization each gene has mean 0 and variance 1 across the 7 samples, so the squared values sum to 7. The residuals `y_i - 0.5` are all plus or minus 0.5. By the Cauchy-Schwarz inequality:

```text
| sum of (y_i - 0.5) * x_ij |  <=  sqrt(7 * 0.25) * sqrt(7)  =  3.5
```

So the strongest possible gene, one that tracks the labels perfectly, can reach a pull of at most 3.5. With `C = 0.1` the threshold is `1 / C = 10`. Since 3.5 is far below 10, every coefficient stays at zero no matter how informative the gene is.

In general, any `C` below about `1 / 3.5 ≈ 0.29` gives an empty model on a dataset of this size. Inside a LOOCV fold there are only 6 training samples, so the ceiling drops to 3 and the practical limit is about 0.33.

I confirmed this on synthetic data with the same shape (7 samples, 29,000 features, with real signal planted in 50 genes). `C = 0.1` and `C = 0.2` selected zero genes and scored 42.9%, and genes only started to appear at about `C = 0.3`.

The lesson is that the useful range of `C` depends on the number of samples. Values that work on a dataset of thousands of rows can be impossibly strict on a dataset of seven.

## 4. Why the accuracy was exactly 42.9%

42.9% is 3 out of 7. When every weight is zero, and the intercept is also zero because `liblinear` penalizes it too, the model's score for every sample is exactly 0. scikit-learn assigns class 0 to a score that is not strictly positive, so the model predicts Normal for every sample in every fold.

The dataset has 3 Normal samples, so predicting Normal each time is correct 3 times out of 7, which is 42.9%. The number is the fraction of Normal samples in the dataset. It says nothing about whether the genes can predict fibrosis, because the model never looked at any gene.

## 5. Experiment 2: C = 1.0

With the cause understood, the fix was to relax the penalty. At `C = 1.0` the threshold is `1 / C = 1`, comfortably below the ceiling of 3.5, so genes can enter the model. The code is identical to Experiment 1 except for one line:

```python
model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
```

Output (the middle of the list is truncated here exactly as it was in the notebook):

```text
Leave-One-Out Cross-Validation Accuracy: 100.0%

The model selected 91 predictive genes:
   Gene_Ensembl_ID  Model_Weight
ENSMUSG00000030255      1.116470
ENSMUSG00000068739      0.092687
ENSMUSG00000031422      0.081391
ENSMUSG00000055737     -0.069485
ENSMUSG00000059208      0.031942
ENSMUSG00000074916      0.024387
ENSMUSG00000028274      0.020418
ENSMUSG00000043289      0.019496
ENSMUSG00000020051     -0.017822
ENSMUSG00000087179      0.016791
ENSMUSG00000035529      0.016532
ENSMUSG00000038429      0.016081
ENSMUSG00000022142      0.015957
ENSMUSG00000006498      0.015372
ENSMUSG00000006342     -0.015352
ENSMUSG00000027510      0.014373
ENSMUSG00000026639      0.014333
ENSMUSG00000035683      0.013271
ENSMUSG00000000976      0.012076
ENSMUSG00000021359      0.010070
ENSMUSG00000024056      0.009769
...
ENSMUSG00000024535      0.000373
ENSMUSG00000022329      0.000252
ENSMUSG00000053007      0.000157
ENSMUSG00000022500      0.000028
```

The model now selects genes, and every held-out sample is classified correctly.

## 6. Why 100% should not be taken at face value

Reaching 100% was a relief after 42.9%, but a few features of this result call for caution.

**One gene carries the model.** The top weight is 1.12, the second is 0.09, and the weights drop to about 0.0001 further down the list. Since the inputs were standardized, the weights are directly comparable. This is effectively a one-gene classifier with 90 near-zero additions.

**The metric is very coarse.** With 7 samples, LOOCV accuracy can only be 0, 14, 29, 43, 57, 71, 86, or 100%. One sample is worth 14 points.

**Perfect separation is cheap with this many genes.** If a gene has no real relationship with the condition, the chance that it still splits a 3 vs 4 grouping perfectly is 2 in 35, about 5.7%. Applied to 29,556 genes, that is on the order of 1,700 genes expected to separate the groups perfectly with no real signal at all. So separating the groups perfectly is not what makes a gene interesting. What matters is the size of the difference compared with the spread within each group, and whether the difference survives a proper multiple-testing correction.

**`C` was chosen by looking at the results.** Moving from 0.1 to 1.0 was a justified correction of a mistake, but it was still tuning on the same 7 samples that produce the accuracy. There are too few samples to do nested cross-validation.

The following checks were run to find out how much of the result holds up.

## 7. Validation checks

### 7.1 Gene stability across folds

If the signature is real, the same genes should be chosen whichever sample is left out. This check keeps the genes that were selected in every fold.

```python
model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=1000, random_state=42)

picked = []
for tr, te in LeaveOneOut().split(df_ml):
    sc = StandardScaler().fit(df_ml.iloc[tr])
    model.fit(sc.transform(df_ml.iloc[tr]), y[tr])
    picked.append(set(np.where(model.coef_[0] != 0)[0]))

always = set.intersection(*picked)
print(f"Genes selected in all {len(picked)} folds: {len(always)}")
print(df_ml.columns[sorted(always)].tolist())
```

Output:

```text
Genes selected in all 7 folds: 1
```

Only one gene was selected in every fold. This agrees with the weights above: one gene is robust, and the other 90 in the full-data signature appear or disappear depending on which sample is held out. The 91-gene list should not be reported as a signature. The stable result is the single dominant gene.

### 7.2 Exact permutation test

With 7 samples there are only 35 ways to assign 4 samples to the fibrosis group, so every possible labeling can be tested. For each one, the whole LOOCV procedure is repeated. The question is how often scrambled labels reach the accuracy that the real labels achieved.

```python
from itertools import combinations

n, k = len(y), int(y.sum())
model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=1000)

def loo_acc(labels):
    correct = 0
    for tr, te in LeaveOneOut().split(df_ml):
        sc = StandardScaler().fit(df_ml.iloc[tr])
        model.fit(sc.transform(df_ml.iloc[tr]), labels[tr])
        correct += model.predict(sc.transform(df_ml.iloc[te]))[0] == labels[te][0]
    return correct / n

accs = []
for pos in combinations(range(n), k):
    perm = np.zeros(n, dtype=int)
    perm[list(pos)] = 1
    accs.append(loo_acc(perm))

print(f"Real labels: {loo_acc(y):.1%}")
print(f"Share of all {len(accs)} labelings reaching >= real accuracy: "
      f"{np.mean(np.array(accs) >= loo_acc(y)):.1%}")
```

Output:

```text
Real labels: 100.0%
Share of all 35 labelings reaching >= real accuracy: 2.9%
```

The real labeling is one of the 35, so the smallest possible share is 1/35, about 2.9%. The result is exactly that minimum: only the real labeling reaches 100%, and none of the 34 scrambled labelings do. This supports the result, and it is the more relevant check for the chance-separation worry in section 6, because it runs the whole pipeline on the real, correlated gene data instead of assuming genes behave independently.

Two limits remain. An exact p-value of 1/35 ≈ 0.029 is the smallest this sample size can give, so it passes the usual 0.05 threshold but not by a wide margin. And the test shows that the labels are associated with expression in these 7 samples. It does not rule out a batch or other confound shared with the condition, and it says nothing about new samples.

### 7.3 Biological sanity check: the top gene and SMOC2

SMOC2 is reported to be among the most upregulated genes in mouse kidney fibrosis, so it works as a positive control for the data and labels. This cell checks it directly and compares it with the model's top gene.

```python
# SMOC2 check (mouse Smoc2 Ensembl ID)
smoc2_id = 'ENSMUSG00000023886'
top_gene = signature_df.iloc[0]['Gene_Ensembl_ID']

print("SMOC2 in matrix:", smoc2_id in df_ml.columns)
print("SMOC2 in selected signature:", smoc2_id in set(signature_df['Gene_Ensembl_ID']))
print(signature_df[signature_df['Gene_Ensembl_ID'] == smoc2_id])

print("\nSMOC2 log2(CPM+1) by condition:")
print(df_ml[smoc2_id].groupby(y).describe())
print("log2FC (fibrosis - normal):", df_ml.loc[y == 1, smoc2_id].mean() - df_ml.loc[y == 0, smoc2_id].mean())
print("Correlation of SMOC2 with top gene:", np.corrcoef(df_ml[smoc2_id], df_ml[top_gene])[0, 1])

print("\nTop gene:", top_gene)
print(df_ml[top_gene].groupby(y).describe())
```

Output:

```text
SMOC2 in matrix: True
SMOC2 in selected signature: False
Empty DataFrame
Columns: [Gene_Ensembl_ID, Model_Weight]
Index: []

SMOC2 log2(CPM+1) by condition:
   count      mean       std       min       25%       50%       75%       max
0    3.0  4.113374  0.749717  3.248205  3.883982  4.519759  4.545959  4.572158
1    4.0  6.781576  0.263862  6.615447  6.641346  6.667923  6.808153  7.175013
log2FC (fibrosis - normal): 2.6682022831194185
Correlation of SMOC2 with top gene: 0.9747117656115839

Top gene: ENSMUSG00000030255
   count      mean       std       min       25%       50%       75%       max
0    3.0  2.047019  0.322258  1.682037  1.924383  2.166730  2.229510  2.292291
1    4.0  3.738266  0.181153  3.496952  3.664357  3.766141  3.840049  3.923828
```

An earlier draft of this check used `ENSMUSG00000024037`, which is not the mouse SMOC2 ID, so it was checking the wrong gene. The output above uses the correct ID, `ENSMUSG00000023886`.

**The top gene separates the groups cleanly.** Group 0 (Normal) ranges from 1.68 to 2.29 and group 1 (Fibrosis) from 3.50 to 3.92, so the ranges do not overlap. The mean difference is about 1.69 log2 units, roughly a 3.2-fold change, and the spread within each group is small. A rough two-sample t-test on these summary values gives t of about 9 and p of about 0.0003 before any correction. That is a strong effect, but not enough alone: across 29,556 genes, about nine genes would be expected to reach p = 0.0003 by chance, so the number needs a proper multiple-testing correction.

**The top gene is Sspn (sarcospan).** `ENSMUSG00000030255` maps to the protein-coding gene Sspn (MGI:1353511) on mouse chromosome 6. Sarcospan is a component of the dystrophin-glycoprotein complex, expressed in skeletal and cardiac muscle, where it helps link the cell membrane to the extracellular matrix. In published mouse work, loss of Sspn increases heart fibrosis after beta-adrenergic stress, which suggests a protective role. Here Sspn is higher in fibrosis, so the direction is not what a protective role would predict. Possible explanations include a compensatory response, differences in cell composition between the samples (for example more smooth-muscle-like or myofibroblast-like cells), a batch effect, or chance with n = 7. None of these is tested here, and no source found so far links Sspn to kidney fibrosis, so Sspn is a lead to verify and not an established marker.

**SMOC2 behaves as expected.** SMOC2 is in the matrix and is clearly higher in fibrosis: mean 4.11 in Normal against 6.78 in Fibrosis, a log2 fold change of 2.67 (about 6.4-fold). The ranges do not overlap (Normal 3.25 to 4.57, Fibrosis 6.62 to 7.18). The fold change is smaller than the 10 to 60-fold increases quoted in the dataset description, but those figures come from different mouse models and measurements, so the comparison is loose. What matters is that the direction is right and the groups separate cleanly, which supports the labels and the preprocessing.

**Why SMOC2 was not selected.** SMOC2 separates the groups as well as Sspn, and the two are highly correlated across the 7 samples (r = 0.975). L1 keeps one member of a correlated group and sets the others to zero, and with Sspn slightly tighter within groups (smaller spread, no outlier) it was kept. Not being selected therefore says nothing against SMOC2. One caution: with only 7 samples, any two genes that both separate the groups will correlate strongly, because the group difference drives the correlation. The r of 0.975 does not show that the two genes are co-regulated, and within-group correlation cannot be assessed with 3 and 4 samples.

## 8. Conclusions

What the evidence supports:

- The first model failed for a mathematical reason, not a biological one: with 7 samples and standardized features, `C = 0.1` is too strict for any gene to enter the model, and the 42.9% was the share of Normal samples.
- At `C = 1.0` the model finds one gene, `ENSMUSG00000030255`, that is selected in every fold and separates the two groups with no overlap.
- The exact permutation test over all 35 labelings gives p ≈ 0.029 (1/35): only the real labeling reaches 100% LOOCV accuracy.
- SMOC2, a gene reported to be strongly upregulated in mouse kidney fibrosis, is about 6.4-fold higher in the fibrosis samples with no overlap between groups, which supports the labels and preprocessing.
- The remaining 90 genes in the 91-gene signature are unstable and should not be interpreted.

What the evidence does not support:

- That the classifier would reach 100% accuracy on new samples. It has not been tested on any.
- That the top gene is a validated biomarker. It is Sspn (sarcospan), a muscle membrane-adhesion gene with no established link to kidney fibrosis found so far. Its significance has not yet been checked against a proper differential expression analysis with multiple-testing correction.
- That SMOC2 and Sspn are co-regulated. Their high correlation (r = 0.975) is expected for any two genes that both separate the groups in 7 samples.

## 9. Next steps

1. **Differential expression on raw counts** with DESeq2, edgeR, or `pydeseq2`, which model count data and give adjusted p-values. Check where `ENSMUSG00000030255` ranks and whether its adjusted p-value and log2 fold change hold up.
2. **Follow up on Sspn.** The top gene is identified (sarcospan), so the next step is to check what is known about it in kidney tissue and fibrosis, and whether it also ranks highly in a differential expression analysis. Treat it as a lead until then.
3. **Reduce the feature space** before modeling, either by keeping the most variable genes or genes that pass a differential expression filter, or try an elastic net penalty (`penalty='elasticnet'`, `solver='saga'`), which copes better with correlated genes than pure L1.
4. **Find an external dataset** with a comparable fibrosis versus normal design to use as a held-out test. This is the only way to estimate how the model would perform on new samples.
