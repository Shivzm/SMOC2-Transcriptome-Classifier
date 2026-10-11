# SMOC-Transcriptome-Classifier

## Project Overview

SMOC-Transcriptome-Classifier is an exploratory machine learning pipeline for bulk RNA-sequencing (RNA-seq) data. It asks a focused question: can a small set of genes, selected automatically, separate fibrotic tissue from normal tissue? The work is centered on Secreted Modular Calcium-Binding Protein 2 (SMOC2), a gene widely studied as a driver of fibrosis.

The project takes raw read counts, normalizes them, and fits an L1-regularized logistic regression model that selects a sparse gene signature. The model is evaluated with leave-one-out cross-validation (LOOCV).

The dataset contains only seven samples. The results below should be read as a proof of concept and a source of candidate genes, not as a validated diagnostic tool. The limitations section explains why.

## Biological Context and Dataset

The dataset holds raw RNA-seq read counts for 47,729 genomic features across 7 tissue samples: 4 fibrosis and 3 normal controls. Gene identifiers are mouse Ensembl IDs (`ENSMUSG...`). The data come from a mouse kidney fibrosis study of SMOC2 (Gerarduzzi et al., 2017; dataset credit: Mary Piper, Bioinformatics Consultant and Trainer). The mouse SMOC2 Ensembl ID is `ENSMUSG00000023886`.

SMOC2 is heavily studied as a trigger for the transition of fibroblasts into myofibroblasts, which drives fibrosis in organs such as the kidney, liver, and lung. Raw sequencing counts also vary with sequencing depth and carry a lot of technical noise, so they cannot be passed to a machine learning model directly.

## Data Preprocessing Pipeline

1. **Low-count filtering.** Genes with zero expression in all 7 samples were removed. This dropped nearly 18,000 silent features and left 29,556 genes.

2. **Library size normalization.** Counts Per Million (CPM) corrects for differences in sequencing depth between samples:

   $$CPM = \left( \frac{\text{raw count}}{\text{total reads in sample}} \right) \times 10^6$$

3. **Log transformation.** A log2 transform stabilizes variance so that highly and lowly expressed genes are on a comparable scale:

   $$\text{Expression} = \log_2(CPM + 1)$$

4. **Matrix transposition.** The matrix was transposed so that samples are rows and genes are columns, as scikit-learn expects. The final feature matrix has shape `(7, 29556)`. The target vector comes from the sample names: `0` for Normal, `1` for Fibrosis.

## Exploratory Data Analysis

With 29,556 features and 7 samples, individual scatter plots are not practical. Two checks were used instead.

- **SMOC2 expression.** A boxplot of SMOC2 across the two conditions.
- **Principal Component Analysis (PCA).** Features were centered and scaled to unit variance with `StandardScaler`, then reduced to two components. The aim was to see whether the samples separate by condition before any model is fitted.

**SMOC2 expression:**

![SMOC2 Expression Level](<images/SMOC2 Expression Level.png>)

**PCA clustering:**

![PCA of Fibrosis vs Normal Tissue Samples](<images/Fibrosis vs Normal tissue.png>)

## Machine Learning Approach

### The p >> n problem

There are far more features than samples (29,556 genes against 7 samples). In this setting many different genes can separate the two groups perfectly by chance, so an unconstrained model will fit the training data and generalize poorly. The pipeline therefore uses a sparse model.

### Model choice

The classifier is logistic regression with an L1 (Lasso) penalty.

- **L1 regularization** pushes the weights of uninformative genes to exactly zero, so the genes with non-zero weights form the signature.
- **The `liblinear` solver** supports the L1 penalty and works well on small datasets.

The optimization objective, with labels $y_i \in \{-1, +1\}$, is:

$$\min_{w, c} \; \lVert w \rVert_1 + C \sum_{i=1}^{n} \log\left(1 + \exp\left(-y_i (X_i^T w + c)\right)\right)$$

Here $\lVert w \rVert_1$ is the L1 norm of the weights and $C$ is the inverse regularization strength. A larger $C$ means a weaker penalty and more genes with non-zero weights. A smaller $C$ means a stronger penalty and fewer genes.

### Validation strategy

With 7 samples a conventional train/test split is not workable, so Leave-One-Out Cross-Validation was used. The model is trained on 6 samples and tested on the remaining one, and this is repeated 7 times. Scaling is fitted inside each fold, on the training samples only, to avoid data leakage.

## Model Training and Results

Two values of `C` were tried. The first (`C=0.1`) selected no genes at all. The second (`C=1.0`) is the model reported here. The full story, including the mathematics behind the failure of the first attempt, is in [docs/model_tuning_and_validation.md](docs/model_tuning_and_validation.md).

| C   | Genes selected | LOOCV accuracy | Comment                                          |
|-----|----------------|----------------|--------------------------------------------------|
| 0.1 | 0              | 42.9%          | Penalty too strong. The model predicts Normal for every sample. |
| 1.0 | 91             | 100.0%         | Sparse signature, dominated by one gene.         |

### Final model code (C = 1.0)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# L1-regularized logistic regression. C=1.0 relaxes the penalty enough
# for genes to enter the model (C=0.1 selected none, see the docs).
model = LogisticRegression(l1_ratio=1, solver='liblinear', C=1.0, max_iter=1000, random_state=42)

# Leave-One-Out cross-validation
loo = LeaveOneOut()
y_true, y_pred = [], []

for train_idx, test_idx in loo.split(df_ml):
    X_train, X_test = df_ml.iloc[train_idx], df_ml.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scale inside the fold to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    y_pred.append(model.predict(X_test_scaled)[0])
    y_true.append(y_test[0])

cv_accuracy = accuracy_score(y_true, y_pred)
print(f"Leave-One-Out Cross-Validation Accuracy: {cv_accuracy * 100:.1f}%")

# Fit on the full dataset to see which genes the model selects globally
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(df_ml)
model.fit(X_full_scaled, y)

coefficients = model.coef_[0]
selected_indices = np.where(coefficients != 0)[0]

signature_df = pd.DataFrame({
    'Gene_Ensembl_ID': df_ml.columns[selected_indices],
    'Model_Weight': coefficients[selected_indices]
}).sort_values(by='Model_Weight', key=abs, ascending=False)

print(f"\nThe model selected {len(signature_df)} predictive genes:")
print(signature_df.to_string(index=False))
```

### Output

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

### How to read this result

A cross-validation accuracy of 100% looks impressive, but with 7 samples it is much weaker evidence than it appears. Each sample is worth about 14 percentage points, and with tens of thousands of genes available, a gene that separates the groups perfectly is easy to find by chance. The signature itself tells a more modest story. One gene (`ENSMUSG00000030255`) has a weight of 1.12, the second has 0.09, and the remaining weights fall off to values close to zero. In practice the model is built around one gene, and the other 90 contribute very little.

## Validation Checks

Because the headline number is fragile, follow-up checks were run on the model. The code for each is in [docs/model_tuning_and_validation.md](docs/model_tuning_and_validation.md).

**Gene stability across folds.** Only one gene was selected in all 7 LOOCV folds. The 91-gene list from the full-data fit is therefore not a stable signature. Most of it changes from fold to fold.

**Exact permutation test.** With 7 samples there are only 35 ways to assign 4 samples to the fibrosis group, so every labeling was tested with the full LOOCV pipeline. With `random_state=42`, 2 of the 35 labelings (the real one and one scrambled one) reached 100%, giving an exact p of 2/35 ≈ 0.057. An earlier run without a seed gave 1/35 ≈ 0.029, so the result moves by one labeling with the solver's random seed. At this sample size 1/35 is the smallest possible p-value, so the test sits right at the 0.05 line. It therefore does not clearly rule out that the 100% is a chance effect of having far more genes than samples. It also does not rule out a batch effect, and it says nothing about new samples.

**Stability selection.** To see how consistently genes are chosen when the data change, the model was refit on all 21 five-sample subsets at three values of `C` (0.5, 1.0, 2.0), 63 fits in total. `ENSMUSG00000030255` was selected in 87% of fits (55 of 63). The next most frequent genes, `ENSMUSG00000027510` and `ENSMUSG00000028885`, were selected in 37% and 35%, and every other gene was at or below 22%. SMOC2 was selected in 1.6% of fits (1 of 63). This is descriptive only: the subsets overlap heavily, and the formal error guarantees of stability selection need far more than 7 samples. The result agrees with the fold-stability check: one gene is robust and the rest of the 91-gene list is not.

**Behavior of the top gene.** `ENSMUSG00000030255` (Sspn, sarcospan, a muscle membrane-adhesion gene) separates the groups with no overlap (values are log2 CPM + 1):

| Group    | n | Mean  | Std. dev. | Min   | Max   |
|----------|---|-------|-----------|-------|-------|
| Normal   | 3 | 2.047 | 0.322     | 1.682 | 2.292 |
| Fibrosis | 4 | 3.738 | 0.181     | 3.497 | 3.924 |

The gap of about 1.7 log2 units corresponds to roughly a 3-fold difference, and the within-group spread is small.

**SMOC2.** SMOC2 (`ENSMUSG00000023886`) is clearly higher in fibrosis: mean log2(CPM + 1) of 4.11 in Normal against 6.78 in Fibrosis, a log2 fold change of 2.67 (about 6.4-fold), with no overlap between groups. This supports the labels and preprocessing, since SMOC2 is reported to be strongly upregulated in mouse kidney fibrosis. It was not selected by the model because it is highly correlated with the top gene (r = 0.975 across the 7 samples) and L1 keeps one gene from a correlated group. Not being selected says nothing against SMOC2. (An earlier version of the notebook used the wrong Ensembl ID for SMOC2, `ENSMUSG00000024037`.)

## Limitations

- **Sample size.** Seven samples (4 vs 3) cannot support a reliable classifier. LOOCV accuracy can only take eight values, and one sample changes it by 14 points.
- **Chance separation.** Under no real association, a single gene still splits a 3 vs 4 grouping perfectly with probability 2/35 (about 6%). Across roughly 29,500 genes, well over a thousand genes would be expected to do so by chance alone. A perfect split is not, on its own, evidence of a real effect.
- **Selection instability.** L1 chooses somewhat arbitrarily among many genes that separate the classes. Only one gene survives every LOOCV fold, and in stability selection only one gene is selected in more than 40% of subset fits.
- **Solver randomness.** The `liblinear` L1 solver uses a random coordinate order. With 7 samples, a figure such as the permutation share can shift by a whole labeling between seeds, so seeded values should be read as one draw, not an exact constant.
- **No independent test.** The model has never been evaluated on data it was not built from, so the LOOCV figure should not be quoted as expected real-world accuracy.
- **Tuning on the evaluation data.** Choosing `C` using the same 7 samples that are used for LOOCV is optimistic. There are too few samples for nested cross-validation.

## Next Steps

1. Run differential expression on the raw counts with DESeq2, edgeR, or `pydeseq2`. These methods model count data properly and give adjusted p-values. Check whether `ENSMUSG00000030255` also ranks near the top there.
2. Follow up on the top gene, Sspn (sarcospan), and on SMOC2: check what is known about them in kidney tissue and fibrosis, whether they rank highly in differential expression, and, if they hold up, validate by qPCR on independent animals.
3. Try pre-filtering to the most variable genes, or an elastic net penalty, which handles correlated genes better than pure L1.
4. Look for an external fibrosis dataset to use as a real held-out test.

## Technology Stack

- **Environment management:** `uv` for fast dependency resolution and virtual environments, producing `pyproject.toml` and `uv.lock` for reproducible installs.
- **Data handling:** `pandas` and `numpy` for matrix operations and filtering.
- **Modeling and metrics:** `scikit-learn` for PCA, scaling, logistic regression, and cross-validation.
- **Visualization:** `matplotlib` and `seaborn` for the PCA plot and SMOC2 boxplot.

## Project Structure

```text
SMOC2-Transcriptome-Classifier/
├── Data/
│   └── fibrosis_smoc2_rawcounts.csv
├── docs/
│   └── model_tuning_and_validation.md
├── images/
├── notebooks/
│   └── 01_eda_and_fibrosis_classification.ipynb
├── .gitignore
├── .python-version
├── main.py
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

## Installation and Execution

1. Clone the repository and move into the root directory.
2. Make sure `uv` is installed.
3. Create the environment and install the dependencies:

```bash
uv venv
source .venv/bin/activate
uv add -r requirements.txt
```

4. Launch Jupyter and run the notebook:

```bash
jupyter notebook
```
