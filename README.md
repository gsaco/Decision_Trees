<div align="center">

# 🌳 Decision Trees for Heart Disease Prediction and Causal Inference

[![R](https://img.shields.io/badge/R-4.5.2-276DC3?style=for-the-badge&logo=r&logoColor=white)](https://www.r-project.org/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Machine Learning](https://img.shields.io/badge/Machine_Learning-Classification-orange?style=for-the-badge)](https://github.com/gsaco/Decision_Trees)
[![Causal Inference](https://img.shields.io/badge/Causal_Inference-Forest-green?style=for-the-badge)](https://github.com/gsaco/Decision_Trees)

*Advanced machine learning and causal inference techniques for personalized medicine*

[Key Features](#-key-features) •
[Results](#-results) •
[Installation](#-installation) •
[Usage](#-usage) •
[Documentation](#-dataset)

</div>

---

## 📋 Overview

This repository implements state-of-the-art machine learning and causal inference techniques for **predicting heart disease** and **estimating heterogeneous treatment effects**. Using the Cleveland Heart Disease dataset, we demonstrate:

- **Classification Trees** with cross-validation and pruning optimization
- **Causal Forests** for personalized treatment effect estimation
- **Rigorous statistical validation** with interpretable results
- **Dual implementation** in both R and Python for reproducibility


## 📂 Project Structure

```
Decision_Trees/
├── 📁 Python/
│   ├── 📂 input/                    # Python input data files
│   ├── 📂 output/                   # Generated visualizations and results
│   │   ├── classification_tree_before_pruning_Python.png
│   │   ├── classification_tree_after_pruning_Python.png
│   │   ├── confusion_matrix_before_pruning_Python.png
│   │   ├── confusion_matrix_after_pruning_Python.png
│   │   └── inaccuracy_vs_alpha_Python.png
│   └── 📓 scripts/
│       ├── assignment4_part1_classification_tree.ipynb
│       └── assignment4_part2_causal_forest.ipynb
│
├── 📁 R/
│   ├── 📂 input/                    # R input data files
│   ├── 📂 output/                   # Generated visualizations and results
│   │   ├── classification_tree_before_pruning_R.png
│   │   ├── classification_tree_after_pruning_R.png
│   │   ├── confusion_matrix_before_pruning_R.png
│   │   ├── confusion_matrix_after_pruning_R.png
│   │   ├── inaccuracy_vs_alpha_R.png
│   │   ├── representative_tree_R.pdf
│   │   ├── feature_importance_R.png
│   │   └── cate_heatmap_R.png
│   └── 📓 scripts/
│       ├── assignment4_part1_classification_tree.ipynb
│       └── assignment4_part2_causal_forest.ipynb
│
└── 📄 processed.cleveland.data      # Cleveland Heart Disease Dataset
```

---

## 🔬 Methodology

### Part 1: Classification Tree for Heart Disease Prediction

**🎯 Objective:** Build a classification tree model to predict the presence of heart disease based on patient characteristics.

#### Pipeline

```mermaid
graph LR
    A[Raw Data] --> B[Data Cleaning]
    B --> C[Feature Engineering]
    C --> D[Train-Test Split]
    D --> E[Model Training]
    E --> F[Cross-Validation]
    F --> G[Hyperparameter Tuning]
    G --> H[Model Pruning]
    H --> I[Final Evaluation]
```

#### Methods Employed

| Step | Technique | Details |
|------|-----------|---------|
| **Data Preprocessing** | Missing value handling | Removed 6 samples with missing data |
| **Feature Engineering** | Binary target creation | Converted multi-class to binary (disease vs. no disease) |
| **Splitting** | Stratified sampling | 70-30 train-test split (208 train, 89 test) |
| **Model Training** | CART algorithm | Recursive partitioning with Gini impurity |
| **Validation** | 4-fold CV | Tested 50 alpha values (log scale: 4.54e-05 to 0.05) |
| **Optimization** | Pruning | Selected optimal α = 0.0001643517 |
| **Evaluation** | Multiple metrics | Accuracy, sensitivity, specificity, Kappa |

#### Visualizations Generated
- 📊 Decision tree structure (pre/post pruning)
- 📈 Confusion matrices with performance metrics
- 📉 Inaccuracy rate vs. complexity parameter curve

---

### Part 2: Causal Forest for Treatment Effect Estimation

**🎯 Objective:** Estimate heterogeneous treatment effects and identify patient subgroups with different treatment responses.

#### Pipeline

```mermaid
graph LR
    A[Dataset] --> B[Define Treatment]
    B --> C[Create Outcome Variable]
    C --> D[Random Forest]
    D --> E[Causal Forest]
    E --> F[CATE Estimation]
    F --> G[Stratification]
    G --> H[Feature Importance]
    H --> I[Visualization]
```

#### Methods Employed

| Step | Technique | Details |
|------|-----------|---------|
| **Treatment Definition** | Binary assignment | 48.5% treated, 51.5% control |
| **Outcome Variable** | Continuous Y | Range: -1.46 to 45.15 |
| **Base Model** | Random Forest | 500 trees, outcome prediction |
| **Causal Model** | Causal Forest | Honest splitting for unbiased estimates |
| **CATE Calculation** | Individual effects | Predicted for each patient |
| **Stratification** | Tercile split | Low/Medium/High response groups (n=99 each) |
| **Interpretation** | Representative tree | Max depth=2 for interpretability |
| **Feature Analysis** | Importance ranking | Variable contribution to predictions |

#### Visualizations Generated
- 🌲 Representative decision tree (simplified structure)
- 📊 Feature importance bar chart
- 🔥 Heatmap of patient characteristics by CATE terciles

---


---

## 🎯 Key Features

<table>
<tr>
<td width="50%">

### 🔬 Classification Tree Analysis
- ✅ Decision tree implementation with pruning
- ✅ 4-fold cross-validation optimization
- ✅ Comprehensive performance metrics
- ✅ Visual tree structure analysis
- ✅ Confusion matrix evaluation

</td>
<td width="50%">

### 🌲 Causal Forest Analysis
- ✅ Heterogeneous treatment effect estimation
- ✅ CATE calculation and visualization
- ✅ Patient stratification by response
- ✅ Feature importance ranking
- ✅ Subgroup identification

</td>
</tr>
</table>

---

## 📊 Results

### 🏆 Classification Performance

<table>
<tr>
<td align="center"><b>Metric</b></td>
<td align="center"><b>Value</b></td>
<td align="center"><b>95% CI</b></td>
</tr>
<tr>
<td><b>Overall Accuracy</b></td>
<td><code>78.65%</code></td>
<td>68.69% - 86.63%</td>
</tr>
<tr>
<td><b>Sensitivity</b></td>
<td><code>70.83%</code></td>
<td>-</td>
</tr>
<tr>
<td><b>Specificity</b></td>
<td><code>87.80%</code></td>
<td>-</td>
</tr>
<tr>
<td><b>PPV</b></td>
<td><code>87.18%</code></td>
<td>-</td>
</tr>
<tr>
<td><b>Kappa</b></td>
<td><code>0.5771</code></td>
<td>-</td>
</tr>
<tr>
<td><b>CV Accuracy</b></td>
<td><code>80.77%</code></td>
<td>-</td>
</tr>
</table>

**Optimal Complexity Parameter (α):** `0.0001643517`

### 🎯 Causal Inference Results

<table>
<tr>
<td align="center"><b>Statistic</b></td>
<td align="center"><b>Value</b></td>
</tr>
<tr>
<td><b>Mean CATE</b></td>
<td><code>26.68</code></td>
</tr>
<tr>
<td><b>Min CATE</b></td>
<td><code>20.00</code></td>
</tr>
<tr>
<td><b>Max CATE</b></td>
<td><code>32.16</code></td>
</tr>
<tr>
<td><b>CATE Range</b></td>
<td><code>12.16</code></td>
</tr>
<tr>
<td><b>Effect Variation</b></td>
<td><code>~60%</code></td>
</tr>
<tr>
<td><b>Treatment Rate</b></td>
<td><code>48.5%</code></td>
</tr>
</table>

**Key Finding:** Treatment effects vary by 60% across patient subgroups, demonstrating significant heterogeneity and the value of personalized treatment strategies.

---


## 📊 Dataset

### Cleveland Heart Disease Dataset

**Source:** UCI Machine Learning Repository  
**Samples:** 303 observations (297 after cleaning)  
**Target:** Heart disease diagnosis (0 = no disease, 1-4 = disease severity)

#### Feature Description

<details>
<summary><b>📋 Click to expand complete feature list</b></summary>

| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| `age` | Numeric | Patient age in years | 29 - 77 |
| `sex` | Binary | Sex | 0 = female, 1 = male |
| `cp` | Categorical | Chest pain type | 1 = typical angina<br>2 = atypical angina<br>3 = non-anginal pain<br>4 = asymptomatic |
| `restbp` | Numeric | Resting blood pressure | mm Hg (94 - 200) |
| `chol` | Numeric | Serum cholesterol | mg/dl (126 - 564) |
| `fbs` | Binary | Fasting blood sugar > 120 mg/dl | 0 = false, 1 = true |
| `restecg` | Categorical | Resting ECG results | 0 = normal<br>1 = ST-T wave abnormality<br>2 = left ventricular hypertrophy |
| `thalach` | Numeric | Maximum heart rate achieved | bpm (71 - 202) |
| `exang` | Binary | Exercise-induced angina | 0 = no, 1 = yes |
| `oldpeak` | Numeric | ST depression induced by exercise | 0 - 6.2 |
| `slope` | Categorical | Slope of peak exercise ST segment | 1 = upsloping<br>2 = flat<br>3 = downsloping |
| `ca` | Numeric | Number of major vessels | 0 - 3 (colored by fluoroscopy) |
| `thal` | Categorical | Thalassemia | 3 = normal<br>6 = fixed defect<br>7 = reversible defect |
| `hd` | Target | Heart disease diagnosis | 0 = no disease<br>1-4 = disease severity |

</details>

#### Data Quality

- **Missing Values:** 6 observations removed (2% of data)
- **Class Distribution:** 
  - No disease: 160 samples (53.9%)
  - Heart disease: 137 samples (46.1%)
- **Train-Test Split:** 70-30 stratified split
  - Training: 208 samples
  - Testing: 89 samples

---


## 🛠 Technical Requirements

### R Environment

![R Version](https://img.shields.io/badge/R-4.5.2-276DC3?style=flat-square&logo=r)

<details>
<summary><b>Required R Packages</b></summary>

| Package | Version | Purpose |
|---------|---------|---------|
| `rpart` | 4.1.24 | Decision tree modeling (CART algorithm) |
| `rpart.plot` | 3.1.3 | Tree visualization with enhanced graphics |
| `caret` | 7.0-1 | Model training, evaluation, and cross-validation |
| `ggplot2` | 4.0.0 | Publication-quality data visualization |
| `dplyr` | 1.1.4 | Data manipulation and transformation |
| `randomForest` | 4.7-1.2 | Random forest implementation |
| `IRkernel` | 1.3.2 | R kernel for Jupyter notebooks |

</details>

### Python Environment

![Python Version](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python)

*Package details available in individual notebooks*

### Development Environment

- **Jupyter Notebook** for interactive analysis
- **Conda** for environment management
- **Git** for version control

---

## 🚀 Installation

### Option 1: Quick Setup (Conda - Recommended)

```bash
# Clone the repository
git clone https://github.com/gsaco/Decision_Trees.git
cd Decision_Trees

# Create and activate conda environment with R
conda create -n forest r-base=4.5.2 -c conda-forge -y
conda activate forest

# Install all required R packages
conda install -c conda-forge \
  r-rpart \
  r-rpart.plot \
  r-caret \
  r-ggplot2 \
  r-dplyr \
  r-randomforest \
  r-irkernel \
  jupyter -y

# Register R kernel for Jupyter
R -e "IRkernel::installspec(name = 'forest-r', displayname = 'R (forest)')"
```

### Option 2: Manual R Installation

```r
# Install packages in R
install.packages(c(
  "rpart",
  "rpart.plot",
  "caret",
  "ggplot2",
  "dplyr",
  "randomForest",
  "IRkernel"
))

# Register Jupyter kernel
IRkernel::installspec(name = 'forest-r', displayname = 'R (forest)')
```

### Verify Installation

```bash
# Check R version
R --version

# List installed packages
R -e "installed.packages()[c('rpart', 'rpart.plot', 'caret', 'ggplot2', 'dplyr', 'randomForest'), c('Package', 'Version')]"

# Verify Jupyter kernel
jupyter kernelspec list
```

---

## 💻 Usage

### Running Classification Tree Analysis

```bash
# Navigate to the R scripts directory
cd R/scripts

# Launch Jupyter Notebook
jupyter notebook assignment4_part1_classification_tree.ipynb
```

**In Jupyter:**
1. Select kernel: **R (forest)**
2. Run cells sequentially (Cell → Run All)
3. Outputs saved to `R/output/`

**Expected Runtime:** ~2-3 minutes

### Running Causal Forest Analysis

```bash
# Navigate to the R scripts directory
cd R/scripts

# Launch Jupyter Notebook
jupyter notebook assignment4_part2_causal_forest.ipynb
```

**In Jupyter:**
1. Select kernel: **R (forest)**
2. Run cells sequentially (Cell → Run All)
3. Outputs saved to `R/output/`

**Expected Runtime:** ~3-5 minutes

### Output Files

After successful execution, find generated visualizations in:
- `R/output/` - All R-generated plots and analysis results
- `Python/output/` - All Python-generated plots and analysis results

---


## 🔍 Key Findings

### Classification Tree Analysis

#### Performance Summary

Our classification tree model demonstrates **strong predictive performance** for heart disease diagnosis:

```
┌─────────────────────────────────────────────────────────────────┐
│  MODEL PERFORMANCE METRICS                                      │
├─────────────────────────────────────────────────────────────────┤
│  Overall Accuracy:        78.65%  [95% CI: 68.69% - 86.63%]   │
│  Sensitivity (Recall):    70.83%  (True Negative Rate)         │
│  Specificity:             87.80%  (True Positive Rate)         │
│  Positive Predictive Value: 87.18%                             │
│  Balanced Accuracy:       79.32%                               │
│  Kappa Statistic:         0.5771  (Moderate Agreement)         │
├─────────────────────────────────────────────────────────────────┤
│  CROSS-VALIDATION RESULTS                                       │
├─────────────────────────────────────────────────────────────────┤
│  4-Fold CV Accuracy:      80.77%                               │
│  Optimal Alpha (cp):      0.0001643517                         │
│  Alpha Search Range:      4.54e-05 to 0.05 (50 values)        │
└─────────────────────────────────────────────────────────────────┘
```

#### Clinical Insights

- ✅ **High Specificity (87.80%):** Excellent at identifying true heart disease cases
- ✅ **Good Sensitivity (70.83%):** Effective at ruling out disease in healthy patients
- ✅ **Low False Negatives (5):** Minimizes missing actual disease cases
- ✅ **Balanced Performance:** Strong results across both disease and non-disease classes
- ✅ **Generalization:** CV accuracy (80.77%) validates model robustness

#### Model Interpretation

The pruning analysis revealed that the **optimal tree complexity** balances bias and variance:
- Minimal pruning was required (α = 0.0001643517)
- Original tree structure was well-regularized
- Cross-validation confirmed appropriate model complexity
- No evidence of severe overfitting

---

### Causal Forest Analysis

#### Treatment Effect Heterogeneity

Our causal forest identified **significant variation** in treatment effects across patient subgroups:

```
┌─────────────────────────────────────────────────────────────────┐
│  CONDITIONAL AVERAGE TREATMENT EFFECT (CATE) DISTRIBUTION      │
├─────────────────────────────────────────────────────────────────┤
│  Mean CATE:              26.68                                 │
│  Minimum CATE:           20.00  (Low responders)               │
│  Maximum CATE:           32.16  (High responders)              │
│  CATE Range:             12.16  (60% variation)                │
│  Standard Deviation:     ~2.5                                  │
├─────────────────────────────────────────────────────────────────┤
│  TERCILE STRATIFICATION (n=99 each)                            │
├─────────────────────────────────────────────────────────────────┤
│  Low Tercile:            20.00 - 24.79                         │
│  Medium Tercile:         24.79 - 28.68                         │
│  High Tercile:           28.68 - 32.16                         │
├─────────────────────────────────────────────────────────────────┤
│  TREATMENT DISTRIBUTION                                         │
├─────────────────────────────────────────────────────────────────┤
│  Treated:                144 patients (48.5%)                  │
│  Control:                153 patients (51.5%)                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Personalized Medicine Insights

The **60% variation** in treatment effects reveals:

1. **Patient Stratification Matters:** Not all patients benefit equally from treatment
2. **Identifiable Subgroups:** Specific patient characteristics predict treatment response
3. **Clinical Decision Support:** The model can guide personalized treatment allocation
4. **Resource Optimization:** Target treatment to high-responder groups for maximum benefit

#### Feature Importance Findings

The causal forest analysis identified **key predictors** of treatment heterogeneity:
- Patient baseline characteristics significantly influence treatment response
- Certain covariates show strong association with CATE terciles
- Feature importance ranking guides clinical attention to relevant factors

---

## Clinical Implications

1. **Risk Stratification:** The classification model can assist in identifying patients at high risk for heart disease
2. **Personalized Medicine:** The causal forest identifies which patients are most likely to benefit from specific treatments
3. **Resource Optimization:** Treatment can be targeted to patients with the highest expected benefit
4. **Decision Support:** Both models provide interpretable insights for clinical decision-making

## Limitations

- Dataset size is relatively small (297 samples after cleaning)
- Missing values reduced the available sample size
- Treatment assignments in the causal analysis are observational, not randomized
- Results should be validated on larger, independent datasets before clinical application

## Future Work

- Validate models on external datasets
- Incorporate additional clinical features
- Explore ensemble methods for improved prediction
- Conduct sensitivity analysis for causal estimates
- Develop interactive clinical decision support tools

## Contributors

Group 1 - Causal Inference and Machine Learning

## License

This project is for educational purposes.

## Acknowledgments

- Cleveland Clinic Foundation for providing the heart disease dataset
- UCI Machine Learning Repository for hosting the dataset