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

#### Model Interpretation

The pruning analysis revealed that the **optimal tree complexity** balances bias and variance:
- Minimal pruning was required (α = 0.0001643517)
- Original tree structure was well-regularized
- Cross-validation confirmed appropriate model complexity
- No evidence of severe overfitting


---

## 👥 Contributors

**Group 1 - Causal Inference and Machine Learning**

This project represents a collaborative effort in applying advanced statistical methods to personalized medicine.

---

## 📄 License

This project is for **educational and research purposes only**.

⚠️ **Disclaimer:** This software is not intended for clinical use. Do not use for medical diagnosis or treatment decisions without proper validation and regulatory approval.


<div align="center">

### ⭐ If you find this project useful, please consider giving it a star!

Made with ❤️ by Group 1 - Causal Inference and Machine Learning

**[↑ Back to Top](#-decision-trees-for-heart-disease-prediction-and-causal-inference)**

</div>