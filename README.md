# Decision Trees for Heart Disease Prediction and Causal Inference

## Overview

This repository implements advanced machine learning techniques for predicting heart disease and estimating treatment effects using decision trees and causal forests. The project demonstrates both classification and causal inference methods applied to the Cleveland Heart Disease dataset.

## Project Structure

```
Decision_Trees/
├── Python/
│   ├── input/          # Python input data files
│   ├── output/         # Python generated visualizations and results
│   └── scripts/        # Python implementation notebooks
├── R/
│   ├── input/          # R input data files
│   ├── output/         # R generated visualizations and results
│   └── scripts/        # R implementation notebooks
└── processed.cleveland.data  # Cleveland Heart Disease dataset
```

## Implementations

### Part 1: Classification Tree for Heart Disease Prediction

**Objective:** Build a classification tree model to predict the presence of heart disease based on patient characteristics.

**Key Results:**
- Dataset: 297 samples (160 without heart disease, 137 with heart disease)
- Model Accuracy: 78.65%
- Sensitivity: 70.83%
- Specificity: 87.80%
- Cross-Validation Accuracy: 80.77%

**Methods:**
- Data cleaning and preprocessing
- Binary classification with decision trees
- Hyperparameter tuning using 4-fold cross-validation
- Optimal alpha (complexity parameter): 0.0001643517
- Model pruning to prevent overfitting
- Confusion matrix analysis

**Visualizations:**
- Decision tree structure (before and after pruning)
- Confusion matrices
- Inaccuracy rate vs. alpha parameter plot

### Part 2: Causal Forest for Treatment Effect Estimation

**Objective:** Estimate heterogeneous treatment effects and identify patient subgroups with different treatment responses.

**Key Results:**
- Treatment distribution: 48.5% treated, 51.5% control
- Mean Conditional Average Treatment Effect (CATE): 26.68
- CATE range: 20.00 to 32.16 (60% variation)
- Treatment effect heterogeneity successfully identified

**Methods:**
- Random Forest for outcome prediction
- Causal Forest for heterogeneous treatment effect estimation
- Conditional Average Treatment Effect (CATE) calculation
- Patient stratification by treatment response (terciles)
- Feature importance analysis

**Visualizations:**
- Representative decision tree (max_depth=2)
- Feature importance bar plots
- Heatmap of patient characteristics by CATE terciles

## Dataset

The Cleveland Heart Disease dataset contains 303 observations with 14 attributes:

**Clinical Features:**
- `age`: Age in years
- `sex`: Sex (1 = male, 0 = female)
- `cp`: Chest pain type (1-4)
- `restbp`: Resting blood pressure (mm Hg)
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl
- `restecg`: Resting electrocardiographic results
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise-induced angina
- `oldpeak`: ST depression induced by exercise
- `slope`: Slope of the peak exercise ST segment
- `ca`: Number of major vessels colored by fluoroscopy
- `thal`: Thalassemia type
- `hd`: Heart disease diagnosis (0 = no disease, 1-4 = disease severity)

## Technical Requirements

### R Environment
- R version 4.5.2
- Required packages:
  - `rpart`: Decision tree modeling
  - `rpart.plot`: Tree visualization
  - `caret`: Model training and evaluation
  - `ggplot2`: Data visualization
  - `dplyr`: Data manipulation
  - `randomForest`: Random forest implementation

### Python Environment
- Python 3.x
- Required packages (see individual notebooks for specific versions)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gsaco/Decision_Trees.git
cd Decision_Trees
```

2. Set up the R environment:
```bash
conda create -n forest r-base -c conda-forge
conda activate forest
conda install -c conda-forge r-rpart r-rpart.plot r-caret r-ggplot2 r-dplyr r-randomforest r-irkernel -y
```

3. Register the R kernel for Jupyter:
```r
R -e "IRkernel::installspec(name = 'forest-r', displayname = 'R (forest)')"
```

## Usage

### Running the Classification Tree Notebook

1. Navigate to the R scripts directory
2. Open `assignment4_part1_classification_tree.ipynb`
3. Select the "R (forest)" kernel
4. Run all cells sequentially

### Running the Causal Forest Notebook

1. Navigate to the R scripts directory
2. Open `assignment4_part2_causal_forest.ipynb`
3. Select the "R (forest)" kernel
4. Run all cells sequentially

## Key Findings

### Classification Results

The classification tree model demonstrates strong performance in predicting heart disease:
- The model achieves 78.65% accuracy on the test set
- High specificity (87.80%) indicates excellent ability to identify heart disease cases
- Cross-validation confirms the model generalizes well to unseen data
- Pruning analysis shows the optimal tree complexity balances bias and variance

### Causal Inference Results

The causal forest analysis reveals significant treatment effect heterogeneity:
- Treatment effects vary by 60% across patient subgroups (CATE range: 20.00 to 32.16)
- Patient characteristics can predict treatment response effectiveness
- Specific patient profiles are associated with substantially higher or lower treatment benefits
- The analysis supports personalized treatment decision-making

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