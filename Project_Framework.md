# Project: Improving Healthcare Accessibility using ANN

## 1. Data Overview

This section outlines the dataset used for the project.

### Dataset Description
The dataset contains multiple independent variables related to transportation and sociodemographic factors, which are used to predict healthcare accessibility. The key variables are:

- **Dependent Variable**: `Q_19` (Healthcare Accessibility Assessment)
- **Independent Variables**:
  - `cost`: Associated costs of transportation.
  - `ability`: Physical ability to access healthcare services.
  - `safety`: Perception of safety during transit.
  - `ln_population_density`: Log of population density.
  - `ln_employment_entropy`: Log of employment entropy, representing job variety.
  - `ln_network_density`: Log of transportation network density.
  - `county_cate`: Categorical variable representing the type of county (urban, suburban, rural).
  - **Demographic Variables**:
    - `Female`: Gender (binary).
    - `Black`: Race (binary).
    - `low_income`: Income level (binary).
    - `no_vehicle`: Access to a vehicle (binary).


The dataset will be split into training (80%) and testing (20%) sets for model evaluation.

---

## 2. Experiment Setup

### Model Architecture

A simple **Artificial Neural Network (ANN)** is built to predict healthcare accessibility (`Q_19`). The model architecture consists of:
- **Input Layer**: Matching the number of independent variables after preprocessing.
- **Hidden Layers**: Two fully connected layers:
  - Layer 1: 64 neurons with ReLU activation.
  - Layer 2: 32 neurons with ReLU activation.
- **Output Layer**: Single neuron for predicting the healthcare accessibility score.

### Model Training

The model is compiled with the following parameters:
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Training Configuration**: 
  - 50 epochs
  - Batch size of 32
  - 20% validation split during training for monitoring performance.

### Model Evaluation
- **Evaluation Metrics**: Evaluate your model on the test set using metrics like RMSE (Root Mean Square Error) or MAE (Mean Absolute Error) for regression tasks. If your task is categorical (binary or multi-class classification), use accuracy, precision, recall, F1-score, and ROC curves.
- Compare the performance of your ANN model with a baseline model, perhaps a simpler regression model or logistic regression, to highlight the improvement or benefits of using ANN.
