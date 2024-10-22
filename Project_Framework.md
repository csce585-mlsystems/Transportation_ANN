# Project: Predicting Near-Road Air Quality Using ANN

## 1. Data Overview

### Dataset Description
This project utilizes high-resolution air quality and traffic-related data collected at the intersection of Taylor Street and Pine Street, Columbia, SC. The dataset combines traffic, environmental, and air quality measurements to model the relationships between these factors.

- **Response Variables:**
  - PM1.0, PM2.5, PM10 (log-transformed), and NO2 concentrations.
- **Predictors:**
  - **Traffic-Related Factors:** Counts of cars, trucks, multi-trailers; average vehicle speed; vehicle gaps.
  - **Environmental Factors:** Temperature, humidity, atmospheric pressure, wind speed, wind direction.

### Data Preprocessing
- **Data Cleaning:** Missing values are handled using imputation or removed if necessary. Outliers are managed to prevent skewing of model results.
- **Normalization/Standardization:** Continuous variables are standardized to improve model convergence.
- **Feature Engineering:** 
  - Log-transform response variables to reduce skewness.
  - Create interaction terms between environmental and traffic factors where appropriate.
- **Integration:** Synchronize traffic and environmental data to maintain consistent 15-minute intervals.

---

## 2. Experiment Setup

### Model Architecture
The ANN model is designed to predict pollutant concentrations using the following structure:
- **Input Layer:** Number of neurons equal to the number of predictors after preprocessing.
- **Hidden Layers:** 
  - Layer 1: 128 neurons with ReLU activation.
  - Layer 2: 64 neurons with ReLU activation.
  - Layer 3: 32 neurons with ReLU activation.
  - **Regularization:** Dropout layers (rate = 0.3) to mitigate overfitting.
- **Output Layer:** Single neuron for regression (predicting pollutant concentration).

### Training and Hyperparameter Tuning
- **Training Configuration:**
  - 80% of the dataset for training, 20% for testing.
  - Early stopping based on validation loss to prevent overfitting.
- **Hyperparameters:** Learning rate, batch size, and number of neurons are tuned using cross-validation.
- **Comparative Models:** 
  - Baseline models like Multiple Linear Regression (MLR) and Decision Trees are included for performance comparison.

---

## 3. Model Evaluation

### Evaluation Metrics
- **Root Mean Squared Error (RMSE):** Measures the prediction error.
- **Coefficient of Determination (R²):** Assesses how well the model explains variance in the data.
- **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual values.
- **Cross-Validation:** k-fold cross-validation to ensure robustness of results.

### Model Comparison
- **Baseline Comparison:** Evaluate ANN against MLR and Decision Trees to justify the need for ANN's complexity.

---

## 4. Interpretability and Analysis

### Techniques for Interpretability
- **SHAP Values:** Provide insights into feature contributions to model predictions.
- **Feature Importance Analysis:** Compare feature importance between ANN and traditional models to identify key factors.
- **Visual Analysis:** 
  - Scatter plots (actual vs. predicted values) for assessing fit.
  - Heat maps and residual plots to evaluate model performance.

---

## 5. Broader Implications and Policy Recommendations

### Policy Insights
- Recommendations for reducing pollution in high-traffic areas, such as optimizing traffic signal timing or restricting certain vehicle types during peak hours.
- Explore the generalizability of the model to other urban areas, identifying challenges and solutions.

### Ethical Considerations
- Transparency in model predictions for informing public policy.
- Address potential biases to ensure that recommendations do not disproportionately affect certain communities.

---

## 6. Deliverables

### Presentation Slides
- Clear explanations of the methodology, key results, and visualizations.
- Insights on the impact of traffic and environmental factors on air quality.

### GitHub Repository
- **Code:** Scripts for data preprocessing, model training, evaluation, and visualizations.
- **Documentation:** README and other files to guide replication of the experiments.

### Final Report
- Comprehensive sections on background, methods, results, and policy recommendations.
- Discussion of limitations, future work, and potential extensions of the model.

---

## Conclusion
This project uses ANN to model the complex relationships between traffic-related and environmental factors and their effect on near-road air quality. With a detailed framework that includes preprocessing, model training, evaluation, and visualizations, this approach aims to provide actionable insights for policymakers and urban planners to improve air quality in high-traffic areas.
