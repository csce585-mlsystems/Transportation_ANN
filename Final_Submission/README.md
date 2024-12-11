YouTube Link for the presentation: 
# Predicting Near-Road Air Quality Using Artificial Neural Networks

This repository contains the resources and instructions needed to replicate the results from the study: **Predicting Near-Road Air Quality Using Artificial Neural Networks: Exploring Traffic-Related Influences**. The project leverages machine learning to predict near-road pollutant concentrations using traffic and environmental variables.


## Project Overview
This project develops an Artificial Neural Network (ANN) to predict air pollutant concentrations, focusing on PM1.0, PM2.5, PM10, and NO₂. The model uses traffic-related and environmental predictors, addressing limitations of traditional models by capturing complex, nonlinear interactions.

### Key Components:
- **Data Collection:** Input datasets for PM and NO₂ pollutants.
- **Model Development:** ANN with three hidden layers using dropout and ReLU activation.
- **Evaluation:** Performance measured using R², RMSE, and Local Interpretable Model-Agnostic Explanations (LIME).

## Repository Contents
- `ann_model.py`: Script for building and training the ANN model.
- `ann_cross_validation.py`: Script for performing cross-validation to assess model robustness.
- `ann_lime.py`: Script for interpreting the ANN predictions using LIME.
- `dataset_PM.csv`: Dataset containing PM1.0, PM2.5, and PM10 data.
- `dataset_NO2.csv`: Dataset containing NO₂ data.

## Getting Started
### Prerequisites
To run this project, you need:
- Python 3.8 or higher
- Required libraries: TensorFlow, NumPy, pandas, scikit-learn, matplotlib, lime

Install the required libraries using pip:
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib lime
```

### Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/your_username/ANN-AirQuality-Prediction.git
cd ANN-AirQuality-Prediction
```

## How to Run the Code
### Step 1: Prepare the Data
Ensure the datasets `dataset_PM.csv` and `dataset_NO2.csv` are in the repository directory. These contain the input data for PM and NO₂ predictions.

### Step 2: Train the ANN Model
Run the `ann_model.py` script to train the model:
```bash
python ann_model.py
```
This script will:
- Load and preprocess the data
- Build the ANN architecture
- Train the model and save the results

### Step 3: Perform Cross-Validation
Run the `ann_cross_validation.py` script to validate the model’s robustness:
```bash
python ann_cross_validation.py
```
This script will:
- Perform k-fold cross-validation
- Output performance metrics like R² and RMSE

### Step 4: Interpret Results with LIME
Run the `ann_lime.py` script to analyze the model’s predictions:
```bash
python ann_lime.py
```
This script will:
- Generate LIME explanations for the ANN predictions
- Visualize feature contributions for selected instances

## Results
The trained ANN model achieves high predictive performance:
- **R²**: Up to 0.88 for PM concentrations and 0.86 for NO₂
- **RMSE**: Low prediction errors across pollutants
- **Insights**: LIME reveals that heavy-duty trucks, temperature, and humidity significantly influence pollutant levels.

## Acknowledgements
This project is part of the research by Yihong Ning, Ph.D. student in Civil Engineering, University of South Carolina. The datasets were collected near Taylor Street, Columbia, SC, as part of the study on traffic-related air quality influences.

A video presentation summarizing the project will be available soon. The link will be provided here once created.

If you have questions or need further assistance, feel free to raise an issue in this repository.
