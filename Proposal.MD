# Predicting Near-Road Air Quality Using Artificial Neural Networks: Exploring Traffic-Related Influences

## Introduction

The proposed project aims to investigate the relationships between near-road air quality and traffic-related factors using Artificial Neural Networks (ANN). This study addresses a critical issue: understanding how vehicular emissions and traffic dynamics affect air pollutant levels, such as Particulate Matter (PM1.0, PM2.5, PM10) and Nitrogen Dioxide (NO2), near high-traffic areas. The insights from this research could guide urban planners and policymakers in mitigating air pollution impacts on public health.

The motivation for this project arises from the increasing concern over air pollution and its health implications, especially in urban areas with high traffic volumes. Traditional approaches, such as linear regression models, have been effective but limited in capturing the nonlinear and complex relationships between traffic behavior and pollutant levels. By employing ANN, this project seeks to improve predictive accuracy and provide more nuanced insights into the interactions between traffic factors (e.g., vehicle counts, speed) and air quality.

## Data

The dataset consists of high-resolution air quality and traffic data collected at the intersection of Taylor Street and Pine Street in downtown Columbia, SC. Variables include:
Response Variables: Concentrations of PM1.0, PM2.5, PM10 (log-transformed) and NO2.
Predictors: Traffic-related factors (e.g., counts of cars, trucks, multi-trailers, average speed, gaps between vehicles) and environmental variables (temperature, humidity, atmospheric pressure).

## Methodology

The project will use ANN to model and predict pollutant concentrations. The ANN model will be compared against traditional methods like Multiple Linear Regression (MLR) to evaluate its performance, particularly in handling complex and nonlinear interactions.

## Evaluation and Expected Outcomes

The performance of the ANN model will be evaluated using metrics such as Root Mean Squared Error (RMSE) and Coefficient of Determination (R²). Additionally, the effectiveness of ANN will be compared with MLR and Bayesian Model to assess improvements in predictive accuracy.

## Expected Outcomes

A detailed analysis of how traffic behaviors influence near-road air quality.
Insights into which factors are most significant in predicting pollutant levels, enabling data-driven recommendations for traffic and air quality management.
