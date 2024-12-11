import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Import necessary libraries
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score


# Function to preprocess and train ANN
def run_ann(data_path, response_variable, predictors, dataset_name, epochs=100, batch_size=32, validation_split=0.2):
    print(f"\nRunning ANN for {dataset_name} dataset")

    # Load the dataset
    data = pd.read_csv(data_path)

    # Split the dataset into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=123)

    # Preprocess data: Centering and Scaling (Standardization)
    scaler = StandardScaler()
    scaler.fit(train_data[predictors])
    x_train = scaler.transform(train_data[predictors])
    x_test = scaler.transform(test_data[predictors])

    # Extract output variable
    y_train = train_data[response_variable].values
    y_test = test_data[response_variable].values

    # Build the ANN model
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1],)))  # Use Input layer to avoid the warning
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1))  # Output layer for regression

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )

    # Early stopping configuration
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        x=x_train, y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=0  # Set to 1 to see training progress
    )

    # Evaluate the model on the test set
    test_loss, test_mse = model.evaluate(x=x_test, y=y_test, verbose=0)
    rmse = np.sqrt(test_mse)
    print(f"Test RMSE for {dataset_name}: {rmse}")

    # Make predictions
    predictions = model.predict(x=x_test, verbose=0).flatten()

    # Calculate R-squared
    r_squared = r2_score(y_test, predictions)
    print(f"R-squared for {dataset_name}: {r_squared}")

    return {
        'dataset': dataset_name,
        'rmse': rmse,
        'r_squared': r_squared,
        'history': history
    }

 # Evaluate the model on the test set
    test_loss, test_mse = model.evaluate(x=x_test, y=y_test, verbose=0)
    rmse = np.sqrt(test_mse)
    print(f"Test RMSE for {dataset_name}: {rmse}")

    # Make predictions
    predictions = model.predict(x=x_test, verbose=0).flatten()

    # Calculate R-squared
    r_squared = r2_score(y_test, predictions)
    print(f"R-squared for {dataset_name}: {r_squared}")

    # Calculate SHAP values
    # Use a small subset of x_train for the background dataset to save computation time
    background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(x_test)

    # Plot SHAP summary
    shap.summary_plot(shap_values, x_test, feature_names=predictors, show=False)
    plt.title(f"SHAP Summary Plot for {dataset_name}")
    plt.savefig(f"shap_summary_{dataset_name}.png")
    plt.close()

    # Return the results
    return {
        'dataset': dataset_name,
        'rmse': rmse,
        'r_squared': r_squared,
        'history': history,
        'shap_values': shap_values,
        'x_test': x_test,
        'predictors': predictors
    }

# List of datasets to process
datasets = [
    {
        'name': 'NO₂',
        'data_path': 'dataset_NO2.csv',
        'response_variable': 'NO2',
        'predictors': ['car', 'truck', 'multi_trailer', 'temp_f', 'humidity', 'pressure', 'speed', 'gap']
    },
    {
        'name': 'PM₁.₀',
        'data_path': 'dataset_PM1_subset.csv',
        'response_variable': 'logPM1.0',
        'predictors': ['car', 'truck', 'multi_trailer', 'temperature', 'humidity', 'pressure', 'speed', 'gap']
    },
    {
        'name': 'PM₂.₅',
        'data_path': 'dataset_PM2.5_subset.csv',
        'response_variable': 'logPM2.5',
        'predictors': ['car', 'truck', 'multi_trailer', 'temperature', 'humidity', 'pressure', 'speed', 'gap']
    },
    {
        'name': 'PM₁₀',
        'data_path': 'dataset_PM10_subset.csv',
        'response_variable': 'logPM10',
        'predictors': ['car', 'truck', 'multi_trailer', 'temperature', 'humidity', 'pressure', 'speed', 'gap']
    }
]

# Initialize a list to collect results
results = []

# Loop over each dataset and run the model
for dataset in datasets:
    result = run_ann(
        data_path=dataset['data_path'],
        response_variable=dataset['response_variable'],
        predictors=dataset['predictors'],
        dataset_name=dataset['name']
    )
    results.append(result)

# Display the results together
print("\nSummary of Results:")
for res in results:
    print(f"{res['dataset']}: RMSE = {res['rmse']:.4f}, R-squared = {res['r_squared']:.4f}")
