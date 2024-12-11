# ann_shap_lime.py

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from lime import lime_tabular
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


def run_ann(data_path, response_variable, predictors, dataset_name, epochs=100, batch_size=32, validation_split=0.2):
    print(f"\nRunning ANN for {dataset_name} dataset")

    # Load the dataset
    data = pd.read_csv(data_path)

    # Extract features and target
    X = data[predictors].values
    y = data[response_variable].values

    # Split the data into training and testing sets
    split_idx = int(len(X) * (1 - validation_split))
    x_train, x_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Standardize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Build the model
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1],), name='input_layer'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )

    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0
    )

    # Evaluate the model on the test set
    test_loss, test_mse = model.evaluate(x_test, y_test, verbose=0)
    rmse = np.sqrt(test_mse)
    print(f"Test RMSE for {dataset_name}: {rmse:.4f}")

    # Make predictions
    predictions = model.predict(x_test, verbose=0).flatten()

    # Calculate R-squared
    r_squared = r2_score(y_test, predictions)
    print(f"R-squared for {dataset_name}: {r_squared:.4f}")

    # Calculate SHAP values
    background = x_train[np.random.choice(x_train.shape[0], size=100, replace=False)]
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(x_test)

    # Debug: Print shapes
    print("x_test shape:", x_test.shape)
    print("shap_values shape:", shap_values.shape)
    print("Number of predictors:", len(predictors))

    # Convert x_test to DataFrame
    x_test_df = pd.DataFrame(x_test, columns=predictors)

    # Plot SHAP summary
    shap.summary_plot(shap_values, x_test_df, show=False)
    plt.title(f"SHAP Summary Plot for {dataset_name}")
    plt.savefig(f"shap_summary_{dataset_name}.png")
    plt.close()

    # Apply LIME
    explainer_lime = lime_tabular.LimeTabularExplainer(
        training_data=x_train,
        feature_names=predictors,
        class_names=[response_variable],
        mode='regression'
    )

    # Choose an instance to explain (e.g., the first test sample)
    idx = 0
    exp = explainer_lime.explain_instance(
        data_row=x_test[idx],
        predict_fn=lambda x: model.predict(x).flatten()
    )

    # Save the explanation as an HTML file
    exp.save_to_file(f'lime_explanation_{dataset_name}.html')
    # Visualize the explanation and save it as a PNG file
    fig = exp.as_pyplot_figure()
    plt.title(f'LIME Explanation for {dataset_name} (Instance {idx})')
    plt.savefig(f'lime_explanation_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
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


def main():
    # Define datasets
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

    # Loop over each dataset and run the model
    for dataset in datasets:
        result = run_ann(
            data_path=dataset['data_path'],
            response_variable=dataset['response_variable'],
            predictors=dataset['predictors'],
            dataset_name=dataset['name'],
            epochs=50,  # Adjust as needed
            batch_size=32
        )

if __name__ == "__main__":
    main()
