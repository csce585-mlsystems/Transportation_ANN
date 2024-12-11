# ann_cross_validation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

def run_ann_cv(data_path, response_variable, predictors, dataset_name, epochs=100, batch_size=32, n_splits=5):
    print(f"\nRunning ANN with Cross-Validation for {dataset_name} dataset")

    # Load the dataset
    data = pd.read_csv(data_path)

    # Extract features and target
    X = data[predictors].values
    y = data[response_variable].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize lists to collect results
    rmse_list = []
    r2_list = []
    y_test_all = []
    y_pred_all = []

    # Set up K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

    fold = 1
    for train_index, test_index in kf.split(X_scaled):
        print(f"Fold {fold}/{n_splits}")
        x_train, x_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Build the model
        model = Sequential()
        model.add(Input(shape=(x_train.shape[1],)))
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

        # Evaluate the model
        test_loss, test_mse = model.evaluate(x_test, y_test, verbose=0)
        rmse = np.sqrt(test_mse)
        y_pred = model.predict(x_test, verbose=0).flatten()
        r2 = r2_score(y_test, y_pred)

        print(f"Fold {fold} RMSE: {rmse:.4f}, R-squared: {r2:.4f}")

        rmse_list.append(rmse)
        r2_list.append(r2)
        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred)

        fold += 1

    # Calculate average metrics
    avg_rmse = np.mean(rmse_list)
    avg_r2 = np.mean(r2_list)
    print(f"\nAverage RMSE for {dataset_name}: {avg_rmse:.4f}")
    print(f"Average R-squared for {dataset_name}: {avg_r2:.4f}")

    # Convert lists to numpy arrays
    y_test_all = np.array(y_test_all)
    y_pred_all = np.array(y_pred_all)

    # Return the results
    return {
        'dataset': dataset_name,
        'rmse_list': rmse_list,
        'r2_list': r2_list,
        'avg_rmse': avg_rmse,
        'avg_r2': avg_r2,
        'y_test_all': y_test_all,
        'y_pred_all': y_pred_all
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

    # Initialize a list to collect results
    results = []

    # Loop over each dataset and run cross-validation
    for dataset in datasets:
        result = run_ann_cv(
            data_path=dataset['data_path'],
            response_variable=dataset['response_variable'],
            predictors=dataset['predictors'],
            dataset_name=dataset['name'],
            epochs=50,  # Adjust as needed
            batch_size=32,
            n_splits=10  # Adjust as needed
        )
        results.append(result)

    # Plot RMSE and R-squared across folds for each dataset
    for res in results:
        dataset_name = res['dataset']
        rmse_list = res['rmse_list']
        r2_list = res['r2_list']
        y_test_all = res['y_test_all']
        y_pred_all = res['y_pred_all']

        # RMSE Plot
        plt.figure()
        plt.plot(range(1, len(rmse_list)+1), rmse_list, marker='o')
        plt.title(f"Cross-Validation RMSE for {dataset_name}")
        plt.xlabel('Fold')
        plt.ylabel('RMSE')
        plt.xticks(range(1, len(rmse_list)+1))
        plt.grid(True)
        plt.savefig(f"cv_rmse_{dataset_name}.png")
        plt.show()

        # R-squared Plot
        plt.figure()
        plt.plot(range(1, len(r2_list)+1), r2_list, marker='o')
        plt.title(f"Cross-Validation R-squared for {dataset_name}")
        plt.xlabel('Fold')
        plt.ylabel('R-squared')
        plt.xticks(range(1, len(r2_list)+1))
        plt.grid(True)
        plt.savefig(f"cv_r2_{dataset_name}.png")
        plt.show()

        # Predicted vs. Actual Plot
        plt.figure()
        plt.scatter(y_test_all, y_pred_all, alpha=0.5)
        min_val = min(y_test_all.min(), y_pred_all.min())
        max_val = max(y_test_all.max(), y_pred_all.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.title(f"Predicted vs. Actual for {dataset_name}")
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.savefig(f"predicted_vs_actual_{dataset_name}.png")
        plt.show()

if __name__ == "__main__":
    main()
