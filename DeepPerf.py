import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from scipy.stats import ttest_rel, spearmanr, wilcoxon

def build_deep_perf_model(input_dim):
    """Creates a Deep Neural Network for performance prediction."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer (regression task)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Mean Squared Error for regression
    return model

def main():
    """DeepPerf-based Performance Prediction"""
    
    # Parameters
    systems = ['jump3r', 'kanzi', 'x264', 'z3']
    num_repeats = 3  
    train_frac = 0.7  
    random_seed = 1  

    for current_system in systems:
        datasets_location = f'datasets/{current_system}'  # Adjust dataset location
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(f'\n> System: {current_system}, Dataset: {csv_file}, Training fraction: {train_frac}, Repeats: {num_repeats}')
            
            # Load data
            data = pd.read_csv(os.path.join(datasets_location, csv_file))
            
            # Initialize results storage
            metrics = {'MAPE': [], 'MAE': [], 'RMSE': [], 'T-test p-value': [], 'Shapiro p-value': [], 'Spearman Corr': [], 'Wilcoxon p-value': []}  

            for current_repeat in range(num_repeats):  
                # Split data into training & testing
                train_data = data.sample(frac=train_frac, random_state=random_seed * current_repeat)
                test_data = data.drop(train_data.index)

                # Separate features (X) and target (Y)
                training_X = train_data.iloc[:, :-1]
                training_Y = train_data.iloc[:, -1]
                testing_X = test_data.iloc[:, :-1]
                testing_Y = test_data.iloc[:, -1]

                # Standardize the features (Deep Learning models perform better with normalization)
                scaler = StandardScaler()
                training_X = scaler.fit_transform(training_X)
                testing_X = scaler.transform(testing_X)

                # Build the DeepPerf Model
                model = build_deep_perf_model(training_X.shape[1])

                # Train the model
                model.fit(training_X, training_Y, epochs=50, batch_size=16, verbose=0)

                # Make predictions
                predictions = model.predict(testing_X).flatten()

                # Evaluate model
                mape = mean_absolute_percentage_error(testing_Y, predictions)
                mae = mean_absolute_error(testing_Y, predictions)
                rmse = np.sqrt(mean_squared_error(testing_Y, predictions))

                # Perform Statistical Tests
                try:
                    # Paired T-Test: Checks if predictions significantly differ from actual values
                    t_stat, t_p_value = ttest_rel(testing_Y, predictions)

                    # Spearman Correlation: Measures how well predictions follow actual values
                    spearman_corr, spearman_p_value = spearmanr(testing_Y, predictions)

                    # Wilcoxon Signed-Rank Test: Non-parametric alternative to T-Test
                    wilcoxon_stat, wilcoxon_p_value = wilcoxon(testing_Y, predictions)

                    # Store metrics & test results
                    metrics['MAPE'].append(mape)
                    metrics['MAE'].append(mae)
                    metrics['RMSE'].append(rmse)
                    metrics['T-test p-value'].append(t_p_value)
                    metrics['Spearman Corr'].append(spearman_corr)
                    metrics['Wilcoxon p-value'].append(wilcoxon_p_value)

                except Exception as e:
                    print(f"Statistical test error: {e}")

            # Display average results
            print(f'Average MAPE: {np.mean(metrics["MAPE"]):.2f}')
            print(f'Average MAE: {np.mean(metrics["MAE"]):.2f}')
            print(f'Average RMSE: {np.mean(metrics["RMSE"]):.2f}')
            print(f'Average T-test p-value: {np.mean(metrics["T-test p-value"]):.5f}')
            print(f'Average Spearman Correlation: {np.mean(metrics["Spearman Corr"]):.3f}')
            print(f'Average Wilcoxon p-value: {np.mean(metrics["Wilcoxon p-value"]):.5f}')

if __name__ == "__main__":
    main()
