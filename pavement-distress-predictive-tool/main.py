# Import necessary modules
from data_preprocessing import load_data, clean_data, save_cleaned_data
from feature_engineering import extract_features, transform_features, scale_features, save_features
from model_training import split_data, train_model, evaluate_model, save_model
from model_evaluation import load_model, evaluate_performance, plot_results, save_evaluation_metrics
from utils import check_directory, save_dataframe, one_hot_encode

# Define file paths and parameters
raw_data_file_path = 'data/pavement.csv'
cleaned_data_file_path = 'output/cleaned_data.csv'
features_data_file_path = 'output/engineered_features.csv'
model_file_path = 'output/pavement_distress_model.pkl'
evaluation_metrics_file_path = 'output/evaluation_metrics.txt'
plots_directory = 'output/plots'

# Ensure directories exist
check_directory('output')
check_directory(plots_directory)

# Load and clean data
print("Loading data...")
df = load_data(raw_data_file_path)
print("Cleaning data...")
df_cleaned = clean_data(df)
save_cleaned_data(df_cleaned, cleaned_data_file_path)

# Feature engineering
print("Extracting features...")
df_features = extract_features(df_cleaned)
print("Transforming features...")
df_transformed = transform_features(df_features)

# Select specific columns for final use including the target variable 'PCI'
selected_columns = ['True Area (Ft2)', 'Length (Ft)', 'Quantity', 'Severity Numeric', 'Distress Density', 'Log Quantity', 'PCI']
df_selected = df_transformed[selected_columns]

# Select columns to scale excluding the target variable 'PCI'
columns_to_scale = ['True Area (Ft2)', 'Length (Ft)', 'Quantity', 'Severity Numeric', 'Distress Density', 'Log Quantity']
print("Scaling features...")
df_scaled = scale_features(df_selected, columns_to_scale)

# Save the scaled features including the target variable 'PCI'
save_features(df_scaled, features_data_file_path)

# Split data into train and test sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = split_data(df_scaled, "PCI")

# Train the model
print("Training the model...")
model = train_model(X_train, y_train)
save_model(model, model_file_path)

# Evaluate the model
print("Evaluating the model...")
metrics = evaluate_model(model, X_test, y_test)
save_evaluation_metrics(metrics, evaluation_metrics_file_path)

# Load the model for evaluation
print("Loading the model for further evaluation...")
loaded_model = load_model(model_file_path)

# Predict and evaluate on the test set
print("Predicting and evaluating performance on test set...")
y_pred = loaded_model.predict(X_test)
evaluation_metrics = evaluate_performance(y_test, y_pred)
save_evaluation_metrics(evaluation_metrics, evaluation_metrics_file_path)

# Plot results
print("Plotting results...")
plot_results(y_test.values, y_pred)

print("Process completed successfully.")
