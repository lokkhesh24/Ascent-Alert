import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
# For regression if predicting exact number of casualties:
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.svm import SVR
# from sklearn.neural_network import MLPRegressor
import joblib
import os
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# --- Configuration ---
MODEL_DIR = 'models'
DATASET_PATH = 'ghat_road_traffic_Indian_accidents.csv'
TARGET_VARIABLE = 'Casualties' # This is what we want to predict

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load Data ---
try:
    data = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: Dataset file '{DATASET_PATH}' not found.")
    exit(1)

# --- Preprocessing ---
# Handle missing values (example: fill with mode or mean, or drop)
for col in data.columns:
    if data[col].isnull().any():
        if data[col].dtype == 'object' or data[col].dtype == 'bool': # Categorical
            data[col] = data[col].fillna(data[col].mode()[0])
        else: # Numerical
            data[col] = data[col].fillna(data[col].median())


# Convert Time to Hour (numerical feature)
def convert_time_to_hour(time_str):
    try:
        if pd.isna(time_str): return 0 # Handle potential NaNs if not caught by earlier fillna
        time_str = str(time_str) # Ensure it's a string
        # Attempt multiple formats if necessary, or standardize your input
        return pd.to_datetime(time_str, format='%I:%M:%S %p', errors='coerce').hour
    except ValueError: # Fallback for unexpected formats
        try:
            return pd.to_datetime(time_str, errors='coerce').hour
        except:
            return 0 # Default hour if parsing fails

data['Hour'] = data['Time'].apply(convert_time_to_hour)
data['Hour'] = data['Hour'].fillna(0).astype(int) # Ensure no NaNs after conversion


# Define features (X) and target (y)
# 'Slope' and 'Radius' are important for ghat roads.
# 'Vehicles Involved' is also a critical factor.
# Assuming 'Day' and 'Month' might be relevant if you have them, or derive from a 'Date' column.
# For this example, we'll use the columns that seem most relevant and are typically available.
# Ensure these column names exactly match your CSV.
categorical_features = ['Location', 'Weather Condition', 'Road Condition']
numerical_features = ['Hour', 'Vehicles Involved', 'Slope', 'Radius'] # Make sure these exist in your CSV

# Check if all features exist
required_features = categorical_features + numerical_features
for feature in required_features:
    if feature not in data.columns:
        print(f"Error: Feature '{feature}' not found in the dataset. Please check your CSV headers.")
        # As a fallback, if a feature is missing, you could drop it or create a dummy one, but it's better to fix the data.
        # For example, creating a dummy 'Slope' and 'Radius' if missing:
        if feature == 'Slope' and 'Slope' not in data.columns:
            print("Warning: 'Slope' column missing. Creating a dummy column with mean 0.1.")
            data['Slope'] = 0.1 
        if feature == 'Radius' and 'Radius' not in data.columns:
            print("Warning: 'Radius' column missing. Creating a dummy column with mean 100.")
            data['Radius'] = 100.0
        # If a critical feature is missing, you might want to exit.
        # exit(1)

X = data[required_features]
y = data[TARGET_VARIABLE]

# --- Determine if it's a classification or regression problem ---
# If casualties are distinct classes (e.g., 0, 1, 2, 3+ mapped to categories like Low, Medium, High)
# or if you're predicting the exact number and treating it as regression.
# For this example, let's assume we're predicting the exact number of casualties,
# but since the original request implies distinct outcomes (0, 1, 2+),
# Random Forest Classifier can handle this if y is integer-coded classes.
# If y has many unique continuous values, a Regressor would be more appropriate.
# Let's treat it as a classification problem where y represents discrete casualty counts.
# Ensure y is integer type for classification
y = y.astype(int)


# --- Preprocessing Pipelining ---
# Create transformers for categorical and numerical data
# OneHotEncoder for categorical features, StandardScaler for numerical features

# IMPORTANT: Store LabelEncoders separately for inverse_transform if needed for interpretation,
# or for consistent transform/inverse_transform in the app.
# However, OneHotEncoder is generally preferred over LabelEncoder for nominal categorical features in linear models, SVMs, NNs.
# For tree-based models, LabelEncoding is often fine.
# For simplicity and consistency with the app.py provided earlier, we'll use LabelEncoders for app-side transformation
# and save them. But for the pipeline, OneHotEncoder is more robust for many model types.

# Let's save LabelEncoders for the app
le_location = LabelEncoder()
X['Location'] = le_location.fit_transform(X['Location'].astype(str))
joblib.dump(le_location, os.path.join(MODEL_DIR, 'le_location.pkl'))

le_weather = LabelEncoder()
X['Weather Condition'] = le_weather.fit_transform(X['Weather Condition'].astype(str))
joblib.dump(le_weather, os.path.join(MODEL_DIR, 'le_weather.pkl'))

le_road = LabelEncoder()
X['Road Condition'] = le_road.fit_transform(X['Road Condition'].astype(str))
joblib.dump(le_road, os.path.join(MODEL_DIR, 'le_road.pkl'))


# Now, for the pipeline, we'll use the already label-encoded columns as numerical inputs,
# and then scale all of them. This is a common approach.
# The features going into the scaler will be:
# 'Hour', 'Vehicles Involved', 'Slope', 'Radius', 'Location' (encoded), 'Weather Condition' (encoded), 'Road Condition' (encoded)
# Ensure the order matches how you'll prepare data in app.py
# The features in X are now all numerical (original numerical + label encoded categorical)
# So we just need a StandardScaler.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None)

# Scaler will be fit on X_train and used to transform X_train and X_test
# It will also be saved for the Flask app
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))


# --- Model Selection and Training ---
# Define models to try (using classifiers as casualties are discrete counts)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, solver='liblinear'), # Good for multiclass if y has few unique values
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100), # Good general purpose
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
    # 'SVM': SVC(random_state=42, probability=True), # Can be slow on large datasets
    # 'Neural Network (MLP)': MLPClassifier(random_state=42, max_iter=500, early_stopping=True)
}

best_model_name = None
best_model_instance = None
best_score = 0

print("Evaluating models...")
for name, model_instance in models.items():
    # Train on scaled data
    model_instance.fit(X_train_scaled, y_train)
    score = model_instance.score(X_test_scaled, y_test) # Using accuracy for classification
    print(f"{name}: Test Accuracy = {score:.4f}")

    if score > best_score:
        best_score = score
        best_model_name = name
        best_model_instance = model_instance

print(f"\nBest model: {best_model_name} with Test Accuracy = {best_score:.4f}")

# --- Save the Best Model ---
if best_model_instance:
    model_filename = os.path.join(MODEL_DIR, 'model.pkl')
    joblib.dump(best_model_instance, model_filename)
    print(f"Best model ({best_model_name}) saved to {model_filename}")
else:
    print("No model was trained successfully.")

# --- Feature Importance (if using a model that supports it, like Random Forest) ---
if hasattr(best_model_instance, 'feature_importances_') and best_model_name in ['Random Forest', 'Gradient Boosting']:
    importances = best_model_instance.feature_importances_
    feature_names = X.columns # X still holds the original feature names before scaling
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    print("\nFeature Importances:")
    print(feature_importance_df)

print("\n--- Training complete ---")
print(f"Saved Label Encoders: le_location.pkl, le_weather.pkl, le_road.pkl in '{MODEL_DIR}/'")
print(f"Saved Scaler: scaler.pkl in '{MODEL_DIR}/'")
print(f"Saved Model: model.pkl in '{MODEL_DIR}/'")
