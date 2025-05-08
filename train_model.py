import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)

# Load the dataset
try:
    data = pd.read_csv('ghat_road_traffic_Indian_accidents.csv')
except FileNotFoundError:
    print("Error: Dataset file 'ghat_road_traffic_Indian_accidents.csv' not found.")
    exit(1)

# Convert Time to Hour
def convert_time_to_hour(time_str):
    try:
        if time_str is None or not isinstance(time_str, str):
            return 0
        time_obj = pd.to_datetime(time_str, format='%I:%M:%S %p')
        return time_obj.hour
    except (ValueError, TypeError):
        return 0

data['Hour'] = data['Time'].apply(convert_time_to_hour)

# Encode categorical variables
le_location = LabelEncoder()
le_weather = LabelEncoder()
le_road = LabelEncoder()

data['Location'] = le_location.fit_transform(data['Location'])
data['Weather Condition'] = le_weather.fit_transform(data['Weather Condition'])
data['Road Condition'] = le_road.fit_transform(data['Road Condition'])

# Define features and target
X = data[['Hour', 'Location', 'Weather Condition', 'Road Condition', 'Vehicles Involved']]
y = data['Severity']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

# Evaluate models using cross-validation
best_model_name = None
best_model = None
best_score = 0

print("Evaluating models...")
for name, model in models.items():
    # Use scaled data for models that require it (SVM, Neural Network)
    if name in ['SVM', 'Neural Network']:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    mean_score = np.mean(cv_scores)
    print(f"{name}: Cross-validation accuracy = {mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_model_name = name
        best_model = model

print(f"\nBest model: {best_model_name} with accuracy = {best_score:.4f}")

# Train the best model on the full training data
if best_model_name in ['SVM', 'Neural Network']:
    best_model.fit(X_train_scaled, y_train)
else:
    best_model.fit(X_train, y_train)

# Save the best model, scaler, and encoders
joblib.dump(best_model, 'models/model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le_location, 'models/le_location.pkl')
joblib.dump(le_weather, 'models/le_weather.pkl')
joblib.dump(le_road, 'models/le_road.pkl')

print("Best model and encoders saved successfully.")