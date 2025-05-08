import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted
from datetime import datetime
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load the dataset
try:
    data = pd.read_csv('ghat_road_traffic_Indian_accidents.csv')
    print(f"Dataset loaded successfully. Shape: {data.shape}")
except FileNotFoundError:
    print("Error: Dataset file 'ghat_road_traffic_Indian_accidents.csv' not found.")
    exit(1)

# Select relevant columns
features = ['Time', 'Location', 'Weather Condition', 'Road Condition', 'Vehicles Involved']
additional = ['Casualties']
data = data[features + additional]

# Create accident severity target based on casualties
def categorize_severity(casualties):
    if casualties <= 2:
        return 0  # Low Accident
    elif 3 <= casualties <= 6:
        return 1  # Medium Accident
    else:
        return 2  # High Accident

data['Accident_Severity'] = data['Casualties'].apply(categorize_severity)

# Preprocess the data
# Convert Time to hour of the day
def convert_time(time_str):
    try:
        time_obj = datetime.strptime(time_str, '%I:%M:%S %p')
        return time_obj.hour
    except ValueError:
        return np.nan

data['Hour'] = data['Time'].apply(convert_time)
data['Hour'].fillna(data['Hour'].median(), inplace=True)

# Encode categorical variables
le_location = LabelEncoder()
le_weather = LabelEncoder()
le_road = LabelEncoder()

data['Location_Encoded'] = le_location.fit_transform(data['Location'])
data['Weather Condition'] = le_weather.fit_transform(data['Weather Condition'])
data['Road Condition'] = le_road.fit_transform(data['Road Condition'])

# Prepare features and target
X = data[['Hour', 'Location_Encoded', 'Weather Condition', 'Road Condition', 'Vehicles Involved']]
y = data['Accident_Severity']

# Handle missing values
X = X.dropna()
y = y[X.index]
print(f"Data after preprocessing. Shape: {X.shape}")

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
if X.shape[0] == 0:
    print("Error: No data available after preprocessing.")
    exit(1)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# Initialize Random Forest model
model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Train the model
print("\nTraining Random Forest...")
try:
    model.fit(X_train, y_train)
    check_is_fitted(model)
    print("Random Forest fitted successfully.")
except Exception as e:
    print(f"Error training Random Forest: {e}")
    exit(1)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nRandom Forest Accuracy: {accuracy:.4f}")

# Save the model and encoders
joblib.dump(model, 'models/model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le_location, 'models/le_location.pkl')
joblib.dump(le_weather, 'models/le_weather.pkl')
joblib.dump(le_road, 'models/le_road.pkl')
print("\nModel and encoders saved successfully.")