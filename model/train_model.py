import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("C:/Users/lokkh\Downloads/Test Project/dataset/Book1.csv")

# Preprocessing
# Handle missing numeric values
data = data.fillna(data.median(numeric_only=True))

# List of all categorical columns to encode
categorical_cols = ['Weather', 'Road_Type', 'Road_Condition', 'Time_of_Day', 
                    'Vehicle_Type', 'Road_Light_Condition', 'Traffic_Density', 
                    'Accident_Severity']

# Handle missing values and encode categorical columns
for col in categorical_cols:
    if col in data.columns:  # Check if column exists (e.g., Accident_Severity might be missing in some rows)
        data[col] = data[col].fillna(data[col].mode()[0])  # Fill missing with mode
        data[col] = LabelEncoder().fit_transform(data[col])  # Convert to numeric

# Features and target
# Drop Accident_Severity if it's not needed as a feature
X = data.drop(['Accident', 'Accident_Severity'], axis=1, errors='ignore')
y = data['Accident']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.pkl')

# Evaluate
print("Accuracy:", model.score(X_test, y_test))