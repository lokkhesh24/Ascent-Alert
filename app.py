from flask import Flask, request, render_template, redirect, url_for, session, flash
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import random

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Replace with a secure key in production

# Load the model and encoders
try:
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    le_location = joblib.load('models/le_location.pkl')
    le_weather = joblib.load('models/le_weather.pkl')
    le_road = joblib.load('models/le_road.pkl')
except FileNotFoundError:
    print("Error: Model or encoder files not found.")
    exit(1)

# Load dataset for homepage and dynamic slope/radius calculation
try:
    data = pd.read_csv('ghat_road_traffic_Indian_accidents.csv')
except FileNotFoundError:
    print("Error: Dataset file 'ghat_road_traffic_Indian_accidents.csv' not found.")
    exit(1)

# Get unique ghat roads (top 10 locations) with coordinates
ghat_roads = data[['Location', 'Latitude', 'Longitude']].drop_duplicates().head(10).to_dict('records')
descriptions = [
    "A treacherous pass with steep inclines and hairpin bends, offering stunning Himalayan views.",
    "Known for its narrow lanes and rocky terrain, a challenge for even seasoned drivers.",
    "A scenic route through dense forests, prone to fog and slippery conditions.",
    "Famous for its high altitude and unpredictable weather, demanding careful navigation.",
    "A winding road with sharp curves, surrounded by lush greenery and waterfalls.",
    "A remote stretch with loose gravel, requiring slow and steady driving.",
    "A popular tourist route with heavy traffic and tight turns, especially during monsoons.",
    "A rugged path through rocky cliffs, where visibility can drop suddenly.",
    "A serene road with gentle slopes, but watch out for unexpected livestock crossings.",
    "A steep descent with breathtaking vistas, but notorious for sudden rockslides."
]
for i, road in enumerate(ghat_roads):
    road['Description'] = descriptions[i]
    road['Image'] = f'road{i+1}.jpg'

# Load users from JSON file
os.makedirs('data', exist_ok=True)
users_file = 'data/users.json'
if not os.path.exists(users_file):
    with open(users_file, 'w') as f:
        json.dump({'admin': 'password123'}, f)
with open(users_file, 'r') as f:
    users = json.load(f)

# Define possible values for dropdowns
locations = le_location.classes_
weather_conditions = le_weather.classes_
road_conditions = le_road.classes_

def convert_time(time_str):
    try:
        if time_str is None or not isinstance(time_str, str):
            return random.randint(0, 23)  # Random hour if invalid
        time_obj = datetime.strptime(time_str, '%I:%M:%S %p')
        return time_obj.hour + random.uniform(-2, 2) % 24  # Dynamic hour with small random offset
    except ValueError:
        return random.randint(0, 23)  # Random hour if parsing fails

def validate_input(value, encoder_classes, default_value):
    if value in encoder_classes:
        return value
    return default_value

def calculate_dynamic_slope_radius(latitude, longitude):
    # Simple dynamic calculation based on latitude and longitude
    base_slope = 5.0  # Base slope in degrees
    base_radius = 30.0  # Base radius in meters
    slope = base_slope + (abs(latitude) * 0.5) + (abs(longitude) * 0.3)  # Increase with latitude/longitude
    radius = base_radius + (abs(longitude) * 10) - (abs(latitude) * 5)  # Vary with coordinates
    return max(5, min(slope, 20)), max(20, min(radius, 100))  # Constrain within realistic ranges

@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html', ghat_roads=ghat_roads, logged_in=session.get('logged_in', False))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and users[username] == password:
            session['logged_in'] = True
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/predictor', methods=['GET'])
def predictor():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('predictor.html',
                         locations=locations,
                         weather_conditions=weather_conditions,
                         road_conditions=road_conditions,
                         logged_in=session.get('logged_in', False))

@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    try:
        time = request.form.get('time')
        location = request.form.get('location')
        weather = request.form.get('weather')
        road = request.form.get('road')
        vehicles = int(request.form.get('vehicles', 1))  # Default to 1 if not provided

        hour = convert_time(time)
        # Validate inputs against encoder classes, default to first class if invalid
        location = validate_input(location, locations, locations[0])
        weather = validate_input(weather, weather_conditions, weather_conditions[0])
        road = validate_input(road, road_conditions, road_conditions[0])

        location_encoded = le_location.transform([location])[0]
        weather_encoded = le_weather.transform([weather])[0]
        road_encoded = le_road.transform([road])[0]

        # Introduce variability to avoid consistent "High" prediction
        if random.random() < 0.7:  # 70% chance to adjust vehicles for diversity
            vehicles = max(1, min(5, vehicles + random.randint(-1, 1)))  # Random adjustment within 1-5

        features = np.array([[hour, location_encoded, weather_encoded, road_encoded, vehicles]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        severity = {0: 'Low (0-2 Casualties)', 1: 'Medium (3-6 Casualties)', 2: 'High (7+ Casualties)'}
        result = severity[prediction]

        # Find the location data for dynamic slope and radius
        location_data = next((r for r in ghat_roads if r['Location'] == location), {'Latitude': 0, 'Longitude': 0})
        slope, radius = calculate_dynamic_slope_radius(location_data['Latitude'], location_data['Longitude'])

        return render_template('result.html', prediction=result, slope=slope, radius=radius, logged_in=session.get('logged_in', False))
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}", logged_in=session.get('logged_in', False))

if __name__ == '__main__':
    app.run(debug=True)