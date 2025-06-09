from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import requests
import redis
import json
from dotenv import load_dotenv
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Replace with a secure key in production

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize Flask-SocketIO
socketio = SocketIO(app, async_mode='eventlet')

# Load environment variables
load_dotenv()
OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')

# Initialize Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Define Incident model
class Incident(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(50), nullable=False)  # e.g., Accident, Roadblock
    severity = db.Column(db.String(20), nullable=False)  # Low, Medium, High
    description = db.Column(db.Text, nullable=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('incidents', lazy=True))

# Create database tables
with app.app_context():
    db.create_all()
    # Add default admin user if not exists
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin')
        admin.set_password('password123')
        db.session.add(admin)
        db.session.commit()

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

# Load dataset for homepage, dashboard, and dynamic slope/radius calculation
try:
    data = pd.read_csv('ghat_road_traffic_Indian_accidents.csv')
    print("Columns in the dataset:", data.columns.tolist())  # Debug statement
except FileNotFoundError:
    print("Error: Dataset file 'ghat_road_traffic_Indian_accidents.csv' not found.")
    exit(1)

# Preprocess the dataset
# Derive 'Severity' from 'Casualties'
def categorize_severity(casualties):
    if casualties <= 2:
        return 'Low'
    elif casualties <= 6:
        return 'Medium'
    else:
        return 'High'

data['Severity'] = data['Casualties'].apply(categorize_severity)

# Extract hour from 'Time' column (assuming format like 'HH:MM:SS')
def extract_hour(time_str):
    try:
        return int(time_str.split(':')[0])  # Extract hour from 'HH:MM:SS'
    except (AttributeError, ValueError):
        return 0  # Default to 0 if parsing fails

data['Hour'] = data['Time'].apply(extract_hour)

# Function to shorten location names to the first word
def shorten_location_name(location):
    return location.split()[0].capitalize()

# Create a mapping of full location names to shortened names
location_mapping = {loc: shorten_location_name(loc) for loc in data['Location'].unique()}
# Apply the mapping to the dataset
data['Short_Location'] = data['Location'].map(location_mapping)

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
    road['Short_Location'] = shorten_location_name(road['Location'])

# Define possible values for dropdowns
locations = le_location.classes_
locations_shortened = {loc: shorten_location_name(loc) for loc in locations}
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
    base_slope = 5.0  # Base slope in degrees
    base_radius = 30.0  # Base radius in meters
    slope = base_slope + (abs(latitude) * 0.5) + (abs(longitude) * 0.3)
    radius = base_radius + (abs(longitude) * 10) - (abs(latitude) * 5)
    return max(5, min(slope, 20)), max(20, min(radius, 100))

# Update the fetch_weather_data function to return both the weather condition and raw response
def fetch_weather_data(latitude, longitude):
    cache_key = f"weather:{latitude}:{longitude}"
    cached = None
    try:
        cached = redis_client.get(cache_key)
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection failed: {e}. Bypassing cache.")
    
    if cached:
        return json.loads(cached), None  # No raw response if cached
    
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
        response = requests.get(url)
        if response.status_code == 401:
            print("Error: Invalid or missing OpenWeatherMap API key. Please check your .env file.")
            return 'Clear', {'error': 'Invalid API key'}
        response.raise_for_status()
        data = response.json()
        print(f"OpenWeatherMap API response: {data}")
        
        if 'weather' not in data or not data['weather']:
            print("Error: 'weather' key missing or empty in API response.")
            return 'Clear', data
        
        weather_main = data['weather'][0]['main'].lower()
        weather_condition = 'Cloudy'
        if 'rain' in weather_main or 'drizzle' in weather_main:
            weather_condition = 'Rainy'
        elif 'fog' in weather_main or 'mist' in weather_main:
            weather_condition = 'Foggy'
        elif 'clear' in weather_main:
            weather_condition = 'Clear'
        elif 'snow' in weather_main:
            weather_condition = 'Snowy'
        
        try:
            redis_client.setex(cache_key, timedelta(minutes=10), json.dumps(weather_condition))
        except redis.exceptions.ConnectionError as e:
            print(f"Failed to cache weather data: {e}")
        
        return weather_condition, data
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return 'Clear', {'error': str(e)}

# Infer road condition from weather
def infer_road_condition(weather_condition):
    if weather_condition in ['Rainy', 'Snowy']:
        return 'Wet'
    elif weather_condition == 'Foggy':
        return 'Slippery'
    else:
        return 'Dry'

@app.route('/')
def loading():
    return render_template('loading.html')

@app.route('/index')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html', ghat_roads=ghat_roads, logged_in=session.get('logged_in', False))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['logged_in'] = True
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if len(username) < 3:
            flash('Username must be at least 3 characters long.', 'error')
            return render_template('register.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('register.html')
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different one.', 'error')
            return render_template('register.html')
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

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
    
    live_weather = None
    live_road_condition = None
    weather_data = None
    
    selected_location = request.args.get('location', locations[0])
    location_data = next((r for r in ghat_roads if r['Location'] == selected_location), None)
    
    if location_data:
        latitude = location_data['Latitude']
        longitude = location_data['Longitude']
        live_weather, weather_data = fetch_weather_data(latitude, longitude)
        live_road_condition = infer_road_condition(live_weather)
    
    return render_template('predictor.html',
                         locations=locations,
                         locations_shortened=locations_shortened,
                         weather_conditions=weather_conditions,
                         road_conditions=road_conditions,
                         live_weather=live_weather,
                         live_road_condition=live_road_condition,
                         weather_data=weather_data,
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
        vehicles = int(request.form.get('vehicles', 1))

        hour = convert_time(time)
        location = validate_input(location, locations, locations[0])
        weather = validate_input(weather, weather_conditions, weather_conditions[0])
        road = validate_input(road, road_conditions, road_conditions[0])

        location_encoded = le_location.transform([location])[0]
        weather_encoded = le_weather.transform([weather])[0]
        road_encoded = le_road.transform([road])[0]

        if random.random() < 0.7:
            vehicles = max(1, min(5, vehicles + random.randint(-1, 1)))

        features = np.array([[hour, location_encoded, weather_encoded, road_encoded, vehicles]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        severity = {0: 'Low (0-2 Casualties)', 1: 'Medium (3-6 Casualties)', 2: 'High (7+ Casualties)'}
        result = severity[prediction]

        location_data = next((r for r in ghat_roads if r['Location'] == location), {'Latitude': 0, 'Longitude': 0})
        slope, radius = calculate_dynamic_slope_radius(location_data['Latitude'], location_data['Longitude'])

        return render_template('result.html', prediction=result, slope=slope, radius=radius, logged_in=session.get('logged_in', False))
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}", logged_in=session.get('logged_in', False))

@app.route('/fetch_live_weather')
def fetch_live_weather():
    try:
        location = request.args.get('location')
        location_data = next((r for r in ghat_roads if r['Location'] == location), None)
        if not location_data:
            return jsonify({'success': False, 'error': 'Location not found'}), 404
        
        latitude = location_data['Latitude']
        longitude = location_data['Longitude']
        weather, weather_data = fetch_weather_data(latitude, longitude)
        road_condition = infer_road_condition(weather)
        
        return jsonify({
            'success': True,
            'weather': weather,
            'road_condition': road_condition,
            'weather_data': weather_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/report_incident', methods=['POST'])
def report_incident():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    try:
        data = request.json
        location = data.get('location')
        incident_type = data.get('type')
        severity = data.get('severity')
        description = data.get('description')
        
        location_data = next((r for r in ghat_roads if r['Location'] == location), None)
        if not location_data:
            return jsonify({'success': False, 'error': 'Invalid location'}), 400
        
        user = User.query.filter_by(username=session['username']).first()
        incident = Incident(
            location=location,
            type=incident_type,
            severity=severity,
            description=description,
            latitude=location_data['Latitude'],
            longitude=location_data['Longitude'],
            user_id=user.id
        )
        db.session.add(incident)
        db.session.commit()
        
        # Broadcast alert via WebSocket
        socketio.emit('new_incident', {
            'location': location,
            'type': incident_type,
            'severity': severity,
            'description': description,
            'latitude': location_data['Latitude'],
            'longitude': location_data['Longitude'],
            'timestamp': incident.timestamp.isoformat()
        }, namespace='/alerts', room=location)
        
        return jsonify({'success': True, 'message': 'Incident reported successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/delete_incident/<int:incident_id>', methods=['DELETE'])
def delete_incident(incident_id):
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    try:
        incident = Incident.query.get(incident_id)
        if not incident:
            return jsonify({'success': False, 'error': 'Incident not found'}), 404

        # Check if the user has permission to delete (either the owner or admin)
        current_user = User.query.filter_by(username=session['username']).first()
        if incident.user_id != current_user.id and session['username'] != 'admin':
            return jsonify({'success': False, 'error': 'Unauthorized: You can only delete your own incidents'}), 403

        # Store location for WebSocket broadcast before deletion
        incident_location = incident.location

        # Delete the incident
        db.session.delete(incident)
        db.session.commit()

        # Broadcast deletion via WebSocket
        socketio.emit('incident_deleted', {
            'incident_id': incident_id,
            'location': incident_location
        }, namespace='/alerts', room=incident_location)

        return jsonify({'success': True, 'message': 'Incident deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_incidents', methods=['GET'])
def get_incidents():
    if not session.get('logged_in'):
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        incidents = Incident.query.all()
        incident_list = []
        for idx, incident in enumerate(incidents):
            incident_list.append({
                'id': idx,  # Use index as ID for client-side reference
                'location': incident.location,
                'type': incident.type,
                'severity': incident.severity,
                'description': incident.description,
                'latitude': incident.latitude,
                'longitude': incident.longitude,
                'timestamp': incident.timestamp.isoformat(),
                'username': incident.user.username
            })
        return jsonify(incident_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # Prepare data for charts
    severity_counts = data['Severity'].value_counts().to_dict()
    weather_counts = data['Weather Condition'].value_counts().to_dict()
    road_counts = data['Road Condition'].value_counts().to_dict()
    hourly_counts = data['Hour'].value_counts().sort_index().to_dict()
    
    # Debug: Print the data being passed to the template
    print("Severity Counts:", severity_counts)
    print("Weather Counts:", weather_counts)
    print("Road Counts:", road_counts)
    print("Hourly Counts:", hourly_counts)
    
    return render_template('dashboard.html',
                         severity_counts=severity_counts,
                         weather_counts=weather_counts,
                         road_counts=road_counts,
                         hourly_counts=hourly_counts,
                         logged_in=session.get('logged_in', False))

@app.route('/about')
def about():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('about.html', logged_in=session.get('logged_in', False))

# WebSocket handlers for real-time alerts
@socketio.on('join', namespace='/alerts')
def on_join(location):
    join_room(location)
    print(f"User joined room: {location}")

@socketio.on('leave', namespace='/alerts')
def on_leave(location):
    leave_room(location)
    print(f"User left room: {location}")

if __name__ == '__main__':
    socketio.run(app, debug=True)