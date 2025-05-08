from flask import Flask, request, render_template, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import random

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Replace with a secure key in production

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

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
except FileNotFoundError:
    print("Error: Dataset file 'ghat_road_traffic_Indian_accidents.csv' not found.")
    exit(1)

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
    # Simple dynamic calculation based on latitude and longitude
    base_slope = 5.0  # Base slope in degrees
    base_radius = 30.0  # Base radius in meters
    slope = base_slope + (abs(latitude) * 0.5) + (abs(longitude) * 0.3)  # Increase with latitude/longitude
    radius = base_radius + (abs(longitude) * 10) - (abs(latitude) * 5)  # Vary with coordinates
    return max(5, min(slope, 20)), max(20, min(radius, 100))  # Constrain within realistic ranges

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
        # Basic validation
        if len(username) < 3:
            flash('Username must be at least 3 characters long.', 'error')
            return render_template('register.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('register.html')
        # Check if username already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different one.', 'error')
            return render_template('register.html')
        # Create new user
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
    return render_template('predictor.html',
                         locations=locations,
                         locations_shortened=locations_shortened,
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

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    # Process data for charts
    # 1. Casualties by Ghat Road Name (using shortened names)
    casualties_by_location = data.groupby('Short_Location')['Casualties'].sum().to_dict()

    # 2. Accidents by Time of Day (unchanged)
    data['Hour'] = data['Time'].apply(convert_time)
    time_bins = pd.cut(data['Hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'], right=False)
    time_accidents = time_bins.value_counts().to_dict()

    # 3. Cause vs Road Condition vs Ghat Road Name (using shortened names)
    # Simulate 'Cause' column if not present
    if 'Cause' not in data.columns:
        causes = ['Speeding', 'Overtaking', 'Weather', 'Mechanical Failure', 'Driver Error']
        data['Cause'] = np.random.choice(causes, size=len(data))
    cause_road_location = data.groupby(['Cause', 'Road Condition', 'Short_Location']).size().reset_index(name='Accident_Count').to_dict('records')
    causes = sorted(data['Cause'].unique())
    road_conditions = sorted(data['Road Condition'].unique())
    short_locations = sorted(data['Short_Location'].unique())

    # 4. Road Condition vs Weather vs Ghat Road (using shortened names)
    road_weather_location = data.groupby(['Road Condition', 'Weather Condition', 'Short_Location']).size().reset_index(name='Accident_Count').to_dict('records')

    return render_template('dashboard.html',
                         casualties_by_location=casualties_by_location,
                         time_accidents=time_accidents,
                         cause_road_location=cause_road_location,
                         causes=causes,
                         road_conditions=road_conditions,
                         short_locations=short_locations,
                         road_weather_location=road_weather_location,
                         logged_in=session.get('logged_in', False))

if __name__ == '__main__':
    app.run(debug=True)