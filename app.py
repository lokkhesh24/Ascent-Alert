from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json # For Gemini API interaction
import requests # For making HTTP requests to Gemini API

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-super-secret-key-fallback-v3')

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users_v3.db' # New DB name to avoid conflicts
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Database Model (no changes needed from previous version) ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        admin_email = 'admin@example.com'
        if not User.query.filter_by(email=admin_email).first():
            admin = User(username='admin', email=admin_email)
            admin.set_password('password123')
            db.session.add(admin)
            db.session.commit()

# --- Load ML Model and Encoders ---
MODEL_PATH = 'models/'
model = None
scaler = None
le_location = None
le_weather = None
le_road = None

try:
    model = joblib.load(os.path.join(MODEL_PATH, 'model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
    le_location = joblib.load(os.path.join(MODEL_PATH, 'le_location.pkl'))
    le_weather = joblib.load(os.path.join(MODEL_PATH, 'le_weather.pkl'))
    le_road = joblib.load(os.path.join(MODEL_PATH, 'le_road.pkl'))
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: Model/encoder file not found: {e}. AscentAlert predictor will NOT function.")
    print(f"Please ensure 'train_model.py' has been run successfully and all .pkl files are in '{MODEL_PATH}'.")
    # To prevent app crash, but functionality will be impaired.
    from sklearn.preprocessing import LabelEncoder
    if not model: print("model.pkl is missing.")
    if not scaler: print("scaler.pkl is missing.")
    if not le_location: le_location = LabelEncoder(); print("le_location.pkl missing, using dummy.")
    if not le_weather: le_weather = LabelEncoder(); print("le_weather.pkl missing, using dummy.")
    if not le_road: le_road = LabelEncoder(); print("le_road.pkl missing, using dummy.")


# --- Load Dataset for Dashboard and Predictor Options ---
DATASET_PATH = 'ghat_road_traffic_Indian_accidents.csv'
data = pd.DataFrame()
locations_available = ["Unknown Location"]
weather_conditions_available = ["Unknown Weather"]
road_conditions_available = ["Unknown Road Condition"]
vehicles_involved_options = ["1"]
ghat_road_info_static = [ # Static data for home page, replace with actuals or load from CSV
    {"id": 1, "name": "Khardung La Pass (Ladakh)", "latitude": 34.2787, "longitude": 77.6046, "description": "A treacherous pass with steep inclines and hairpin bends, offering stunning Himalayan views.", "image": "road1.jpg"},
    {"id": 2, "name": "Gata Loops (Himachal Pradesh)", "latitude": 33.2068, "longitude": 78.2974, "description": "Known for its narrow lanes and rocky terrain, a challenge for even seasoned drivers.", "image": "road2.jpg"},
    {"id": 3, "name": "Rohtang Pass (Himachal Pradesh)", "latitude": 32.3643, "longitude": 77.2422, "description": "A scenic route through dense forests, prone to fog and slippery conditions.", "image": "road3.jpg"},
    {"id": 4, "name": "Kinnaur Road (Himachal Pradesh)", "latitude": 31.5267, "longitude": 76.9629, "description": "Famous for its high altitude and unpredictable weather, demanding careful navigation.", "image": "road4.jpg"},
    {"id": 5, "name": "Nathula Pass Road (Sikkim)", "latitude": 27.3869, "longitude": 88.8309, "description": "A winding road with sharp curves, surrounded by lush greenery and waterfalls.", "image": "road5.jpg"},
    {"id": 6, "name": "Kallati Ghat Road (Tamil Nadu)", "latitude": 11.4637, "longitude": 76.7028, "description": "A remote stretch with loose gravel, requiring slow and steady driving.", "image": "road6.jpg"},
    {"id": 7, "name": "Munnar Road (Kerala)", "latitude": 10.0889, "longitude": 77.0595, "description": "A popular tourist route with heavy traffic and tight turns, especially during monsoons.", "image": "road7.jpg"},
    {"id": 8, "name": "Kolli Hills Road (Tamil Nadu)", "latitude": 11.2485, "longitude": 78.3386, "description": "A rugged path through rocky cliffs, where visibility can drop suddenly.", "image": "road8.jpg"},
    {"id": 9, "name": "Zoji La Pass (Jammu & Kashmir)", "latitude": 34.2704, "longitude": 75.4657, "description": "A serene road with gentle slopes, but watch out for unexpected livestock crossings.", "image": "road9.jpg"},
    {"id": 10, "name": "Valparai Ghat Road (Tamil Nadu)", "latitude": 10.3269, "longitude": 76.9541, "description": "A steep descent with breathtaking vistas, but notorious for sudden rockslides.", "image": "road10.jpg"}
]


try:
    data = pd.read_csv(DATASET_PATH)
    if not data.empty:
        if le_location and hasattr(le_location, 'fit'):
            le_location.fit(data['Location'].astype(str).unique())
        if le_weather and hasattr(le_weather, 'fit'):
            le_weather.fit(data['Weather Condition'].astype(str).unique())
        if le_road and hasattr(le_road, 'fit'):
            le_road.fit(data['Road Condition'].astype(str).unique())

        locations_available = sorted(data['Location'].astype(str).unique())
        weather_conditions_available = sorted(data['Weather Condition'].astype(str).unique())
        road_conditions_available = sorted(data['Road Condition'].astype(str).unique())
        vehicles_involved_options = sorted(data['Vehicles Involved'].astype(str).unique())
    else:
        print(f"Warning: Dataset file '{DATASET_PATH}' is empty.")
except FileNotFoundError:
    print(f"Error: Dataset file '{DATASET_PATH}' not found.")
except Exception as e:
    print(f"Error loading dataset '{DATASET_PATH}': {e}")


# --- Helper Functions ---
def convert_time_to_hour(time_str): # Changed from convert_time_to_features
    try:
        dt_obj = datetime.strptime(time_str, '%I:%M:%S %p') # Format from predictor.html
        return dt_obj.hour
    except ValueError:
        try: # Fallback for HH:MM from HTML time input
            dt_obj = datetime.strptime(time_str, '%H:%M')
            return dt_obj.hour
        except ValueError:
            return 0 # Default if parsing fails


# --- Gemini API Configuration ---
# IMPORTANT: In a real application, DO NOT hardcode API keys. Use environment variables.
# For Canvas environment, the API key is injected if left as empty string for specific models.
GEMINI_API_KEY = "AIzaSyBwnF1gvVclP4BLWGxLOa04bcAYzQkklm0" # For gemini-2.0-flash, this can be empty in Canvas
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"


# --- Routes ---
@app.route('/loading')
def loading():
    # The loading.html you provided has its own JS for redirection.
    # This route just serves the page.
    return render_template('loading.html', title="Initializing AscentAlert")

@app.route('/')
def index():
    # Logic for showing loading screen only once per session
    if not session.get('loaded_once_v3'): # Use a new session variable
        session['loaded_once_v3'] = True
        return redirect(url_for('loading'))
    return render_template('home.html', title="Home - AscentAlert",
                           logged_in=session.get('logged_in', False),
                           ghat_roads=ghat_road_info_static)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('logged_in'):
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return redirect(url_for('register'))
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title="Register - AscentAlert")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['logged_in'] = True
            session['username'] = user.username
            session['user_id'] = user.id
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'error')
    return render_template('login.html', title="Login - AscentAlert")

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    session.pop('user_id', None)
    session.pop('loaded_once_v3', None) # Reset loading screen flag
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/predictor', methods=['GET', 'POST'])
def predictor():
    if not session.get('logged_in'):
        flash('Please login to use the predictor.', 'warning')
        return redirect(url_for('login'))

    prediction_result = None # Will be a dict: {'text': "...", 'severity': "low/medium/high"}
    if request.method == 'POST':
        if not model or not scaler or not le_location or not le_weather or not le_road:
            flash("Prediction service is critically unavailable. Model or encoders not loaded.", "error")
            # Return with current form data if possible, or just the empty form
            return render_template('predictor.html', title="Safety Predictor", logged_in=session.get('logged_in', False),
                                   locations=locations_available, weather_conditions=weather_conditions_available,
                                   road_conditions=road_conditions_available, vehicles_options=vehicles_involved_options,
                                   prediction_result=None) # Explicitly None
        try:
            time_input = request.form['time'] # This is now HH:MM from <input type="time">
            location_form = request.form['location']
            road_condition_form = request.form['road_condition']
            weather_condition_form = request.form['weather_condition']
            vehicles_involved = int(request.form['vehicles_involved'])

            hour = convert_time_to_hour(time_input)

            # Transform categorical features
            location_encoded = le_location.transform([str(location_form)])[0]
            weather_encoded = le_weather.transform([str(weather_condition_form)])[0]
            road_encoded = le_road.transform([str(road_condition_form)])[0]

            slope_value = data['Slope'].mean() if 'Slope' in data.columns and not data.empty else 0.1
            radius_value = data['Radius'].mean() if 'Radius' in data.columns and not data.empty else 100.0
            if not data.empty and 'Location' in data.columns and 'Slope' in data.columns and 'Radius' in data.columns:
                location_data_df = data[data['Location'] == location_form] # Ensure using original string form
                if not location_data_df.empty:
                    slope_value = location_data_df['Slope'].iloc[0]
                    radius_value = location_data_df['Radius'].iloc[0]

            # Feature order as per train_model.py
            feature_names = ['Location', 'Weather Condition', 'Road Condition', 'Hour', 'Vehicles Involved', 'Slope', 'Radius']
            feature_values = [location_encoded, weather_encoded, road_encoded, hour, vehicles_involved, slope_value, radius_value]
            input_features_df = pd.DataFrame([feature_values], columns=feature_names)

            scaled_features = scaler.transform(input_features_df)
            prediction = model.predict(scaled_features)
            predicted_casualties = int(prediction[0])
            if predicted_casualties >= 0 and predicted_casualties <=2 :
                severity = "low"
                popup_title = "Low Risk"
                popup_message = f"Predicted: Low Risk (Likely 0 Casualties)"
            if predicted_casualties >= 3 and predicted_casualties <=6 :
                severity = "medium"
                popup_title = "Moderate Risk"
                popup_message = f"Predicted: Moderate Risk (Potential for {predicted_casualties} Casualty)"
            elif predicted_casualties >= 7:
                severity = "high"
                popup_title = "High Risk Alert!"
                popup_message = f"Predicted: High Risk (Potential for {predicted_casualties} Casualties)"

            prediction_result = {
                "text": popup_message,
                "severity": severity,
                "title": popup_title,
                "raw_casualties": predicted_casualties # For potential further use
            }
            # Flash message is now handled by the popup on the frontend
            # flash(f"Prediction successful: {popup_message}", "success_prediction")

        except ValueError as ve:
             flash(f"Input Error: {str(ve)}. One of the selected values might not be recognized by the model.", "error")
             print(f"Predictor ValueError: {ve}")
        except Exception as e:
            flash(f"Error during prediction: {str(e)}. Please check your inputs.", "error")
            print(f"Predictor Exception: {e}")

    return render_template('predictor.html', title="Safety Predictor - AscentAlert", logged_in=session.get('logged_in', False),
                           locations=locations_available, weather_conditions=weather_conditions_available,
                           road_conditions=road_conditions_available, vehicles_options=vehicles_involved_options,
                           prediction_result=prediction_result) # Pass the whole dict

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        flash('Please login to view the dashboard.', 'warning')
        return redirect(url_for('login'))
    
    if data.empty:
        flash('Dashboard data is unavailable. Dataset could not be loaded.', 'error')
        return render_template('dashboard.html', title="Dashboard - AscentAlert", logged_in=session.get('logged_in', False), charts_data={})

    charts_data = {}
    try:
        if 'Time' in data.columns and 'Hour' not in data.columns:
             data['Hour'] = data['Time'].apply(lambda x: convert_time_to_hour(str(x)))

        if 'Location' in data.columns and 'Casualties' in data.columns:
            casualties_by_loc = data.groupby('Location')['Casualties'].sum().sort_values(ascending=False).reset_index()
            charts_data['casualties_by_location'] = {"labels": casualties_by_loc['Location'].tolist(), "values": casualties_by_loc['Casualties'].tolist()}
        if 'Hour' in data.columns:
            accidents_by_hour = data['Hour'].value_counts().sort_index().reset_index(); accidents_by_hour.columns = ['Hour', 'Count']
            charts_data['accidents_by_hour'] = {"labels": accidents_by_hour['Hour'].tolist(), "values": accidents_by_hour['Count'].tolist()}
        if 'Weather Condition' in data.columns:
            accidents_by_weather = data['Weather Condition'].value_counts().reset_index(); accidents_by_weather.columns = ['Weather', 'Count']
            charts_data['accidents_by_weather'] = {"labels": accidents_by_weather['Weather'].tolist(), "values": accidents_by_weather['Count'].tolist()}
        if 'Road Condition' in data.columns:
            accidents_by_road = data['Road Condition'].value_counts().reset_index(); accidents_by_road.columns = ['RoadCondition', 'Count']
            charts_data['accidents_by_road'] = {"labels": accidents_by_road['RoadCondition'].tolist(), "values": accidents_by_road['Count'].tolist()}
        if 'Vehicles Involved' in data.columns and 'Casualties' in data.columns:
            casualties_by_vehicles = data.groupby('Vehicles Involved')['Casualties'].sum().reset_index()
            charts_data['casualties_by_vehicles'] = {"labels": casualties_by_vehicles['Vehicles Involved'].astype(str).tolist(), "values": casualties_by_vehicles['Casualties'].tolist()}
    except Exception as e:
        flash(f"Error generating dashboard data: {str(e)}", "error")
        print(f"Dashboard error: {e}")
    return render_template('dashboard.html', title="Dashboard - AscentAlert", logged_in=session.get('logged_in', False), charts_data=charts_data)

@app.route('/about')
def about():
    team_members = [
        {"name": "Alex Chen", "role": "Lead Developer & AI Specialist", "bio": "Alex is passionate about leveraging AI for social good, with a focus on safety and predictive analytics. He designed the core prediction engine for AscentAlert.", "image": "alex.jpg"},
        {"name": "Maria Garcia", "role": "UX/UI Designer & Frontend Developer", "bio": "Maria crafted the user experience of AscentAlert, ensuring an intuitive and impactful interface. She believes in design that empowers users.", "image": "maria.jpg"},
        {"name": "Samira Khan", "role": "Data Scientist & Backend Engineer", "bio": "Samira was instrumental in data analysis, model refinement, and building the robust backend infrastructure that powers AscentAlert.", "image": "samira.jpg"}
    ]
    return render_template('about.html', title="About Us - AscentAlert", logged_in=session.get('logged_in', False), team_members=team_members)

@app.route('/ai_chat_interactive') # New route for the AI chat page
def ai_chat_interactive():
    if not session.get('logged_in'):
        flash('Please login to use the AI Chat.', 'warning')
        return redirect(url_for('login'))
    return render_template('ai_chat.html', title="AI Safety Chat - AscentAlert", logged_in=session.get('logged_in', False))


@app.route('/ask_gemini', methods=['POST'])
def ask_gemini():
    if not session.get('logged_in'):
        return jsonify({"error": "User not logged in"}), 401

    user_prompt = request.json.get('prompt')
    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Constructing the payload for Gemini API
    # Prepend a context to guide the AI for safety-related queries
    contextual_prompt = (
        "You are a helpful AI assistant for 'AscentAlert', a ghat road safety prediction application. "
        "Provide concise and relevant information related to road safety, ghat road conditions, "
        "safe driving practices, or interpreting potential accident risks. "
        "If the query is unrelated to these topics, politely state that you are specialized in road safety. "
        "User query: " + user_prompt
    )

    payload = {
        "contents": [{
            "parts": [{"text": contextual_prompt}]
        }]
    }
    headers = {
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=20) # Added timeout
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        result = response.json()

        if (result.get('candidates') and
                result['candidates'][0].get('content') and
                result['candidates'][0]['content'].get('parts') and
                result['candidates'][0]['content']['parts'][0].get('text')):
            ai_response = result['candidates'][0]['content']['parts'][0]['text']
            return jsonify({"response": ai_response})
        else:
            # Log the unexpected response for debugging
            print(f"Unexpected Gemini API response structure: {result}")
            # Check for promptFeedback if candidates are missing
            if result.get('promptFeedback') and result['promptFeedback'].get('blockReason'):
                reason = result['promptFeedback']['blockReason']
                error_message = f"AI response blocked due to: {reason}. Please rephrase your query."
                return jsonify({"error": error_message}), 503 # Service Unavailable or custom code
            return jsonify({"error": "AI could not generate a response or response format was unexpected."}), 500

    except requests.exceptions.HTTPError as http_err:
        print(f"Gemini API HTTP error: {http_err} - Response: {response.text}")
        return jsonify({"error": f"AI service error: {http_err}"}), response.status_code
    except requests.exceptions.RequestException as req_err: # Catches network errors, timeout, etc.
        print(f"Gemini API Request error: {req_err}")
        return jsonify({"error": f"Could not connect to AI service: {req_err}"}), 503
    except Exception as e:
        print(f"Error processing Gemini request: {e}")
        return jsonify({"error": "An unexpected error occurred while contacting the AI service."}), 500


if __name__ == '__main__':
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs('static/videos', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    os.makedirs('static/images/team', exist_ok=True)
    os.makedirs('static/images/ghat_roads', exist_ok=True) # For ghat road images
    app.run(debug=True, port=5001)
