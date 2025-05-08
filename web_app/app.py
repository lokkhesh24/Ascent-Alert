from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sqlite3
import bcrypt

app = Flask(__name__)
app.secret_key = 'your_secret_key'
model = joblib.load("C:/Users/lokkh/Downloads/Test Project/model/model.pkl")

# SQLite database setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT,
        email TEXT,
        reset_token TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return redirect(url_for('loading'))

@app.route('/loading')
def loading():
    return render_template('loading.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        if user and bcrypt.checkpw(password, user[0].encode('utf-8')):
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('home'))
        return render_template('login.html', error="Invalid credentials", hide_header=True)
    return render_template('login.html', error=None, hide_header=True)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')
        email = request.form['email']
        hashed = bcrypt.hashpw(password, bcrypt.gensalt())
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)", 
                      (username, hashed.decode('utf-8'), email))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error="Username already exists", hide_header=True)
    return render_template('register.html', error=None, hide_header=True)

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT username FROM users WHERE email = ?", (email,))
        user = c.fetchone()
        if user:
            reset_token = "temp_reset_token_" + user[0]
            c.execute("UPDATE users SET reset_token = ? WHERE email = ?", (reset_token, email))
            conn.commit()
            return render_template('reset_password.html', token=reset_token, error=None, hide_header=True)
        conn.close()
        return render_template('forgot_password.html', error="Email not found", hide_header=True)
    return render_template('forgot_password.html', error=None, hide_header=True)

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if request.method == 'POST':
        new_password = request.form['new_password'].encode('utf-8')
        hashed = bcrypt.hashpw(new_password, bcrypt.gensalt())
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("UPDATE users SET password = ?, reset_token = NULL WHERE reset_token = ?", 
                  (hashed.decode('utf-8'), token))
        conn.commit()
        conn.close()
        return redirect(url_for('login'))
    return render_template('reset_password.html', token=token, error=None, hide_header=True)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

# Protected routes
def login_required(f):
    def wrap(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route('/predictor', methods=['GET', 'POST'])
@login_required
def predictor():
    if request.method == 'POST':
        input_data = {
            'Weather': request.form['weather'],
            'Road_Type': 'Kolli Ghat Road',
            'Time_of_Day': request.form['time'],
            'Traffic_Density': int(request.form['traffic']),
            'Speed_Limit': float(request.form['speed']),
            'Number_of_Vehicles': 5,
            'Driver_Alcohol': int(request.form['alcohol']),
            'Road_Condition': request.form['road'],
            'Vehicle_Type': request.form['vehicle'],
            'Driver_Age': int(request.form['age']),
            'Driver_Experience': int(request.form['experience']),
            'Road_Light_Condition': request.form['light']
        }
        df = pd.DataFrame([input_data])
        for col in ['Weather', 'Road_Condition', 'Time_of_Day', 'Vehicle_Type', 'Road_Light_Condition', 'Road_Type']:
            df[col] = LabelEncoder().fit_transform(df[col])
        prob = model.predict_proba(df)[0][1]
        risk = "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"
        notification = None
        if prob > 0.7:
            notification = f"CAUTION: High accident risk! Slow down to 20 km/h on sharp turns."
        elif prob > 0.3:
            notification = f"MENACE: Moderate risk. Drive cautiously on this road."
        return render_template('predictor.html', prediction=f"Accident Risk: {prob*100:.1f}% ({risk})", prob=prob, notification=notification)
    return render_template('predictor.html', prediction=None)

@app.route('/dashboard')
@login_required
def dashboard():
    data = pd.read_csv("../dataset/kolli_ghat_data.csv", encoding='latin1')
    data_json = data.to_dict(orient='records')
    weather_accidents = data.groupby('Weather')['Accident'].sum().to_dict()
    time_accidents = data.groupby('Time_of_Day')['Accident'].sum().to_dict()
    # Multiple bar graph: Alcohol vs Accident vs Driver Age
    alcohol_accident_age = data.groupby(['Driver_Alcohol', 'Accident']).agg({
        'Driver_Age': ['mean', 'count']
    }).reset_index()
    # Flatten the multi-level column index
    alcohol_accident_age.columns = ['Driver_Alcohol', 'Accident', 'Driver_Age_mean', 'Driver_Age_count']
    alcohol_accident_age_dict = alcohol_accident_age.to_dict(orient='records')
    vehicle_accidents = data.groupby('Vehicle_Type')['Accident'].sum().to_dict()
    road_condition_accidents = data.groupby('Road_Condition')['Accident'].sum().to_dict()
    return render_template('dashboard.html', data=data_json, weather_accidents=weather_accidents, 
                          time_accidents=time_accidents, alcohol_accident_age=alcohol_accident_age_dict,
                          vehicle_accidents=vehicle_accidents, road_condition_accidents=road_condition_accidents)

if __name__ == '__main__':
    app.run(debug=True)