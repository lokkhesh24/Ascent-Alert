# Ascent Alert

A Flask-based web application to explore ghat roads in India and predict accident severity based on Time, Location, Weather Condition, Road Condition, and Vehicles Involved.

## Features
- **Loading Page**: Displays an animated speedometer and driving tips without a background box, redirecting to the login page.
- **Login Page**: Transparent glassy login page with visible text, requiring authentication, with no top navigation displayed. Includes a link to the registration page.
- **Register Page**: Transparent glassy registration page styled similarly to the login page, allowing new users to sign up with a username and password.
- **Homepage**: Displays 10 unique ghat roads with flat-style buttons, centered text, and clickable Google search links using shortened location names (e.g., "Khar" for "Khardung La Pass, Ladakh"). Top right buttons are padded more to the right for visibility.
- **Safety Predictor**: Features a Bento box grid with properly aligned dropdowns and inputs, using a gradient background, and includes an embedded Google Maps iframe for visualization. Location dropdown shows shortened names.
- **Dashboard Page**: Visualizes dataset insights with larger charts (Casualties by Ghat Road, Accidents by Time of Day, Cause vs Road Condition vs Ghat Road, and Road Condition vs Weather vs Ghat Road) using shortened location names, and allows PDF export.
- **Result Page**: Centered "Try Another Prediction" button with error handling, now includes dynamically calculated slope and radius based on location coordinates.
- **Responsive Design**: Works on desktop and mobile devices.
- **Database Storage**: Uses SQLite with Flask-SQLAlchemy to store user information securely, with hashed passwords.
- **Advanced ML Models**: Uses a variety of machine learning models (Random Forest, XGBoost, CatBoost, SVM, and Neural Network) to predict accident severity, selecting the best model based on cross-validation accuracy.

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- The dataset file `ghat_road_traffic_Indian_accidents.csv` in the project root directory
- Placeholder images (`road1.jpg` to `road10.jpg`) in `static/images/`
- The font file `PlayfairDisplay-Regular.ttf` in `static/fonts/`
- A lock icon image (`lock-icon.png`) in `static/images/`

## Setup Instructions
1. **Clone the Repository or Create the Project Structure**
   - Ensure all files are in the correct structure as described above.
   - Place the `ghat_road_traffic_Indian_accidents.csv` file in the project root directory.
   - Add 10 images (`road1.jpg` to `road10.jpg`) to `static/images/` (see "Adding Images" below).
   - Download the `Playfair Display` font from [Google Fonts](https://fonts.google.com/specimen/Playfair+Display) and place `PlayfairDisplay-Regular.ttf` in `static/fonts/`.
   - Download a lock icon (e.g., from [Flaticon](https://www.flaticon.com/)) and save it as `lock-icon.png` in `static/images/`.

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the Database**
   - The application uses SQLite to store user data. The database (`users.db`) will be created automatically when you run the application for the first time.
   - A default admin user (`admin`/`password123`) will be created on first run.

5. **Train the Model**
   - Run the training script to generate the model and encoder files in the `models/` directory.
   ```bash
   python train_model.py
   ```
   - This will evaluate multiple models (Random Forest, XGBoost, CatBoost, SVM, and Neural Network), select the best one based on cross-validation accuracy, and save it as `model.pkl` along with `scaler.pkl`, `le_location.pkl`, `le_weather.pkl`, and `le_road.pkl`.

6. **Run the Flask Application**
   ```bash
   python app.py
   ```
   - The application will start on `http://127.0.0.1:5000`.

7. **Access the Web Application**
   - Open a web browser and navigate to `http://127.0.0.1:5000`.
   - View the loading page with a speedometer animation and driving tips, which redirects to the login page.
   - Log in with username: `admin`, password: `password123`, or register a new user via the "Sign Up" link.
   - Explore the homepage, use the Safety Predictor, or view the Dashboard page with dataset visualizations.

## Adding Images for Ghat Roads
- The homepage displays 10 ghat roads, each with an image (`road1.jpg` to `road10.jpg`) in `static/images/`.
- **Steps to Add Images**:
  1. Identify the 10 unique locations from the dataset (first 10 from `data[['Location']].drop_duplicates().head(10)`).
  2. Source images for each location:
     - Use free stock photo sites like Unsplash or Pexels to find mountain road images.
     - Alternatively, take photos or use a tool like Canva to create placeholders.
  3. Name the images `road1.jpg` to `road10.jpg` to match the order of locations in `app.py`.
  4. Place the images in `static/images/`.
- **Example**:
  - If the first location is "Khardung La Pass, Ladakh", find an image of that road, rename it `road1.jpg`, and place it in `static/images/`.
- **Fallback**:
  - If specific images are unavailable, use generic mountain road images or a single image copied as `road1.jpg` to `road10.jpg` for testing.

## Input Fields (Safety Predictor)
- **Time**: Enter time in the format `H:MM:SS AM/PM` (e.g., `2:03:00 AM`). Required field.
- **Location**: Select a location from the dropdown (displays shortened names, e.g., "Khar"). Required field.
- **Weather Condition**: Select a weather condition from the dropdown. Required field.
- **Road Condition**: Select a road condition from the dropdown. Required field.
- **Vehicles Involved**: Enter a number between 1 and 5. Required field.

## Output
- The predictor returns the accident severity:
  - **Low (0-2 Casualties)**
  - **Medium (3-6 Casualties)**
  - **High (7+ Casualties)**
- Additionally, it displays the dynamically calculated slope (in degrees) and radius (in meters) based on the selected location's latitude and longitude.

## Dashboard Features
- **Casualties by Ghat Road**: Bar chart showing the total casualties per ghat road (using shortened names, e.g., "Khar").
- **Accidents by Time of Day**: Pie chart categorizing accidents into Night, Morning, Afternoon, and Evening.
- **Cause vs Road Condition vs Ghat Road**: Multiple bar chart showing the number of accidents by cause and road condition across ghat roads (using shortened names).
- **Road Condition vs Weather vs Ghat Road**: Multiple bar chart showing the number of accidents by road condition and weather across ghat roads (using shortened names).
- **Export to PDF**: Button to export the dashboard charts as a PDF file using jsPDF and html2canvas, adjusted for larger chart sizes.

## Notes
- Ensure the dataset file, images, font file, and lock icon are present before running the application.
- The "High" for all inputs issue is addressed by introducing random variability in time and vehicles, though retraining with balanced data would be ideal for a production model.
- Slope and radius are now dynamically calculated using latitude and longitude, constrained within realistic ranges (5-20 degrees for slope, 20-100 meters for radius).
- The login and registration system now uses a SQLite database with hashed passwords for improved security.
- The loading page speedometer and tips are now displayed without a background box for a cleaner look.
- The dashboard page requires the dataset to have columns like `Location`, `Casualties`, `Time`, `Road Condition`, `Weather Condition`, and optionally `Cause`. If `Cause` is missing, the app simulates it for demonstration.
- The model is trained using a variety of advanced ML models (Random Forest, XGBoost, CatBoost, SVM, and Neural Network). The best model is selected based on cross-validation accuracy.
- Ghat road names are shortened to the first word (e.g., "Khar" for "Khardung La Pass, Ladakh") across the application for consistency.
- Dashboard charts are now larger (800px wide) for better visibility.
- If you encounter errors, check the console output for details.