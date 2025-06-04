# Ascent Alert

A Flask-based web application to explore ghat roads in India and predict accident severity based on Time, Location, Weather Condition, Road Condition, and Vehicles Involved.

## Features
- **Loading Page**: Displays an animated speedometer and driving tips without a background box, redirecting to the login page.
- **Login Page**: Transparent glassy login page with visible text, requiring authentication, with no top navigation displayed. Includes a link to the registration page.
- **Register Page**: Transparent glassy registration page styled similarly to the login page, allowing new users to sign up with a username and password.
- **Homepage**: Displays 10 unique ghat roads with flat-style buttons, centered text, and clickable Google search links using shortened location names (e.g., "Khar" for "Khardung La Pass, Ladakh"). Top right buttons are padded more to the right for visibility.
- **Safety Predictor**: Features a Bento box grid with properly aligned dropdowns and inputs, using a gradient background, and includes a downloadable KML file to view ghat road locations in 3D using Google Earth. Supports voice input for Location, Weather Condition, and Road Condition fields using the Web Speech API (best supported in Chrome and Edge). Includes an embedded Google Map and a chat dialog box to ask questions about ghat roads, safety, or the predictor tool, with voice input support.
- **Dashboard Page**: Visualizes dataset insights with larger charts (Casualties by Ghat Road, Accidents by Time of Day, Cause vs Road Condition vs Ghat Road, and Road Condition vs Weather vs Ghat Road) using shortened location names, and allows PDF export.
- **Result Page**: Centered "Try Another Prediction" button with error handling, now includes dynamically calculated slope and radius based on location coordinates.
- **About Us Page**: Introduces the three-member team behind Ascent Alert with a carousel (thumbnail navigation linking to LinkedIn profiles), tabbed content for team member details, a parallax background, and fade-in/fade-out animations. Includes a footer with colorful social media links (X, Instagram, Facebook, GitHub).
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
- Team member images (`member1.jpg`, `member2.jpg`, `member3.jpg`) in `static/images/team/`
- A mountain road image (`mountain_road.jpg`) in `static/images/team/` for the parallax background
- Social media icons (`x.png`, `instagram.png`, `facebook.png`, `github.png`) in `static/images/social/`

## Setup Instructions
1. **Clone the Repository or Create the Project Structure**
   - Ensure all files are in the correct structure as described above.
   - Place the `ghat_road_traffic_Indian_accidents.csv` file in the project root directory.
   - Add