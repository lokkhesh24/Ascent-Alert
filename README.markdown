# Ascent Alert

A Flask-based web application to explore ghat roads in India and predict accident severity based on Time, Location, Weather Condition, Road Condition, and Vehicles Involved.

## Features
- **Login System**: Transparent glassy login page with visible text, requiring authentication, with no top navigation displayed.
- **Homepage**: Displays 10 unique ghat roads with flat-style buttons, centered text, and clickable Google search links. Top right buttons are padded more to the right for visibility.
- **Safety Predictor**: Features a Bento box grid with properly aligned dropdowns and inputs, using a gradient background, and includes an embedded Google Maps iframe for visualization.
- **Result Page**: Centered "Try Another Prediction" button with error handling, now includes dynamically calculated slope and radius based on location coordinates.
- **Stylish Design**: Flat buttons with gradient backgrounds using colors like cyan (#00c6ff), purple (#d0aaff), and orange (#ff8c42).
- **Responsive Design**: Works on desktop and mobile devices.

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

4. **Train the Model**
   - Run the training script to generate the model and encoder files in the `models/` directory.
   ```bash
   python train_model.py
   ```
   - This will create `model.pkl`, `scaler.pkl`, `le_location.pkl`, `le_weather.pkl`, and `le_road.pkl`.

5. **Run the Flask Application**
   ```bash
   python app.py
   ```
   - The application will start on `http://127.0.0.1:5000`.

6. **Access the Web Application**
   - Open a web browser and navigate to `http://127.0.0.1:5000`.
   - Log in with username: `admin`, password: `password123` (or `user1`/`pass456`).
   - Explore the homepage and use the Safety Predictor, which now includes a Google Maps embed.

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
- **Location**: Select a location from the dropdown. Required field.
- **Weather Condition**: Select a weather condition from the dropdown. Required field.
- **Road Condition**: Select a road condition from the dropdown. Required field.
- **Vehicles Involved**: Enter a number between 1 and 5. Required field.

## Output
- The predictor returns the accident severity:
  - **Low (0-2 Casualties)**
  - **Medium (3-6 Casualties)**
  - **High (7+ Casualties)**
- Additionally, it displays the dynamically calculated slope (in degrees) and radius (in meters) based on the selected location's latitude and longitude.

## Notes
- Ensure the dataset file, images, font file, and lock icon are present before running the application.
- The "High" for all inputs issue is addressed by introducing random variability in time and vehicles, though retraining with balanced data would be ideal for a production model.
- Slope and radius are now dynamically calculated using latitude and longitude, constrained within realistic ranges (5-20 degrees for slope, 20-100 meters for radius).
- The login system uses a JSON file for demo purposes. In production, use a database and secure authentication.
- The Safety Predictor now includes a Google Maps iframe below the predict button for enhanced visualization.
- The model is trained using a Random Forest Classifier on the provided dataset.
- If you encounter errors, check the console output for details.