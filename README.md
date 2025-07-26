# ETA-Predictor


## 📌 Problem Statement

Given trip data (`train.csv`) on Uber/Lyft ride origin-destination (OD) locations, start time, and durations, the goal is to **predict the duration of trips** between two locations at a specific departure time using machine learning models like Support Vector Regression (SVR), Random Forest, XGBoost, and Feedforward Neural Networks (FFNN).

The performance of each model is measured using **Root Mean Squared Error (RMSE)** on the test dataset. This problem was originally featured in a [Kaggle competition](https://www.kaggle.com/c/ce263n-hw4) where participants were evaluated on public test data comprising 20% of the total test set.

---

## 📂 Datasets

### 🔹 Training Set (`train.csv`)

Contains the following fields:


- **start_lng/start_lat**: Start location in WGS84 coordinates  
- **end_lng/end_lat**: End location in WGS84 coordinates  
- **datetime**: Local start time of the trip  
- **duration**: Target variable (trip duration in seconds)

---

### 🔹 Test Set (`test.csv`)

Contains the same structure except the target duration:


You are expected to **predict the duration** for each row using the trained model.

---

## 🔍 Solution Strategy

### ✅ Step 1: Exploratory Data Analysis

- **Data Cleaning**:
  - Dropped rows with missing coordinates
  - Corrected invalid positive longitude values
  - Removed trips ending in the ocean (e.g., `end_lng ≈ -50`)
  - Dropped rows with:
    - `duration == 0` unless start & end locations were identical
    - Duration > 20,000 seconds (~6 hours)
    - Duration capped at 40,000 seconds (~11 hours)

- **Regions Identified**:
  - San Francisco (2012)
  - New York (2015)

---

### ⚙️ Step 2: Feature Engineering

1. **Distance Metrics**:
   - **Manhattan Distance** (`|lat1 - lat2| + |lng1 - lng2|`)
   - **Haversine Distance**
   - **Google Maps Distance/Duration** via GCP API (with fallback for Treasure Island routes)

2. **Time-Based Features**:
   - Extracted **weekday**, **hour**, and **holiday** flags from `datetime`

3. **Geolocation Clustering**:
   - Used **DBSCAN** to categorize coordinates into:
     - `'citycenter'`
     - `'airport'` (JFK, LaGuardia, SFO)
     - `'standalone'`  
   - One-hot encoded for model input

4. **Weather Integration**:
   - Added historical **precipitation data** from **NOAA** for both SF and NY

5. **Routing Quality Flags**:
   - `short_trip`: Small manhattan & gmaps distances  
   - `routing_error`: Small manhattan, large gmaps distance  

---

### 🤖 Step 3: Model Training & Evaluation

Tested models:

- Linear Regression  
- Ridge, Lasso  
- SVM Regressor  
- Random Forest  
- **XGBoost** ✅ (Best)  
- FFNN  
- (Stacked model attempted, underperformed)

**Best Model:**  
```python
XGBoost(
  max_depth = 9,
  learning_rate = 0.045,
  n_estimators = 500,
  reg_lambda = 0.5
)
Kaggle Public Test RMSE: 287.02604
Real-time Use Case
This model can be used in real-time for:

Fare estimation

Traffic forecasting

Fleet sizing

SLA compliance in ride-sharing platforms

Inputs required:

Start & End Coordinates

Trip Start Time

Weather Conditions

Once inputted, the model provides an accurate duration estimate, improving decisions across pricing, routing, and customer experience.

 Real-time Use Case
This model can be used in real-time for:

Fare estimation

Traffic forecasting

Fleet sizing

SLA compliance in ride-sharing platforms

Inputs required:

Start & End Coordinates

Trip Start Time

Weather Conditions

Once inputted, the model provides an accurate duration estimate, improving decisions across pricing, routing, and customer experience.

🧠 Tools & Libraries Used
Python 3.10

numpy, pandas, matplotlib, seaborn

scikit-learn, xgboost

keras, tensorflow

Google Maps API

NOAA Climate Data
Project Structure
markdown
Copy
Edit
├── dataset/
│   ├── train.csv
│   └── test.csv
├── gmaps/
│   └── *.html
├── img/
│   └── *.png
├── output/
│   └── *.csv, *.pkl
├── trip-duration-prediction.ipynb
├── trip-duration-prediction.py
├── README.md
✍️ Authored by Amrutha Vamshi Goud
🔗 [LinkedIn](https://www.linkedin.com/in/vamshi-a-b5b7692b9)  
🔗 [GitHub](https://github.com/vamsi9090)
