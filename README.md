# 🌲 Forest Cover Type Prediction

## Project Overview
A machine learning web application that predicts the type of forest cover based on cartographic variables such as elevation, soil type, wilderness area, and hillshade. The model classifies a given land patch into one of seven forest cover types using a trained Random Forest classifier, and presents the result through an interactive web interface with real forest imagery.

---

## Problem Statement
Determining forest cover type traditionally requires extensive field surveys, which are time-consuming and costly. This project automates that process by training a machine learning model on cartographic data — information that can be derived remotely — to accurately predict the dominant tree species in any given area without physical inspection.

---

## Dataset
- **Source:** UCI Machine Learning Repository — Covertype Dataset
- **Size:** 15,120 training samples, 55 features
- **Target:** 7 forest cover types
  - Type 1 — Spruce / Fir
  - Type 2 — Lodgepole Pine
  - Type 3 — Ponderosa Pine
  - Type 4 — Cottonwood / Willow
  - Type 5 — Aspen
  - Type 6 — Douglas-fir
  - Type 7 — Krummholz

---

## Features Used
The dataset contains 54 input features across three categories:

- **Numerical (10):** Elevation, Aspect, Slope, Horizontal/Vertical Distance to Hydrology, Horizontal Distance to Roadways, Hillshade at 9am/Noon/3pm, Horizontal Distance to Fire Points
- **Binary — Wilderness Area (4):** One-hot encoded wilderness zone indicators
- **Binary — Soil Type (40):** One-hot encoded soil type indicators

---

## Machine Learning Pipeline
1. **Exploratory Data Analysis** — distribution plots, correlation matrix, class balance check
2. **Preprocessing** — feature scaling, handling any missing values
3. **Model Training** — Random Forest Classifier with hyperparameter tuning via RandomizedSearchCV
4. **Evaluation** — Accuracy score, cross-validation with StratifiedKFold
5. **Model Export** — saved as `model.pkl` using Python's pickle library

**Final Model Accuracy: ~86%**

---

## Web Application
Built with **Flask** (backend) and **HTML/CSS/JavaScript** (frontend).

**How it works:**
- User pastes all 54 feature values as a single comma-separated list
- The frontend parses and previews each value mapped to its feature name
- On prediction, Flask loads the model, runs inference, and returns the result
- The UI displays the predicted forest type with a real photograph, description, elevation range, climate type, and model confidence score

---

## Tech Stack
| Layer | Technology |
|---|---|
| Machine Learning | scikit-learn, Random Forest |
| Data Analysis | pandas, numpy|
| Backend | Python, Flask |
| Frontend | HTML, CSS, JavaScript |
| Model Serialization | pickle |
| Deployment | Render (via gunicorn) |
| Development Environment | Google Colab, VS Code |

---

## Key Learnings
- Handled a high-dimensional binary feature space (44 one-hot encoded columns) without dropping features, as Random Forest manages irrelevant features internally
- Explored multiple models (XGBoost, LightGBM, Extra Trees) and found Random Forest to be the best performer on this dataset
- Built and deployed a full end-to-end ML pipeline from raw data to a live web application
- Gained hands-on experience with Flask API development, virtual environments, and cloud deployment on Render

---

## How to Run Locally
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/forest-cover-prediction

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py

# 5. Open in browser
http://127.0.0.1:5000
```

---

## Project Structure
```
forest_cover_app/
│
├── app.py                  ← Flask backend & prediction API
├── model.pkl               ← Trained Random Forest model
├── requirements.txt        ← Python dependencies
└── templates/
      └── index.html        ← Frontend web interface
```
