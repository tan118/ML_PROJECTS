from flask import Flask, request, jsonify
import joblib
import webbrowser
from flask_cors import CORS
# CREATE APP

app = Flask(__name__)
CORS(app)

# LOAD MODELS

try:
    stress_model = joblib.load("stress_model.pkl")
    prod_model = joblib.load("productivity_model.pkl")
    print("Models loaded successfully!")
except Exception as e:
    print("Error loading models:", e)


# HOME ROUTE

@app.route("/")
def home():
    return "Work Mode Analyzer API is running"


# PREDICTION ROUTE

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Validate input
        required_fields = [
            "work_location",
            "hours",
            "meetings",
            "work_life_balance",
            "isolation",
            "sleep_quality"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Mapping
        work_map = {
            "remote": 0,
            "hybrid": 1,
            "onsite": 2
        }

        sleep_map = {
            "poor": 0,
            "average": 1,
            "good": 2
        }

        stress_labels = {0: "Low", 1: "Medium", 2: "High"}
        prod_labels = {0: "Decreased", 1: "No Change", 2: "Increased"}

        # Clean input
        work_location = data["work_location"].strip().lower()
        sleep_quality = data["sleep_quality"].strip().lower()

        # Convert to model input
        features = [[
            work_map.get(work_location),
            float(data["hours"]),
            float(data["meetings"]),
            float(data["work_life_balance"]),
            float(data["isolation"]),
            sleep_map.get(sleep_quality)
        ]]

        # Check for invalid mapping
        if None in features[0]:
            return jsonify({"error": "Invalid categorical input"}), 400

        # Predict
        stress = stress_model.predict(features)
        productivity = prod_model.predict(features)

        # Return result
        return jsonify({
            "stress": stress_labels[int(stress[0])],
            "productivity": prod_labels[int(productivity[0])]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# RUN SERVER

if __name__ == "__main__":
     webbrowser.open("http://127.0.0.1:5000/")
     app.run(debug=True)


