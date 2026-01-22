from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model, scaler = joblib.load("model/wine_cultivar_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        features = [
            float(request.form["alcohol"]),
            float(request.form["malic_acid"]),
            float(request.form["alcalinity_of_ash"]),
            float(request.form["magnesium"]),
            float(request.form["flavanoids"]),
            float(request.form["color_intensity"])
        ]

        features_scaled = scaler.transform([features])
        pred_class = model.predict(features_scaled)[0]
        prediction = f"Cultivar {pred_class + 1}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
