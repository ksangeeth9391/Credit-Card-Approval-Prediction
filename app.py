from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load Model, Scaler and Encoders
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")


@app.route("/")
def home():

    dropdowns = {}

    for col, encoder in encoders.items():
        dropdowns[col] = list(encoder.classes_)

    return render_template("index.html", dropdowns=dropdowns)


@app.route("/predict", methods=["POST"])
def predict():

    try:

        gender = request.form["CODE_GENDER"]
        own_car = request.form["FLAG_OWN_CAR"]
        own_realty = request.form["FLAG_OWN_REALTY"]
        children = int(request.form["CNT_CHILDREN"])
        income = float(request.form["AMT_INCOME_TOTAL"])
        income_type = request.form["NAME_INCOME_TYPE"]
        education = request.form["NAME_EDUCATION_TYPE"]
        family = request.form["NAME_FAMILY_STATUS"]
        housing = request.form["NAME_HOUSING_TYPE"]
        age = int(request.form["AGE"])
        employment = int(request.form["EMPLOYMENT_YEARS"])
        mobile = int(request.form["FLAG_MOBIL"])
        work_phone = int(request.form["FLAG_WORK_PHONE"])
        phone = int(request.form["FLAG_PHONE"])
        email = int(request.form["FLAG_EMAIL"])
        occupation = request.form["OCCUPATION_TYPE"]
        family_members = float(request.form["CNT_FAM_MEMBERS"])

        # Encode categorical values
        gender = encoders["CODE_GENDER"].transform([gender])[0]
        own_car = encoders["FLAG_OWN_CAR"].transform([own_car])[0]
        own_realty = encoders["FLAG_OWN_REALTY"].transform([own_realty])[0]
        income_type = encoders["NAME_INCOME_TYPE"].transform([income_type])[0]
        education = encoders["NAME_EDUCATION_TYPE"].transform([education])[0]
        family = encoders["NAME_FAMILY_STATUS"].transform([family])[0]
        housing = encoders["NAME_HOUSING_TYPE"].transform([housing])[0]
        occupation = encoders["OCCUPATION_TYPE"].transform([occupation])[0]

        features = np.array([[
            gender,
            own_car,
            own_realty,
            children,
            income,
            income_type,
            education,
            family,
            housing,
            age,
            employment,
            mobile,
            work_phone,
            phone,
            email,
            occupation,
            family_members
        ]])

        features = scaler.transform(features)

        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "✅ Credit Card Approved"
        else:
            result = "❌ Credit Card Rejected"

        return render_template("result.html", prediction=result)

   except Exception as e:
    return str(e)



if __name__ == "__main__":
    app.run(debug=True)
