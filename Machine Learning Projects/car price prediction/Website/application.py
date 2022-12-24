from flask import Flask, render_template, request
import pandas as pd
import pickle
app = Flask(__name__)
model = pickle.load(open("LinearRegression.pkl", 'rb'))
# import clean data
car = pd.read_csv(
    "E:\Mechine learning\Projects\car price prediction\Website\clean car.csv")


@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_model = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    # kms_driven = car['kms_driven'].unique()
    return render_template('index.html', companies=companies,
                           car_model=car_model,
                           year=year, fuel_type=fuel_type)


@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('companies')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kms_driven'))
    prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]], columns=[
                               'name', 'company', 'year', 'kms_driven', 'fuel_type']))
    return str(prediction[0])


if __name__ == "__main__":
    app.run(debug=True)
