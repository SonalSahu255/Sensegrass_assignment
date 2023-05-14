from flask import Flask, render_template, url_for, flash, redirect
import joblib
from flask import request
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__, template_folder='template')


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/top_5")
def top_5():
    return render_template("top_5.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route('/wine_predict', methods = ["POST"])
def wine_predict():
    if request.method == "POST":
        country=request.form["country"]
        rating=request.form["rating"]
        price=request.form["price"]
        Sentiment=request.form["Sentiment"]

        countries = {'Argentina': 0, 'Australia': 1, 'Brazil': 2, 'Bulgaria': 3, 'Canada': 4, 'Chile': 5, 'Croatia': 6, 'Cyprus': 7,
     'Czech Republic': 8, 'England': 9, 'France': 10, 'Georgia': 11, 'Germany': 12, 'Greece': 13, 'Hungary': 14, 
     'Israel': 15, 'Italy': 16, 'Lebanon': 17, 'Macedonia': 18, 'Mexico': 19, 'Moldova': 20, 'Morocco': 21, 'New Zealand': 22,
     'Peru': 23, 'Portugal': 24, 'Romania': 25, 'Serbia': 26, 'Slovenia': 27, 'South Africa': 28, 'Spain': 29, 
     'Switzerland': 30, 'Turkey': 31, 'US': 32, 'Ukraine': 33, 'Uruguay': 34}
        
        if country in countries:
            country=countries[country]
        else:
            country= 0
                
        data = [int(country),int(rating),int(price),int(Sentiment)]
        data=np.array(data).reshape(1,-1)

        wine_loaded_model = joblib.load(r'D:\PROJECTS\Sensegrass Assignment\models\model.pkl')

        result = wine_loaded_model.predict(data)
        
        if result==0:
            prediction = "the Wine Variety is: Cabernet Sauvignon"
        elif result==1:
            prediction ="the Wine Variety is: Chardonnay"
        elif result==2:
            prediction ="the Wine Variety is: Pinot Noir"
        else:
            prediction ="the Wine Variety is: Red Blend"


    return(render_template("predict.html", prediction=prediction ))

@app.route('/review_wine_predict', methods = ["POST"])
def review_wine_predict():
    if request.method == "POST":
        review=request.form["review"]

        data=np.array([review],dtype="O")

        #it's taken long time
        vectorizer = TfidfVectorizer()
        data = vectorizer.fit_transform(data)
        
        review_loaded_model = joblib.load(r'D:\PROJECTS\Sensegrass Assignment\models\TfidfVectorizer_model.pkl')

        result = review_loaded_model.predict(data)
        
        

    return(render_template("predict.html", prediction_text=result))


if __name__ == "__main__" :
    app.run(debug=True,port=9000)