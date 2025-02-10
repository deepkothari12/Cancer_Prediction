from flask import Flask , redirect , render_template , url_for , request
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

df = pd.read_csv("thyroid_cancer_risk_data.csv")

with open("Ohe-encoder.pkl" , 'rb') as File:
    Ohe = pickle.load(File)

with open("DT_model.pkl" , 'rb') as file:
    dt = pickle.load(file)

with open("RandomForestmodel.pkl" , 'rb') as fil:
    rc = pickle.load(fil)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/action" , methods = ['POST'])
def action():
    x = df.drop(columns= 'Thyroid_Cancer_Risk' , axis=1)
    y = df['Thyroid_Cancer_Risk']
    X_train , x_test , y_train , y_test = train_test_split(x ,y, test_size=0.2 , random_state=42)

    age = int(str(request.form.get("age")))##fetch name
    gender = int(str(request.form.get("gender")))
    country = str(request.form.get("country"))
    Ethnicity = str(request.form.get("Ethnicity"))
    Family_History = int(str(request.form.get("Family_History")))
    Radiation_Exposure = int(str(request.form.get("Radiation_Exposure")))
    Iodine_Deficiency = int(str(request.form.get("Iodine_Deficiency")))
    Smoking = int(str(request.form.get("Smoking")))
    Obesity = int(str(request.form.get("Obesity")))
    Diabetes = int(str(request.form.get("Diabetes") ))
    TSH_Level = int(str(request.form.get("TSH_Level") ))
    T3_Level = int(str(request.form.get("T3_Level") ))
    T4_Level = int(str(request.form.get("T4_Level") ))
    Nodule_Size = int(str(request.form.get("Nodule_Size")))
    #print("Age ---->>>" , gender)

    ##transform the Data into Encoded form
    one_transfer_encoded = Ohe.transform([[country , Ethnicity]]).toarray()
    
    data = [age , gender  , Family_History , 
    Radiation_Exposure , Iodine_Deficiency , Smoking , Obesity ,Diabetes , TSH_Level , T3_Level , T4_Level , Nodule_Size]
    #print(data)
    hstack = np.hstack((one_transfer_encoded , [data]))
    # print(hstack)
    # print(rc.predict(hstack))

    prediction = rc.predict(hstack)
    
    #print(rc.predict_proba(hstack))
    # accuracy_score_ = rc.score(y_test,  prediction)
    # print("accuracct -->" , accuracy_score_)
    return render_template("index.html" , prediction = prediction[0])


if __name__ == "__main__":
    app.run(debug=True , port=5000)


