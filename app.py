import gzip
import dill
from operator import itemgetter
import uuid
import time
from flask import Flask,request,render_template,redirect,url_for, session
from flask_fontawesome import FontAwesome
import sklearn
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import json

app=Flask(__name__)
fa=FontAwesome(app)
app.secret_key='skey'

def sid_gen():
    session['sid']=str(uuid.uuid4())
    session['sip']=str(request.remote_addr)
    session['stime']=str(time.ctime())
    return None

def file_load(filename):
    with gzip.open("%s.dill.gz"%filename,"rb") as f:
        return dill.load(f)

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/index",methods=['GET','POST'])
def index():
    return render_template("predict.html")

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method=="GET":
        name=request.args.get("name")
        age=int(request.args.get("age"))
        sex=int(request.args.get("sex"))
        chest_pain_type=int(request.args.get("chest_pain_type"))
        resting_blood_pressure=int(request.args.get("resting_blood_pressure"))
        serum_cholesterol_mg_per_dl=int(request.args.get("serum_cholesterol_mg_per_dl"))
        fasting_blood_sugar_gt_120_mg_per_dl=int(request.args.get("fasting_blood_sugar_gt_120_mg_per_dl"))
        resting_ekg_results=int(request.args.get("resting_ekg_results"))
        max_heart_rate_achieved=int(request.args.get("max_heart_rate_achieved"))
        exercise_induced_angina=int(request.args.get("exercise_induced_angina"))
        oldpeak_eq_st_depression=float(request.args.get("oldpeak_eq_st_depression"))
        slope_of_peak_exercise_st_segment=int(request.args.get("slope_of_peak_exercise_st_segment"))
        num_major_vessels=int(request.args.get("num_major_vessels"))
        thal=int(request.args.get("thal"))
    elif request.method=="POST":
        name=request.form["name"]
        age=int(request.form["age"])
        sex=int(request.form["sex"])
        chest_pain_type=int(request.form["chest_pain_type"])
        resting_blood_pressure=int(request.form["resting_blood_pressure"])
        serum_cholesterol_mg_per_dl=int(request.form["serum_cholesterol_mg_per_dl"])
        fasting_blood_sugar_gt_120_mg_per_dl=int(request.form["fasting_blood_sugar_gt_120_mg_per_dl"])
        resting_ekg_results=int(request.form["resting_ekg_results"])
        max_heart_rate_achieved=int(request.form["max_heart_rate_achieved"])
        exercise_induced_angina=int(request.form["exercise_induced_angina"])
        oldpeak_eq_st_depression=float(request.form["oldpeak_eq_st_depression"])
        slope_of_peak_exercise_st_segment=int(request.form["slope_of_peak_exercise_st_segment"])
        num_major_vessels=int(request.form["num_major_vessels"])
        thal=int(request.form["thal"])
    
    if request.method in ["POST","GET"]:
        sid_gen()
        session['name']=name
        session['age']=age
        session['sex']=sex
        session['chest_pain_type']=chest_pain_type
        session['resting_blood_pressure']=resting_blood_pressure
        session['serum_cholesterol_mg_per_dl']=serum_cholesterol_mg_per_dl
        session['fasting_blood_sugar_gt_120_mg_per_dl']=fasting_blood_sugar_gt_120_mg_per_dl
        session['resting_ekg_results']=resting_ekg_results
        session['max_heart_rate_achieved']=max_heart_rate_achieved
        session['exercise_induced_angina']=exercise_induced_angina
        session['oldpeak_eq_st_depression']=oldpeak_eq_st_depression
        session['slope_of_peak_exercise_st_segment']=slope_of_peak_exercise_st_segment
        session['num_major_vessels']=num_major_vessels
        session['thal']=thal

    if sex==0:
        sex_arr=[1,0]
    else:
        sex_arr=[0,1]

    if thal==0:
        thal_arr=[0,1,0]
    elif thal==1:
        thal_arr=[1,0,0]
    else:
        thal_arr=[0,0,1]
    
    predict_arr=[slope_of_peak_exercise_st_segment, resting_blood_pressure,chest_pain_type,num_major_vessels,fasting_blood_sugar_gt_120_mg_per_dl,resting_ekg_results,serum_cholesterol_mg_per_dl,oldpeak_eq_st_depression,age,max_heart_rate_achieved,exercise_induced_angina]+thal_arr+sex_arr
    selected_column_indices=np.array([2,3,10,12,13])
    
    scaler=file_load("scaler")
    pca=file_load("pca")
    lda=file_load("lda")
    model=file_load("model-1_rf_rscv")

    scaled_data_full=scaler.transform(np.array(predict_arr).reshape(-1,1).T)
    pca_data=pca.transform(scaled_data_full)
    scaled_data_selected=scaled_data_full.T[selected_column_indices].T
    new_data=np.append(scaled_data_selected,pca_data)
    lda_data=lda.transform(new_data.reshape(-1,1).T)
    new_data=np.append(new_data,lda_data)

    percent_proba=round((model.predict_proba(pd.DataFrame(new_data.reshape(-1,1).T)))[0][1]*100,2)
    #percent_proba=round((model.predict_proba(pd.DataFrame(new_data.reshape(-1,1).T,columns=model.get_booster().feature_names)))[0][1]*100,2)
    
    session['result']=percent_proba

    data = {
            'session_id' : session['sid'],
            'ip_address' : session['sip'],
            'session_start' : session['stime'],
            'form_data' : request.form,
            'prediction' : session['result']
        }
    
    with open('data.json','a') as f:
        json.dump(data,f)

    return redirect("/results")

@app.route("/results")
def results():
    return render_template("results.html",percent_proba=session['age'],name=session['name'])

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html")

if __name__=='__main__':
    app.run(debug=True)