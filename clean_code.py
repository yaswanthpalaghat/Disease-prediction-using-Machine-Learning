import tkinter as tk
from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import logging
import hashlib
import os
import json
from cryptography.fernet import Fernet
import re
import tempfile

# Setup logging
logging.basicConfig(filename='disease_predictor.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# List of symptoms and diseases
l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
      'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
      'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
      'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
      'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
      'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
      'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
      'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
      'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
      'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
      'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
      'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
      'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
      'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
      'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
      'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
      'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
      'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
      'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
      'yellow_crust_ooze']

disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
           'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
           'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox',
           'Dengue', 'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
           'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
           'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
           'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
           'Impetigo']

# Security functions
def hash_patient_name(name):
    salt = os.urandom(32)
    hashed = hashlib.pbkdf2_hmac('sha256', name.encode('utf-8'), salt, 100000)
    return salt + hashed

def generate_key():
    return Fernet.generate_key()

def encrypt_data(data, key):
    f = Fernet(key)
    return f.encrypt(json.dumps(data).encode())

def decrypt_data(encrypted_data, key):
    f = Fernet(key)
    return json.loads(f.decrypt(encrypted_data))

def sanitize_input(input_string):
    return re.sub(r'[^a-zA-Z0-9\s]', '', input_string)

def secure_write(data, filename):
    fd, temp_path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'w') as temp_file:
            temp_file.write(data)
        os.replace(temp_path, filename)
    except Exception as e:
        os.remove(temp_path)
        raise e

# Data loading function
def load_data():
    try:
        df = pd.read_csv("Training.csv")
        tr = pd.read_csv("Testing.csv")
        
        # Data preprocessing
        df.replace({'prognosis': {disease[i]: i for i in range(len(disease))}}, inplace=True)
        tr.replace({'prognosis': {disease[i]: i for i in range(len(disease))}}, inplace=True)
        
        return df, tr
    except FileNotFoundError:
        messagebox.showerror("Error", "CSV files not found. Please ensure 'Training.csv' and 'Testing.csv' are in the correct directory.")
        return None, None
    except pd.errors.EmptyDataError:
        messagebox.showerror("Error", "One or both CSV files are empty.")
        return None, None
    except Exception as e:
        logging.error(f"An error occurred while loading data: {str(e)}")
        messagebox.showerror("Error", "An unexpected error occurred while loading data. Please check the log file.")
        return None, None

# Load data
df, tr = load_data()
if df is None or tr is None:
    # Handle the error 
    pass
else:
    X = df[l1]
    y = df[["prognosis"]]
    X_test = tr[l1]
    y_test = tr[["prognosis"]]

# Prediction functions
def DecisionTree():
    if not validate_inputs():
        return
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    
    y_pred = clf.predict(X_test)
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred)}")
    
    inputtest = [convert_symptoms_to_binary(get_selected_symptoms())]
    predicted = clf.predict(inputtest)[0]
    
    display_prediction(predicted, t1)

def RandomForest():
    if not validate_inputs():
        return
    
    clf = RandomForestClassifier()
    clf = clf.fit(X, np.ravel(y))
    
    y_pred = clf.predict(X_test)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred)}")
    
    inputtest = [convert_symptoms_to_binary(get_selected_symptoms())]
    predicted = clf.predict(inputtest)[0]
    
    display_prediction(predicted, t2)

def NaiveBayes():
    if not validate_inputs():
        return
    
    gnb = GaussianNB()
    gnb = gnb.fit(X, np.ravel(y))
    
    y_pred = gnb.predict(X_test)
    print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred)}")
    
    inputtest = [convert_symptoms_to_binary(get_selected_symptoms())]
    predicted = gnb.predict(inputtest)[0]
    
    display_prediction(predicted, t3)

# Helper functions
def validate_inputs():
    symptoms = get_selected_symptoms()
    if any(symptom == 'None' for symptom in symptoms):
        messagebox.showwarning("Incomplete Input", "Please select all 5 symptoms before predicting.")
        return False
    return True

def get_selected_symptoms():
    return [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]

def convert_symptoms_to_binary(symptoms):
    binary_symptoms = [0] * len(l1)
    for symptom in symptoms:
        if symptom in l1:
            binary_symptoms[l1.index(symptom)] = 1
    return binary_symptoms

def display_prediction(predicted, text_widget):
    if 0 <= predicted < len(disease):
        text_widget.delete("1.0", END)
        text_widget.insert(END, disease[predicted])
    else:
        text_widget.delete("1.0", END)
        text_widget.insert(END, "Not Found")

# GUI setup
root = tk.Tk()
root.configure(background='blue')

# Entry variables
Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)
Name = StringVar()

# Heading
w2 = Label(root, justify=LEFT, text="Disease Predictor using Machine Learning", fg="white", bg="blue")
w2.config(font=("Elephant", 30))
w2.grid(row=1, column=0, columnspan=2, padx=100)
w2 = Label(root, justify=LEFT, text="A Project by Yaswanth Sai Palaghat", fg="white", bg="blue")
w2.config(font=("Aharoni", 30))
w2.grid(row=2, column=0, columnspan=2, padx=100)

# labels
NameLb = Label(root, text="Name of the Patient", fg="yellow", bg="black")
NameLb.grid(row=6, column=0, pady=15, sticky=W)


S1Lb = Label(root, text="Symptom 1", fg="yellow", bg="black")
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="yellow", bg="black")
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="yellow", bg="black")
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="yellow", bg="black")
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="yellow", bg="black")
S5Lb.grid(row=11, column=0, pady=10, sticky=W)


lrLb = Label(root, text="DecisionTree", fg="white", bg="red")
lrLb.grid(row=15, column=0, pady=10,sticky=W)

destreeLb = Label(root, text="RandomForest", fg="white", bg="red")
destreeLb.grid(row=17, column=0, pady=10, sticky=W)

ranfLb = Label(root, text="NaiveBayes", fg="white", bg="red")
ranfLb.grid(row=19, column=0, pady=10, sticky=W)

# entries
OPTIONS = sorted(l1)

NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.grid(row=7, column=1)

S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.grid(row=8, column=1)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.grid(row=9, column=1)

S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.grid(row=10, column=1)

S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.grid(row=11, column=1)


dst = Button(root, text="DecisionTree", command=DecisionTree,bg="green",fg="yellow")
dst.grid(row=8, column=3,padx=10)

rnf = Button(root, text="Randomforest", command=randomforest,bg="green",fg="yellow")
rnf.grid(row=9, column=3,padx=10)

lr = Button(root, text="NaiveBayes", command=NaiveBayes,bg="green",fg="yellow")
lr.grid(row=10, column=3,padx=10)

#textfileds
t1 = Text(root, height=1, width=40,bg="orange",fg="black")
t1.grid(row=15, column=1, padx=10)

t2 = Text(root, height=1, width=40,bg="orange",fg="black")
t2.grid(row=17, column=1 , padx=10)

t3 = Text(root, height=1, width=40,bg="orange",fg="black")
t3.grid(row=19, column=1 , padx=10)

root.mainloop()
