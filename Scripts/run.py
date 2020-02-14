#Authors: Kevin Quinn / Samuel Orekoya
#Purpose: This file is to be used with the setup.py file to run the exported models with a prediction

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

def dummyEncode(df):
          columnsToEncode = list(df.select_dtypes(include=['category','object','bool']))
          le = LabelEncoder()
          for feature in columnsToEncode:
              try:
                  df[feature] = le.fit_transform(df[feature])
              except:
                  print('Error encoding '+feature)
          return df

mlpc = pickle.load(open('neuralnet.sav', 'rb'))
rfc = pickle.load(open('randomforest.sav', 'rb'))

Xnew = [["01/05/2018 11:30:00 PM", True, False, "90D"]]
le = LabelEncoder()
sc = pickle.load(open('scaler.sav', 'rb'))

df=pd.DataFrame(Xnew, columns=['Date',
                               'Arrest', 
                               'Domestic',
                               'FBI Code'])

df = dummyEncode(df)

df = sc.transform(df)

print("Neural Network Prediction")
pred_xnew = mlpc.predict(df)
print(pred_xnew)

print("Random Forest Prediction")
pred_xnew = rfc.predict(df)
print(pred_xnew)