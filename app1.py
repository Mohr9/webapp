# on importe les librairies
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# on crée le titre
st.title('Welcome to Diabetes Prediction Application using Machine Learning Algorithms') 

file1 = open('diabetes_prediction.pkl', 'rb')
rf = pickle.load(file1)
file1.close()


data = pd.read_csv("diabete_population.csv")
print(data)
  
age = st.number_input("Enter your age") 
grossesses = st.number_input("Enter your grossesses") 
insuline= st.number_input("Enter your insuline") 

#pour la normalisation################################
moy_age=data['age'].mean()
std_age=data['age'].std()

moy_grossesses=data['grossesses'].mean()
std_grossesses=data['grossesses'].std()

moy_insuline=data['insuline'].mean()
std_insuline=data['insuline'].std()

age = (age-moy_age)/std_age
grossesses = (grossesses -moy_grossesses)/ std_grossesses
insuline = (insuline - moy_insuline) / std_insuline

if(st.button('Predict Diabete')): 
    query = np.array([grossesses, age, insuline])
    # on recupère et on crée un tableau à 1 ligne et 3 colonnes
    query = query.reshape(1, 3)
    print(query)
    prediction = rf.predict(query)[0]
    st.title("Predicted value " +
             str(prediction))

# on doit normaliser les données par rapport au normalized_train = (X_train-X_train.mean())/X_train.std()
# normalized_test = (X_test-X_train.mean())/X_train.std()


    
 