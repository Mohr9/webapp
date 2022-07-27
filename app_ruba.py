import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle

# une petite documentation: https://fr.acervolima.com/un-guide-du-debutant-pour-streamlit/

st.title('Welcome to Diabetes Prediction Application using Machine Learning Algorithms') 

file1 = open('diabetes_prediction.pkl', 'rb')
knn = pickle.load(file1)
file1.close()

file2 = open('diabetes_prediction_LR.pkl', 'rb')
lg_reg = pickle.load(file2)
file2.close()

file3 = open('diabetes_prediction_forest.pkl', 'rb')
dtc = pickle.load(file3)
file3.close()

data = pd.read_csv("diabete_population.csv")
print(data)

# Pour normaliser chaque colonne, exécuter le code suivant: data_norm=(data[colonne]-moyenne)/ecart_type

age = st.number_input("Enter your age") 
age= (age- data['age'].mean())/data['age'].std()

#pour choisir le nb de grossessses sur un interval
grossesses = st.slider("Select the grossesses value", 0, 14) 
st.text('Selected: {}'.format(grossesses)) 
#grossesses = st.number_input("Enter your grossesses") 
grossesses=(grossesses- data['grossesses'].mean())/data['grossesses'].std()

insuline= st.number_input("Enter your insuline") 
insuline= (insuline- data['insuline'].mean())/data['insuline'].std()


#1. as sidebar menu

#with st.sidebar:
#    selected= option_menu(menu_title= "Main Menu",   #required
#    options= ["Home", "Projects", "Contact"],   #required
 #   icons= ["house","book","envelope"] ,  #optional
#    menu_icon= "cast",   #optional
#    default_index= 0,   #optional
#                             )
    
#2. horizontal menu 
selected= option_menu(menu_title= "Model Prediction Diabet",   #required
    options= ["KNeighborsClassifier","logistic_Regression", "randomForestClassifier", "DecisionTreeClassifier"],   #required
    icons= ["caret-right","caret-right","caret-right","caret-right"] ,  #optional
    menu_icon= "cast",   #optional
    default_index= 0,   #optional
    orientation= "horizontal",
    styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "green"},
        })




if(st.button('Predict Diabete')): 
    if selected == "KNeighborsClassifier": 
        query = np.array([grossesses, age, insuline])
        query = query.reshape(1, 3)
        print(query)
        prediction = knn.predict(query)[0]
        st.title(f" Predicted value {prediction}  {selected}" )

    elif selected == "logistic_Regression":
        query = np.array([grossesses, age, insuline])
        query = query.reshape(1, 3)
        print(query)
        prediction = lg_reg.predict(query)[0]
        st.title(f" Predicted value {prediction}  {selected}" )
    

    elif selected == "DecisionTreeClassifier": 
        query = np.array([grossesses, age, insuline])
        query = query.reshape(1, 3)
        print(query)
        prediction = dtc.predict(query)[0]
        st.title(f" Predicted value {prediction}  {selected}" )
                 
    
    elif selected == "randomForestClassifier":
        query = np.array([grossesses, age, insuline])
        query = query.reshape(1, 3)
        print(query)
        prediction = rfc.predict(query)[0]
        st.title(f" Predicted value {prediction}  {selected}" )
    if prediction == 0:
        st.success("You have a good health!") 
    else:
        st.warning("Be Carful!") 
            
            
 #1) Améliorer votre page web en utilisant d'autres composants du module streamlit  (modifié)
# 2)Vous pouvez aussi créer d'autres modèles de machine learning avec logistic_Regression(), randomForestClassifier() et DecisionTreeClassifier() et les utiliser dans la page web
             
    


    
