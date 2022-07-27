# on importe les librairies
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import base64


####Code pour mettre un fond d'ecran
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('fond.jpg')


# on crée le titre
st.title("Bienvenu sur l'application de prédiction de diabète grâce aux algorithmes de Machine Learning") 
nom_fichier = ["KNN","LR","RandomForest"]

### Ajout d'une image
image = Image.open('pika.jpg')

st.image(image, caption='pika pika ')



### Creation d'un volet contenant les 3 modèles

with st.sidebar:
    selected = option_menu("Main Menu", nom_fichier)

if selected == nom_fichier[0]:
    st.title(f"You selected {selected}")
if selected == nom_fichier[1]:
    st.title(f"You selected {selected}") 
if selected == nom_fichier[2]:
    st.title(f"selected {selected}")


file1 = open('diabetes_prediction.pkl', 'rb')
rf = pickle.load(file1)
file1.close()

file2= open('diabetes_prediction_LR.pkl', 'rb')
LR = pickle.load(file2)
file2.close()

file3 = open('diabetes_prediction_forest.pkl', 'rb')
forest = pickle.load(file3)
file3.close()


data = pd.read_csv("diabete_population.csv")
print(data)
  
 ### Recuperation des features :
age = st.slider("Select your age", 0, 100) 
grossesses = st.slider("Select your grossesses", 0, 10) 
insuline= st.number_input("Enter your insuline") 

# ##### Normalisation des features################################
moy_age=data['age'].mean()
std_age=data['age'].std()

moy_grossesses=data['grossesses'].mean()
std_grossesses=data['grossesses'].std()

moy_insuline=data['insuline'].mean()
std_insuline=data['insuline'].std()


#### Remplacement des input par leur valeur normalisées : 
age = (age-moy_age)/std_age
grossesses = (grossesses -moy_grossesses)/ std_grossesses
insuline = (insuline - moy_insuline) / std_insuline


if(st.button('Predict Diabete')):
    if(selected == nom_fichier[0]):
        query = np.array([grossesses, age, insuline])

        query = query.reshape(1, 3)
        print(query)
        prediction = rf.predict(query)[0]
        if prediction == 0:
            st.title("Valeur prédite : " + str(prediction) + " Vous êtes probablement non diabétique ! modèle utilisé :" + str(nom_fichier[0]))
        else:
            st.title("Valeur prédite : " + str(prediction) + " Vous êtes probablement diabétique !  modèle utilisé :" + str(nom_fichier[0]))

    elif(selected == nom_fichier[1]):
        query = np.array([grossesses, age, insuline])

        query = query.reshape(1, 3)
        print(query)
        prediction = LR.predict(query)[0]
        if prediction == 0:
            st.title("Valeur prédite : " + str(prediction) + " Vous êtes probablement non diabétique ! modèle utilisé : " + str(nom_fichier[1]))
        else:
            st.title("Valeur prédite : " + str(prediction) + " Vous êtes probablement diabétique !  modèle utilisé :" + str(nom_fichier[1]))
        
    elif(selected == nom_fichier[2]):
        query = np.array([grossesses, age, insuline])

        query = query.reshape(1, 3)
        print(query)
        prediction = forest.predict(query)[0]
        if prediction == 0:
            st.title("Valeur prédite : " + str(prediction) + " Vous êtes probablement non diabétique ! modèle utilisé :" + str(nom_fichier[2]))
        else:
            st.title("Valeur prédite : " + str(prediction) + " Vous êtes probablement diabétique !  modèle utilisé :" + str(nom_fichier[2]))
