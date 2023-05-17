import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.write('''
# PredictCarPrice : Application pour prédire le prix d'une voiture
''')

user = st.sidebar.selectbox("Etes-vous un vendeur ou un acheteur", ("Vendeur", "Acheteur"))

st.sidebar.header("Les paramètres d'entrées")

def user_input():
    year=st.sidebar.slider("L'année du véhicule",1996,2023,2020)
    mileage=st.sidebar.slider('Le kilométrage de la voiture',0,200000,10000)
    mpg=st.sidebar.slider('La consommation de carburant en mpg',5,200,50)
    engineSize=st.sidebar.slider('La capacité du moteur',0.0,6.0,3.0)
    data={'year':year,
    'mileage':mileage,
    'mpg':mpg,
    'engineSize':engineSize
    }
    arr = np.array([[year, mileage,  mpg, engineSize]])
    return arr

df=user_input()

st.subheader('On veut trouver le prix de cette voiture:')
st.write(df)

# Chargement du modèle
model = joblib.load(filename = 'predictPriceCar.pkl')
prediction=model.predict(df)


if user == "Acheteur":
    Prix = st.text_input("Entrez le prix affiché sur l'annonce de la voiture qui vous interesse")
    if Prix: # Vérifie que Prix est non vide
        try:
            Prix = float(Prix)
            if prediction >= Prix:
                st.subheader(f"Vous allez faire une bonne affaire car pour une voiture avec ces caractéristiques, notre modèle prédit un prix égal à {prediction[0]:.2f} $")
            else:
                st.subheader(f"Le prix affiché sur l'annonce est élevé car pour une voiture avec ces caractéristiques, notre modèle prédit un prix égal à {prediction[0]:.2f} $")
        except:
            st.write("Veuillez entrer un prix valide")
    else:
        st.warning("Veuillez entrez un prix")
 
else:
    Prix = st.text_input("Entrez le prix auquel vous voulez vendre votre voiture")
    if Prix == "":
        st.warning("Veuillez entrer un prix")
    else:
        try:
            Prix = float(Prix)
        except:
            st.write("Veuillez entrer un prix valide")
        if prediction > Prix:
            st.subheader(f"Le prix auquel vous voulez vendre votre voiture est faible car pour une voiture avec ces caractéristiques, notre modèle prédit un prix égal à {prediction[0]:.2f} $")
        elif prediction == Prix:
            st.subheader(f"Le prix auquel vous voulez vendre votre voiture est adéquat car pour une voiture avec ces caractéristiques, notre modèle prédit un prix égal à {prediction[0]:.2f} $")
        else:
            st.subheader(f"Le prix auquel vous voulez vendre votre voiture est élevé car pour une voiture avec ces caractéristiques, notre modèle prédit un prix égal à {prediction[0]:.2f} $")

    