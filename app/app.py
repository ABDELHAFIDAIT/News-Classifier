import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_model() :
    model = joblib.load('models/model.pkl')
    model_emb = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    classes = ["World", "Sports", "Business", "Sci/Tech"]
    
    return model, model_emb, classes

model, model_emb, classes = load_model()

st.title("Bienvenue dans News Classifier - Le Meilleur Système de Classification des Articles de News")
st.write("Cette IA classe vos articles en 4 catégories : **World**, **Sports**, **Business** et **Sci/Tech**.")

article = st.text_area("Renseigner l'Article ici :", height=200)

if st.button("Classer l'Article") :
    if article.strip() == "" :
        st.warning("Veuillez saisir l'Article !")
    else :
        vector = model_emb.encode(article)

        vector_2d = vector.reshape(1, -1)
        
        pred_label = model.predict(vector_2d)[0]
        pred_proba = model.predict_proba(vector_2d)[0]
        
        classe_pred = classes[pred_label]
        confidence = pred_proba[pred_label]
        
        st.success(f"Classe Prédite est : **{classe_pred}** - [label_**{pred_label}**]")
        st.info(f"Score de Confiance est : **{confidence:.2%}**")
        
        st.write("---")
        st.write("Détails des probabilités :")
        st.bar_chart(
            {name: prob for name, prob in zip(classes, pred_proba)}
        )