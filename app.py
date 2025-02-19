import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from version_1.correlation_analyzer import CorrelationAnalyzer
import ollama

# Configuration de la page pour un affichage large
st.set_page_config(layout="wide")
# Dans la barre latérale (où se trouve le chatbot)
with st.sidebar:
    st.markdown("## 💬 Assistant IA")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "knowledge_base" not in st.session_state:
        st.session_state["knowledge_base"] = []

    # Ajouter un état pour contrôler l'interruption
    if "stop_response" not in st.session_state:
        st.session_state["stop_response"] = False

    # Afficher les messages du chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Boîte de saisie du chatbot
    user_input = st.chat_input("Posez-moi une question...")

    # Bouton pour arrêter la réponse en cours
    if st.button("Arrêter la réponse"):
        st.session_state["stop_response"] = True        
        st.write("Réponse interrompue par l'utilisateur.")

    def retrieve_relevant_info(query, knowledge_base):
        """ Récupère les informations pertinentes de la base de connaissances. """
        relevant_info = []
        for info in knowledge_base:
            if any(word in info.lower() for word in query.lower().split()):
                relevant_info.append(info)

        print(f"knowledge_base: {knowledge_base}")        
        print(f"relevant_info: {relevant_info}")
        return "\n".join(relevant_info)
    
    if user_input:
        # Réinitialiser l'état d'interruption
        st.session_state["stop_response"] = False

        relevant_context = retrieve_relevant_info(user_input, st.session_state["knowledge_base"])

        messages = [{"role": "system", "content": "Tu es un assistant d'analyse de données. Utilise les informations fournies pour aider l'utilisateur."}]
        messages += [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]

        full_prompt = f"Contexte disponible :\n{relevant_context}\n\nQuestion : {user_input}"
        messages.append({"role": "user", "content": full_prompt})

        st.session_state.messages.append({"role": "user", "content": user_input})

        # Zone vide pour afficher la réponse progressivement
        response_placeholder = st.empty()
        streamed_response = ""

        # Générer la réponse en flux
        for chunk in ollama.chat(model="mistral", messages=messages, stream=True):
            # Vérifier si l'utilisateur a demandé d'arrêter la réponse
            if st.session_state["stop_response"]:
                streamed_response += "\n\n**Réponse interrompue par l'utilisateur.**"
                response_placeholder.markdown(streamed_response)
                break

            streamed_response += chunk["message"]["content"]
            response_placeholder.markdown(streamed_response + "▌")  # Affichage progressif avec curseur clignotant

        # Enlever le curseur clignotant à la fin
        response_placeholder.markdown(streamed_response)

        # Ajouter la réponse complète dans l'historique
        st.session_state.messages.append({"role": "assistant", "content": streamed_response})

        st.rerun()
# Zone principale pour l'analyse
st.title("📊 Analyse de Corrélation et Détection des Outliers")

# Charger la classe CorrelationAnalyzer
temp_file = "Datas_Total_RAST.xlsx"
analyzer = CorrelationAnalyzer(temp_file)

# Ajouter le logo
logo_path = "hipe.png"
st.image(logo_path, width=800)

# Charger le fichier Excel pour détecter les feuilles
excel_file = pd.ExcelFile(temp_file)
sheet_names = excel_file.sheet_names  

# Sélectionner une feuille
sheet_name = st.selectbox("Sélectionner la feuille", sheet_names)
numSprint = st.number_input("Numéro de Sprint", min_value=1, max_value=10, value=1)

if st.button("Charger les Données"):
    cmj, sprint = analyzer.load_data(numSprint, sheet_name)
    sprint = analyzer.filter_data(sprint, ['Participant', 'SujetsSauts', 'J_Dominante'])
    cmj = analyzer.filter_data(cmj, ['DatasCMJ', 'T°piste', 'J_Dominante', 'SujetsSauts', 'Participant'])

    st.session_state["cmj"] = cmj
    st.session_state["sprint"] = sprint

    # Mise à jour de la base de connaissances
    st.session_state["knowledge_base"].append(f"CMJ : {cmj.shape} lignes, colonnes: {list(cmj.columns)}")
    st.session_state["knowledge_base"].append(f"Sprint : {sprint.shape} lignes, colonnes: {list(sprint.columns)}")

if "cmj" in st.session_state and "sprint" in st.session_state:
    cmj = st.session_state["cmj"]
    sprint = st.session_state["sprint"]

    st.write("### 📄 Aperçu des Données CMJ")
    st.dataframe(cmj.head())

    st.write("### 📄 Aperçu des Données Sprint")
    st.dataframe(sprint.head())

    st.write("## 📊 Détection des Outliers")
    col_select = st.selectbox("Sélectionner une variable numérique", cmj.select_dtypes(include=np.number).columns)
    
    outliers = analyzer.detect_outliers(cmj)
    if col_select in outliers and not outliers[col_select].empty:
        st.write(f"### 🔍 Valeurs aberrantes pour `{col_select}`")
        st.dataframe(outliers[col_select])
        fig = analyzer.generate_box_plot(cmj, col_select)
        st.plotly_chart(fig)
    else:
        st.info(f"✅ Aucune valeur aberrante détectée pour `{col_select}`.")

    if st.button("Supprimer les Outliers"):
        cmj_filtered = analyzer.count_outliers_and_filter(cmj)
        sprint_filtered = analyzer.count_outliers_and_filter(sprint)
        
        st.session_state["cmj_filtered"] = cmj_filtered
        st.session_state["sprint_filtered"] = sprint_filtered

        st.session_state["knowledge_base"].append(f"Données CMJ filtrées : {cmj_filtered.shape} lignes")
        st.session_state["knowledge_base"].append(f"Données Sprint filtrées : {sprint_filtered.shape} lignes")

    if "cmj_filtered" in st.session_state and "sprint_filtered" in st.session_state:
        cmj_filtered = st.session_state["cmj_filtered"]
        sprint_filtered = st.session_state["sprint_filtered"]

        st.write("### 🔍 Données après suppression des valeurs aberrantes")
        st.dataframe(cmj_filtered.head())

        st.write("## 🔄 Aligner les données")
        id_column = st.selectbox("Sélectionner la colonne pour l'alignement", cmj_filtered.columns)

        if st.button("Aligner les Données"):
            cmj_aligned, sprint_aligned = analyzer.align_dataframes_on_participants(cmj_filtered, sprint_filtered, id_column)
            st.session_state["cmj_aligned"] = cmj_aligned
            st.session_state["sprint_aligned"] = sprint_aligned

            st.session_state["knowledge_base"].append(f"CMJ aligné avec Sprint : {cmj_aligned.shape} lignes")

            st.write("### 🔍 Données après alignement")
            st.dataframe(cmj_aligned.head())

    if st.button("Calculer corrélation"):
        st.session_state["cmj_aligned"] = cmj_aligned
        st.session_state["sprint_aligned"] = sprint_aligned

    if "cmj_aligned" in st.session_state and "sprint_aligned" in st.session_state:
        cmj_aligned = st.session_state["cmj_aligned"]
        sprint_aligned = st.session_state["sprint_aligned"]

        st.write("## 🔥 Sélection des Variables pour la Corrélation")
        cmj_selected_columns = st.multiselect("Sélectionner les variables pour CMJ", cmj_aligned.columns)
        sprint_selected_columns = st.multiselect("Sélectionner les variables pour Sprint", sprint_aligned.columns)

        if cmj_selected_columns and sprint_selected_columns:
            method = st.selectbox("Méthode de Corrélation", ["spearman", "pearson", "kendall"])
            corr_matrix = analyzer.compute_correlation_matrix(sprint_aligned[sprint_selected_columns], cmj_aligned[cmj_selected_columns], "sprint", "cmj", method)
            
            st.dataframe(corr_matrix)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr_matrix.astype(float), annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
            st.pyplot(fig)
            # Stocker la matrice de corrélation dans la base de connaissances
            st.session_state["knowledge_base"].append(f"Matrice de corrélation ({method}):\n{corr_matrix.to_string()}")

    
        else:
            st.warning("Veuillez sélectionner des variables.")