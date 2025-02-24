import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from correlation_analyzer import CorrelationAnalyzer
import ollama

OLLAMA_API_URL = "http://localhost:11434"  # Connexion à Ollama via Docker
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
        for chunk in ollama.chat(model="mistral", messages=messages, base_url=OLLAMA_API_URL, stream=True):
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
# 📊 **Zone principale : Analyse de Corrélation et Détection des Outliers**
st.title("📊 Analyse de Corrélation et Détection des Outliers")

# 🚀 **Ajout d'un File Uploader**
uploaded_file = st.file_uploader("📂 **Chargez un fichier Excel**", type=["xlsx"])
selected_sheets = []
num_sheets=np.inf
if uploaded_file:
    # Charger le fichier Excel
    excel_file = pd.ExcelFile(uploaded_file)
    sheet_names = excel_file.sheet_names

    # Sélection du nombre de feuilles
    num_sheets = st.radio("Nombre de feuilles à charger", [1, 2])

    if num_sheets == 1:
        selected_sheets = st.selectbox("Sélectionnez la feuille", sheet_names)
    else:
        selected_sheets = st.multiselect("Sélectionnez deux feuilles", sheet_names, default=sheet_names[:2])

    analyzer = CorrelationAnalyzer(excel_file)

# Charger les données
if st.button("Charger les Données"):

    data = analyzer.load_data(selected_sheets)
    
    if num_sheets == 1:
        st.session_state["df"] = data
        st.write("### 📄 Aperçu des Données")
        st.dataframe(data.head())
    else:
        st.session_state["df1"], st.session_state["df2"] = data
        st.write("### 📄 Aperçu des Données")
        for i, df in enumerate(data):
            st.write(f"**{selected_sheets[i]}**")
            st.dataframe(df.head())

# Détection des outliers

# Vérifier si au moins un DataFrame est présent
if "df" in st.session_state or ("df1" in st.session_state and "df2" in st.session_state):
    st.write("## 📊 Détection des Outliers")
    
    # Si plusieurs feuilles sont chargées, laisser l'utilisateur choisir
    if "df1" in st.session_state and "df2" in st.session_state:
        sheet_choice = st.radio("Sélectionner la feuille", ["Feuille 1", "Feuille 2"])
        df = st.session_state["df1"] if sheet_choice == "Feuille 1" else st.session_state["df2"]
    else:
        # Si une seule feuille est chargée, la récupérer proprement
        df = st.session_state.get("df")
        if df is None:
            df = st.session_state.get("df1")

    # Vérifier que `df` est bien défini avant de continuer
    if df is not None and not df.empty:
        # Sélection des colonnes numériques
        numeric_columns = df.select_dtypes(include=np.number).columns
        if len(numeric_columns) > 0:
            col_select = st.selectbox("Sélectionner une variable", numeric_columns)
            
            # Détection des outliers
            outliers = analyzer.detect_outliers(df)
            
            if col_select in outliers and not outliers[col_select].empty:
                st.write(f"### 🔍 Valeurs aberrantes pour `{col_select}`")
                st.dataframe(outliers[col_select])
                fig = analyzer.generate_box_plot(df, col_select)
                st.plotly_chart(fig)
            else:
                st.info(f"✅ Aucune valeur aberrante détectée pour `{col_select}`.")
                fig = analyzer.generate_box_plot(df, col_select)
                st.plotly_chart(fig)
        else:
            st.warning("Aucune colonne numérique disponible pour la détection des outliers.")
    else:
        st.error("Le DataFrame sélectionné est vide ou introuvable.")


if  num_sheets == 1 and "df" in st.session_state:
    if st.button("Calculer corrélation"):
        df_corr = st.session_state["df"]
        st.session_state["df_corr"] = df_corr
        
        selected_columns = st.multiselect("Sélectionner les variables", st.session_state["df_corr"].columns)

        if selected_columns:
            method = st.selectbox("Méthode de Corrélation", ["spearman", "pearson", "kendall"])
            corr_matrix = analyzer.compute_correlation_matrix(st.session_state["df_corr"][selected_columns])
            st.dataframe(corr_matrix)
            st.session_state["knowledge_base"].append(f"Matrice de corrélation ({method}):\n{corr_matrix.to_string()}")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr_matrix.astype(float), annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Veuillez sélectionner des variables.")


elif num_sheets == 2 and "df1" in st.session_state and "df2" in st.session_state:
    st.write("Alignement des données")
    
    # Ajouter un selectbox pour permettre à l'utilisateur de choisir la colonne sur laquelle faire l'alignement
    colonne_alignement = st.selectbox("Choisissez la colonne pour l'alignement", st.session_state["df1"].columns)

    if st.button("Aligner des données"):
        # Utiliser la colonne sélectionnée par l'utilisateur pour l'alignement
        df_corr, df2_corr = analyzer.align_dataframes(st.session_state["df1"], st.session_state["df2"], colonne_alignement)
        
        # Vérifier si l'alignement a fonctionné
        if df_corr is not None and df2_corr is not None:
            st.session_state["df_corr"], st.session_state["df2_corr"] = df_corr, df2_corr
            
            st.write(f"### 🔍 Données après alignement: {selected_sheets[0]}")
            st.dataframe(df_corr.head())
            st.write(f"### 🔍 Données après alignement: {selected_sheets[1]}")
            st.dataframe(df2_corr.head())
        else:
            st.error(f"Erreur : Impossible d'aligner les données. Vérifiez la présence de la colonne '{colonne_alignement}'.")


if num_sheets == 2:
    if st.button("Calculer la corrélation"):
        #st.session_state["df_corr"] = df_corr
        #st.session_state["df2_corr"] = df2_corr

        if "df_corr" in st.session_state and "df2_corr" in st.session_state:
            df_corr = st.session_state["df_corr"]
            df2_corr = st.session_state["df2_corr"]

            st.write("## 🔥 Sélection des Variables pour la Corrélation")
            df_selected_columns = st.multiselect("Sélectionner les variables pour data frame 1", df_corr.columns)
            df2_selected_columns = st.multiselect("Sélectionner les variables pour data frame 2", df2_corr.columns)

            if df_selected_columns and df2_selected_columns:
                method = st.selectbox("Méthode de Corrélation", ["spearman", "pearson", "kendall"])
                corr_matrix = analyzer.compute_correlation_matrix(df2_corr[df2_selected_columns], df_corr[df_selected_columns], method=method)
                
                st.dataframe(corr_matrix)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(corr_matrix.astype(float), annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
                st.pyplot(fig)
                # Stocker la matrice de corrélation dans la base de connaissances
                st.session_state["knowledge_base"].append(f"Matrice de corrélation ({method}):\n{corr_matrix.to_string()}")

    
        else:
            st.warning("Veuillez sélectionner des variables.")





































exit()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from correlation_analyzer import CorrelationAnalyzer
import ollama

# Configuration large de la page
st.set_page_config(layout="wide")

# Barre latérale : Chatbot IA
with st.sidebar:
    st.markdown("## 💬 Assistant IA")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "knowledge_base" not in st.session_state:
        st.session_state["knowledge_base"] = []
    if "stop_response" not in st.session_state:
        st.session_state["stop_response"] = False

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Posez-moi une question...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        response = ollama.chat(model="mistral", messages=st.session_state.messages, stream=True)
        response_content = "".join(chunk["message"]["content"] for chunk in response)
        st.session_state.messages.append({"role": "assistant", "content": response_content})
        st.rerun()

# Zone principale : Analyse des données
st.title("📊 Analyse de Corrélation et Détection des Outliers")

# Charger l’analyseur
temp_file = "Datas_Total_RAST.xlsx"
analyzer = CorrelationAnalyzer(temp_file)

# Ajouter le logo
logo_path = "hipe.png"
st.image(logo_path, width=800)

# Charger la liste des feuilles
excel_file = pd.ExcelFile(temp_file)
sheet_names = excel_file.sheet_names

# Sélection du nombre de feuilles
num_sheets = st.radio("Nombre de feuilles à charger", [1, 2])

if num_sheets == 1:
    selected_sheets = st.selectbox("Sélectionnez la feuille", sheet_names)
   
else:
    selected_sheets = st.multiselect("Sélectionnez deux feuilles", sheet_names, default=sheet_names[:2])


# Charger les données
if st.button("Charger les Données"):
    data = analyzer.load_data(selected_sheets)
    
    if num_sheets == 1:
        st.session_state["df"] = data
        st.write("### 📄 Aperçu des Données")
        st.dataframe(data.head())
    else:
        st.session_state["df1"], st.session_state["df2"] = data
        st.write("### 📄 Aperçu des Données")
        for i, df in enumerate(data):
            st.write(f"**{selected_sheets[i]}**")
            st.dataframe(df.head())

# Détection des outliers
if "df" in st.session_state or ("df1" in st.session_state and "df2" in st.session_state):
    st.write("## 📊 Détection des Outliers")
    
    df = st.session_state.get("df") if num_sheets == 1 else st.session_state.get("df1")
    col_select = st.selectbox("Sélectionner une variable", df.select_dtypes(include=np.number).columns)
    outliers = analyzer.detect_outliers(df)
    
    if col_select in outliers and not outliers[col_select].empty:
        st.write(f"### 🔍 Valeurs aberrantes pour `{col_select}`")
        st.dataframe(outliers[col_select])
        fig = analyzer.generate_box_plot(df, col_select)
        st.plotly_chart(fig)
    else:
        st.info(f"✅ Aucune valeur aberrante détectée pour `{col_select}`.")

if st.button("Calculer corrélation"):
    if num_sheets == 1 and "df" in st.session_state:
        df_corr = st.session_state["df"]
        st.session_state["df_corr"] = df_corr
    elif num_sheets == 2 and "df1" in st.session_state and "df2" in st.session_state:
        df_corr, df2_corr = analyzer.align_dataframes(st.session_state["df1"], st.session_state["df2"], "Participant")
        st.session_state["df_corr"], st.session_state["df2_corr"] = df_corr, df2_corr

if "df_corr" in st.session_state:
    st.write("## 🔥 Sélection des Variables pour la Corrélation")

    if num_sheets == 1:
        selected_columns = st.multiselect("Sélectionner les variables", st.session_state["df_corr"].columns)

        if selected_columns:
            method = st.selectbox("Méthode de Corrélation", ["spearman", "pearson", "kendall"])
            corr_matrix = analyzer.compute_correlation_matrix(st.session_state["df_corr"][selected_columns])
            st.dataframe(corr_matrix)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr_matrix.astype(float), annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Veuillez sélectionner des variables.")

    elif num_sheets == 2:
        cmj_selected = st.multiselect("Variables Feuille 1", st.session_state["df_corr"].columns)
        sprint_selected = st.multiselect("Variables Feuille 2", st.session_state["df2_corr"].columns)

        if cmj_selected and sprint_selected:
            method = st.selectbox("Méthode de Corrélation", ["spearman", "pearson", "kendall"])
            corr_matrix = analyzer.compute_correlation_matrix(
                st.session_state["df_corr"][cmj_selected],
                st.session_state["df2_corr"][sprint_selected],
                method
            )
            st.dataframe(corr_matrix)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr_matrix.astype(float), annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Veuillez sélectionner des variables.")
