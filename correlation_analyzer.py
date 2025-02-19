import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway
import plotly.express as px

class CorrelationAnalyzer:
    def __init__(self, file):
        self.file = file
    
    def load_data(self, numSprint, sheet_name):
        sheet = pd.read_excel(self.file, sheet_name=sheet_name)
        sheet.columns = sheet.columns.str.replace(" ", "")
        
        data = pd.read_excel(self.file, sheet_name='RAST_FV')
        data.columns = data.columns.str.replace(" ", "")
        
        sprint_dataframes = {}
        current_sprint = None
        sprint_columns = {}
        
        for col in data.columns:
            if col.endswith("_Sprint") or col.endswith("_Fatigue"):
                current_sprint = col
                sprint_columns[current_sprint] = []
            elif current_sprint:
                sprint_columns[current_sprint].append(col)
                
        for sprint, cols in sprint_columns.items():
            sprint_dataframes[sprint] = data[cols]
        
        sprint_dataframes[f"{numSprint}_Sprint"].insert(0, 'SujetsRAST', sheet['SujetsRAST'].reset_index(drop=True))
        
        return sheet, sprint_dataframes[f"{numSprint}_Sprint"]

    def filter_data(self, df, columns_to_drop):
        df = df.drop(columns=columns_to_drop, errors='ignore')
        df = df.dropna(axis=1, how='all')  # Supprimer les colonnes totalement vides
        na_values = df[df.isnull().any(axis=1)]['SujetsRAST']
        return df[~df['SujetsRAST'].isin(na_values)]
    
    def count_outliers_and_filter(self, df):
        df_filtered = df.copy()
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound)]
        return df_filtered
    
    @staticmethod
    def align_dataframes_on_participants(df1, df2, id_column):
        """
        Aligne deux DataFrames en ne gardant que les lignes correspondant aux mêmes valeurs dans la colonne spécifiée.
        
        Args:
            df1 (pd.DataFrame): Premier DataFrame.
            df2 (pd.DataFrame): Deuxième DataFrame.
            id_column (str): Nom de la colonne utilisée pour l'alignement.

        Returns:
            pd.DataFrame, pd.DataFrame: Deux DataFrames alignés.
        """
        if id_column not in df1.columns or id_column not in df2.columns:
            raise KeyError(f"La colonne '{id_column}' est absente dans l'un des DataFrames.")
        
        participants_communs = set(df1[id_column].dropna()).intersection(set(df2[id_column].dropna()))

        df1_aligned = df1[df1[id_column].isin(participants_communs)].copy()
        df2_aligned = df2[df2[id_column].isin(participants_communs)].copy()

        print(f"Taille avant alignement : df1 = {df1.shape}, df2 = {df2.shape}")
        print(f"Taille après alignement : df1 = {df1_aligned.shape}, df2 = {df2_aligned.shape}")

        return df1_aligned, df2_aligned  # ✅ Garde la colonne choisie par l'utilisateur

        
    def get_corr_cat_to_cat(data, cat_column_1, cat_column_2):
        """Coefficient de corrélation de Cramér entre 2 variables catégorielles."""
        contingency_table = pd.crosstab(data[cat_column_1], data[cat_column_2])
        chi2_stat, _, _, _ = chi2_contingency(contingency_table)
        n = data.shape[0]
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2_stat / (n * min_dim))
        return cramers_v

    """def get_corr_cat_to_num(data, cat_column, num_column):
        
        grouped_data = {r: data[data[cat_column] == r][num_column].values for r in data[cat_column].unique()}
        _, p_value = f_oneway(*grouped_data.values()) if len(grouped_data) > 1 else (None, np.nan)
        return p_value
        """
    @staticmethod
    def get_corr_cat_to_num(data, cat_column, num_column):
        """Calcule η² (force de l'association) entre une variable catégorielle et une variable numérique."""
        categories = data[cat_column].dropna().unique()
        #print(f"categories : {categories}")
        if len(categories) < 2:
            return np.nan  # Impossible de calculer l'ANOVA avec une seule catégorie
        
        anova_results = f_oneway(*(data.loc[data[cat_column] == cat, num_column] for cat in categories))
        
        # Calcul de η²
        ss_between = sum(data.loc[data[cat_column] == cat, num_column].var() * 
                        len(data.loc[data[cat_column] == cat]) for cat in categories)
        ss_total = data[num_column].var() * len(data)
        eta_squared = ss_between / ss_total if ss_total > 0 else np.nan
        
        return eta_squared

    @staticmethod
    def get_corr_num_to_num(data, num_column_1, num_column_2, method='spearman'):
        """Calcule la corrélation entre 2 variables numériques : 'pearson', 'kendall', 'spearman'."""
        return data[num_column_1].corr(data[num_column_2], method=method)
    def compute_correlation_matrix(self,df1, df2, df1_name="df1", df2_name="df2", method='spearman'):
        """
        Génère une matrice de corrélation où les lignes sont les features de df1 
        et les colonnes celles de df2 en tenant compte du type de variables.
        """

        # Ajouter des suffixes personnalisés en fonction du nom des DataFrames
        df1 = df1.add_suffix(f"_{df1_name}")
        df2 = df2.add_suffix(f"_{df2_name}")

        num_cols_df1 = df1.select_dtypes(include=['number']).columns
        cat_cols_df1 = df1.select_dtypes(exclude=['number']).columns
        num_cols_df2 = df2.select_dtypes(include=['number']).columns
        cat_cols_df2 = df2.select_dtypes(exclude=['number']).columns

        # DataFrame où les lignes sont df1 et les colonnes sont df2
        correlation_df = pd.DataFrame(index=df1.columns, columns=df2.columns, dtype=float)

        # Fusionner les DataFrames après ajout des suffixes
        combined_df = pd.concat([df1, df2], axis=1)

        # Numérique vs Numérique (Spearman, Pearson, Kendall)
        for col1 in num_cols_df1:
            for col2 in num_cols_df2:
                correlation_df.loc[col1, col2] =CorrelationAnalyzer.get_corr_num_to_num(combined_df, col1, col2, method)

        # Numérique vs Catégorielle 
        for num_col in num_cols_df1:
            for cat_col in cat_cols_df2:
                correlation_df.loc[num_col, cat_col] = CorrelationAnalyzer.get_corr_cat_to_num(combined_df, cat_col, num_col)

        for num_col in num_cols_df2:
            for cat_col in cat_cols_df1:
                correlation_df.loc[cat_col, num_col] = CorrelationAnalyzer.get_corr_cat_to_num(combined_df, cat_col, num_col)

        # Catégorielle vs Catégorielle (Cramér’s V)
        for col1 in cat_cols_df1:
            for col2 in cat_cols_df2:
                correlation_df.loc[col1, col2] = CorrelationAnalyzer.get_corr_cat_to_cat(combined_df, col1, col2)

        return correlation_df

    def detect_outliers(self, df):
            """Retourne un DataFrame des valeurs aberrantes détectées."""
            outliers = {}
            for col in df.select_dtypes(include=[np.number]).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][['SujetsRAST', col]]
            return outliers

    def generate_box_plot(self, df, column):
            """Génère un box plot interactif avec Plotly."""
            fig = px.box(df, y=column, title=f"📦 Box Plot - {column}")
            return fig

# Utilisation de la classe
""" 

analyzer = CorrelationAnalyzer('Datas_Total_RAST (2).xlsx')
cmj, sprint_1 = analyzer.load_data(1, 'CMJ')
sprint_1 = analyzer.filter_data(sprint_1, ['Participant', 'SujetsSauts', 'J_Dominante'])
cmj = analyzer.filter_data(cmj, ['DatasCMJ', 'T°piste','J_Dominante','SujetsSauts','Participant'])
corrcmj_filtered = analyzer.count_outliers_and_filter(cmj)
corrsprint_filtered = analyzer.count_outliers_and_filter(sprint_1)
corr_matrix = analyzer.compute_correlation_matrix(corrsprint_filtered, corrcmj_filtered, "sprint", "cmj")

"""