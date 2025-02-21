import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2_contingency, f_oneway


class CorrelationAnalyzer:
    def __init__(self, file):
        self.file = file

    def load_data(self, sheet_names):
        """
        Charge les données depuis le fichier Excel.

        - Si `sheet_names` est une seule feuille, retourne un seul DataFrame.
        - Si `sheet_names` contient plusieurs feuilles, retourne plusieurs DataFrames.
        """
        if isinstance(sheet_names, str):  # Une seule feuille
            df = pd.read_excel(self.file, sheet_name=sheet_names)
            df.columns = df.columns.str.replace(" ", "")
            return df

        elif isinstance(sheet_names, list):  # Plusieurs feuilles
            dfs = [pd.read_excel(self.file, sheet_name=sheet) for sheet in sheet_names]
            for df in dfs:
                df.columns = df.columns.str.replace(" ", "")
            return dfs if len(dfs) > 1 else dfs[0]

        else:
            raise ValueError("`sheet_names` doit être un nom de feuille (str) ou une liste de feuilles.")

    def clean_data(self, df):
        """
        Nettoie un DataFrame :
        - Supprime les colonnes entièrement vides.
        - Supprime les lignes avec des valeurs NaN sur toutes les colonnes.
        """
        df = df.dropna(axis=1, how='all')  # Supprime les colonnes vides
        df = df.dropna(axis=0, how='all')  # Supprime les lignes vides
        return df

    def filter_outliers(self, df):
        """
        Supprime les outliers basés sur l'IQR pour les colonnes numériques.
        """
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
    def align_dataframes(df1, df2, id_column):
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
    
    @staticmethod
    def categorical_correlation(data, cat1, cat2):
        """
        Calcule la corrélation entre deux variables catégorielles (Cramér's V).
        """
        table = pd.crosstab(data[cat1], data[cat2])
        chi2, _, _, _ = chi2_contingency(table)
        n = data.shape[0]
        min_dim = min(table.shape) - 1
        return np.sqrt(chi2 / (n * min_dim))

    @staticmethod
    def categorical_to_numeric_correlation(data, cat_col, num_col):
        """
        Calcule la corrélation entre une variable catégorielle et une variable numérique (Eta Squared).
        """
        categories = data[cat_col].dropna().unique()
        if len(categories) < 2:
            return np.nan

        anova_results = f_oneway(*(data[data[cat_col] == cat][num_col] for cat in categories))
        ss_between = sum(data[data[cat_col] == cat][num_col].var() * len(data[data[cat_col] == cat]) for cat in categories)
        ss_total = data[num_col].var() * len(data)
        return ss_between / ss_total if ss_total > 0 else np.nan

    @staticmethod
    def numeric_correlation(data, col1, col2, method='spearman'):
        """
        Calcule la corrélation entre deux variables numériques.
        """
        return data[col1].corr(data[col2], method=method)

    def compute_correlation_matrix(self, df1, df2=None, method='spearman'):
        """
        Calcule la matrice de corrélation :
        - Si `df2` est fourni, la corrélation est calculée entre les colonnes de `df1` et `df2`.
        - Sinon, la corrélation est calculée entre les colonnes de `df1` uniquement.
        """
        if df2 is None:
            df = df1.copy()
        else:
            df1 = df1.add_prefix("DF1_")
            df2 = df2.add_prefix("DF2_")
            df = pd.concat([df1, df2], axis=1)

        num_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(exclude=['number']).columns
        correlation_df = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)

        for col1 in num_cols:
            for col2 in num_cols:
                correlation_df.loc[col1, col2] = self.numeric_correlation(df, col1, col2, method)

        for num_col in num_cols:
            for cat_col in cat_cols:
                correlation_df.loc[num_col, cat_col] = self.categorical_to_numeric_correlation(df, cat_col, num_col)

        for cat_col1 in cat_cols:
            for cat_col2 in cat_cols:
                correlation_df.loc[cat_col1, cat_col2] = self.categorical_correlation(df, cat_col1, cat_col2)

        return correlation_df

    def detect_outliers(self, df):
        """
        Détecte les outliers dans un DataFrame.
        """
        outliers = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][[col]]
        return outliers

    def generate_box_plot(self, df, column):
        """
        Génère un box plot pour une colonne donnée.
        """
        fig = px.box(df, y=column, title=f"📦 Box Plot - {column}")
        return fig
