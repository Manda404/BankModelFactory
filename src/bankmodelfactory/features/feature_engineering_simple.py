# feature_engineering_simple.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class FeatureEngineerSimple(BaseEstimator, TransformerMixin):
    """
    A lightweight and robust transformer for the Bank Marketing dataset.

    Features:
    - Converts binary columns ('yes'/'no') to numeric (0/1)
    - Performs one-hot encoding on categorical features
    - Standardizes numeric features (mean=0, std=1)
    - Optionally adds interpretable business features (controlled by a parameter)
    """

    def __init__(self, target_col="deposit", add_business_features=True):
        """
        Parameters
        ----------
        target_col : str
            Name of the target column.
        add_business_features : bool
            Whether to create extra business / interaction features.
        """
        self.target_col = target_col
        self.add_business_features = add_business_features
        self.numeric_features = None
        self.categorical_features = None
        self.scaler = StandardScaler()

    # ==========================================================
    # 1. Base preprocessing (binary conversion)
    # ==========================================================
    def _convert_binary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        binary_cols = ["default", "housing", "loan", self.target_col]
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({"no": 0, "yes": 1})
        return df

    # ==========================================================
    # 2. Optional business features
    # ==========================================================
    def _add_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if not self.add_business_features:
            return df

        # Exemple de nouvelles features cohérentes avec ton premier modèle
        if {"balance", "age"}.issubset(df.columns):
            df["balance_to_age_ratio"] = df["balance"] / (df["age"] + 1)
            df["balance_log"] = np.log1p(df["balance"].clip(lower=0))

        if {"housing", "loan"}.issubset(df.columns):
            df["financial_engagement"] = df[["housing", "loan"]].sum(axis=1)

        if {"default", "loan", "housing"}.issubset(df.columns):
            df["risk_score"] = 0.5 * df["default"] + 0.3 * df["loan"] + 0.2 * df["housing"]

        if {"campaign", "previous"}.issubset(df.columns):
            df["contact_intensity"] = df["campaign"] + df["previous"]

        if "pdays" in df.columns:
            df["was_contacted_before"] = (df["pdays"] != -1).astype(int)

        return df

    # ==========================================================
    # 3. Fit
    # ==========================================================
    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        df = self._convert_binary_columns(df)
        df = self._add_business_features(df)

        # Identifier les colonnes numériques et catégorielles
        self.numeric_features = df.select_dtypes(exclude="object").columns.tolist()
        self.categorical_features = df.select_dtypes(include="object").columns.tolist()

        # One-hot encoding temporaire pour éviter perte de structure
        df_encoded = pd.get_dummies(df, columns=self.categorical_features, drop_first=False)

        # Fit du scaler sur colonnes numériques uniquement
        self.scaler.fit(df_encoded[self.numeric_features])

        return self

    # ==========================================================
    # 4. Transform
    # ==========================================================
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = self._convert_binary_columns(df)
        df = self._add_business_features(df)

        # Encodage des variables catégorielles
        df_encoded = pd.get_dummies(df, columns=df.select_dtypes(include="object").columns, drop_first=False)

        # Standardisation des colonnes numériques
        num_cols = [col for col in df_encoded.columns if col in self.numeric_features]
        df_encoded[num_cols] = self.scaler.transform(df_encoded[num_cols])

        # Assurer la cohérence des colonnes avec le fit initial (pour train/test alignement)
        if hasattr(self, "feature_names_"):
            missing_cols = set(self.feature_names_) - set(df_encoded.columns)
            for col in missing_cols:
                df_encoded[col] = 0
            df_encoded = df_encoded[self.feature_names_]
        else:
            self.feature_names_ = df_encoded.columns.tolist()

        return df_encoded

    # ==========================================================
    # 5. Fit-Transform
    # ==========================================================
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    # ==========================================================
    # 6. Vérification
    # ==========================================================
    @staticmethod
    def check_numeric(df: pd.DataFrame):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if len(cat_cols) == 0:
            print("Dataset is fully numeric and ready for modeling.")
        else:
            print(f"Remaining categorical columns: {cat_cols}")
