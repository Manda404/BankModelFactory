import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A robust and explainable transformer for the Bank Marketing dataset.

    Responsibilities:
    - Performs consistent preprocessing and data cleaning.
    - Creates business-relevant and interpretable features.
    - Encodes categorical variables (frequency or target encoding).
    - Prevents target leakage between train and test sets.
    - Ensures the dataset is fully numerical after transformation.
    
    Note: This transformer does NOT drop columns. Column selection is left to the user.
    """

    def __init__(self, encoding_strategy="frequency", target_col="y"):
        """
        Parameters
        ----------
        encoding_strategy : str, optional
            Type of encoding to apply ('frequency' or 'target').
        target_col : str, optional
            Name of the target column (default is 'y').
        """
        self.encoding_strategy = encoding_strategy
        self.target_col = target_col

        # Learned mappings from training data (stored for reuse on test)
        self.freq_maps = {}
        self.target_means = {}

        # Map month names to numerical values for temporal feature creation
        self.month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }

    # ======================================================
    # 1. Preprocessing
    # ======================================================
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs initial cleaning and conversion of binary and textual data.

        Steps:
        - Converts target column ('yes'/'no') into 1/0 if present.
        - Replaces 'unknown' with NaN for proper imputation.
        - Fills missing values for selected categorical features.
        - Converts binary ('yes'/'no') features into numeric 1/0.
        """
        df = df.copy()

        # Convert target variable if present
        if self.target_col in df.columns:
            df[self.target_col] = df[self.target_col].map({"yes": 1, "no": 0})

        # Replace "unknown" values with NaN
        for col in df.select_dtypes(include='object'):
            df[col] = df[col].replace('unknown', np.nan)

        # Simple imputation for categorical columns
        if 'job' in df.columns:
            df['job'] = df['job'].fillna('other')
        if 'education' in df.columns:
            df['education'] = df['education'].fillna('unknown')
        if 'contact' in df.columns:
            df['contact'] = df['contact'].fillna('unknown')
        if 'poutcome' in df.columns:
            df['poutcome'] = df['poutcome'].fillna('unknown')

        # Convert yes/no binary columns to 1/0
        binary_cols = ['default', 'housing', 'loan']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({'yes': 1, 'no': 0})

        return df

    # ======================================================
    # 2. Business and behavioral feature engineering
    # ======================================================
    def _add_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates explainable, business-oriented, and behavioral features.

        Each new feature reflects a plausible relationship between
        customer characteristics, marketing behavior, and financial engagement.
        """
        df = df.copy()

        # --- DEMOGRAPHIC FEATURES ---
        if 'age' in df.columns:
            # Segment customers by age ranges (useful for marketing targeting)
            df['age_group'] = pd.cut(
                df['age'],
                bins=[17, 25, 35, 45, 55, 65, 120],
                labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
            )
            # Ratio between balance and age – proxy for financial maturity
            if 'balance' in df.columns:
                df['balance_to_age_ratio'] = df['balance'] / (df['age'] + 1)
        
        # Combined socio-professional indicator
        if 'education' in df.columns and 'job' in df.columns:
            df['education_job_level'] = df['education'].astype(str) + "_" + df['job'].astype(str)

        # --- FINANCIAL FEATURES ---
        if 'balance' in df.columns:
            # Log transformation for skewed balance distribution (handle negative values)
            df['balance_log'] = np.log1p(df['balance'].clip(lower=0))
        
        # Measures how financially active a client is (loans + housing)
        if 'housing' in df.columns and 'loan' in df.columns:
            df['financial_engagement'] = df[['housing', 'loan']].sum(axis=1)
            # Synthetic risk indicator combining financial commitments
            if 'default' in df.columns:
                df['risk_score'] = 0.5 * df['default'] + 0.3 * df['loan'] + 0.2 * df['housing']

        # --- MARKETING FEATURES ---
        if 'campaign' in df.columns and 'previous' in df.columns:
            # Total number of contacts (proxy for marketing exposure)
            df['contact_intensity'] = df['campaign'] + df['previous']
            # Ratio indicating contact effectiveness over campaigns
            df['contact_efficiency'] = df['previous'] / (df['campaign'] + 1)
            
            # Weighted score of marketing engagement
            if 'pdays' in df.columns:
                df['was_contacted_before'] = (df['pdays'] != -1).astype(int)
                df['marketing_score'] = (
                    0.6 * df['previous'] +
                    0.3 * df['was_contacted_before'] +
                    0.1 * df['contact_intensity']
                )

        # --- HISTORICAL FEATURES ---
        if 'poutcome' in df.columns:
            # Capture success in past marketing outcomes
            df['poutcome_success_flag'] = (df['poutcome'] == 'success').astype(int)

        # --- TEMPORAL FEATURES ---
        if 'month' in df.columns:
            # Encode month cyclically (sin/cos) to preserve temporal continuity
            df['month_num'] = df['month'].map(self.month_map)
            df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
            # Indicator for summer campaigns (lower engagement periods)
            df['is_summer_call'] = df['month'].isin(['jun', 'jul', 'aug']).astype(int)

        # --- INTERACTION FEATURES ---
        if 'age' in df.columns and 'balance_log' in df.columns:
            # Interaction between age and wealth level
            df['interaction_age_balance'] = df['age'] * df['balance_log']

        return df

    # ======================================================
    # 3. Encoding categorical variables
    # ======================================================
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes categorical variables using learned mappings.
        """
        df = df.copy()
        
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Apply learned encodings
        for col in cat_cols:
            if self.encoding_strategy == "frequency" and col in self.freq_maps:
                # Convert to numeric first, then fillna
                df[col] = pd.to_numeric(df[col].map(self.freq_maps[col]), errors='coerce').fillna(0)
            elif self.encoding_strategy == "target" and col in self.target_means:
                # Convert to numeric first, then fillna with mean
                mapped_values = df[col].map(self.target_means[col])
                df[col] = pd.to_numeric(mapped_values, errors='coerce').fillna(self.target_means[col].mean())

        # Encode any remaining categorical variables as integer codes
        for col in df.select_dtypes(include=['object', 'category']).columns:
            df[col] = df[col].astype('category').cat.codes

        return df

    # ======================================================
    # 4. Fit: learn encoding mappings ONLY
    # ======================================================
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Learns encoding mappings (frequency or target-based) from training data.
        
        This method ONLY learns statistics, it does NOT transform the data.

        Notes:
        - Target encoding uses the mean target value per category.
        - Frequency encoding replaces each category with its occurrence rate.
        """
        # Prepare data to learn from
        df = self._preprocess(X)
        df = self._add_business_features(df)
        
        if y is not None:
            df[self.target_col] = y

        # Learn encoding mappings from categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in cat_cols:
            if self.encoding_strategy == "frequency":
                self.freq_maps[col] = df[col].value_counts(normalize=True)
            elif self.encoding_strategy == "target":
                if self.target_col not in df.columns:
                    raise ValueError(f"The target column '{self.target_col}' is missing for target encoding.")
                # Learn target means per category
                self.target_means[col] = df.groupby(col, observed=True)[self.target_col].mean()

        return self

    # ======================================================
    # 5. Transform: apply all transformations
    # ======================================================
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies preprocessing, feature creation, and encoding to any dataset
        (train or test) using mappings learned during fit.
        
        Steps:
        1. Preprocess (clean, impute, convert binary)
        2. Add business features
        3. Encode categorical variables using learned mappings
        """
        df = self._preprocess(X)
        df = self._add_business_features(df)
        df = self._encode_categorical(df)
        
        return df

    # ======================================================
    # 6. Fit + Transform: training pipeline
    # ======================================================
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Runs the complete training pipeline:
        1. Learn encoding mappings from training data
        2. Transform the data
        """
        self.fit(X, y)
        return self.transform(X)

    # ======================================================
    # 7. Validation: check for categorical columns
    # ======================================================
    @staticmethod
    def check_no_categorical_columns(df: pd.DataFrame):
        """
        Validates that the dataset is fully numeric after transformation.
        Prints remaining categorical columns if any.
        """
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(cat_cols) == 0:
            print("✓ No categorical columns remaining — dataset is ready for modeling.")
            return True
        else:
            print(f"✗ Remaining categorical columns: {cat_cols}")
            return False