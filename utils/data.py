import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(df=None):
    """
    Load and clean the Telco Customer Churn dataset.
    
    Parameters:
    df: pandas DataFrame (optional) - If provided, uses this data instead of loading from file
    
    Returns:
    pandas DataFrame - Cleaned dataset
    """
    if df is None:
        try:
            # Load default dataset
            df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        except FileNotFoundError:
            return None
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Step 1: Drop customerID as it's not predictive
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Step 2: Handle TotalCharges column
    if 'TotalCharges' in df.columns:
        # Convert TotalCharges to numeric, handling blanks as NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Impute missing values with median
        median_total_charges = df['TotalCharges'].median()
        df['TotalCharges'].fillna(median_total_charges, inplace=True)
    
    # Step 3: Ensure SeniorCitizen is treated as categorical
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
    
    # Step 4: Handle any other missing values
    df = df.dropna()
    
    return df

def identify_feature_types(df, target_column='Churn'):
    """
    Automatically identify categorical and numerical columns.
    
    Parameters:
    df: pandas DataFrame
    target_column: str - name of target column to exclude
    
    Returns:
    tuple: (categorical_columns, numerical_columns)
    """
    # Exclude target column
    feature_columns = [col for col in df.columns if col != target_column]
    
    categorical_columns = []
    numerical_columns = []
    
    for col in feature_columns:
        if df[col].dtype in ['object', 'category'] or col == 'SeniorCitizen':
            categorical_columns.append(col)
        elif df[col].dtype in ['int64', 'float64']:
            numerical_columns.append(col)
    
    return categorical_columns, numerical_columns

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the data for machine learning.
    
    Parameters:
    df: pandas DataFrame - cleaned dataset
    test_size: float - proportion of data for testing
    random_state: int - random seed for reproducibility
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test, preprocessor, feature_names)
    """
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = (df['Churn'] == 'Yes').astype(int)  # Convert to binary
    
    # Identify feature types
    categorical_columns, numerical_columns = identify_feature_types(df)
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    # Fit preprocessor on training data only
    preprocessor.fit(X_train)
    
    # Get feature names after preprocessing
    feature_names = get_feature_names(preprocessor, categorical_columns, numerical_columns)
    
    return X_train, X_test, y_train, y_test, preprocessor, feature_names

def get_feature_names(preprocessor, categorical_columns, numerical_columns):
    """
    Get feature names after preprocessing.
    
    Parameters:
    preprocessor: sklearn ColumnTransformer
    categorical_columns: list of categorical column names
    numerical_columns: list of numerical column names
    
    Returns:
    list: feature names after preprocessing
    """
    feature_names = []
    
    # Add numerical feature names (unchanged)
    feature_names.extend(numerical_columns)
    
    # Add categorical feature names (one-hot encoded)
    try:
        cat_encoder = preprocessor.named_transformers_['cat']
        if hasattr(cat_encoder, 'get_feature_names_out'):
            cat_features = cat_encoder.get_feature_names_out(categorical_columns)
        else:
            # Fallback for older sklearn versions
            cat_features = []
            for i, col in enumerate(categorical_columns):
                categories = cat_encoder.categories_[i]
                # Skip first category due to drop='first'
                for cat in categories[1:]:
                    cat_features.append(f"{col}_{cat}")
        
        feature_names.extend(cat_features)
    except Exception as e:
        # Fallback: create generic names
        cat_encoder = preprocessor.named_transformers_['cat']
        n_cat_features = len(cat_encoder.transform(
            preprocessor.transformers_[1][2]
        ).toarray()[0]) if hasattr(cat_encoder, 'transform') else 0
        
        for i in range(n_cat_features):
            feature_names.append(f"cat_feature_{i}")
    
    return feature_names

def create_tenure_bins(df):
    """
    Create tenure bins for cohort analysis.
    
    Parameters:
    df: pandas DataFrame
    
    Returns:
    pandas Series: tenure bins
    """
    bins = [0, 12, 24, 36, float('inf')]
    labels = ['0-12 months', '12-24 months', '24-36 months', '36+ months']
    
    return pd.cut(df['tenure'], bins=bins, labels=labels, right=False)

def get_churn_rate_by_segment(df, segment_col):
    """
    Calculate churn rate by segment.
    
    Parameters:
    df: pandas DataFrame
    segment_col: str - column name for segmentation
    
    Returns:
    pandas Series: churn rates by segment
    """
    return df.groupby(segment_col)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
