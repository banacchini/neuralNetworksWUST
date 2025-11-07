from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def fetch_heart_data():
    # fetch dataset
    heart_disease = fetch_ucirepo(id=45)

    # data (as pandas dataframes)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    df = pd.concat([X, y], axis=1)

    return df

def preprocess_heart_data(df, scale=True):
    # Start with a fresh copy of the original dataframe
    df_processed = df.copy()

    # Impute missing values using the strategies we decided on
    df_processed['ca'] = df_processed['ca'].fillna(df_processed['ca'].median())
    df_processed['thal'] = df_processed['thal'].fillna(df_processed['thal'].mode()[0])

    # --- 2. Separate Features (X) and Target (y) ---

    # The 'num' column is our target variable (what we want to predict)
    X = df_processed.drop('num', axis=1)
    y = df_processed['num']

    # --- 3. Define Column Types ---

    # Identify which columns are categorical and which are numerical
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

    # --- 4. Create the Preprocessing Pipeline ---

    # Create a transformer for numerical features: scale them
    numerical_transformer = StandardScaler()

    # Create a transformer for categorical features: one-hot encode them
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Use ColumnTransformer to apply the correct transformer to each column type
    if scale:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough')  # Keep numerical features as they are

    # --- 5. Apply the Transformations ---

    # The preprocessor is now ready to transform the data
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y