from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from src.config import NUMERICAL_FEATURES, NOMINAL_FEATURES, ORDINAL_FEATURES



def create_preprocessor():

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[
                        ["Low", "Moderate", "High"],
                        ["Poor", "Average", "Good"],
                    ],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("nominal", nominal_pipeline, NOMINAL_FEATURES),
            ("ordinal", ordinal_pipeline, ORDINAL_FEATURES),
        ]
    )

    return preprocessor
