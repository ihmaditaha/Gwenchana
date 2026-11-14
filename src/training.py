import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from src.preprocessing import create_preprocessor
from src.config import MODEL_PATH


def train_models(X_train, y_train):
    models = {
        "Logistic Regression": (
            Pipeline(
                steps=[
                    ("pre_processing", create_preprocessor()),
                    ("classifier", LogisticRegression(max_iter=1000,random_state=42)),
                ]
            ),
            [
                {
                    "classifier__C": [1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03],
                    "classifier__solver": ["saga"],
                },
                {
                    "classifier__C": [1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03],
                    "classifier__solver": ["lbfgs", "newton-cg"],
                }
            ],
        ),
        "Random Forest Classifier": (
            Pipeline(
                steps=[
                    ("pre_processing", create_preprocessor()),
                    ("classifier", RandomForestClassifier(max_depth=200,random_state=42)),
                ]
            ),
            {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [None, 10, 20, 30],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__max_features": ["sqrt", "log2"],
                "classifier__criterion": ["gini", "entropy"],
            },
        ),
        "K Nearest Neighbors Classifier": (
            Pipeline(
                steps=[
                    ("pre_processing", create_preprocessor()),
                    ("classifier", KNeighborsClassifier()),
                ]
            ),
            {
                "classifier__n_neighbors": [1,  3,  5,  7,  9, 11, 13, 15, 17, 19],
                "classifier__weights": ["uniform", "distance"]
            },
        ),
    }

    results = []
    best_model = None
    best_score = 0

    for model_name, (pipeline,params) in models.items():


        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1,
        )

        grid_search.fit(X_train, y_train)

        results.append(
            {
                "Model": model_name,
                "Best CV F1": grid_search.best_score_,
                "Best Parameters": grid_search.best_params_,
            }
        )

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_

    results_df = pd.DataFrame(results)

    joblib.dump(best_model , MODEL_PATH)

    return best_model , results_df

