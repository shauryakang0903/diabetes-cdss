"""Fold-fitted Core-10 preprocessing; no dataset loading or model fitting."""

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.preprocessing import CORE_FEATURES

NUMERIC_FEATURES = [
    "age", "bmi", "waist_cm", "mean_systolic_bp", "mean_diastolic_bp",
    "hdl_mg_dl", "total_cholesterol_mg_dl",
]
CATEGORICAL_FEATURES = ["sex", "race_ethnicity", "smoking_status"]
STRATEGIES = ("A_median", "B_iterative")


def make_preprocessor(strategy="A_median", branch="logistic", random_seed=42):
    """Return a fresh transformer; callers must fit it within training folds.

    The experimental import above is required to enable IterativeImputer.
    Its default BayesianRidge estimator is used with deterministic, non-posterior
    imputation. This is single imputation, not multiple-imputation inference.
    The XGBoost branch defines preprocessing only; it does not fit a model.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown preprocessing strategy: {strategy}")
    if branch not in ("logistic", "xgboost"):
        raise ValueError(f"Unknown model branch: {branch}")
    imputer = (
        SimpleImputer(strategy="median", keep_empty_features=True)
        if strategy == "A_median" else
        IterativeImputer(random_state=random_seed, max_iter=10, tol=1e-3,
                         sample_posterior=False, keep_empty_features=True)
    )
    numeric_steps = [("imputer", imputer)]
    if branch == "logistic":
        numeric_steps.append(("scaler", StandardScaler()))
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop", verbose_feature_names_out=True)
