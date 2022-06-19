from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

def create_pipeline(
        model: str, use_grid_search: bool, use_scaler: bool, pca: int,  # General
        max_iter: int, logreg_C: float, random_state: int,  # LogisticRegression
        max_depth: int, n_estimators: int,  # RandomForestClassifier
) -> Pipeline:
    pipeline_steps = []

    # Scaler
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))

    # PCA
    if pca:
        pipeline_steps.append(("PCA", PCA(pca)))
    # Classifier

    if use_grid_search:
        # using grid search
        if model == "logreg":
            clf = LogisticRegression(random_state=random_state)
        elif model == "tree":
            clf = RandomForestClassifier(random_state=random_state)
        else:
            raise ValueError
    else:
        # manual parameter setting
        if model == "logreg":
            clf = LogisticRegression(random_state=random_state, max_iter=max_iter, C=logreg_C)
        elif model == "tree":
            clf = RandomForestClassifier(random_state=random_state, max_depth=max_depth, n_estimators=n_estimators)
        else:
            raise ValueError

    pipeline_steps.append(("clf", clf))
    return Pipeline(steps=pipeline_steps)
