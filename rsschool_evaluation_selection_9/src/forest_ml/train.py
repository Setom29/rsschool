from pathlib import Path
from joblib import dump

import click
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, v_measure_score, f1_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV

from .get_data import get_dataset
from .pipeline import create_pipeline

from warnings import filterwarnings

filterwarnings("ignore")


@click.command()
@click.option(
    "-m",
    "--model",
    default="logreg",
    type=str,
    show_default=True,
)
@click.option(
    "-a",
    "--use-grid-search",
    default="False",
    type=bool,
    show_default=True,
)
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--use-pca",
    default=0,
    type=click.IntRange(0, 20, min_open=False, max_open=False),
    show_default=True,
)
@click.option(
    "--max-iter",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
@click.option(
    "--max-depth",
    default=8,
    type=int,
    show_default=True,
)
@click.option(
    "--n-estimators",
    default=100,
    type=int,
    show_default=True,
)
def train(
        model: str,
        use_grid_search: bool,
        dataset_path: Path,
        save_model_path: Path,
        random_state: int,
        test_split_ratio: float,
        use_scaler: bool,
        use_pca: bool,
        max_iter: int,
        logreg_c: float,
        max_depth: int,
        n_estimators: int,
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    print(features_train.shape)
    with mlflow.start_run():
        pipeline = create_pipeline(model, use_grid_search, use_scaler, use_pca,  # General
                                   max_iter, logreg_c, random_state,  # LogisticRegression
                                   max_depth, n_estimators)  # RandomForestClassifier
        # use grid search
        if use_grid_search:

            cv = KFold(n_splits=6, shuffle=True, random_state=random_state)
            # params
            if model == "tree":
                params = {'clf__max_depth': np.arange(6, 12, 1),
                          "clf__n_estimators": np.arange(20, 300, 20)}
            else:
                params = {'clf__max_iter': np.arange(200, 500, 100),
                          "clf__C": np.arange(0.5, 2, 0.2)}

            # fit
            gs_clf = GridSearchCV(pipeline, params, cv=cv, n_jobs=-1, scoring="accuracy")
            gs_clf.fit(features_train, target_train)

            # predict
            y_pred = gs_clf.best_estimator_.predict(features_val)

            # fill mlflow params
            if model == "logreg":
                max_iter = gs_clf.best_params_["clf__max_iter"]
                logreg_c = gs_clf.best_params_["clf__C"]
            else:
                max_depth = gs_clf.best_params_["clf__max_depth"]
                n_estimators = gs_clf.best_params_["clf__n_estimators"]
            # save best model
            click.echo(gs_clf.best_estimator_)
            best_model = gs_clf.best_estimator_

        # use manual setting
        else:

            pipeline.fit(features_train, target_train)
            y_pred = pipeline.predict(features_val)
            best_model = pipeline

        # metrics calculation
        accuracy = accuracy_score(target_val, y_pred)
        v_score = v_measure_score(target_val, y_pred)
        f1 = f1_score(target_val, y_pred, average="weighted")
        kf = KFold(n_splits=6, random_state=random_state, shuffle=True)
        cvs = cross_val_score(pipeline, features_train, target_train, scoring='accuracy', cv=kf).mean()

        # model params
        mlflow.log_param("model", model)
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("use_pca", use_pca)
        if model == "logreg":
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("logreg_c", logreg_c)
        else:
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("n_estimators", n_estimators)

        # metrics log_param
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("v_measure_score", v_score)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("cross_val_score", cvs)

        # metrics click.echo
        click.echo(f"V_measure_score: {v_score}")
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"F1_score: {f1}")
        click.echo(f"Cross_val_score: {cvs}\n")

        # saving model
        dump(best_model, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
