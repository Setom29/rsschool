![score](https://github.com/Setom29/rsschool_evaluation_selection_9/blob/master/score.png)

Homework for RS School Machine Learning course.

In this progect the [Forest cover type](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset is used.
## Usage
This package allows you to train model for detecting the type of forest cover.
1. Clone this repository to your machine.
2. Download [Forest cover type](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.13).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Run train with the following command:
```sh
poetry run train
```
### Additional options
```md
  -m, --model TEXT                [default: logreg]
  -a, --use-grid-search BOOLEAN   [default: False]
  -d, --dataset-path FILE         [default: data/train.csv]
  -s, --save-model-path FILE      [default: data/model.joblib]
  --random-state INTEGER          [default: 42]
  --test-split-ratio FLOAT RANGE  [default: 0.2; 0<x<1]
  --use-scaler BOOLEAN            [default: True]
  --use-pca INTEGER RANGE         [default: 0; 0<=x<=20]
  --max-iter INTEGER              [default: 100]
  --logreg-c FLOAT                [default: 1.0]
  --max-depth INTEGER             [default: 8]
  --n-estimators INTEGER          [default: 100]
  --help                          Show this message and exit.

```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```
## Experiments
![mlflow_exp](https://github.com/Setom29/rsschool_evaluation_selection_9/blob/master/mlflow_exp.jpg)

The best scores using GridSearchCV for the models are:
![best_models](https://github.com/Setom29/rsschool_evaluation_selection_9/blob/master/best_models.png)
