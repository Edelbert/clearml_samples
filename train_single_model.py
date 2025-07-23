import pandas as pd

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from clearml import Task, OutputModel
import argparse


def get_model_params(model_name):
    cfg = {
        "GradientBoostingRegressor": {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 5,
            "random_state": 42,
        },
        "KNeighborsRegressor": {"n_neighbors": 5},
        "LinearRegression": {},
    }

    return cfg[model_name]


def get_model(model_name: str):
    models = {
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "KNeighborsRegressor": KNeighborsRegressor,
        "LinearRegression": LinearRegression,
    }

    return models[model_name]


def main(model_name: str):
    task = Task.init(project_name="uber", reuse_last_task_id=False, output_uri=True, auto_connect_frameworks=False,)
    task.execute_remotely(queue_name = 'default')
    logger = task.get_logger()

    # 1. Загрузка и предобработка данных
    df = pd.read_csv("uber.csv")
    # Логгирование артефакта
    task.register_artifact(
        name="uber_fares_dataset",
        artifact=df,
        metadata={
            "description": "Uber fares dataset with pickup and dropoff locations, fare amount, and passenger count."
        },
    )

    # Удалим пропуски и выбросы
    df = df.dropna()
    df = df[
        (df["fare_amount"] > 0)
        & (df["passenger_count"] > 0)
        & (df["passenger_count"] <= 6)
    ]

    # Выбор признаков
    X = df[
        [
            "pickup_latitude",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
            "passenger_count",
        ]
    ]
    y = df["fare_amount"]

    stats = X.describe()

    task.register_artifact(
        name="dataset_statistics",
        artifact=stats,
        metadata={"description": "Dataset statistics after post-processing."},
    )

    logger.report_table(
        title="Datasets statts", series="Stats", iteration=0, table_plot=stats
    )
    params = {
        "data": {
            "test_size": 0.2,
            "random_state": 42,
        },
        "model": {"name": model_name, "params": get_model_params(model_name)},
    }
    # Логгирование параметров
    task.connect(params)

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, **params["data"])
    # создаем OutputModel модель
    output_model = OutputModel(
        task=task,
        framework="ScikitLearn",
        name=model_name,
        comment=f"{model_name} for uber",
        tags=["uber", "lr_model"],
    )
    # Модель
    model = get_model(model_name)(**get_model_params(model_name))
    model.fit(X_train, y_train)
    joblib.dump(model, f"{model_name}.pkl", compress=True)
    predicts = model.predict(X_test)
    output_model.update_weights(f"{model_name}.pkl")

    # Оценка мдели и логирование метрик
    def print_metrics(name, y_true, y_pred):
        logger.report_scalar(
            name, "RMSE", value=mean_squared_error(y_true, y_pred), iteration=0
        )
        logger.report_scalar(
            name, "MAE", value=mean_absolute_error(y_true, y_pred), iteration=0
        )
        logger.report_scalar(name, "R2", value=r2_score(y_true, y_pred), iteration=0)

    print_metrics(model_name, y_test, predicts)
    print_metrics(model_name, y_test, predicts)
    print_metrics(model_name, y_test, predicts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        choices=[
            "GradientBoostingRegressor",
            "KNeighborsRegressor",
            "LinearRegression",
        ],
        required=True
    )
    args = parser.parse_args()
    main(args.model_name)
