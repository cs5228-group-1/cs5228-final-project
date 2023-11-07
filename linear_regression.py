import pandas as pd
import numpy as np
from common import RANDOM_SEED, PREPROCESSORS
import sklearn
from sklearn.compose import make_column_transformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV
from category_encoders.cat_boost import CatBoostEncoder
from path import Path
import typer
from omegaconf import OmegaConf
from typing import Dict

# numpy random seed also applies to pandas functions
np.random.seed(RANDOM_SEED)
sklearn.set_config(transform_output="pandas")


def fit_and_predict(cfg: Dict):
    output = Path(cfg.output)
    outdir = Path(cfg.outdir) / "linear"
    outdir.makedirs_p()

    preprocess = PREPROCESSORS[cfg["preprocess"]](cfg)
    datadir = Path(cfg["datadir"])

    train_df = pd.read_csv(datadir / "train.csv")
    test_df = pd.read_csv(datadir / "test.csv")

    # Data preprocessing
    train_df = preprocess.apply(train_df)
    targets = train_df.monthly_rent
    train_df = train_df.drop(columns="monthly_rent")

    test_df = preprocess.apply(test_df)\
        .drop(columns="monthly_rent", errors='ignore')

    categorical_feats = train_df.select_dtypes(include=['object']).columns.tolist()

    transformer = make_column_transformer(
        (CatBoostEncoder(), categorical_feats),
        remainder='passthrough'
    )

    train = transformer.fit_transform(train_df, targets)
    train_df = pd.DataFrame(train, columns=transformer.get_feature_names_out())
    test = transformer.transform(test_df)
    test_df = pd.DataFrame(test, columns=transformer.get_feature_names_out())

    model = Ridge(random_state=RANDOM_SEED)

    param_grid = {
        "alpha": cfg.alpha,
        "solver": cfg.solver
    }

    gsearch = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           scoring="neg_root_mean_squared_error",
                           n_jobs=-1,
                           refit=True,
                           cv=10)

    gsearch.fit(train_df, targets)
    print(f"Best score: {gsearch.best_score_}")
    print(f"Best params: {gsearch.best_params_}")

    model = gsearch.best_estimator_
    score = cross_val_score(
        model, train_df, targets,
        scoring="neg_root_mean_squared_error", cv=10, n_jobs=-1)
    print(f"CV score: {score}")

    model.fit(train_df, targets)
    predictions = model.predict(test_df)
    submission_df = pd.DataFrame(
        {
            'Id': list(range(len(test_df))),
            'Predicted': predictions
        }
    )
    submission_file = outdir / output
    submission_df.to_csv(submission_file, index=False)
    print(f"Save submission to {submission_file}")

    # save cross validation results
    cv_results = pd.DataFrame(-score, columns=["val-rmse"])
    cv_filename = output.stem + "_cross_validation.csv"
    cv_results.to_csv(outdir / cv_filename, index=False)


def main(
        config_path: Path = typer.Argument(
            ..., exists=True, file_okay=True, readable=True,
            help="Path to config file", path_type=Path
        )):
    cfg = OmegaConf.load(config_path)
    fit_and_predict(cfg)


if __name__ == "__main__":
    typer.run(main)
