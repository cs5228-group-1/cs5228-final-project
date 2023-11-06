import pandas as pd
import numpy as np
from preprocessing import cat_attr_to_id
from common import RANDOM_SEED, PREPROCESSORS
from catboost import CatBoostRegressor
import sklearn
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from path import Path
import typer
from omegaconf import OmegaConf
from typing import Dict

# numpy random seed also applies to pandas functions
np.random.seed(RANDOM_SEED)

def fit_and_predict(cfg: Dict):
    submission_file = Path(cfg["outdir"]) / "linear" / Path(cfg["output"])
    Path(submission_file.parent[0]).makedirs_p()
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
        (OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_feats),
        remainder='passthrough'
    )

    train = transformer.fit_transform(train_df)
    train_df = pd.DataFrame(train, columns=transformer.get_feature_names_out())
    test = transformer.transform(test_df)
    test_df = pd.DataFrame(test, columns=transformer.get_feature_names_out())



    # param_grid = {
    #     "n_estimators": [1000],
    #     "max_depth": [15,20],
    #     #"min_samples_split": [2,3],
    #     "min_samples_leaf": [10],
    # }

    # gsearch = GridSearchCV(estimator=model,
    #                        param_grid=param_grid,
    #                        scoring="neg_root_mean_squared_error",
    #                        n_jobs=-1,
    #                        refit=True,
    #                        cv=10)
    #
    # gsearch.fit(train_df, targets)
    # print(f"Best score: {gsearch.best_score_}")
    # print(f"Best params: {gsearch.best_params_}")

    # rf = gsearch.best_estimator_
    # with sklearn.config_context(assume_finite=True):
    model = Ridge(alpha=1e-9, random_state=RANDOM_SEED, solver="cholesky")
    score = cross_val_score(model, train_df, targets, scoring="neg_root_mean_squared_error", cv=10, n_jobs=-1)
    print(f"CV score: {score}")

    #cv = KFold(n_splits=2, shuffle=True, random_state=42)

    #scores = cross_val_score(pipeline, train_df, targets, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1)

    #print(f"Mean: {np.mean(scores)}, std dev: {np.std(scores)}")

    # features, feature_importances = transformer.get_feature_names_out(), rf.feature_importances_
    # feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    # feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

    predictions = model.predict(test_df)
    submission_df = pd.DataFrame(
        {
            'Id': list(range(len(test_df))),
            'Predicted': predictions
        }
    )
    submission_df.to_csv(submission_file, index=False)
    print(f"Save submission to {submission_file}")

    # df = pd.DataFrame(gsearch.cv_results_)
    # with pd.ExcelWriter("rf_results_.xlsx") as writer:
    #     df.to_excel(writer, sheet_name='CV_results', index=False)
    #     feat_imp_df.to_excel(writer, sheet_name='Feat_imp', index=False)

def main(
        config_path: Path = typer.Argument(
            ..., exists=True, file_okay=True, readable=True,
            help="Path to config file", path_type=Path
        )):
    cfg = OmegaConf.load(config_path)
    fit_and_predict(cfg)


if __name__ == "__main__":
    typer.run(main)
