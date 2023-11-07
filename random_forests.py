from omegaconf import OmegaConf
import pandas as pd
from preprocessing import TARGET_ATTR
from common import (
    RANDOM_SEED, PREPROCESSORS, N_FOLDS,
    create_submission_file, cross_validation_on_model
)
from path import Path
import typer
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from typing import Dict


def fit_and_predict(cfg: Dict):
    output = Path(cfg.output)
    outdir = Path(cfg.outdir) / "random_forests"
    outdir.makedirs_p()
    submission_file = outdir / output

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
        # (OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_feats),
        (CatBoostEncoder(), categorical_feats),
        remainder='passthrough'
    )
    train = transformer.fit_transform(train_df, targets)
    train_df = pd.DataFrame(train, columns=transformer.get_feature_names_out())
    test = transformer.transform(test_df)
    test_df = pd.DataFrame(test, columns=transformer.get_feature_names_out())

    model = RandomForestRegressor(n_jobs=-1)

    param_grid = {
        "n_estimators": cfg['n_estimators'],
        "max_depth": cfg['max_depth'],
        "min_samples_leaf": cfg['min_samples_leaf'],
    }

    gsearch = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           scoring="neg_root_mean_squared_error",
                           n_jobs=None,
                           refit=True,
                           cv=N_FOLDS, verbose=100)

    gsearch.fit(train_df, targets)
    print(f"Best score: {gsearch.best_score_}")
    print(f"Best params: {gsearch.best_params_}")

    model = gsearch.best_estimator_
    cv_filename = output.stem + "_cross_validation.csv"
    cv_path = outdir / cv_filename
    cross_validation_on_model(model, train_df, targets, cv_path)

    features = transformer.get_feature_names_out()
    feature_importances = model.feature_importances_

    feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

    create_submission_file(model, test_df, submission_file)

    feat_imp_file = outdir / output.stem + "_Feat_imp.csv"
    feat_imp_df.to_csv(feat_imp_file, index=False)


def main(
        config_path: Path = typer.Argument(
            ..., exists=True, file_okay=True, readable=True,
            help="Path to config file", path_type=Path
        )):
    cfg = OmegaConf.load(config_path)
    fit_and_predict(cfg)


if __name__ == "__main__":
    typer.run(main)
