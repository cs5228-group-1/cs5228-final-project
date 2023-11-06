import pandas as pd
import numpy as np
from preprocessing import cat_attr_to_id
from common import RANDOM_SEED, PREPROCESSORS
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.ensemble import RandomForestRegressor
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
    submission_file = Path(cfg["outdir"]) / Path(cfg["output"])
    preprocess = PREPROCESSORS[cfg["preprocess"]](cfg)
    datadir = Path(cfg["datadir"])

    train_df = pd.read_csv(datadir / "train.csv").sample(frac=1.0)
    test_df = pd.read_csv(datadir / "test.csv")

    # Data preprocessing
    train_df, val_df = train_test_split(preprocess.apply(train_df), test_size=0.1)
    targets = train_df.monthly_rent
    train_df = train_df.drop(columns="monthly_rent")

    val_targets = val_df.monthly_rent
    val_df = val_df.drop(columns="monthly_rent")

    test_df = preprocess.apply(test_df)\
        .drop(columns="monthly_rent", errors='ignore')

    print(train_df.info())
    print(train_df.iloc[:5,:6])
    print(train_df.iloc[:5,6:12])
    print(train_df.iloc[:5,12:])
    assert True==False, print("yo")

    trainer = CatBoostRegressor(
        learning_rate=cfg["learning_rate"],
        iterations=cfg["iterations"],
        random_seed=RANDOM_SEED,
        l2_leaf_reg=cfg["l2_leaf_reg"],
        depth=cfg["depth"],
        langevin=cfg["langevin"],
        od_type="IncToDec", od_wait=20, od_pval=1e-3,
        verbose=200
    )
    trainer.fit(
        train_df, targets,
        cat_features=cat_attr_to_id(train_df),
        eval_set=(val_df, val_targets),
        use_best_model=True)

    predictions = trainer.predict(test_df)
    submission_df = pd.DataFrame(
        {
            'Id': list(range(len(test_df))),
            'Predicted': predictions
        }
    )
    submission_df.to_csv(submission_file, index=False)
    print(f"Save submission to {submission_file}")

def fit_and_predict_rf(cfg: Dict):
    submission_file = Path(cfg["outdir"]) / Path(cfg["output"])
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

    # print(train_df.info())
    # print(train_df.iloc[:5,:6])
    # print(train_df.iloc[:5,6:12])
    # print(train_df.iloc[:5,12:])

    #numerical_feats = train_df.select_dtypes(include=['float64', 'int64']).columns
    categorical_feats = train_df.select_dtypes(include=['object']).columns.tolist()

    transformer = make_column_transformer(
        (OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_feats),
        remainder='passthrough'
    )
    #print(f"train df {train_df.iloc[0,:]}")
    train = transformer.fit_transform(train_df)
    train_df = pd.DataFrame(train, columns=transformer.get_feature_names_out())
    test = transformer.transform(test_df)
    test_df = pd.DataFrame(test, columns=transformer.get_feature_names_out())

    # t = [('cat', OneHotEncoder(handle_unknown='ignore'), categorical_feats)]
    # col_transform = ColumnTransformer(transformers=t, remainder="passthrough")

    model = RandomForestRegressor(n_jobs=-1)

    # pipeline = Pipeline([('prep', col_transform), ('model', model)])  # param_grid 'model__' prefix

    param_grid = {
        "n_estimators": [500],
        "max_depth": [15,20,25],
        #"min_samples_split": [2,3],
        "min_samples_leaf": [5,7],
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

    rf = gsearch.best_estimator_
    score = cross_val_score(rf, train_df, targets, scoring="neg_root_mean_squared_error", cv=10)
    print(f"CV score: {score}")

    #cv = KFold(n_splits=2, shuffle=True, random_state=42)

    #scores = cross_val_score(pipeline, train_df, targets, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1)

    #print(f"Mean: {np.mean(scores)}, std dev: {np.std(scores)}")

    predictions = rf.predict(test_df)
    submission_df = pd.DataFrame(
        {
            'Id': list(range(len(test_df))),
            'Predicted': predictions
        }
    )
    submission_df.to_csv(submission_file, index=False)
    print(f"Save submission to {submission_file}")

    df = pd.DataFrame(gsearch.cv_results_)
    df.to_excel("rf_cv_results_.xlsx", index=False)

def main(
        config_path: Path = typer.Argument(
            ..., exists=True, file_okay=True, readable=True,
            help="Path to config file", path_type=Path
        )):
    cfg = OmegaConf.load(config_path)
    fit_and_predict_rf(cfg)


if __name__ == "__main__":
    typer.run(main)
