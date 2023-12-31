import pandas as pd
import numpy as np
from preprocessing import cat_attr_to_id, TARGET_ATTR
from common import RANDOM_SEED, PREPROCESSORS, N_FOLDS
from catboost import CatBoostRegressor
from path import Path
import typer
from omegaconf import OmegaConf
from typing import Dict
from sklearn.model_selection import KFold
import seaborn as sns

# numpy random seed also applies to pandas functions
np.random.seed(RANDOM_SEED)


def cross_validation(cfg: Dict):
    output = Path(cfg.output)
    outdir = Path(cfg.outdir) / "catboost"
    outdir.makedirs_p()
    preprocess = PREPROCESSORS[cfg["preprocess"]](cfg)
    datadir = Path(cfg["datadir"])

    data = pd.read_csv(datadir / "train.csv").sample(frac=1.0, random_state=RANDOM_SEED)

    # Data preprocessing
    data = preprocess.apply(data)
    kf = KFold(n_splits=N_FOLDS)

    # Model Setup
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
    cv_data = []

    for train_idx, val_idx in kf.split(data):
        train_df = data.iloc[train_idx].copy()
        targets = train_df.monthly_rent

        train_df = train_df.drop(columns=TARGET_ATTR)

        val_df = data.iloc[val_idx].copy()
        val_targets = val_df.monthly_rent
        val_df = val_df.drop(columns=TARGET_ATTR)

        trainer.fit(
            train_df, targets,
            cat_features=cat_attr_to_id(train_df),
            eval_set=(val_df, val_targets),
            use_best_model=True)

        cv_data.append(trainer.get_best_score()['validation']['RMSE'])

    print(cv_data)
    cv_results = pd.DataFrame(cv_data, columns=["val-rmse"])
    cv_filename = output.stem + "_cross_validation.csv"
    cv_results.to_csv(outdir / cv_filename, index=False)


def fit_and_predict(cfg: Dict):
    output = Path(cfg.output)
    outdir = Path(cfg.outdir) / "catboost"
    outdir.makedirs_p()
    submission_file = outdir / output
    preprocess = PREPROCESSORS[cfg["preprocess"]](cfg)
    datadir = Path(cfg["datadir"])

    train_df = pd.read_csv(datadir / "train.csv").sample(frac=1.0, random_state=RANDOM_SEED)
    test_df = pd.read_csv(datadir / "test.csv")

    # Data preprocessing
    train_df = preprocess.apply(train_df)
    targets = train_df.monthly_rent

    train_df = train_df.drop(columns=TARGET_ATTR)

    test_df = preprocess.apply(test_df)\
        .drop(columns=TARGET_ATTR, errors='ignore')

    cat_features = cat_attr_to_id(train_df)

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
        cat_features=cat_features)

    predictions = trainer.predict(test_df)
    submission_df = pd.DataFrame(
        {
            'Id': list(range(len(test_df))),
            'Predicted': predictions
        }
    )
    submission_df.to_csv(submission_file, index=False)
    print(f"Save submission to {submission_file}")

    # plot feature importances
    feature_importance = trainer.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    feat_names = np.array(test_df.columns)[sorted_idx]
    feat_plot = sns.barplot(y=feat_names, x=feature_importance[sorted_idx], hue=feat_names)
    fig = feat_plot.get_figure()
    fig.savefig(f"figures/catboost_{cfg.preprocess}_feat_importances.pdf", dpi=350, bbox_inches="tight")


def main(
        config_path: Path = typer.Argument(
            ..., exists=True, file_okay=True, readable=True,
            help="Path to config file", path_type=Path
        )):
    cfg = OmegaConf.load(config_path)
    fit_and_predict(cfg)
    cross_validation(cfg)


if __name__ == "__main__":
    typer.run(main)
