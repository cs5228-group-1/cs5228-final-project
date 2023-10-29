import pandas as pd
import numpy as np
from preprocessing import cat_attr_to_id
from common import RANDOM_SEED, PREPROCESSORS
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
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


def main(
        config_path: Path = typer.Argument(
            ..., exists=True, file_okay=True, readable=True,
            help="Path to config file", path_type=Path
        )):
    cfg = OmegaConf.load(config_path)
    fit_and_predict(cfg)


if __name__ == "__main__":
    typer.run(main)
