import pandas as pd
import numpy as np
from preprocessing import cat_attr_to_id
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from path import Path
import typer
from omegaconf import OmegaConf
from common import RANDOM_SEED, PREPROCESSORS
from typing import Dict

# numpy random seed also applies to pandas functions
np.random.seed(RANDOM_SEED)

param_space = {
    'learning_rate': hp.choice('learning_rate', list(np.arange(0.01, 0.11, 0.03))),
    'l2_leaf_reg': hp.choice('l2_leaf_reg', [20.0, 40.0, 60.0, 80.0]),
    'depth': hp.randint('depth', 4, 6),
    'bootstrap_type': hp.choice('bootstrap_type', ['Bayesian', 'Bernoulli', "MVS"]),
    'langevin': hp.choice('langevin', [True, False]),
}


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

    def objective(param_space):
        trainer = CatBoostRegressor(
            **param_space,
            random_seed=RANDOM_SEED,
            od_type="IncToDec", od_wait=20, od_pval=1e-3,
            iterations=5000,
            verbose=False
        )
        trainer.fit(
            train_df, targets,
            cat_features=cat_attr_to_id(train_df),
            eval_set=(val_df, val_targets),
            use_best_model=True)
        return {
            'loss': trainer.get_best_score()['validation']['RMSE'], 
            'status': STATUS_OK
        }


    best_params = fmin(
        objective, space=param_space, algo=tpe.suggest, max_evals=100)
    hyper_params = space_eval(param_space, best_params)

    candidate = CatBoostRegressor(
        learning_rate=hyper_params['learning_rate'],
        l2_leaf_reg=hyper_params['l2_leaf_reg'],
        depth=hyper_params['depth'],
        bootstrap_type=hyper_params['bootstrap_type'],
        langevin=hyper_params['langevin'],
        random_seed=RANDOM_SEED,
        od_type="IncToDec", od_wait=20, od_pval=1e-3,
        iterations=5000,
    )

    candidate.fit(
        train_df, targets,
        cat_features=cat_attr_to_id(train_df),
        eval_set=(val_df, val_targets),
        use_best_model=True)

    predictions = candidate.predict(test_df)
    submission_df = pd.DataFrame(
        {
            'Id': list(range(len(test_df))),
            'Predicted': predictions
        }
    )
    submission_df.to_csv(submission_file, index=False)
    print(f"Best params: {hyper_params}")
    print("Done !!")


def main(
        config_path: Path = typer.Argument(
            ..., exists=True, file_okay=True, readable=True,
            help="Path to config file", path_type=Path
        )):
    cfg = OmegaConf.load(config_path)
    fit_and_predict(cfg)


if __name__ == "__main__":
    typer.run(main)
