import pandas as pd
import numpy as np
from preprocessing import preprocess_v1, preprocess_v2, preprocess_v3
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from path import Path


RANDOM_SEED = 42
SUBMISSION_FILE = Path("./submissions") / "submission_v3_hyper3.csv"

param_space = {
    'learning_rate': hp.choice('learning_rate', list(np.arange(0.01, 0.11, 0.05))),
    'l2_leaf_reg': hp.choice('l2_leaf_reg', [20.0, 40.0, 60.0, 80.0]),
    'depth': hp.randint('depth', 3, 6),
    'bootstrap_type': hp.choice('bootstrap_type', ['Bayesian', 'Bernoulli', "MVS"]),
    'langevin': hp.choice('langevin', [True, False]),
}

# numpy random seed also applies to pandas functions
np.random.seed(RANDOM_SEED)

train_df = pd.read_csv("./data/train.csv").sample(frac=1.0)
test_df = pd.read_csv("./data/test.csv")

# choose preprocess steps
preprocess_func = preprocess_v3

# Data preprocessing
train_df, val_df = train_test_split(preprocess_func(train_df), test_size=0.1)
targets = train_df.monthly_rent
train_df = train_df.drop(columns="monthly_rent")

val_targets = val_df.monthly_rent
val_df = val_df.drop(columns="monthly_rent")

floor_area_sqm = test_df.floor_area_sqm.copy()
test_df = preprocess_func(test_df).drop(columns="monthly_rent", errors='ignore')

cat_features = [
    'street_name',
    'block',
    'flat_type',
    'flat_model',
    'subzone',
    'nearest_mrt_code',
    'nearest_mall_name',
    'nearest_school_name',
]
cat_features_ids = [idx for idx in range(len(train_df.columns)) if train_df.columns[idx] in cat_features]


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
        cat_features=cat_features_ids,
        eval_set=(val_df, val_targets),
        use_best_model=True)
    return {'loss': trainer.get_best_score()['validation']['RMSE'], 'status': STATUS_OK}


best_params = fmin(
    objective, space=param_space, algo=tpe.suggest, max_evals=64)
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
    cat_features=cat_features_ids,
    eval_set=(val_df, val_targets),
    use_best_model=True)

predictions = candidate.predict(test_df)
submission_df = pd.DataFrame(
    {
        'Id': list(range(len(test_df))),
        'Predicted': predictions
    }
)
submission_df.to_csv(SUBMISSION_FILE, index=False)
print(f"Best params: {hyper_params}")
print("Done !!")
