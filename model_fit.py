import pandas as pd
import numpy as np
from preprocessing import preprocess_v1, preprocess_v2, preprocess_v3, preprocess_v4
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
SUBMISSION_FILE = "submission_v3.csv"

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
    # 'town',
    'block',
    'flat_type',
    'flat_model',
    'subzone',
    'nearest_mrt_code',
    'nearest_mall_name',
    'nearest_school_name',
]
cat_features_ids = [idx for idx in range(len(train_df.columns))
                    if train_df.columns[idx] in cat_features]

trainer = CatBoostRegressor(
    learning_rate=0.05,
    iterations=3000,
    random_seed=RANDOM_SEED,
    l2_leaf_reg=50.0,
    depth=5,
    langevin=True,
    od_type="IncToDec", od_wait=20
)
trainer.fit(
    train_df, targets,
    cat_features=cat_features_ids,
    eval_set=(val_df, val_targets),
    use_best_model=True)

predictions = trainer.predict(test_df)
submission_df = pd.DataFrame(
    {
        'Id': list(range(len(test_df))),
        'Predicted': predictions
    }
)
submission_df.to_csv(SUBMISSION_FILE, index=False)



