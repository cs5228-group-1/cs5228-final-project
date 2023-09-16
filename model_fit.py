import pandas as pd
from preprocessing import preprocess_v1, preprocess_v2
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

# Data preprocessing
train_df, val_df = train_test_split(preprocess_v2(train_df), test_size=0.1)
targets = train_df.monthly_rent
train_df = train_df.drop(columns="monthly_rent")

val_targets = val_df.monthly_rent
val_df = val_df.drop(columns="monthly_rent")

test_df = preprocess_v2(test_df)

cat_features = [
    # 'town',
    'flat_type',
    'flat_model',
    'subzone',
    'nearest_mrt_code'
]
cat_features_ids = [idx for idx in range(len(train_df.columns)) if train_df.columns[idx] in cat_features]

trainer = CatBoostRegressor(
    learning_rate=0.1,
    iterations=2000,
    od_type="Iter", od_wait=40
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
submission_df.to_csv("submission_v1.csv", index=False)



