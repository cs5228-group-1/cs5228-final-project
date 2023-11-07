import pandas as pd
from sklearn.model_selection import cross_val_score

from preprocessing import V1, V2, V3, V4, V5
RANDOM_SEED = 42
N_FOLDS = 10

PREPROCESSORS = {
    'V1': V1,
    'V2': V2,
    'V3': V3,
    'V4': V4,
    'V5': V5,
}


def create_submission_file(model, test_df, submission_file):
    predictions = model.predict(test_df)
    submission_df = pd.DataFrame(
        {
            'Id': list(range(len(test_df))),
            'Predicted': predictions
        }
    )
    submission_df.to_csv(submission_file, index=False)
    print(f"Save submission to {submission_file}")


def cross_validation_on_model(model, train_df, targets, cv_path):
    score = cross_val_score(
        model, train_df, targets,
        scoring="neg_root_mean_squared_error",
        cv=N_FOLDS, n_jobs=-1)

    print(f"CV score: {-score}")

    model.fit(train_df, targets)

    # save cross validation results
    cv_results = pd.DataFrame(-score, columns=["val-rmse"])
    cv_results.to_csv(cv_path, index=False)
    print(f"Saved cv score to {cv_path}")

