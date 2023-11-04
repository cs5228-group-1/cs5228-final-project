# cs5228-final-project

## Installation

- Python 3.8

```bash
pip install -r requirements.txt
```

Following [kaggle-api setup](https://github.com/Kaggle/kaggle-api#api-credentials) to add your credentials.

Next, run `download_data.sh` script. The data is stored in `data` folder.

## Submission

Use `submit.sh` to submit with the following syntax:

```bash
./submit.sh <submit file> <message>
```

# Model Training

All configuration related to the training is defined in a YAML file. Below are parameters needed for the training:

- `preprocess`: Name of the preprocess classes. All defined in `preprocess.py` file.
- `datadir`: Directory contains the provided data.
- `outdir`: Directory of the output predictions.
- `output`: Filename of the output predictions.
- `learning_rate`: The learning rate of tree-based learning methods.
- `iterations`: Number of iterations of the training.
- `depth`: Maximum depth of the tree.
- `l2_leaf_reg`: Regularization tern in each leaf.
- `langevin`: (true/false) whether to enable Stochastic Gradient Langevin Boosting.
- `logging_level`: string or number: logging level or number of iterations to output logs.

To run model training:

```bash
python model_fit.py <config-file>
```
