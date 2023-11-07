"""
Plot cross-validation results
"""
import typer
import numpy as np
import pandas as pd
from path import Path
import matplotlib.pyplot as plt

INPUT_DIR = Path("submissions")
SETTING_LISTS = [
    f"setting_{idx}" for idx in range(1, 5)
]


def main(output_path: Path = typer.Argument(..., exists=False, file_okay=True, writable=True, help="Output figure path", path_type=Path), 
         model_name: str = typer.Argument(..., exists=False, help="Model Name")):
    cross_validations = {
        name: pd.read_csv(INPUT_DIR / f"{name}_cross_validation.csv")['val-rmse'].values
        for name in SETTING_LISTS
    }

    for setting_name, losses in cross_validations.items():
        plt.scatter([setting_name] * len(losses), losses)

    loss_mean = np.array([np.mean(v) for k,v in sorted(cross_validations.items())])
    loss_std = np.array([np.std(v) for k,v in sorted(cross_validations.items())])
    
    plt.title(f"{model_name}")
    plt.errorbar(sorted(cross_validations.keys()), loss_mean, yerr=loss_std)

    plt.xlabel("Setting")
    plt.ylabel("Cross-validation RSME score")
    plt.savefig(output_path, dpi=350, bbox_inches="tight")


if __name__ == "__main__":
    typer.run(main)
