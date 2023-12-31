"""
Plot cross-validation results
"""
import typer
import numpy as np
import pandas as pd
from path import Path
import matplotlib.pyplot as plt


SETTING_LISTS = range(1, 6)


def main(
        input_dir: Path = typer.Argument(..., exists=True, dir_okay=True, readable=True, help="Input dir contains all cross validation results", path_type=Path), 
        output_path: Path = typer.Argument(..., exists=False, file_okay=True, writable=True, help="Output figure path", path_type=Path), 
        model_name: str = typer.Argument(..., exists=False, help="Model Name")):

    cross_validations = {
        f"V{name}": pd.read_csv(input_dir / f"setting_{name}_cross_validation.csv")['val-rmse'].values
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
