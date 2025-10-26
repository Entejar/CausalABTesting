"""
Basic plotting helpers.
"""
import matplotlib.pyplot as plt
import pandas as pd

def plot_uplift(curve: pd.DataFrame, path: str | None = None):
    plt.figure()
    plt.plot(curve["decile"], curve["uplift"], marker="o")
    plt.xlabel("Uplift decile (highest first)")
    plt.ylabel("Estimated uplift")
    plt.title("Uplift curve")
    if path:
        plt.savefig(path, bbox_inches="tight")
    return plt.gcf()
