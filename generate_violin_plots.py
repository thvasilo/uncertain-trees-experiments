"""
Script to create violin plots for quantile loss and utility measurements
"""

import argparse
from pathlib import Path

import seaborn as sns
import pandas as pd
from pylab import rcParams

large_text = 16
small_text = 14
params = {
    'axes.labelsize': large_text,
    'font.size': small_text,
    'legend.fontsize': small_text,
    'xtick.labelsize': small_text,
    'ytick.labelsize': small_text,
    'text.usetex': True,
    'figure.figsize': [9, 6]
}
rcParams.update(params)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="Input directory, should contain quantile_loss.means.csv"
                                                       "or adjusted_utility.csv file.")

    parser.add_argument("--metric", required=True, help="Metric to plot.", choices=['quantile_loss', "utility"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load results CSV
    filename = "quantile_loss.means.csv" if args.metric == "quantile_loss" else "exp_adjusted_utility.csv"
    input_path = Path(args.input)
    df = pd.read_csv(input_path / filename)[:-3].set_index("Dataset")

    # Ensure method order matches rest of paper
    try:
        # Some old experiments used MF for MondrianForest
        df.rename(index=str, columns={"MF": "MondrianForest"}, inplace=True)
    except KeyError:
        pass
    df = df[["MondrianForest", "OnlineQRF", "CPApproximate", "CPExact"]]

    # Melt data and add plots
    y_label = "Quantile Loss" if args.metric == "quantile_loss" else "Utility"
    melted_df = pd.melt(df, value_vars=["MondrianForest", "OnlineQRF", "CPApproximate", "CPExact"],
                        value_name=y_label, var_name="Method")
    ax = sns.violinplot(x="Method", y=y_label, data=melted_df, cut=0, inner=None, bw=0.4,
                        scale="width")
    ax = sns.swarmplot(x="Method", y=y_label, data=melted_df, edgecolor="grey", color="black")

    ax.set_ylabel(y_label)
    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    # Remove vertical grid lines
    ax.grid(axis='x', color="1.0", linestyle='-', linewidth=1)
    ax.set_axisbelow(True)

    ax.get_figure().savefig(str(input_path / (args.metric + "-violin.pdf")), bbox_inches='tight')

