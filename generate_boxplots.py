"""
Script to create boxplots for quantile loss and utility measurements
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
from pylab import rcParams
plt.style.use('seaborn-whitegrid')

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

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

    # Ensure method order matches rest of paper
    try:
        df.rename(index=str, columns={"MF": "MondrianForest"}, inplace=True)
    except KeyError:
        pass
    df = df[["MondrianForest", "OnlineQRF", "CPApproximate", "CPExact"]]

    # Get color palette
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    bp = ax.boxplot(df.transpose(), labels=df.columns)

    # Color boxes, from https://github.com/jbmouret/matplotlib_for_papers#colored-boxes
    for i in range(0, len(bp['boxes'])):
        bp['boxes'][i].set_color(colors[i])
        # we have two whiskers!
        bp['whiskers'][i*2].set_color(colors[i])
        bp['whiskers'][i*2 + 1].set_color(colors[i])
        bp['whiskers'][i*2].set_linewidth(2)
        bp['whiskers'][i*2 + 1].set_linewidth(2)
        # top and bottom fliers
        # (set allows us to set many parameters at once)
        bp['fliers'][i].set(markerfacecolor=colors[i],
                            marker='o', alpha=0.75, markersize=6,
                            markeredgecolor='none')
        bp['medians'][i].set_color('black')
        bp['medians'][i].set_linewidth(2)
        # and 4 caps to remove
        for c in bp['caps']:
            c.set_linewidth(0)

    # Fil boxes with color
    for i in range(0, len(bp['boxes'])):
        box = bp['boxes'][i]
        box.set_linewidth(0)
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
            boxCoords = np.array([boxX, boxY]).transpose()
            boxPolygon = Polygon(boxCoords, facecolor=colors[i], linewidth=0)
            ax.add_patch(boxPolygon)

    y_label = "Quantile Loss" if args.metric == "quantile_loss" else "Utility"
    ax.set_ylabel(y_label)
    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    # Remove vertical grid lines
    ax.grid(axis='x', color="1.0", linestyle='-', linewidth=1)
    ax.set_axisbelow(True)

    plt.savefig(str(input_path / (args.metric + ".pdf")), bbox_inches='tight')
