"""
Used to create combined MER/RIS figures, where we plot the MER on the x axis, and the RIS
on the y axis, with each dataset/experiment being a single point.
"""
import argparse
import itertools
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
rcParams = matplotlib.rcParams


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-mer", type=float, default=0.1)

    return parser.parse_args()


def main():
    args = parse_args()

    large_text = 16
    small_text = 14
    fig_width = 9
    fig_height = 6
    params = {
      'axes.labelsize': large_text,
      'font.size': small_text,
      'legend.fontsize': small_text,
      'xtick.labelsize': small_text,
      'ytick.labelsize': small_text,
      'text.usetex': True,
      'figure.figsize': [fig_width, fig_height]
    }
    rcParams.update(params)

    prefix = Path(args.input)

    mer_path = prefix / "mean_error_rate.means.csv"
    ris_path = prefix / "relative_interval_size.means.csv"

    mers = pd.read_csv(mer_path)
    mer = mers.set_index("Dataset").iloc[:-3]

    ris = pd.read_csv(ris_path).set_index("Dataset").iloc[:-3]

    def scatter_points(x_points, y_points, marker):
        xy_df = pd.DataFrame([x_points, y_points])
        plt.scatter(xy_df.iloc[0], xy_df.iloc[1], marker=marker)

    markers = itertools.cycle(('o', '^', 's', 'x', '*', 'D', 'J'))
    # TODO: Support any method available in csv read in
    for method in ["MondrianForest", "OnlineQRF", "CPApproximate", "CPExact"]:
        scatter_points(mer[method], ris[method], marker=next(markers))  # Can use util or ris on the y axis

    plt.legend()
    plt.xlabel("MER")
    plt.ylabel("RIS")
    plt.ylim(ymax=1.05, ymin=-0.05)

    plt.axvline(x=args.expected_mer, linestyle='dashed', color='grey')

    plt.savefig(prefix / "combined-mer-ris.pdf", bbox_inches='tight')


if __name__ == '__main__':
    main()