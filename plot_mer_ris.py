import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools

plt.style.use('seaborn-whitegrid')

rcParams = matplotlib.rcParams

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

prefix = "/home/tvas/Dropbox/SICS/uncertain-trees/results/figures/small-mid-old-nooutliers/"

mer_path = prefix + "mean_error_rate.means.csv"
util_path = prefix + "utility.csv"
ris_path = prefix + "relative_interval_size.means.csv"

mers = pd.read_csv(mer_path)
mer = mers.set_index("Dataset").iloc[:-3]
util = pd.read_csv(util_path).set_index("Dataset")
ris = pd.read_csv(ris_path).set_index("Dataset").iloc[:-3]


def scatter_points(x_points, y_points, marker):
    xy_df = pd.DataFrame([x_points, y_points])
    plt.scatter(xy_df.iloc[0], xy_df.iloc[1], marker=marker)


markers = itertools.cycle(('o', '^', 's', 'x', '*'))
for method in ["MondrianForest", "OnlineQRF", "CPApproximate", "CPExact"]:
    scatter_points(mer[method], ris[method], marker=next(markers))  # Can use util or ris on the y axis


plt.legend()
plt.xlabel("MER")
plt.ylabel("RIS")
plt.ylim(ymax=1.05)

plt.axvline(x=0.10, linestyle='dashed', color='grey')

plt.savefig(prefix + "combined-mer-ris.pdf", bbox_inches='tight')