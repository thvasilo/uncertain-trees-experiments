import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools

plt.style.use(['seaborn-whitegrid'])

rcParams = matplotlib.rcParams

large_text = 16
small_text = 14
params = {
  'axes.labelsize': large_text,
  'font.size': small_text,
  'legend.fontsize': small_text,
  'xtick.labelsize': small_text,
  'ytick.labelsize': small_text,
  'text.usetex': False,
  'figure.figsize': [6, 6]
}
rcParams.update(params)

mer_path = "/home/tvas/Dropbox/SICS/uncertain-trees/results/figures/friedman/mean_error_rate.means.csv"
util_path = "/home/tvas/Dropbox/SICS/uncertain-trees/results/figures/friedman/utility.csv"
ris_path = "/home/tvas/Dropbox/SICS/uncertain-trees/results/figures/friedman/relative_interval_size.means.csv"

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

# mf_xy  = pd.DataFrame([mer["MondrianForest"].iloc[:-3], util["MondrianForest"]])
# plt.scatter(mf_xy.iloc[0], mf_xy.iloc[1])

# qrf_xy  = pd.DataFrame([mer["OnlineQRF"].iloc[:-3], util["OnlineQRF"]])
# plt.scatter(qrf_xy.iloc[0], qrf_xy.iloc[1])

# cpapprox_xy  = pd.DataFrame([mer["CPApproximate"].iloc[:-3], util["CPApproximate"]])
# plt.scatter(cpapprox_xy.iloc[0], cpapprox_xy.iloc[1])

# cpexact_xy  = pd.DataFrame([mer["CPExact"].iloc[:-3], util["CPExact"]])
# plt.scatter(cpexact_xy.iloc[0], cpexact_xy.iloc[1])

plt.legend()
plt.xlabel("MER")
plt.ylabel("RIS")
plt.ylim(ymax=1.05)

plt.axvline(x=0.10, linestyle='dashed', color='grey')
