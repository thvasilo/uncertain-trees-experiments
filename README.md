# Experiments scripts for online trees with uncertainty.

This is a collection of scripts that we used during our work
on predictive intervals for online random forests.

It is meant to be used as a part of larger reproducibility
repo, available [here](https://github.com/thvasilo/uncertain-trees-reproducible).

We describe each file separately.

## Experiment automation

The are python scripts that automate the experiments.

### moa_experiments.py

Used to run the MOA experiments of the online conformal
prediction and OnlineQRF methods.
It requires access to a compiled JAR based on [our fork](https://github.com/thvasilo/moa/tree/uncertain-trees)
of MOA that includes these classes.

The expected data format is arff. The argument help contains all the relevant options,
check with `python moa_experiments.py -h`

Usage example:

```bash
python moa_experiments.py --moajar /path/to/moa.jar --input /path/to/data --meta OnlineQRF
```

### skgarden_experiments.py

Used to run the Mondrian Forest experiments using our [fork](https://github.com/thvasilo/scikit-garden/tree/interval-predictions)
of scikit-garden.


Usage: `python skgarden_experiments.py --input path/to/data`

As above, you can check the argument help to get all the relevant
options.

The output is one csv file per dataset, per experiment repeat.
Will also output two additional files per experiment:
\<name\>.time.csv
\<name\>.pred

These contain timing measurements and each individual prediction.

Makes use of `evaluation_functions.py` that contains `skleaner`-like evaluation
functions adjusted for interval predictors.

### parameter_sweep.py

This executable script is designed to call the two previous scripts in order to facilitate
sweeping over algorithm parameters. Example usage:

```bash
./parameter_sweep.py --command "python moa_experiments.py  --moajar $MOA_JAR \
  --input /home/user/data/moa  --repeats 10 --window 1000 --njobs 4 \
  --learner-threads 1 --verbose 1" \
  --output-prefix /home/user/output/uncertain-trees/moa --sweep-argument meta\
  --argument-list OnlineQRF OoBConformalApproximate OoBConformalRegressor
```

The above will call the `moa_experiments.py` with the provided command,
substituting the `sweep-argument` for each element of the `argument-list`.
The output will be created under the prefix directory, creating a subdir
for each parameter setting, in this case each method will have its own
subdir:
```bash
ls /home/user/output/uncertain-trees/moa
# OnlineQRF OoBConformalApproximate OoBConformalRegressor
```

It's also possible to call with a range argument:

```bash
./parameter_sweep.py --command "python moa_experiments.py  --moajar $MOA_JAR \
  --input /home/user/data/moa  --repeats 10 --window 1000 --njobs 2 --meta OoBConformalRegressor \
  --learner-threads 2 --overwrite --verbose 1" --sweep-argument max-calibration-instances \
  --output-prefix /home/user/output/moa/onlinecp-max-cal-instances --argument-range 100 1001 100
```

The above will call the `moa_experiments.py` script, setting the `--max-calibration-instances` parameter
to a value of 100-1000 with a step of 100.

Finally the script can do two levels of sweeps using `--inner-sweep-parameter` and
`--inner-argument-list` or `--inner-argument-range` with the same syntax.

## Generating ouput

The following scripts use the output of the experiment automation scripts
to produce figures and TeX tables.

### generate_figures.py

This script can take as input a directory that contains sub-directories,
each corresponding to a method or parameter setting and produces TeX
tables and figures if requested.

For example given the example run in `parameter_sweep.py`, we could run:

```bash
python generate_figures.py --input /home/user/output/uncertain-trees/moa \
  --input /home/user/figures/moa
```

This would create comparison figures and TeX tables under the input directory.

### interval_metrics.py

This script is used to generate post-hoc metrics for the methods, like Relative
Interval Size and overall correctness. The syntax is the same as for `generate_figures.py`.

It does so by parsing the produced prediction files, selecting the correct parser
to use based on the directory name. The fall back parser is for Mondrian Forests,
we can use `--force-moa` to enforce MOA parsing (e.g. when parsing parameter sweeps where dirs
have non-method names)