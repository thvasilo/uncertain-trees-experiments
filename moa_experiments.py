import argparse
import json
from collections import defaultdict
from pathlib import Path
from subprocess import run

from joblib import Parallel, delayed


def main(argv):
    """
    Runs a number of experiments using MOA meta-learners.
    The user provides a datadir which contains a number of arff files for classification, and a MOA
    EvaluatePrequentialRegression(IntervalRegressionPerformanceEvaluator) task is ran on each one.


    Usage: python moa_experiments.py --moajar /path/to/moa.jar --datadir /path/to/data --meta meta.OnlineQRF
    """
    parser = argparse.ArgumentParser(description="Runs MOA experiments and stores results into files")
    parser.add_argument("--moajar", help="Path to MOA jar", required=True)
    parser.add_argument("--datadir", help="A directory containing one or more arff data files", required=True)
    parser.add_argument("--meta", help="The meta algorithm to use for training", required=True,
                        choices=["meta.OnlineQRF", "meta.OoBConformalRegressor"])
    parser.add_argument("--calibration-file", default="",
                        help="An arff file containing calibration instances, "
                             "to be used only with meta.ConformalRegressor.")
    # parser.add_argument("--base", help="The base algorithm to use for training")
    parser.add_argument("--repeats", help="Number of times to repeat each experiment", type=int, default=1)
    parser.add_argument("--window", help="Performance report window size", type=int, default=1000)
    parser.add_argument("--njobs", help="Number of experiment jobs to run in parallel, max one per input file", type=int, default=1)
    parser.add_argument("--learner-threads", help="Number of threads to use for the learner", type=int, default=1)
    parser.add_argument("--outputdir", help="The directory to place the output files. If not given, creates "
                                            "dir under datadir.")
    parser.add_argument("--ensemble-size", help="The size of the ensemble", type=int, default=10)
    parser.add_argument("--max-calibration-instances", type=int, default=1000,
                        help="The max size of the calibration set.")
    parser.add_argument("--confidence", help="The confidence level of the predictions", type=float, default=0.9)
    parser.add_argument("--stdout", default=False, action="store_true",
                        help="When given, output results to stdout only instead of file")
    parser.add_argument("--overwrite", default=False, action="store_true",
                        help="When given, it will not check if the output folder exists already.")
    # TODO: Other params I want to investigate?

    args = parser.parse_args(argv)

    # Tailor the command to each framework

    task = "EvaluatePrequentialRegression -e (IntervalRegressionPerformanceEvaluator -w {})".format(args.window)
    moa_jar_path = Path(args.moajar)
    moa_dir = moa_jar_path.parent
    # TODO: Remove the sizeofag to speed up experiment runtime? Only keep if we want actually investigate
    command_prefix = "java -cp \"{moa_jar}:{moa_dir}/dependency-jars/*\" " \
                     "-javaagent:{moa_dir}/dependency-jars/sizeofag-1.0.0.jar " \
                     "moa.DoTask ".format(moa_dir=moa_dir, moa_jar=moa_jar_path)

    # Set up input and output dirs
    data_path = Path(args.datadir)

    # TODO: Customization for base learner (buckets will be necessary)
    if args.meta == "meta.OnlineQRF":
        base_learner = "(trees.FIMTQR -e)"
    else:
        base_learner = "(trees.FIMTDD -e)"

    # If the user did not provide an output dir, put results under the data folder
    if args.outputdir is None and not args.stdout:
        output_path = data_path / args.meta.split('.')[1]
        print("Will try to create directory {} to store the results".format(output_path))
    else:
        output_path = Path(args.outputdir)

    # Create the output dir if needed
    output_path.mkdir(parents=True,
                      exist_ok=args.overwrite)

    # Run experiments for each data file
    commands = []
    commands_per_file = defaultdict(list)
    for arff_file in data_path.glob("*.arff"):
        learner = "{meta} -l {base} -s {size} -a {confidence} -j {threads}".format(
            meta=args.meta, base=base_learner, size=args.ensemble_size,
            confidence=args.confidence, threads=args.learner_threads)
        if args.meta != "meta.OnlineQRF":
            learner += " -i {cal_size} -c {cal_file}".format(
                cal_size=args.max_calibration_instances, cal_file=args.calibration_file)
        for i in range(args.repeats):
            command = command_prefix + "\"" + "{task} -l ({learner}) " \
                                              "-s (ArffFileStream -f {arff_file}) " \
                                              "-f {window}".format(task=task, learner=learner,
                                                                   arff_file=data_path / arff_file,
                                                                   window=args.window)
            if not args.stdout:
                command += " -d {}".format(output_path / (arff_file.stem + "_{}.csv".format(i)))
            command += "\""  # Quotes necessary because of parentheses
            commands_per_file[arff_file].append(command)
            commands.append(command)
            # print("Executing command: {}".format(command))

    # Parallelize over files, ensuring that there's only one process at any time reading the
    # same file. Otherwise parallel performance is crap. If not enough files, compensate with
    # learner_threads > 1
    with Parallel(n_jobs=args.njobs) as parallel:
        for i in range(args.repeats):
            parallel(delayed(run)(commands[i], shell=True, check=True)
                     for arff_file, commands in commands_per_file.items())

    # Write the settings for the experiment
    json_file = output_path / "settings.json"
    settings = vars(args)
    json_file.write_text(json.dumps(settings))
    print("\nResults files created under {}".format(output_path))


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
