"""
Runs a number of experiments using MOA meta-learners.
The user provides an input which contains a number of arff files for regression, and a MOA
EvaluatePrequentialRegression(IntervalRegressionPerformanceEvaluator) task is run on each one.

The output is one csv file per dataset, per experiment repeat.

Usage: python moa_experiments.py --moajar /path/to/moa.jar --input /path/to/data --meta OnlineQRF
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from subprocess import run

from joblib import Parallel, delayed


def main():
    parser = argparse.ArgumentParser(description="Runs MOA experiments and stores results into files")
    parser.add_argument("--moajar", help="Path to MOA jar", required=True)
    parser.add_argument("--input", help="A directory containing one or more arff data files", required=True)
    parser.add_argument("--meta", help="The meta algorithm to use for training", required=True,
                        choices=["OnlineQRF", "OoBConformalRegressor", "PredictiveVarianceRF",
                                 "OoBConformalApproximate"])
    parser.add_argument("--repeats", help="Number of times to repeat each experiment", type=int, default=1)
    parser.add_argument("--window", help="Performance report window size", type=int, default=1000)
    parser.add_argument("--njobs", type=int, default=1,
                        help="Number of experiment jobs to run in parallel, max one per input file")
    parser.add_argument("--learner-threads", help="Number of threads to use for the learner", type=int, default=1)
    parser.add_argument("--output", help="The directory to place the output files. If not given, creates "
                                         "directory under input, using the meta learner name.")
    parser.add_argument("--ensemble-size", help="The size of the ensemble", type=int, default=10)
    parser.add_argument("--max-calibration-instances", type=int, default=1000,
                        help="The max size of the calibration set.")
    parser.add_argument("--num-bins", type=int, default=100,
                        help="Number of bins to use for each leaf histogram.")
    parser.add_argument("--confidence", help="The confidence level of the predictions", type=float, default=0.9)
    parser.add_argument("--stdout", default=False, action="store_true",
                        help="When given, output results to stdout only instead of file")
    parser.add_argument("--overwrite", default=False, action="store_true",
                        help="When given, it will not check if the output folder exists already.")
    parser.add_argument("--dont-save-predictions", default=False, action="store_true",
                        help="When given, will not create file with predictions.")
    parser.add_argument("--dont-measure-model-size", default=False, action="store_true",
                        help="When given, will not report the size of the model in the results.")
    parser.add_argument("--verbose", type=int, default=0,
                        help="Set to 1 output per experiment, 2 to write MOA output to stdout.")

    args = parser.parse_args()

    # Tailor the command to each framework

    task = "EvaluatePrequentialRegression -e (IntervalRegressionPerformanceEvaluator -w {})".format(args.window)
    moa_jar_path = Path(args.moajar)
    moa_dir = moa_jar_path.parent

    command_prefix = "java -cp \"{moa_jar}:{moa_dir}/dependency-jars/*\" ".format(moa_dir=moa_dir, moa_jar=moa_jar_path)
    if not args.dont_measure_model_size:
        command_prefix += "-javaagent:{}/dependency-jars/sizeofag-1.0.0.jar ".format(moa_dir)

    # Set up input and output dirs
    data_path = Path(args.input).absolute()

    if args.meta == "OnlineQRF":
        base_learner = "(trees.FIMTQR -e)"
    else:
        base_learner = "(trees.FIMTDD -e)"

    # If the user did not provide an output dir, put results under the data folder
    if args.output is None and not args.stdout:
        output_path = (data_path / args.meta).absolute()
        print("Will try to create directory {} to store the results".format(output_path))
    else:
        output_path = Path(args.output).absolute()

    # Create the output dir if needed
    output_path.mkdir(parents=True,
                      exist_ok=args.overwrite)

    # Run experiments for each data file
    commands = []
    commands_per_file = defaultdict(list)
    for arff_file in data_path.glob("*.arff"):
        learner = "meta.{meta} -l {base} -s {size} -a {confidence} -j {threads}".format(
            meta=args.meta, base=base_learner, size=args.ensemble_size,
            confidence=args.confidence, threads=args.learner_threads)
        if args.meta != "OnlineQRF":
            learner += " -i {cal_size} ".format(cal_size=args.max_calibration_instances)
        else:
            learner += " -b {}".format(args.num_bins)
        for i in range(args.repeats):
            command = command_prefix + "moa.DoTask \" {task} -l ({learner}) " \
                                              "-s (ArffFileStream -f {arff_file}) " \
                                              "-f {window}".format(task=task, learner=learner,
                                                                   arff_file=data_path / arff_file,
                                                                   window=args.window)
            if not args.stdout:
                command += " -d {}".format(output_path / (arff_file.stem + "_{}.csv".format(i)))
                if not args.dont_save_predictions:
                    command += " -o {}".format(output_path / (arff_file.stem + "_{}.pred".format(i)))
            command += "\""  # Quotes necessary because of parentheses
            commands_per_file[arff_file].append(command)
            commands.append(command)
            # print("Executing command: {}".format(command))

    # Parallelize over files, ensuring that there's only one process at any time reading the
    # same file. Otherwise parallel performance is crap. If not enough files, compensate with
    # learner_threads > 1
    # TODO: This is horrible job separation, long-running jobs are holding back the rest
    # at each repeat iteration. Figure out different way.
    with Parallel(n_jobs=args.njobs, verbose=args.verbose) as parallel:
        for i in range(args.repeats):
            print("Running repeat {}/{}".format(i+1, args.repeats))
            parallel(delayed(run)(commands[i], shell=True)
                     for arff_file, commands in commands_per_file.items())

    # Write the settings for the experiment
    json_file = output_path / "settings.json"
    settings = vars(args)
    json_file.write_text(json.dumps(settings))
    print("\nResults files created under {}".format(output_path))


if __name__ == '__main__':
    main()
