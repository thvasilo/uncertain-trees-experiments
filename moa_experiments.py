from pathlib import Path
import argparse
from subprocess import run
import json


def main(argv):
    """
    Runs a number of experiments using MOA meta-learners.
    The user provides a datadir which contains a number of arff files for classification, and a MOA
    PrequentialEvaluation task is ran on each one.

    The default arguments assume the script is being run with the MOA distribution as the working directory.

    Usage: python moa_experiments.py --framework MOA --moajar /path/to/moa.jar --datadir /arff_data/ --fileoutput
    """
    parser = argparse.ArgumentParser(description="Runs MOA experiments and stores results into files")
    parser.add_argument("--moajar", help="Path to MOA jar", required=True)
    parser.add_argument("--datadir", help="A directory containing one or more arff data files", required=True)
    parser.add_argument("--meta", help="The meta algorithm to use for training", required=True,
                        choices=["meta.OnlineQRF", "meta.OoBConformalRegressor"])
    parser.add_argument("--calibration", help="An arff file containing calibration instances, "
                                              "to be used with meta.ConformalRegressor.")
    # parser.add_argument("--base", help="The base algorithm to use for training")
    parser.add_argument("--repeats", help="Number of times to repeat each experiment", type=int, default=1)
    parser.add_argument("--interval", help="Performance report window size", type=int, default=1000)
    parser.add_argument("--outputdir", help="The directory to place the output files. If not given, creates "
                                            "dir under datadir.")
    parser.add_argument("--ensemble-size", help="The size of the ensemble", type=int, default=10)
    parser.add_argument("--confidence", help="The confidence level of the predictions", type=float, default=0.9)
    parser.add_argument("--fileoutput", default=False, action="store_true",
                        help="When given, output results to file instead of only printing to stdout")
    parser.add_argument("--overwrite", default=False, action="store_true",
                        help="When given, it will overwrite results in the output folder")

    args = parser.parse_args(argv)

    # Tailor the command to each framework

    task = "EvaluatePrequentialRegression -e (IntervalRegressionPerformanceEvaluator -w {})".format(args.interval)
    moa_jar_path = Path(args.moajar)
    moa_dir = moa_jar_path.parent
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
    if args.outputdir is None:
        output_path = data_path / "results-MOA"
    else:
        output_path = Path(args.outputdir)

    # Create the output dir if needed
    output_path.mkdir(parents=True,
                      exist_ok=args.overwrite)

    # Run experiments for each data file
    for arff_file in data_path.glob("*.arff"):
        learner = "{meta} -l {base} -s {size} -a {confidence}".format(
            meta=args.meta, base=base_learner, size=args.ensemble_size,
            confidence=args.confidence)
        for i in range(args.repeats):
            command = command_prefix + "\"" + "{task} -l ({learner}) " \
                                              "-s (ArffFileStream -f {arff_file}) " \
                                              "-f {interval}".format(task=task, learner=learner,
                                                                     arff_file=data_path / arff_file,
                                                                     interval=args.interval)
            if args.fileoutput:
                command += " -d {}".format(output_path / (arff_file.stem + "_{}.csv".format(i)))
            command += "\""  # Quotes necessary because of parentheses
            print("Executing command: {}".format(command))
            run(command, shell=True, check=True)

    # Write the settings for the experiment
    json_file = output_path / "settings.json"
    settings = vars(args)
    json_file.write_text(json.dumps(settings))


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
