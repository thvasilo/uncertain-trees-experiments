import os
import argparse
from subprocess import run


def main(argv):
    """
    Runs a number of experiments using MOA meta-learners.
    The user provides a datadir which contains a number of arff files for classification, and a MOA
    PrequentialEvaluation task is ran on each one.

    The default arguments assume the script is being run with the MOA distribution as the working directory.

    Usage: python ~/path/to/moa_experiments.py --framework MOA --disthome /path/to/moa/ --datadir ~/data/ --fileoutput
    """
    # TODO(tvas): JSON ouput, including runtime for each experiment and the settings used.
    # It would be better to use timings provided by the frameworks however, to not have to worry about JVM init times
    parser = argparse.ArgumentParser(description="Runs MOA experiments and stores results into files")
    parser.add_argument("--framework", help="Which framework to use, choose from MOA or SAMOA",
                        choices=["MOA", "SAMOA"], required=True)
    parser.add_argument("--disthome", help="The home dir of the (SA)MOA distribution", required=True)
    parser.add_argument("--datadir", help="A directory containing one or more arff files", required=True)
    parser.add_argument("--meta", help="The meta algorithm to use for training")
    parser.add_argument("--base", help="The base algorithm to use for training")
    parser.add_argument("--size", help="The size of the ensemble", type=int, default=10)
    parser.add_argument("--interval", help="Report performance every X examples", type=int, default=1000)
    parser.add_argument("--outputdir", help="The directory to place the output files. Creates dir under datadir "
                                            "if not given")
    parser.add_argument("--fileoutput", default=False, action="store_true",
                        help="When given, output results to file instead of only printing to stdout")

    args = parser.parse_args(argv)

    # Tailor the command to each framework
    if args.framework == "MOA":
        task = "EvaluatePrequential"
        command_prefix = "java -cp {disthome}/moa.jar -javaagent:{disthome}/lib/sizeofag-1.0.0.jar " \
                         "moa.DoTask ".format(disthome=args.disthome)
        if args.meta is None:
            args.meta = "meta.OzaBoost"
        if args.base is None:
            args.base = "trees.HoeffdingTree"
    else:  # Using SAMOA framewokr
        task = "PrequentialEvaluation"
        command_prefix = "{disthome}/bin/samoa local {disthome}/target/SAMOA-Local-0.5.0-incubating-SNAPSHOT.jar " \
                         "".format(disthome=args.disthome)
        if args.meta is None:
            args.meta = "classifiers.ensemble.BoostVHT"
        if args.base is None:
            args.base = "classifiers.trees.VerticalHoeffdingTree"

    # Get all the .arff files in the data directory
    arff_file_list = [file for file in os.listdir(args.datadir) if file.endswith(".arff")]
    if len(arff_file_list) == 0:
        raise FileNotFoundError("No arff files found in the provided directory: {}".format(args.datadir))

    # If the user did not provide an output dir, put results under the data folder
    if args.outputdir is None:
        args.outputdir = "{}".format(os.path.join(args.datadir, "results-{}".format(args.framework)))

    # Create the output dir if needed
    if args.fileoutput and not os.path.exists(args.outputdir):
        os.mkdir(args.outputdir)
    # Run one experiment for each data file
    for arff_file in arff_file_list:
        command = command_prefix + " \"{task} -l ({meta} -l {base} -s {size}) " \
                  "-s (ArffFileStream -f {arff_file}) " \
                  "-f {interval}\"".format(task=task, meta=args.meta, base=args.base,
                                           size=args.size, arff_file=os.path.join(args.datadir, arff_file),
                                           interval=args.interval)
        if args.fileoutput:
            command += " -d {}".format(os.path.join(args.outputdir, arff_file + "_out.csv"))
        print("Executing command: {}".format(command))
        run(command, shell=True, check=True)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
