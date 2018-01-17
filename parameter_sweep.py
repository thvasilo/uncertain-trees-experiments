#!/usr/bin/env python
"""
Simple python script to sweep over a single parameter
"""
import argparse
from subprocess import run
from pathlib import Path

import numpy as np

# TODO: Allow multi-parameter sweeps


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--command", required=True,
                        help="The base command to run, this does not change across sweeps")
    parser.add_argument("--sweep-argument", required=True,
                        help="The argument to sweep on")
    parser.add_argument("--output-prefix", required=True,
                        help="The prefix for the output argument")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--argument-list", nargs='+',
                       help="A list of arguments to run")
    group.add_argument("--argument-range", nargs=3,
                       help="A string representation of an argument range, enter"
                            "\"start end step\" as in np.arange(start, end, step)")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.argument_list is not None:
        sweep = args.argument_list
    else:
        # We use this trick to correctly parse ints or float from string, so
        # that the meta-args end up with the correct type
        from ast import literal_eval as le
        range_args = [le(x) for x in args.argument_range]
        start, end, step = range_args
        sweep = np.arange(start, end, step)

    for value in sweep:
        value = str(value)
        outdir = Path(args.output_prefix) / value
        command = args.command + " --{arg} {val} --output {outdir} ".format(
            arg=args.sweep_argument, val=value, outdir=outdir)
        print("Running command:\n{}".format(command))
        run(command, shell=True)


if __name__ == "__main__":
    main()
