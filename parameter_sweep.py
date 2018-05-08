#!/usr/bin/env python
"""
Simple python script to sweep over one or two parameters for our experiments.
"""
import argparse
from subprocess import run
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--command", required=True,
                        help="The base command to run, this does not change across sweeps")
    parser.add_argument("--sweep-argument", required=True,
                        help="The argument to sweep on, e.g. \"meta\"")
    parser.add_argument("--output-prefix", required=True,
                        help="The prefix for the output argument")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--argument-list", nargs='+',
                       help="A space-separated list of arguments to use, e.g. \"OnlineQRF CPExact\"")
    group.add_argument("--argument-range", nargs=3,
                       help="A string representation of an argument range, enter"
                            "\"start end step\" as in np.arange(start, end, step).")
    parser.add_argument("--njobs", type=int, default=1,
                        help="Number of experiment jobs to run in parallel, max one per input file")
    parser.add_argument("--verbose", type=int, default=0,
                        help="Set to 1 output per experiment, 2 to write MOA output to stdout.")
    parser.add_argument("--inner-sweep-argument",
                        help="The argument to sweep on")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--inner-argument-list", nargs='+',
                       help="A list of arguments to run")
    group.add_argument("--inner-argument-range", nargs=3,
                       help="A string representation of an argument range, enter "
                            "\"start end step\" as in np.arange(start, end, step)")

    return parser.parse_args()


def run_print(cur_command):
    print("Running command:\n{}".format(cur_command))
    run(cur_command, shell=True)


def main():
    args = parse_args()

    def parse_sweep(argument_list, argument_range):
        if argument_list is not None:
            sweep = argument_list
        else:
            # We use this trick to correctly parse ints or float from string, so
            # that the meta-args end up with the correct type
            from ast import literal_eval as le
            range_args = [le(x) for x in argument_range]
            start, end, step = range_args
            sweep = np.arange(start, end, step)
        return sweep

    outer_sweep = parse_sweep(args.argument_list, args.argument_range)
    inner_sweep = None
    if args.inner_argument_list is not None or args.inner_argument_range is not None:
        inner_sweep = parse_sweep(args.inner_argument_list, args.inner_argument_range)

    command_list = []
    for outer_value in outer_sweep:
        outer_value = str(outer_value)
        outdir = Path(args.output_prefix) / outer_value
        if inner_sweep is not None:
            for inner_value in inner_sweep:
                inner_value = str(inner_value)
                inner_ouput = outdir / inner_value
                command = args.command + " --{outer_arg} {outer_val} --{inner_arg} {inner_val}  --output {outdir} ".format(
                    outer_arg=args.sweep_argument, outer_val=outer_value, outdir=inner_ouput,
                    inner_arg=args.inner_sweep_argument, inner_val=inner_value)
                command_list.append(command)
        else:
            command = args.command + " --{arg} {val} --output {outdir} ".format(
                arg=args.sweep_argument, val=outer_value, outdir=outdir)
            command_list.append(command)

    with Parallel(n_jobs=args.njobs, verbose=args.verbose) as parallel:
        parallel(delayed(run_print)(command)
                 for command in command_list)


if __name__ == "__main__":
    main()
