"""
Used to create the computational measures for a single method. We create a csv/tex file with
the total runtime and final model size for every dataset for the method, and provide aggregate
metrics.
"""
import argparse
from pathlib import Path
import json

import pandas as pd
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required= True,
                        help="A dir containing csvs of results, one per repeat.")
    parser.add_argument("--output", required=True,
                        help="The folder to create the output in, the last part of the input"
                             " (method name) will be appended.")
    parser.add_argument("--overwrite", action="store_true",
                        help="When given, will not check if the output folder already exists ,"
                             "potentially overwriting its contents.")

    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    # Append the method name to the output as a subdir
    output_path = Path(args.output).joinpath(input_path.name)

    assert len(list(input_path.glob("*.csv"))) > 0, "No experiment files found under {}".format(input_path)
    assert output_path.parent != input_path, "Setting output path under input can cause issues, choose another path."
    output_path.mkdir(parents=True, exist_ok=args.overwrite)
    computational_metrics = ['evaluation time (cpu seconds)',
                             'model serialized size (bytes)']

    json_file = output_path / "computational-measures-settings.json"
    json_file.write_text(json.dumps(vars(args)))

    # This df will have one row for each experiment
    combined_df = pd.DataFrame(columns=computational_metrics)
    experiment_values = {}
    # TODO: Support for multiple method directories, creation of comparison tables
    experiment_index = 0
    # We iterate through each output file, get its last line and extract the computational metrics
    for experiment_file in input_path.glob("*.csv"):
        # Ignore other output files
        if experiment_file.match("*.pred.csv") or experiment_file.match("*.time.csv"):
            continue
        # Get the experiment name, i.e. dataset and repeat number
        experiment_value = experiment_file.stem
        # We maintain a dict from experiment index to name, we use this as the combined_df index later
        experiment_values[experiment_index] = experiment_value
        df = pd.read_csv(experiment_file)
        # Get the last line of the results dataframe, we want the final measurements of the requested metrics
        df = df.iloc[[-1]]
        metrics_df = df[computational_metrics]
        # Append the measurements to the combined dataframe
        combined_df = combined_df.append(metrics_df, ignore_index=True)
        experiment_index += 1

    combined_df = combined_df.rename(experiment_values, axis='index').sort_index(ascending=True)

    # Calculate aggregate metrics
    def include_aggregates(df):
        df.loc['Mean'] = df.mean()
        df.loc['Median'] = df.median()
        df.loc['Std'] = df.std()
        return df

    combined_df = include_aggregates(combined_df)

    def create_table_str(df):
        # float_format = [".2f", ".4f", ".2f"]  # if outpath.name == "mean_error_rate" else ".2f"
        return tabulate(df, headers='keys', tablefmt='latex_booktabs')

    # Create csv and tex output
    combined_df.to_csv(output_path / "computational_metrics.csv")
    table_str = create_table_str(combined_df)
    table_file = output_path / "computational_metrics.tex"
    table_file.write_text(table_str)


if __name__ == '__main__':
    main()