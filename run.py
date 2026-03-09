from typing import Any, Dict, Iterable, List
from oidiq import *

import multiprocessing as mp
import os
import pandas as pd
import tqdm.contrib.concurrent as concurrent
import numpy as np

import argparse


def create_session_factory():
    return OIDIQSessionFactory("config.yaml")


def create_headers(session_factory: OIDIQSessionFactory) -> List[str]:
    return (
        ["Filename"] + session_factory.registered_metric_creators() + ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    )


def create_results(session_factory: OIDIQSessionFactory, headers: List[str], rows: pd.DataFrame) -> List[List[str]]:
    session = session_factory.create_batch_session(*rows["Filename"].tolist())
    scores: Dict[str, Any] = session.get_all_scores()

    corners = np.array(session.get_id_card_corners())
    for corner_idx in range(4):
        scores[f"x{corner_idx+1}"] = corners[:, corner_idx, 0]
        scores[f"y{corner_idx+1}"] = corners[:, corner_idx, 1]

    scores["Filename"] = rows["Filename"].tolist()

    results = []
    for i in range(len(session)):
        row = []
        for header in headers:
            score = scores[header][i]
            if isinstance(score, QualityMetric):
                row.append(f"{score.raw_value:.6f}")
            else:
                row.append(str(score))
        results.append(row)
    return results


def dataframe_batch_iterator(df: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    df = df.reset_index(drop=True)
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        yield df.iloc[start_idx:end_idx]


def run_analysis(
    data_split: pd.DataFrame,
    output_csv: str,
    batch_size: int,
):
    if len(data_split) == 0:
        return
    session_factory = create_session_factory()
    if os.path.exists(output_csv):
        os.remove(output_csv)

    headers = create_headers(session_factory)
    with open(output_csv, "a") as f:
        f.write(",".join(headers) + "\n")
        for idx, rows in enumerate(dataframe_batch_iterator(data_split, batch_size)):
            rows_data = create_results(session_factory, headers, rows)

            for row_idx in range(len(rows)):
                score_values = rows_data[row_idx]
                f.write(",".join(score_values) + "\n")

            if (idx + 1) % 5 == 0:
                f.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Run batched OIDIQ analysis on a dataset.")
    parser.add_argument(
        "--input-csv",
        "-i",
        type=str,
        required=True,
        help="Path to the input CSV file containing image filenames.",
    )
    parser.add_argument(
        "--output-csv",
        "-o",
        type=str,
        required=True,
        help="Path to the output CSV file to store results.",
    )
    parser.add_argument(
        "--data-splits",
        type=int,
        default=50,
        help="Number of data splits to use for parallel processing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing images.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=mp.cpu_count() // 2,
        help="Number of parallel workers to use. Defaults to half the CPU count.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default="data/tmp",
        help="Temporary directory to store intermediate CSV files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_csv(args.input_csv)
    data.rename(columns={data.columns[0]: "Filename"}, inplace=True)
    data = data[["Filename"]]

    if len(data) >= args.data_splits * args.batch_size:
        data_splits = [data.iloc[i :: args.data_splits].reset_index(drop=True) for i in range(args.data_splits)]
    else:
        actual_splits = max(1, len(data) // args.batch_size)
        data_splits = [data.iloc[i::actual_splits].reset_index(drop=True) for i in range(actual_splits)]

    output_csvs = [os.path.join(args.tmp_dir, f"result_part_{i}.csv") for i in range(len(data_splits))]
    os.makedirs(args.tmp_dir, exist_ok=True)
    workers = min(args.workers, len(data_splits))
    print(f"Using {workers} workers to process {len(data)} images in {len(data_splits)} splits...")
    r = concurrent.process_map(
        run_analysis,
        data_splits,
        output_csvs,
        [args.batch_size for _ in range(len(data_splits))],
        desc="Processing images",
        max_workers=workers,
    )

    combined_df = pd.concat([pd.read_csv(csv) for csv in output_csvs if os.path.exists(csv)])
    combined_df.to_csv(args.output_csv, index=False)
    for csv in output_csvs:
        if os.path.exists(csv):
            os.remove(csv)
    try:
        os.rmdir(args.tmp_dir)
    except:
        print(f"Could not remove temporary directory {args.tmp_dir}")


if __name__ == "__main__":
    main()
