import argparse
import glob
import os
import re
from typing import Any, Dict

import pandas as pd

from utils.evaluation import compute_performance_metrics

EMBEDDING_MODEL_SIZE_DICT = {
    "all-mpnet-base-v2": 768,
    "bge-small-en-v1.5": 384,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def parse_column_rank_filepath(filepath: str) -> Dict[str, Any]:
    """
    Parse the filename of a column_rank parquet file to extract metadata.

    Args:
        filepath: Path to the parquet file.

    Returns:
        A dictionary with extracted fields including benchmark, llm_model, embedding_model, n_candidates, and config flags.

    Raises:
        ValueError: If filename does not match expected pattern.
    """
    filename = os.path.basename(filepath)
    pattern = re.compile(
        r"""
        ^column_rank_                                           # prefix
        (?P<benchmark>[a-zA-Z0-9\-]+)_                          # benchmark, e.g. mimic-omop
        openai_model_name_                                      # literal segment
        (?P<llm_model>[a-zA-Z0-9\.\-]+)_                        # LLM model, e.g. gpt-4.1-2025-04-14
        (?P<embedding_model>[a-zA-Z0-9\-\._]+?)_                # embedding model, allow dots, dashes, underscores
        (?P<n_candidates>-?\d+)_                                # number of candidates
        (?P<bool_flags>(?:[a-zA-Z_]+_(?:true|false)_?)+)        # boolean flags
        \.parquet$                                              # file extension
        """,
        re.VERBOSE,
    )

    match = pattern.fullmatch(filename)
    if not match:
        raise ValueError("Filename does not conform to expected pattern.")

    # Parse boolean flags
    bool_flags_str = match.group("bool_flags")
    parts = re.split(r"(true|false)", bool_flags_str)
    config_dict = {}
    i = 0
    while i < len(parts) - 1:
        key = parts[i].strip("_")
        value = parts[i + 1] == "true"
        if key:
            config_dict[key] = value
        i += 2

    result = {
        "benchmark": match.group("benchmark"),
        "llm_model": match.group("llm_model"),
        "embedding_model": match.group("embedding_model"),
        "n_candidates": int(match.group("n_candidates")),
    }
    result.update(config_dict)
    return result


def parse_needle_filepath(filepath: str) -> Dict[str, Any]:
    """
    Extracts the benchmark and OpenAI model name from a 'needle' file path.

    Args:
        filepath: File path string.

    Returns:
        Dictionary with 'benchmark', 'llm_model', and None for other keys.
    """
    bench_match = re.search(r"processed_data/([^/]+)/", filepath)
    benchmark = bench_match.group(1) if bench_match else None

    model_match = re.search(r"openai_model_name_([a-zA-Z0-9\-.]+)", filepath)
    model_name = model_match.group(1) if model_match else None

    return {
        "benchmark": benchmark,
        "llm_model": model_name,
        "embedding_model": None,
        "n_candidates": None,
    }


def main(benchmark: str) -> None:
    """
    Main function to process parquet files for a given benchmark, compute metrics,
    and save the aggregated results to CSV.

    Args:
        config: The benchmark name (e.g., "saki-imdb").
    """
    files = glob.glob(f"processed_data/{benchmark}/column_rank/*")
    rows = []

    for fname in files:
        if "needle" in fname:
            row_i = parse_needle_filepath(fname)
        else:
            try:
                row_i = parse_column_rank_filepath(fname)
            except (ValueError, KeyError) as e:
                print(f"Skipping file {fname} due to parsing error: {e}")
                continue

        df = pd.read_parquet(fname)
        if "query_ind" not in df.columns:
            df["query_ind"] = df.index

        row_i_t = compute_performance_metrics(df, [1, 3, 5])
        row_i.update(row_i_t)
        rows.append(row_i)

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(
        ["llm_model", "embedding_model", "n_candidates"], ascending=[True, True, True]
    )
    results_df["embedding_model_size"] = results_df["embedding_model"].map(
        EMBEDDING_MODEL_SIZE_DICT
    )
    results_df.to_csv(
        f"processed_data/{benchmark}/hitrate.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process column rank files for a given benchmark."
    )
    parser.add_argument(
        "--benchmark", required=True, help="Benchmark name (e.g., saki-imdb)"
    )
    args = parser.parse_args()
    main(benchmark=args.benchmark)
