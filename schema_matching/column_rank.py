import argparse
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from json_repair import repair_json

from prompts.prompts import RANKING_PROMPT
# Local imports
from utils import logger as utils_logger
from utils.evaluation import compute_performance_metrics
from utils.functions import (check_output_file_exists,
                             get_embedding_config_names,
                             get_index_ablation_parameters,
                             get_llm_config_names,
                             get_rank_ablation_parameters, load_config,
                             setup_llm_client)

load_dotenv()
API_KEY = os.getenv("API_KEY")
# Initialize logging
logger = utils_logger.logger_config("column_rank")

# Constants
JSON_PATTERN = r"\{.+\}"


def json_extract_validate(response: str) -> List[int]:
    """Extract and validate 'selected_columns' from JSON response."""
    try:
        match = re.search(JSON_PATTERN, response, re.DOTALL)
        if not match:
            raise ValueError(f"JSON pattern not found in response: {response}")

        out = repair_json(match.group().strip())
        json_out = json.loads(out)

        if "selected_columns" not in json_out or not isinstance(
            json_out["selected_columns"], list
        ):
            raise ValueError(f"Invalid 'selected_columns' format: {json_out}")

        cands_clean = []
        for el in json_out["selected_columns"]:
            if isinstance(el, str):
                cand_ind = el.split("_")[-1]
                cands_clean.append(int(cand_ind))
            else:
                cands_clean.append(el)

        if not cands_clean:
            raise ValueError(f"'selected_columns' should not be empty: {json_out}")

        return cands_clean

    except Exception as e:
        logger.error(f"Response: {response}\nError extracting JSON from response: {e}")
        return []


def table_filter_flag(row: pd.Series) -> int:
    """Determine if a row's 'table_name' is in 'selected_table_names'."""
    return (
        1
        if (
            isinstance(row["selected_table_names"], list)
            or isinstance(row["selected_table_names"], np.ndarray)
        )
        and row["table_name"] in row["selected_table_names"]
        else 0
    )


def custom_agg(
    group: pd.DataFrame, rank_ablation_parameters
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """Aggregate group to create prompts and mapping of indices to candidates."""
    if rank_ablation_parameters["query_enrichment"]:
        query_column_info = {
            "table_name": group["query_table_name"].iloc[0],
            "table_desc": group["query_table_desc"].iloc[0],
            "column_name": group["query_column_name"].iloc[0],
            "column_enriched_names": sorted(set(group["query_enriched_name"].tolist())),
            "column_desc": group["query_column_desc"].iloc[0],
        }
    else:
        query_column_info = {
            "table_name": group["query_table_name"].iloc[0],
            "table_desc": group["query_table_desc"].iloc[0],
            "column_name": group["query_column_name"].iloc[0],
            "column_desc": group["query_column_desc"].iloc[0],
        }
    prompt_template = deepcopy(RANKING_PROMPT)

    prompt = f""" 
    
    Query column information :
    ```
    json
    {{"query_column": {json.dumps(query_column_info, indent=4)}}}
    ```

    """
    if rank_ablation_parameters["document_enrichment"]:

        df_temp = group[
            [
                "table_name",
                "doc_id",
                "target_table_desc",
                "target_column_desc",
                "enriched_name",
                "column_name",
            ]
        ].drop_duplicates()
    else:
        df_temp = group[
            [
                "table_name",
                "doc_id",
                "target_table_desc",
                "target_column_desc",
                "column_name",
            ]
        ].drop_duplicates()

    df_temp = (
        df_temp.groupby(
            [
                "table_name",
                "target_table_desc",
                "column_name",
                "target_column_desc",
                "doc_id",
            ]
        )
        .agg(list)
        .reset_index()
    )

    ind2candidate = {}
    for i, row in df_temp.iterrows():
        if rank_ablation_parameters["document_enrichment"]:
            candidate_info = {
                "table_name": row["table_name"],
                "table_desc": row["target_table_desc"],
                "column_name": row["column_name"],
                "column_enriched_names": sorted(set(row["enriched_name"])),
                "column_desc": row["target_column_desc"],
            }
        else:
            candidate_info = {
                "table_name": row["table_name"],
                "table_desc": row["target_table_desc"],
                "column_name": row["column_name"],
                "column_desc": row["target_column_desc"],
            }
        prompt += f"""{{ "candidate_{i}": {json.dumps(candidate_info, indent=4)} }}\n"""
        ind2candidate[str(i)] = int(row["doc_id"])

    temp = {"role": "user", "content": prompt}
    prompt_template.append(temp)

    return prompt_template, ind2candidate


def get_selected_columns(x: pd.Series) -> List[int]:
    """Retrieve selected columns based on indices."""
    return [
        x["ind2candidate"][str(cand)]
        for cand in x["selected_candidates"]
        if str(cand) in x["ind2candidate"]
    ]


def get_file_paths(
    config: Dict[str, str], rank_ablation_string: str
) -> Tuple[Path, Path, Path]:
    """Get input and output file paths based on configuration."""
    llm_config = get_llm_config_names(config)
    embedding_config = get_embedding_config_names(config)
    benchmark_name = config["data"]["benchmark_name"]

    base_dir = (
        Path(__file__).resolve().parent.parent / "processed_data" / benchmark_name
    )
    n_candidates = config["experiment"]["n_candidates"]
    _, ablation_string = get_index_ablation_parameters(config)

    table_selection_path = (
        base_dir
        / "table_selection"
        / f"table_selection_{benchmark_name}_{llm_config}_{embedding_config}_{n_candidates}_{ablation_string}.parquet"
    )
    candidate_path = (
        base_dir
        / "candidates"
        / f"candidates_{benchmark_name}_{llm_config}_{embedding_config}_{n_candidates}_{ablation_string}.parquet"
    )
    output_path = (
        base_dir
        / "column_rank"
        / f"column_rank_{benchmark_name}_{llm_config}_{embedding_config}_{n_candidates}_{rank_ablation_string}.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return table_selection_path, candidate_path, output_path


def remove_records_without_labels(df: pd.DataFrame) -> pd.DataFrame:
    first_row = df.iloc[0]
    if isinstance(first_row["target_ind"], list) or isinstance(
        first_row["target_ind"], np.ndarray
    ):
        df = df[df["target_ind"].apply(lambda x: len(x) > 0)]
    elif isinstance(first_row["target_ind"], int) or isinstance(
        first_row["target_ind"], np.integer
    ):
        df = df[df["target_ind"] != -1]
    else:
        raise ValueError(
            f"Unexpected type for 'target_ind': {type(first_row['target_ind'])}"
        )
    return df


def rank_columns(config: Dict[str, Any]):
    """Main processing function."""
    llm_client = setup_llm_client(config, API_KEY)
    rank_ablation_params, rank_ablation_string = get_rank_ablation_parameters(config)

    table_selection_path, candidate_path, output_path = get_file_paths(
        config, rank_ablation_string
    )
    overwrite = config["experiment"]["overwrite"]

    if not overwrite and check_output_file_exists(output_path):
        logger.info(f"Output file already exists: {output_path}")
        return
    table_filter = pd.read_parquet(table_selection_path)
    table_filter = table_filter[
        ["query_table_name", "query_column_name", "selected_table_names"]
    ]
    df = pd.read_parquet(candidate_path)

    df = df[
        ((df["q_type"].str.contains("embedding")) & (df["score"] >= 0.50))
        | ((df["q_type"].str.contains("bm25")) & (df["score"] >= 1))
    ]
    if config["data"]["run_only_for_annotated"]:
        df = remove_records_without_labels(df)
    df = df.merge(
        table_filter, on=["query_table_name", "query_column_name"], how="left"
    )

    if rank_ablation_params["table_selection"]:
        df["table_filter_flag"] = df.apply(table_filter_flag, axis=1)
        df = df[df["table_filter_flag"] == 1]

    df["q_type_derived"] = df["q_type"].apply(lambda x: x.split("_")[0])

    temp = (
        df.groupby("query_ind")
        .apply(lambda x: custom_agg(x, rank_ablation_params))
        .reset_index(name="temp")
    )

    prompts = temp["temp"].apply(lambda x: x[0])
    ind2candidate = temp["temp"].apply(lambda x: x[1])
    temp["ind2candidate"] = ind2candidate
    temp["prompt"] = prompts
    temp.drop("temp", axis=1, inplace=True)

    temperature = 0
    responses = llm_client.chat_many(prompts, temperature=temperature)
    temp["response"] = responses
    temp["selected_candidates"] = temp["response"].apply(json_extract_validate)
    temp["selected_columns"] = temp.apply(get_selected_columns, axis=1)

    while temperature < 0.25:

        mask = temp["selected_candidates"].apply(lambda x: len(x) == 0)
        broken_output_count = mask.sum()
        logger.info(
            "Temperature: %s, Broken outputs: %s", temperature, broken_output_count
        )
        if not mask.any():
            break
        temperature += 0.05
        temp.loc[mask, "response"] = llm_client.chat_many(
            temp.loc[mask, "prompt"].tolist(), temperature=temperature
        )
        temp.loc[mask, "selected_candidates"] = temp.loc[mask, "response"].apply(
            json_extract_validate
        )
        temp.loc[mask, "selected_columns"] = temp.loc[mask].apply(
            get_selected_columns, axis=1
        )

    df = df.merge(temp, on="query_ind")
    df.drop(columns=["ind2candidate", "prompt"], inplace=True)

    performance_metrics = compute_performance_metrics(df, [1, 3, 5])
    logger.info(f"Performance Metrics: {performance_metrics}")

    df.to_parquet(output_path, index=False)


def main(args):
    """Main processing function."""
    config = load_config(args.config)
    rank_columns(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schema Enrichment Script")
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config.toml"
    )
    args = parser.parse_args()
    main(args)
