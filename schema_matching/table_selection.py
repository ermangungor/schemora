import argparse
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from json_repair import repair_json

from prompts.prompts import TABLE_SELECTION
# Local imports
from utils import logger as utils_logger
from utils.functions import (check_output_file_exists,
                             get_embedding_config_names,
                             get_index_ablation_parameters,
                             get_llm_config_names, load_config,
                             setup_llm_client)

load_dotenv()
API_KEY = os.getenv("API_KEY")

logger = utils_logger.logger_config("table_selection")

JSON_PATTERN = r"\{.+\}"


def json_extract_validate(response: str) -> List[str]:
    """Extract and validate 'selected_table_names' from JSON response."""
    try:
        match = re.search(JSON_PATTERN, response, re.DOTALL)
        if not match:
            raise ValueError(f"JSON pattern not found in response: {response}")

        json_str = repair_json(match.group().strip())
        json_out = json.loads(json_str)

        selected_table_names = json_out.get("selected_table_names")
        if not isinstance(selected_table_names, list) or not selected_table_names:
            raise ValueError(
                f"Invalid or empty 'selected_table_names' in JSON: {json_out}"
            )

        return selected_table_names

    except Exception as e:
        logger.error(f"Response: {response}")
        logger.error(f"Error extracting JSON: {e}")
        return []


def get_input_file_name(config: Dict[str, str]) -> Path:
    """Get the input file name for candidates."""
    llm_config_values = get_llm_config_names(config)
    embedding_call_type = get_embedding_config_names(config)
    benchmark_name = config["data"]["benchmark_name"]
    n_candidates = config["experiment"]["n_candidates"]
    _, ablation_string = get_index_ablation_parameters(config)

    base_dir = Path(__file__).resolve().parent.parent
    candidates_dir = base_dir / "processed_data" / benchmark_name / "candidates"
    candidate_file_name = f"candidates_{benchmark_name}_{llm_config_values}_{embedding_call_type}_{n_candidates}_{ablation_string}.parquet"

    return candidates_dir / candidate_file_name


def get_output_file_name(config: Dict[str, str]) -> Path:
    """Get the output file name for table selection results."""
    llm_config_values = get_llm_config_names(config)
    embedding_call_type = get_embedding_config_names(config)
    benchmark_name = config["data"]["benchmark_name"]
    n_candidates = config["experiment"]["n_candidates"]
    _, ablation_string = get_index_ablation_parameters(config)

    base_dir = Path(__file__).resolve().parent.parent
    candidates_dir = base_dir / "processed_data" / benchmark_name / "table_selection"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    table_selection_file_name = f"table_selection_{benchmark_name}_{llm_config_values}_{embedding_call_type}_{n_candidates}_{ablation_string}.parquet"

    return candidates_dir / table_selection_file_name


def build_target_name_to_desc_map(target_df: pd.DataFrame) -> Dict[str, str]:
    """Build a mapping from target table names to their descriptions."""

    target_name_to_desc = {}
    for _, row in target_df.iterrows():
        target_name = row["TableName"]
        target_desc = row["TableDesc"]
        if target_name not in target_name_to_desc:
            target_name_to_desc[target_name] = target_desc
        else:
            logger.warning(
                f"Duplicate target table name found: {target_name}. Skipping."
            )
    return target_name_to_desc


def select_tables(config: Dict[str, Any]):
    """Main processing function."""

    input_file_path = get_input_file_name(config)
    llm_client = setup_llm_client(config, API_KEY)
    output_file_name = get_output_file_name(config)
    overwrite = config["experiment"]["overwrite"]

    if not overwrite and check_output_file_exists(output_file_name):
        logger.info(f"Output file already exists: {output_file_name}")
        return

    df = pd.read_parquet(input_file_path)
    threshold_filter = (
        df["q_type"].str.contains("embedding") & (df["score"] >= 0.50)
    ) | (df["q_type"].str.contains("bm25") & (df["score"] >= 1))
    df = df[threshold_filter].drop_duplicates(
        subset=[
            "query_table_name",
            "query_column_name",
            "query_table_desc",
            "table_name",
            "target_table_desc",
        ]
    )
    df = (
        df.groupby(["query_table_name", "query_column_name", "query_table_desc"])
        .agg(list)
        .reset_index()
    )

    table_iterator = df[
        ["query_table_name", "query_table_desc", "table_name", "target_table_desc"]
    ].itertuples(index=False)
    prompts = []

    for query_table, query_desc, target_tables, target_descs in table_iterator:
        prompt_template = deepcopy(TABLE_SELECTION)
        prompt_suffix = (
            f"\nTable Name: {query_table}\nTable Description: {query_desc}\n"
        )

        for idx, (table_name, table_desc) in enumerate(
            zip(target_tables, target_descs)
        ):
            prompt_suffix += f"Candidate Table Name-{idx}: {table_name}\nCandidate Table Description-{idx}: {table_desc}\n"

        prompt_template[0]["content"] += prompt_suffix
        prompts.append(prompt_template)

    logger.info(f"Generating responses for {len(prompts)} prompts.")
    temperature = 0
    responses = llm_client.chat_many(prompts, temperature=0)

    df["prompt"] = prompts
    df["response"] = responses
    df["selected_table_names"] = df["response"].apply(json_extract_validate)

    while temperature < 0.25:

        mask = df["selected_table_names"].apply(lambda x: len(x) == 0)
        broken_output_count = mask.sum()
        logger.info(
            "Temperature: %s, Broken outputs: %s", temperature, broken_output_count
        )
        if not mask.any():
            break
        temperature += 0.05
        df.loc[mask, "response"] = llm_client.chat_many(
            df.loc[mask, "prompt"].tolist(), temperature=temperature
        )
        df.loc[mask, "selected_table_names"] = df.loc[mask, "response"].apply(
            json_extract_validate
        )

    df.to_parquet(output_file_name, index=False)


def main(args: argparse.Namespace):
    """Main processing function."""
    config = load_config(args.config)
    select_tables(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Table Selection Script for Schema Enrichment"
    )
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config.toml"
    )
    args = parser.parse_args()
    main(args)
