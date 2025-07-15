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
from utils.functions import (
    check_output_file_exists,
    get_llm_config_names,
    get_mapping_file_name,
    get_source_target_file_names,
    load_config,
    setup_llm_client,
)

load_dotenv()
API_KEY = os.getenv("API_KEY")
# Initialize logging
logger = utils_logger.logger_config("embedding_enrichment")

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


def gen_target_prompt(df: pd.DataFrame) -> tuple[str, dict]:
    # TableName,TableDesc,ColumnName,ColumnDesc,ETL Conventions,ColumnType,Required,IsPK,IsFK,FK table,FK column,FK
    itr = df[["TableName", "TableDesc", "ColumnName", "ColumnDesc"]].iterrows()
    ind2candidate = {}
    prompt = ""
    for i, row in itr:

        candidate_info = {
            "table_name": row["TableName"],
            "table_desc": row["TableDesc"],
            "column_name": row["ColumnName"],
            "column_desc": row["ColumnDesc"],
        }
        prompt += f"""{{ "candidate_{i}": {json.dumps(candidate_info, indent=4)} }}\n"""

        ind2candidate[(row["TableName"], row["ColumnName"])] = int(i)

    return prompt, ind2candidate


def gen_source_prompt(row, target_prompt) -> List[Dict[str, str]]:
    query_column_info = {
        "table_name": row["TableName"],
        "table_desc": row["TableDesc"],
        "column_name": row["ColumnName"],
        "column_desc": row["ColumnDesc"],
    }
    prompt_template = deepcopy(RANKING_PROMPT)

    prompt = f""" 
    
    Query column information :
    ```
    json
    {{"query_column": {json.dumps(query_column_info, indent=4)}}}
    ```

    """
    prompt += target_prompt

    temp = {"role": "user", "content": prompt}
    prompt_template.append(temp)

    return prompt_template


def get_selected_columns(x: pd.Series) -> List[int]:
    """Retrieve selected columns based on indices."""
    return [
        x["ind2candidate"][str(cand)]
        for cand in x["selected_candidates"]
        if str(cand) in x["ind2candidate"]
    ]


def get_file_paths(config: Dict[str, str]) -> Tuple[Path, Path, Path]:
    """Get input and output file paths based on configuration."""
    llm_config = get_llm_config_names(config)
    benchmark_name = config["data"]["benchmark_name"]
    base_dir = (
        Path(__file__).resolve().parent.parent / "processed_data" / benchmark_name
    )

    output_path = (
        base_dir
        / "column_rank"
        / f"column_rank_{benchmark_name}_{llm_config}_needle_in_the_stack.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mapping_file = get_mapping_file_name(config)

    source_file, target_file = get_source_target_file_names(config)

    return source_file, target_file, mapping_file, output_path


def add_target_index(
    source_data: pd.DataFrame, candidate2ind: Dict[str, int]
) -> pd.DataFrame:
    n = len(source_data)
    target_indexes = []
    for i in range(n):
        target_table_name = source_data["TGT_ENT"].iloc[i]
        target_column_name = source_data["TGT_ATT"].iloc[i]
        if isinstance(target_table_name, str):
            target_ind = candidate2ind.get(
                (target_table_name, target_column_name), None
            )
        elif isinstance(target_table_name, np.ndarray) or isinstance(
            target_table_name, list
        ):
            target_ind = []
            for target_name_i, target_column_name_i in zip(
                target_table_name, target_column_name
            ):
                target_ind.append(
                    candidate2ind.get((target_name_i, target_column_name_i), None)
                )
        else:
            target_ind = None

        target_indexes.append(target_ind)
    source_data["target_ind"] = target_indexes
    return source_data


def rank_columns(config: Dict[str, Any]):
    """Main processing function."""
    llm_client = setup_llm_client(config, API_KEY)
    source_file, target_file, mapping_file, output_path = get_file_paths(config)
    overwrite = config["experiment"]["overwrite"]

    if not overwrite and check_output_file_exists(output_path):
        logger.info(f"Output file already exists: {output_path}")
        return

    if mapping_file.suffix == ".csv":
        mapping_data = pd.read_csv(mapping_file)
    elif mapping_file.suffix == ".parquet":
        mapping_data = pd.read_parquet(mapping_file)
    else:
        raise ValueError(f"Unsupported mapping file format: {mapping_file.suffix}")
    source_data = pd.read_csv(source_file)
    source_data = mapping_data.merge(
        source_data,
        right_on=["TableName", "ColumnName"],
        left_on=["SRC_ENT", "SRC_ATT"],
    )

    target_data = pd.read_csv(target_file)
    target_prompt, candidate2ind = gen_target_prompt(target_data)

    source_data = add_target_index(source_data, candidate2ind)
    source_data = source_data[source_data["target_ind"].notnull()]

    prompts = source_data.apply(lambda x: gen_source_prompt(x, target_prompt), axis=1)

    source_data["prompt"] = prompts

    temperature = 0
    responses = llm_client.chat_many(prompts, temperature=temperature)
    source_data["response"] = responses
    source_data["selected_columns"] = source_data["response"].apply(
        json_extract_validate
    )

    while temperature < 0.25:

        mask = source_data["selected_columns"].apply(lambda x: len(x) == 0)
        broken_output_count = mask.sum()
        logger.info(
            "Temperature: %s, Broken outputs: %s", temperature, broken_output_count
        )
        if not mask.any():
            break
        temperature += 0.05
        source_data.loc[mask, "response"] = llm_client.chat_many(
            source_data.loc[mask, "prompt"].tolist(), temperature=temperature
        )
        source_data.loc[mask, "selected_columns"] = source_data.loc[
            mask, "response"
        ].apply(json_extract_validate)

    source_data["query_ind"] = source_data.index
    performance_metrics = compute_performance_metrics(source_data, [1, 3, 5])
    logger.info(f"Performance Metrics: {performance_metrics}")
    source_data.to_parquet(output_path, index=False)


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
