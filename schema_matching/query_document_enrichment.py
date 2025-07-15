import argparse
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from json_repair import repair_json

from prompts.prompts import (COT_COLUMN_ENRICHMENT,
                             COT_COLUMN_ENRICHMENT_SYNONYMS)
from utils import llm
from utils import logger as utils_logger
from utils.functions import (check_output_files_exists, get_llm_config_names,
                             get_source_target_file_names, load_config,
                             setup_llm_client)

# Initialize logger for the script
logger = utils_logger.logger_config("column_entities")
load_dotenv()
API_KEY = os.getenv("API_KEY")
# Assume these constants and utils are defined elsewhere in your codebase
JSON_PATTERN = r"\{.*?\}"


def extract_and_validate_json(response: str) -> List[str]:
    """Extract and validate JSON from a response string."""
    try:
        match = re.search(JSON_PATTERN, response, re.DOTALL)
        if match is None:
            raise ValueError(f"JSON pattern not found in response: {response}")

        repaired_json_string = repair_json(match.group().strip())
        json_out = json.loads(repaired_json_string)

        names = json_out.values()
        return names

    except ValueError as error:
        logger.error(f"Response: {response}")
        logger.error(f"Error extracting and validating JSON: {error}")
        return []


def create_prompts(df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """Generate prompts for OpenAI API based on the source schema."""

    cot_column_enrichment = deepcopy(COT_COLUMN_ENRICHMENT)
    cot_column_enrichment_synonyms = deepcopy(COT_COLUMN_ENRICHMENT_SYNONYMS)

    itr = df[["TableName", "TableDesc", "ColumnName", "ColumnDesc"]].itertuples(
        index=False
    )
    prompts = []
    prompt_syn = []

    for row in itr:
        prompt_str = f"Column Name: {row.ColumnName}\nTable Name: {row.TableName}\nTable Description: {row.TableDesc}\n"

        prompt = deepcopy(cot_column_enrichment)
        prompt.append({"role": "user", "content": prompt_str})
        prompts.append(prompt)

        prompt_synonyms = deepcopy(cot_column_enrichment_synonyms)
        prompt_synonyms.append({"role": "user", "content": prompt_str})
        prompt_syn.append(prompt_synonyms)

    return prompts, prompt_syn


def load_data(config: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the source and target schema files."""
    source_file, target_file = get_source_target_file_names(config)
    source = pd.read_csv(source_file).fillna("")
    target = pd.read_csv(target_file).fillna("")
    return source, target


def run_enrichment(
    df: pd.DataFrame,
    promtpts: List[str],
    llm_client: llm.LLMClient,
    enrichment_type: str,
) -> pd.DataFrame:
    """Run enrichment using LLM client and handle responses."""
    temperature = 0
    # First iteration: process all rows
    df["prompt_temp"] = promtpts
    df["response_temp"] = llm_client.chat_many(
        df["prompt_temp"].tolist(), temperature=temperature
    )
    df["enriched_name_temp"] = df["response_temp"].apply(extract_and_validate_json)

    # Separate loops for enriched_name and enriched_name_syn
    while temperature < 0.25:

        # Process enriched_name
        mask_name = df["enriched_name_temp"].apply(lambda x: len(x) == 0)
        broken_count_name = mask_name.sum()
        logger.info(
            "Temperature: %s, Broken outputs (enriched_name): %s for enrichment type %s",
            temperature,
            broken_count_name,
            enrichment_type,
        )
        if not mask_name.any():
            break
        # this only happens if the model fails to generate a valid JSON
        # which is rare
        temperature += 0.05
        df.loc[mask_name, "response_temp"] = llm_client.chat_many(
            df.loc[mask_name, "prompt_temp"].tolist(), temperature=temperature
        )
        df.loc[mask_name, "enriched_name_temp"] = df.loc[
            mask_name, "response_temp"
        ].apply(extract_and_validate_json)

    out = (df["response_temp"].tolist(), df["enriched_name_temp"].tolist())
    df.drop(
        columns=["prompt_temp", "response_temp", "enriched_name_temp"], inplace=True
    )
    return out


def enriched_table(df: pd.DataFrame, llm_client: llm.LLMClient) -> pd.DataFrame:
    """Enrich DataFrame with additional information using LLM client."""
    promts, prompts_syn = create_prompts(df)
    df["response"], df["enriched_name"] = run_enrichment(
        df, promts, llm_client, "enriched_name"
    )
    df["response_syn"], df["enriched_name_syn"] = run_enrichment(
        df, prompts_syn, llm_client, "enriched_name_syn"
    )
    return df


def output_file_names(config: Dict[str, str]) -> Tuple[Path, Path]:
    """Generate output file names for enriched data."""

    benchmark_name = config["data"]["benchmark_name"]
    base_dir = (
        Path(__file__).resolve().parent.parent
        / "processed_data"
        / benchmark_name
        / "enrichments"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    llm_config_values = get_llm_config_names(config)

    source_file, target_file = get_source_target_file_names(config)
    source_output = (
        base_dir / f"{source_file.name.split('.')[0]}_{llm_config_values}.parquet"
    )
    target_output = (
        base_dir / f"{target_file.name.split('.')[0]}_{llm_config_values}.parquet"
    )

    return source_output, target_output


def enrich_document(config: Dict[str, Any]) -> pd.DataFrame:
    llm_client = setup_llm_client(config, API_KEY)

    source, target = load_data(config)
    overwrite = config["experiment"]["overwrite"]

    source_file_path, target_file_path = output_file_names(config)
    source_flag, target_flag = check_output_files_exists(
        source_file_path, target_file_path
    )

    if not overwrite and source_flag and target_flag:
        logger.info("Enriched files already exist. Skipping enrichment process.")
        return
    if overwrite or not source_flag:
        source_enriched = enriched_table(source, llm_client)
        source_enriched.to_parquet(source_file_path)
        logger.info(f"Enriched source schema saved to {source_file_path}")
    if overwrite or not target_flag:
        target_enriched = enriched_table(target, llm_client)
        target_enriched.to_parquet(target_file_path)
        logger.info(f"Enriched target schema saved to {target_file_path}")


def main(args: argparse.Namespace) -> None:
    """Main function to enrich schema and save results."""
    config = load_config(args.config)
    enrich_document(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schema Enrichment Script")
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config.toml"
    )
    args = parser.parse_args()
    main(args)
