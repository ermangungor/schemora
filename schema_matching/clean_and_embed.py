import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from camelsplit import camelsplit
from dotenv import load_dotenv

# Local imports
from utils import embedding, llm
from utils import logger as utils_logger
from utils.functions import (check_output_files_exists,
                             get_embedding_config_names, get_llm_config_names,
                             get_source_target_file_names, load_config)

load_dotenv()
API_KEY = os.getenv("API_KEY")

# Initialize logger
logger = utils_logger.logger_config("clean_and_embed")


def setup_embedding_client(config: Dict) -> llm.LLMClient:
    """Setup Embedding Client using configuration."""
    client_config = config["embedding"]["client"]
    if client_config["type"] != "local":
        client_config["args"]["api_key"] = API_KEY
    return embedding.EmbeddingClient.create(
        client_config["type"], **client_config["args"]
    )


def clean_string(s: str) -> str:
    """
    Cleans a given string by removing special characters, numbers, and certain stop words.
    Also splits words based on camel case, underscores, dots, and hyphens.
    """
    s = re.sub(r"\(.*?\)", "", s)  # Remove content inside parentheses
    s = re.sub(r"\d", " ", s)  # Replace all numbers with whitespace
    s = re.sub(r"[^\w\s]", " ", s)  # Replace all punctuation with whitespace
    name_i_c = [
        word for el in re.split(r"[_.-]", s) for word in camelsplit(el) if el.strip()
    ]

    return " ".join("and" if i.strip() == "&" else i.strip() for i in name_i_c).lower()


def add_embeddings(df: pd.DataFrame, embedding_client: llm.LLMClient) -> pd.DataFrame:
    """Add embeddings to DataFrame columns using embedding client."""
    serialized_name = (
        df["ColumnName"]
        + " "
        + df["ColumnDesc"]
        + " "
        + df["TableName"]
        + " "
        + df["TableDesc"]
    )
    df["enriched_name_embeddings"] = embedding_client.embed_many(
        df["enriched_name"].tolist()
    )
    df["enriched_name_syn_embeddings"] = embedding_client.embed_many(
        df["enriched_name_syn"].tolist()
    )
    df["serialized_column_embedding"] = embedding_client.embed_many(
        serialized_name.tolist()
    )
    df["serialized_column"] = serialized_name.tolist()
    return df


def clean_strings(el: List[str]) -> List[str]:
    """Clean individual strings in a list."""
    return [clean_string(e) for e in el]


def clean_and_tokenize(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and tokenize the strings in the DataFrame."""
    df["enriched_name"] = [clean_strings(el) for el in df["enriched_name"].tolist()]
    df["enriched_name_syn"] = [
        clean_strings(el) for el in df["enriched_name_syn"].tolist()
    ]
    return df


def get_source_target_file_names_enriched(config: Dict[str, str]) -> Tuple[str, str]:
    """Get enriched source and target file names based on configuration."""
    base_dir = Path(__file__).resolve().parent.parent
    benchmark_name = config["data"]["benchmark_name"]
    enrichment_dir = base_dir / "processed_data" / benchmark_name / "enrichments"
    enrichment_dir.mkdir(parents=True, exist_ok=True)

    llm_config_values = get_llm_config_names(config)
    source_file, target_file = get_source_target_file_names(config)

    source_file_output = (
        enrichment_dir / f"{source_file.name.split('.')[0]}_{llm_config_values}.parquet"
    )
    target_file_output = (
        enrichment_dir / f"{target_file.name.split('.')[0]}_{llm_config_values}.parquet"
    )

    return source_file_output, target_file_output


def output_file_names(config: Dict[str, str]) -> Tuple[Path, Path]:
    """Generate output file names for embedded data."""
    base_dir = Path(__file__).resolve().parent.parent
    benchmark_name = config["data"]["benchmark_name"]
    embedding_dir = base_dir / "processed_data" / benchmark_name / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    llm_config_values = get_llm_config_names(config)
    embedding_call_type = get_embedding_config_names(config)
    source_file, target_file = get_source_target_file_names(config)

    source_path = (
        embedding_dir
        / f"{source_file.name.split('.')[0]}_{embedding_call_type}_{llm_config_values}.parquet"
    )
    target_path = (
        embedding_dir
        / f"{target_file.name.split('.')[0]}_{embedding_call_type}_{llm_config_values}.parquet"
    )

    return source_path, target_path


def process_file(df: pd.DataFrame, embedding_client: llm.LLMClient) -> pd.DataFrame:
    """Process a DataFrame by cleaning and adding embeddings."""
    df = clean_and_tokenize(df)
    df = add_embeddings(df, embedding_client)
    return df


def preprocess(config: Dict[str, Any]) -> None:
    """Main function to preprocess data and add embeddings."""
    embedding_client = setup_embedding_client(config)

    source_file_name, target_file_name = get_source_target_file_names_enriched(config)
    source_file_path, target_file_path = output_file_names(config)
    overwrite = config["experiment"]["overwrite"]
    source_flag, target_flag = check_output_files_exists(
        source_file_path, target_file_path
    )

    if not overwrite and source_flag and target_flag:
        logger.info("Embedded files already exist. Skipping embedding process.")
        return

    if overwrite or not source_flag:
        logger.info("Processing source file for embeddings.")
        source = pd.read_parquet(source_file_name)
        source = process_file(source, embedding_client)
        source.to_parquet(source_file_path)

    if overwrite or not target_flag:
        logger.info("Processing target file for embeddings.")
        target = pd.read_parquet(target_file_name)
        target = process_file(target, embedding_client)
        target.to_parquet(target_file_path)


def main(args: argparse.Namespace) -> None:
    """Main function to preprocess data and add embeddings."""
    config = load_config(args.config)
    preprocess(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schema Enrichment Script")
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config.toml"
    )
    args = parser.parse_args()
    main(args)
