from pathlib import Path
from typing import Any, Dict, Tuple

import toml

from utils import llm
from utils import logger as utils_logger

logger = utils_logger.logger_config("functions")


def get_source_target_file_names(config: Dict[str, str]) -> Tuple[Path, Path]:
    """Retrieve the file paths for source and target files based on benchmark name."""
    base_dir = Path(__file__).resolve().parent.parent
    benchmark_name = config["data"]["benchmark_name"]
    source_file_path = base_dir / "raw_data" / benchmark_name / "source"
    target_file_path = base_dir / "raw_data" / benchmark_name / "target"

    source_files = list(source_file_path.rglob("*.csv"))
    target_files = list(target_file_path.rglob("*.csv"))

    if len(source_files) != 1:
        raise ValueError(f"Expected one source file, found {len(source_files)}")
    if len(target_files) != 1:
        raise ValueError(f"Expected one target file, found {len(target_files)}")

    return source_files[0], target_files[0]


def get_mapping_file_name(config: Dict[str, str]) -> Path:
    """Retrieve the file path for mapping file based on benchmark name."""
    base_dir = Path(__file__).resolve().parent.parent
    benchmark_name = config["data"]["benchmark_name"]
    mapping_file_path = base_dir / "raw_data" / benchmark_name / "mapping"

    mapping_files_csv = list(mapping_file_path.rglob("*.csv"))
    mapping_files_parquet = list(mapping_file_path.rglob("*.parquet"))
    mapping_files = mapping_files_csv + mapping_files_parquet
    if len(mapping_files) != 1:
        raise ValueError(f"Expected one mapping file, found {len(mapping_files)}")

    return mapping_files[0]


def load_config(config_file: str) -> Dict:
    """Load configuration from a TOML file."""
    with open(config_file, "r") as ifs:
        config = toml.load(ifs)
    return config


def setup_llm_client(config: Dict, api_key: str) -> llm.LLMClient:
    """Setup LLM Client using the provided configuration."""
    client_config = config["llm"]["client"]
    client_config["args"]["api_key"] = api_key
    llm_client = llm.LLMClient.create(client_config["type"], **client_config["args"])
    return llm_client


def get_llm_config_names(config: Dict[str, str]) -> str:
    """Generate a configuration name string for the LLM client."""
    client_config = config["llm"]["client"]
    params = "_".join(
        f"{k}_{v}"
        for k, v in client_config["args"].items()
        if k in ["model_name", "api_version"]
    )
    return f"{client_config['type']}_{params}"


def get_embedding_config_names(config: Dict[str, str]) -> str:
    """Extract the embedding model name from configuration."""
    return config["embedding"]["client"]["args"]["model_name"].split("/")[-1]


def check_output_files_exists(source_output: Path, target_output: Path) -> bool:
    """Check if the output files already exist."""
    source_flag, target_flag = False, False
    if source_output.exists():
        source_flag = True
        logger.info(f"Output file already exists: {source_output}")
    if target_output.exists():
        target_flag = True
        logger.info(f"Output file already exists: {target_output}")
    return source_flag, target_flag


def check_output_file_exists(file_path: Path) -> bool:
    """Check if the output file already exists."""
    if file_path.exists():
        logger.info(f"Output file already exists: {file_path}")
        return True
    return False


def get_index_ablation_parameters(
    config: Dict[str, Any],
) -> Tuple[Dict[str, bool], str]:
    experiment_configs = config["experiment"]
    required_keys = [
        "query_enrichment",
        "document_enrichment",
        "embedding_search",
        "full_text_search",
        "non_sim_prompt",
    ]
    ablation_params = {key: experiment_configs[key] for key in required_keys}
    ablation_string = "_".join(
        [f"{key}_{str(value).lower()}" for key, value in ablation_params.items()]
    )
    return ablation_params, ablation_string


def get_rank_ablation_parameters(config: Dict[str, Any]) -> Tuple[Dict[str, bool], str]:
    experiment_configs = config["experiment"]
    required_keys = [
        "query_enrichment",
        "document_enrichment",
        "embedding_search",
        "full_text_search",
        "non_sim_prompt",
        "table_selection",
    ]
    ablation_params = {key: experiment_configs[key] for key in required_keys}
    ablation_string = "_".join(
        [f"{key}_{str(value).lower()}" for key, value in ablation_params.items()]
    )
    return ablation_params, ablation_string
