import argparse
import os

import toml

from utils import functions
from utils import logger as utils_logger

# Initialize logger for the script
logger = utils_logger.logger_config("column_entities")


def main(args):
    dir_configs = "scripts/omop_configs"
    # dir_configs = "scripts/synt_configs"
    os.makedirs(dir_configs, exist_ok=True)

    n_candidates = [1, 2, 3]
    # n_candidates = [2,3]
    # n_candidates = [3]
    llm_models = ["gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14"]
    # llm_models = ["gpt-4.1-2025-04-14"]
    embedding_models = [
        ("local", "all-mpnet-base-v2"),
        ("openai", "text-embedding-ada-002"),
        ("openai", "text-embedding-3-large"),
        ("openai", "text-embedding-3-small"),
        ("local", "BAAI/bge-small-en-v1.5"),
    ]
    # embedding_models = [("openai","text-embedding-3-small"),("local","BAAI/bge-small-en-v1.5")]
    # embedding_models = [("openai","text-embedding-ada-002"),("openai","text-embedding-3-large"),("openai","text-embedding-3-small")]
    # embedding_models = [("local","BAAI/bge-small-en-v1.5")]
    for n_cand in n_candidates:
        for llm_model in llm_models:
            for type_model, embedding_model in embedding_models:
                config = functions.load_config(args.config)
                config["experiment"]["n_candidates"] = n_cand
                config["llm"]["client"]["args"]["model_name"] = llm_model
                config["embedding"]["client"]["args"]["model_name"] = embedding_model
                if type_model == "local":
                    config["embedding"]["client"]["args"]["nthreads"] = 1
                else:
                    config["embedding"]["client"]["args"]["nthreads"] = 10
                config["embedding"]["client"]["type"] = type_model
                logger.info(
                    f"Running with n_candidates={n_cand}, llm_model={llm_model}, embedding_model={embedding_model}, embedding_model_type:{type_model}"
                )
                fname = f"config_{n_cand}_{llm_model}_{embedding_model}_{type_model}"
                fname = fname.replace(".", "_").replace("/", "_")
                fname = os.path.join(dir_configs, fname + ".toml")
                with open(fname, "w") as f:
                    toml.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Table Selection Script for Schema Enrichment"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config file",
        default="config_omop_template.toml",
    )
    args = parser.parse_args()
    main(args)
