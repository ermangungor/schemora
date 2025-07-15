import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import bm25s
import faiss
import numpy as np
import pandas as pd

from utils.functions import (get_index_ablation_parameters,
                             get_mapping_file_name,
                             get_source_target_file_names, load_config)


def generate_bm25_index(target, ablation_params, n_candidates):
    itr = target[
        [
            "TableName",
            "ColumnName",
            "enriched_name",
            "enriched_name_syn",
            "ColumnType",
            "TableDesc",
            "ColumnDesc",
            "serialized_column",
        ]
    ].itertuples(index=False)
    corpus_json = []
    texts = []
    for i, row in enumerate(itr):
        if ablation_params["document_enrichment"]:
            el, el_syn = row[2][:n_candidates], row[3][:n_candidates]
            table_name = row[0]
            column_name = row[1]
            for doc_i in el:
                corpus_json.append(
                    {
                        "table_name": table_name,
                        "column_name": column_name,
                        "doc_id": i,
                        "enriched_name": doc_i,
                        "type": "enriched",
                        "target_type": row[4],
                        "target_table_desc": row[5],
                        "target_column_desc": row[6],
                    }
                )
                texts.append(doc_i)

            if ablation_params["non_sim_prompt"]:
                for doc_i in el_syn:
                    corpus_json.append(
                        {
                            "table_name": table_name,
                            "column_name": column_name,
                            "doc_id": i,
                            "enriched_name": doc_i,
                            "type": "enriched_syn",
                            "target_type": row[4],
                            "target_table_desc": row[5],
                            "target_column_desc": row[6],
                        }
                    )
                    texts.append(doc_i)
        else:
            serialized_column = row[7]
            table_name = row[0]
            column_name = row[1]
            corpus_json.append(
                {
                    "table_name": table_name,
                    "column_name": column_name,
                    "doc_id": i,
                    "target_type": row[4],
                    "target_table_desc": row[5],
                    "target_column_desc": row[6],
                    "serialized_column": serialized_column,
                }
            )
            texts.append(serialized_column)

    corpus_tokens = bm25s.tokenize(texts, show_progress=False)
    retriever = bm25s.BM25(corpus=corpus_json, method="lucene")
    retriever.index(corpus_tokens)

    return retriever


def generate_faiss_index(target, ablation_params, n_candidates):
    # Create a FAISS index
    # embeddings = model.encode(docs, normalize_embeddings=True)
    dimension = len(target["enriched_name_embeddings"].iloc[0][0])
    print("dimension", dimension)

    index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (dot product)
    ind2doc = {}

    itr = target[
        [
            "enriched_name_embeddings",
            "enriched_name_syn_embeddings",
            "TableName",
            "ColumnName",
            "enriched_name",
            "enriched_name_syn",
            "ColumnType",
            "TableDesc",
            "ColumnDesc",
            "serialized_column",
            "serialized_column_embedding",
        ]
    ].itertuples(index=False)
    embeddings = []
    cnt = 0
    for i, row in enumerate(itr):
        doc = row[0][:n_candidates]
        doc_syn = row[1][:n_candidates]
        table_name = row[2]
        column_name = row[3]
        enriched_names = row[4]
        enriched_names_str = row[5]

        if ablation_params["document_enrichment"]:
            for doc_i, enriched_name in zip(doc, enriched_names):
                embeddings.append(doc_i)
                ind2doc[cnt] = {
                    "table_name": table_name,
                    "column_name": column_name,
                    "doc_id": i,
                    "enriched_name": enriched_name,
                    "type": "enriched",
                    "target_type": row[6],
                    "target_table_desc": row[7],
                    "target_column_desc": row[8],
                }
                cnt += 1

            if ablation_params["non_sim_prompt"]:
                for doc_i_syn, enriched_name_str in zip(doc_syn, enriched_names_str):
                    embeddings.append(doc_i_syn)
                    ind2doc[cnt] = {
                        "table_name": table_name,
                        "column_name": column_name,
                        "doc_id": i,
                        "enriched_name": enriched_name_str,
                        "type": "enriched_syn",
                        "target_type": row[6],
                        "target_table_desc": row[7],
                        "target_column_desc": row[8],
                    }
                    cnt += 1
        else:
            serialized_column_embedding = row[10]
            embeddings.append(serialized_column_embedding)
            serialized_column = row[9]
            ind2doc[cnt] = {
                "table_name": table_name,
                "column_name": column_name,
                "doc_id": i,
                "target_type": row[6],
                "target_table_desc": row[7],
                "target_column_desc": row[8],
                "serialized_column": serialized_column,
            }
            cnt += 1

    embeddings = np.array(embeddings)
    index.add(embeddings)
    return index, ind2doc


def index_docs(target, ablation_params, n_candidates):
    retriever = generate_bm25_index(target, ablation_params, n_candidates)
    faiss_retriever, ind2doc = generate_faiss_index(
        target, ablation_params, n_candidates
    )
    return retriever, faiss_retriever, ind2doc


def get_input_file_names(config: Dict[str, str]) -> Tuple[Path, Path, Path]:

    client_config = config["llm"]["client"]
    imp_args = ""
    for k, v in client_config["args"].items():
        if k in ["model_name", "api_version"]:
            imp_args += f"{k}_{v}_"
    llm_config_values = client_config["type"] + "_" + imp_args[:-1]

    embedding_call_type = config["embedding"]["client"]["args"]["model_name"].split(
        "/"
    )[-1]

    base_dir = Path(__file__).resolve().parent.parent

    # Set up directories
    benchmark_name = config["data"]["benchmark_name"]
    embedding_dir = base_dir / "processed_data" / benchmark_name / "embeddings"

    source_file, target_file = get_source_target_file_names(config)
    source_file_name = (
        source_file.name.split(".")[0]
        + f"_{embedding_call_type}_{llm_config_values}.parquet"
    )
    source_file_path = embedding_dir / source_file_name

    target_file_name = (
        target_file.name.split(".")[0]
        + f"_{embedding_call_type}_{llm_config_values}.parquet"
    )
    target_file_path = embedding_dir / target_file_name

    mapping_file_path = get_mapping_file_name(config)

    return source_file_path, target_file_path, mapping_file_path


def load_data(
    config: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_file_name, target_file_name, mapping_file_name = get_input_file_names(config)
    source = pd.read_parquet(source_file_name)
    target = pd.read_parquet(target_file_name)
    if mapping_file_name.suffix == ".csv":
        mappings = pd.read_csv(mapping_file_name)
    elif mapping_file_name.suffix == ".parquet":
        mappings = pd.read_parquet(mapping_file_name)
    else:
        raise ValueError(
            f"Unsupported file format for mapping file: {mapping_file_name}"
        )
    mappings = mappings.dropna()
    return source, target, mappings


def process_source_target(
    source: pd.DataFrame, target: pd.DataFrame, mappings: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target_row_ind = {}
    for i, (table_name, column_name) in enumerate(
        target[["TableName", "ColumnName"]].itertuples(index=False)
    ):
        target_row_ind[(table_name, column_name)] = i

    source = source.merge(
        mappings,
        left_on=["TableName", "ColumnName"],
        right_on=["SRC_ENT", "SRC_ATT"],
        how="left",
    )

    return source, target_row_ind


def output_file_name(config: Dict[str, str], ablation_string: str) -> str:

    client_config = config["llm"]["client"]
    imp_args = ""
    for k, v in client_config["args"].items():
        if k in ["model_name", "api_version"]:
            imp_args += f"{k}_{v}_"
    llm_config_values = client_config["type"] + "_" + imp_args[:-1]

    embedding_call_type = config["embedding"]["client"]["args"]["model_name"].split(
        "/"
    )[-1]

    base_dir = Path(__file__).resolve().parent.parent
    benchmark_name = config["data"]["benchmark_name"]
    candidates_dir = base_dir / "processed_data" / benchmark_name / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    n_candidates = config["experiment"]["n_candidates"]

    candidate_file_name = f"candidates_{benchmark_name}_{llm_config_values}_{embedding_call_type}_{n_candidates}_{ablation_string}.parquet"
    candidate_file_path = candidates_dir / candidate_file_name
    return candidate_file_path


def search_with_enriched_queries(
    source,
    target_row_ind,
    faiss_retriever,
    bm25_retriever,
    ind2doc,
    ablation_params,
    n_candidates,
):
    itr = source[
        [
            "TableName",
            "ColumnName",
            "enriched_name",
            "enriched_name_syn",
            "enriched_name_embeddings",
            "enriched_name_syn_embeddings",
            "TGT_ENT",
            "TGT_ATT",
            "ColumnType",
            "TableDesc",
            "ColumnDesc",
        ]
    ].itertuples(index=False)
    result_row = []
    k = 30
    list_flag = any(
        source["TGT_ENT"].apply(
            lambda x: isinstance(x, list) or isinstance(x, np.ndarray)
        )
    )
    if list_flag:
        empty_target_ind = []
    else:
        empty_target_ind = -1

    for i, (
        table_name,
        column_name,
        enriched_name,
        enriched_name_syn,
        enriched_name_embeddings,
        enriched_name_syn_embeddings,
        target_table_name,
        target_column_name,
        query_type,
        table_desc,
        query_column_desc,
    ) in enumerate(itr):
        if isinstance(target_table_name, str):
            target_ind = target_row_ind.get((target_table_name, target_column_name), -1)
        elif isinstance(target_table_name, np.ndarray) or isinstance(
            target_table_name, list
        ):
            target_ind = []
            for target_name_i, target_column_name_i in zip(
                target_table_name, target_column_name
            ):
                target_ind.append(
                    target_row_ind.get((target_name_i, target_column_name_i), -1)
                )

        elif pd.isna(target_table_name) or pd.isna(target_column_name):
            target_ind = empty_target_ind
        else:
            raise ValueError(
                f"Unexpected type for target_table_name: {type(target_table_name)} or target_column_name: {type(target_column_name)}"
            )

        row_dict_template = {
            "query_ind": i,
            "query_table_name": table_name,
            "query_table_desc": table_desc,
            "query_column_name": column_name,
            "query_column_desc": query_column_desc,
            "query_type": query_type,
            "target_table_name": target_table_name,
            "target_column_name": target_column_name,
            "target_ind": target_ind,
        }

        if len(enriched_name_embeddings) > 0:
            enriched_name_embeddings = enriched_name_embeddings[:n_candidates]
            enriched_name = enriched_name[:n_candidates]
            if ablation_params["embedding_search"]:
                query_emb = np.stack(enriched_name_embeddings)
                (
                    score_faiss,
                    cand_faiss,
                ) = faiss_retriever.search(query_emb, k)

                for cand_i, score_i, enrich_i in zip(
                    cand_faiss, score_faiss, enriched_name
                ):
                    for cand_ind, score_in in zip(cand_i, score_i):
                        temp = ind2doc[int(cand_ind)]
                        row_dict = {
                            **row_dict_template,
                            **{"query_enriched_name": enrich_i},
                            **temp,
                        }
                        row_dict["score"] = float(score_in)
                        row_dict["q_type"] = "embedding_match"
                        result_row.append(row_dict)

            if ablation_params["full_text_search"]:
                query_tokens = bm25s.tokenize(
                    enriched_name,
                    show_progress=False,
                )
                results, scores = bm25_retriever.retrieve(
                    query_tokens, k=k, show_progress=False, n_threads=0
                )

                for cand_i, score_i, enrich_i in zip(results, scores, enriched_name):
                    for cand_ind, score_in in zip(cand_i, score_i):
                        row_dict = {
                            **row_dict_template,
                            **{"query_enriched_name": enrich_i},
                            **cand_ind,
                        }
                        row_dict["score"] = float(score_in)
                        row_dict["q_type"] = "bm25_match"
                        result_row.append(row_dict)

        if ablation_params["non_sim_prompt"] and len(enriched_name_syn_embeddings) > 0:
            enriched_name_syn_embeddings = enriched_name_syn_embeddings[:n_candidates]
            enriched_name_syn = enriched_name_syn[:n_candidates]
            if ablation_params["embedding_search"]:
                query_syn_emb = np.stack(enriched_name_syn_embeddings)
                score_faiss_n, cand_faiss_syn = faiss_retriever.search(query_syn_emb, k)

                for cand_i, score_i, enrich_i in zip(
                    cand_faiss_syn, score_faiss_n, enriched_name_syn
                ):
                    for cand_ind, score_in in zip(cand_i, score_i):
                        temp = ind2doc[int(cand_ind)]
                        row_dict = {
                            **row_dict_template,
                            **{"query_enriched_name": enrich_i},
                            **temp,
                        }
                        row_dict["score"] = float(score_in)
                        row_dict["q_type"] = "embedding_match_syn"
                        result_row.append(row_dict)

            if ablation_params["full_text_search"]:
                query_tokens_syn = bm25s.tokenize(
                    enriched_name_syn, show_progress=False
                )
                results_syn, scores_syn = bm25_retriever.retrieve(
                    query_tokens_syn, k=k, show_progress=False, n_threads=0
                )

                for cand_i, score_i, enrich_i in zip(
                    results_syn, scores_syn, enriched_name_syn
                ):
                    for cand_ind, score_in in zip(cand_i, score_i):
                        row_dict = {
                            **row_dict_template,
                            **{"query_enriched_name": enrich_i},
                            **cand_ind,
                        }
                        row_dict["score"] = float(score_in)
                        row_dict["q_type"] = "bm25_match_syn"
                        result_row.append(row_dict)

    return result_row


def search_without_enriched_queries(
    source, target_row_ind, faiss_retriever, bm25_retriever, ind2doc, ablation_params
):
    itr = source[
        [
            "TableName",
            "ColumnName",
            "TGT_ENT",
            "TGT_ATT",
            "ColumnType",
            "TableDesc",
            "ColumnDesc",
            "serialized_column",
            "serialized_column_embedding",
        ]
    ].itertuples(index=False)

    result_row = []
    k = 30
    list_flag = any(
        source["TGT_ENT"].apply(
            lambda x: isinstance(x, list) or isinstance(x, np.ndarray)
        )
    )
    if list_flag:
        empty_target_ind = []
    else:
        empty_target_ind = -1

    for i, (
        table_name,
        column_name,
        target_table_name,
        target_column_name,
        query_type,
        table_desc,
        query_column_desc,
        serialized_column,
        serialized_column_embedding,
    ) in enumerate(itr):
        if isinstance(target_table_name, str):
            target_ind = target_row_ind.get((target_table_name, target_column_name), -1)
        elif isinstance(target_table_name, np.ndarray) or isinstance(
            target_table_name, list
        ):
            target_ind = []
            for target_name_i, target_column_name_i in zip(
                target_table_name, target_column_name
            ):
                target_ind.append(
                    target_row_ind.get((target_name_i, target_column_name_i), -1)
                )

        elif pd.isna(target_table_name) or pd.isna(target_column_name):
            target_ind = empty_target_ind
        else:
            raise ValueError(
                f"Unexpected type for target_table_name: {type(target_table_name)} or target_column_name: {type(target_column_name)}"
            )

        if ablation_params["embedding_search"]:
            query_emb = serialized_column_embedding[np.newaxis, :]
            (
                score_faiss,
                cand_faiss,
            ) = faiss_retriever.search(query_emb, k)
            score_faiss, cand_faiss = (
                score_faiss[0],
                cand_faiss[0],
            )  # Get the first row since we have one query embedding
            row_dict_template = {
                "query_ind": i,
                "query_table_name": table_name,
                "query_table_desc": table_desc,
                "query_column_name": column_name,
                "query_column_desc": query_column_desc,
                "query_type": query_type,
                "target_table_name": target_table_name,
                "target_column_name": target_column_name,
                "target_ind": target_ind,
            }
            for cand_i, score_i in zip(cand_faiss, score_faiss):
                temp = ind2doc[int(cand_i)]
                row_dict = {
                    **row_dict_template,
                    **{"serialized_column": serialized_column},
                    **temp,
                }
                row_dict["score"] = float(score_i)
                row_dict["q_type"] = "embedding_match"
                result_row.append(row_dict)

        if ablation_params["full_text_search"]:
            query_tokens = bm25s.tokenize(
                serialized_column,
                show_progress=False,
            )
            results, scores = bm25_retriever.retrieve(
                query_tokens, k=k, show_progress=False, n_threads=0
            )
            results, scores = (
                results[0],
                scores[0],
            )  # Get the first row since we have one query tokenized

            for cand_i, score_i in zip(results, scores):
                row_dict = {
                    **row_dict_template,
                    **{"serialized_column": serialized_column},
                    **cand_i,
                }
                row_dict["score"] = float(score_i)
                row_dict["q_type"] = "bm25_match"
                result_row.append(row_dict)

    return result_row


def generate_candidates(config: Dict[str, Any]):
    n_candidates = config["experiment"]["n_candidates"]
    source, target, mappings = load_data(config)
    source, target_row_ind = process_source_target(source, target, mappings)
    ablation_params, ablation_string = get_index_ablation_parameters(config)
    out_file_name = output_file_name(config, ablation_string)
    overwrite = config["experiment"]["overwrite"]
    if os.path.exists(out_file_name) and not overwrite:
        print(f"Output file already exists: {out_file_name}")
        return
    bm25_retriever, faiss_retriever, ind2doc = index_docs(
        target, ablation_params, n_candidates
    )
    if ablation_params["query_enrichment"]:
        result_row = search_with_enriched_queries(
            source,
            target_row_ind,
            faiss_retriever,
            bm25_retriever,
            ind2doc,
            ablation_params,
            n_candidates,
        )
    else:
        result_row = search_without_enriched_queries(
            source,
            target_row_ind,
            faiss_retriever,
            bm25_retriever,
            ind2doc,
            ablation_params,
        )

    results_df = pd.DataFrame(result_row)
    results_df.to_parquet(out_file_name, index=False)


def main(args):
    config = load_config(args.config)
    generate_candidates(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schema Enrichment Script")
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config.toml"
    )
    args = parser.parse_args()
    main(args)
