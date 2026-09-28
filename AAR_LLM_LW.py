"""
AAR-LLM-LW: listwise LLM-based answerability-aware reranking.

Given top-k candidates from dense retrieval, jointly rank all candidates
in a single LLM call (listwise), rather than scoring each table independently.
"""

import argparse
import ast
import csv
import datetime
import json
import logging
import os
import re
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List

import pandas as pd
import tqdm
from openai import OpenAI

from tools.mklog import mklog
from tools.reformat import (
    df2csv,
    df2html,
    df2json,
    df2latex,
    df2mk,
    df2sentence,
    df2sentence_shuffled,
    df2sql,
    df2xml,
)

warnings.filterwarnings("ignore")


QUERY_PREFIX_MAP = {
    "tableqa": (
        "Given a TableQA query, retrieve a relevant table that can answer the query. "
        "Note that the retrieved table should contain sufficient information to provide an answer, "
        "rather than resulting in an empty answer. "
    ),
    "fact_verification": (
        "Given a claim about tabular data, retrieve a relevant table whose schema and content "
        "can be used to verify the claim. "
        "Focus on tables whose column names and row structure are relevant to the claim, "
        "even if an exact answer cannot be directly read off. "
    ),
    "text2sql": (
        "Given a natural language question that may require joining or querying multiple tables, "
        "retrieve all tables whose schemas are necessary to write a SQL query answering the question. "
        "Prioritize tables whose column names match the entities, attributes, and join keys "
        "mentioned in the question. "
    ),
}

TASK_TYPES = list(QUERY_PREFIX_MAP.keys())

QUERY_LABEL_MAP = {
    "tableqa": "Query",
    "fact_verification": "Claim",
    "text2sql": "Question",
}

RANK_SYSTEM_PROMPT = """\
You are an expert table retrieval assistant.
You will be given a query and K candidate tables, each labeled with an anonymous ID \
like Table_0, Table_1, etc.

Rank ALL candidate tables from most to least likely to answer the query.
Put the table most likely to contain the answer first.
Include every candidate anonymous ID exactly once. Do not invent IDs.

Output format — respond with a JSON object (no markdown fences, no extra text):
{
  "ranked_ids": ["Table_2", "Table_0", "Table_1"]
}
"""


def build_table_key(db_id, table_path):
    return "{}::{}".format(db_id, table_path)


def str_to_list(s):
    # type: (str) -> List[str]
    return ast.literal_eval(s)


def trans_tables(table_name, df, table_format):
    empty_df = df.copy()
    if table_format == "markdown":
        table_infos = df2mk(empty_df)
    elif table_format == "html":
        table_infos = df2html(empty_df)
    elif table_format == "sql":
        table_infos = df2sql(empty_df, table_name)
    elif table_format == "latex":
        table_infos = df2latex(empty_df)
    elif table_format == "xml":
        table_infos = df2xml(empty_df)
    elif table_format == "json":
        table_infos = df2json(empty_df)
    elif table_format == "df":
        table_infos = str(empty_df)
    elif table_format == "csv":
        table_infos = df2csv(empty_df)
    elif table_format == "sentence":
        table_infos = df2sentence(empty_df)
    elif table_format == "sentence_shuffled":
        table_infos = df2sentence_shuffled(empty_df)
    else:
        table_infos = df2mk(empty_df)

    table_infos = re.sub(r" +", " ", table_infos)
    return "{}:\n{}\n\n".format(table_name, table_infos)


def load_all_tables(tables_json_path, tables_path, table_format):
    """Load tables.json, read all tables, build table_store (without embedding)."""
    with open(tables_json_path, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    table_store = {}
    for item in tqdm.tqdm(tables_data, desc="Loading table data"):
        db_id = item["db_id"]
        table_path = item["table_path"]
        table_name = item["table_name"]

        if table_format == "mixed":
            table_format_tmp = item["table_format"]
        else:
            table_format_tmp = table_format

        table_file = os.path.join(tables_path, db_id, "{}.csv".format(table_path))
        df = pd.read_csv(table_file, low_memory=False)
        table_str = trans_tables(table_name, df, table_format_tmp)
        key = build_table_key(db_id, table_path)
        table_store[key] = {"table_str": table_str}

    return table_store


def _truncate_table(table_str, preview_chars):
    if preview_chars <= 0 or len(table_str) <= preview_chars:
        return table_str
    return table_str[:preview_chars] + "\n[... table truncated ...]"


def _strip_think(raw_text):
    think_end = raw_text.rfind("</think>")
    if think_end == -1:
        return raw_text.strip()
    return raw_text[think_end + len("</think>"):].strip()


def _extract_ranked_ids(raw_text, valid_ids):
    answer_text = _strip_think(raw_text)
    answer_text = re.sub(r"^```(?:json)?\s*", "", answer_text.strip())
    answer_text = re.sub(r"\s*```$", "", answer_text.strip())

    ranked_ids = []
    try:
        data = json.loads(answer_text)
        if isinstance(data, dict):
            for key in ("ranked_ids", "order", "ranking", "ids"):
                value = data.get(key)
                if isinstance(value, list):
                    ranked_ids = value
                    break
        elif isinstance(data, list):
            ranked_ids = data
    except (json.JSONDecodeError, TypeError):
        ranked_ids = re.findall(r"Table_\d+", answer_text)

    cleaned = []
    seen = set()
    valid_set = set(valid_ids)
    for item in ranked_ids:
        if not isinstance(item, str):
            item = str(item)
        item = item.strip()
        if item not in valid_set or item in seen:
            continue
        cleaned.append(item)
        seen.add(item)

    for anon_id in valid_ids:
        if anon_id not in seen:
            cleaned.append(anon_id)
    return cleaned


def _build_user_prompt(query, task_type, candidate_previews):
    task_hint = QUERY_PREFIX_MAP.get(task_type, QUERY_PREFIX_MAP["tableqa"])
    query_label = QUERY_LABEL_MAP.get(task_type, QUERY_LABEL_MAP["tableqa"])
    previews_str = "\n\n".join(
        "[{}]\n{}".format(anon_id, preview)
        for anon_id, preview in candidate_previews
    )
    n = len(candidate_previews)
    return (
        "Task context: {}\n\n"
        "{}: {}\n\n"
        "Candidate tables ({}):\n{}\n\n"
        "Rank all {} candidate tables from most to least relevant. "
        "Return every anonymous ID exactly once, best first.".format(
            task_hint, query_label, query, n, previews_str, n
        )
    )


def llm_rank_tables(client, model, query, candidate_previews, task_type, max_retries=3):
    valid_ids = [anon_id for anon_id, _ in candidate_previews]
    if not valid_ids:
        return []

    user_msg = _build_user_prompt(query, task_type, candidate_previews)
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": RANK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=8192,
                timeout=18000,
                top_p=1.0,
                seed=42,
            )
            raw = response.choices[0].message.content or ""
            ranked_ids = _extract_ranked_ids(raw, valid_ids)
            logging.info(
                "LLM listwise rank for [{}]: {}".format(query[:60], ranked_ids)
            )
            return ranked_ids
        except Exception as e:
            last_error = e
            logging.warning(
                "LLM listwise rank attempt {}/{} failed: {}".format(
                    attempt, max_retries, e
                )
            )

    logging.error(
        "All {} listwise rank attempts failed for [{}]: {}".format(
            max_retries, query[:60], last_error
        )
    )
    return list(valid_ids)


def rerank_topk_tables(
        client,
        query,
        table_store,
        model,
        task_type,
        preview_chars,
        topk=1,
        max_retries=3,
):
    """Give the LLM k tables at once; return (key, score) in model order."""
    keys = list(table_store.keys())
    if not keys:
        return []

    candidate_previews = []
    anon_to_key = {}
    for i, key in enumerate(keys):
        anon_id = "Table_{}".format(i)
        anon_to_key[anon_id] = key
        table_str = table_store[key].get("table_str", "") or ""
        candidate_previews.append(
            (anon_id, _truncate_table(table_str, preview_chars))
        )

    ranked_ids = llm_rank_tables(
        client=client,
        model=model,
        query=query,
        candidate_previews=candidate_previews,
        task_type=task_type,
        max_retries=max_retries,
    )

    n = len(ranked_ids)
    scores = []
    for rank, anon_id in enumerate(ranked_ids):
        key = anon_to_key.get(anon_id)
        if key is None:
            continue
        scores.append((key, float(n - rank)))
    return scores[:topk]


def evaluate(
        model_path,
        output_path,
        all_data,
        content_type,
        table_format,
        tables_path,
        key,
        url,
        tables_json_path,
        in_topk,
        top_k,
        max_workers=8,
        task_type="tableqa",
        preview_chars=0,
        max_retries=3,
):
    start_time = time.time()

    client = OpenAI(base_url=url, api_key=key)
    os.makedirs(output_path, exist_ok=True)

    logging.info("output_path: {}".format(output_path))
    logging.info("Loading tables.json: {}".format(tables_json_path))

    table_store_all = load_all_tables(
        tables_json_path, tables_path, table_format,
    )

    output_jsonl = os.path.join(output_path, "output.jsonl")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        f.write("")

    write_lock = Lock()

    def process_one_query(conv):
        try:
            query = conv[content_type]
            retrieved_by_embedding = str_to_list(conv["output_text"])
            candidate_keys = retrieved_by_embedding[:in_topk]

            table_store_subset = {
                k: table_store_all[k]
                for k in candidate_keys
                if k in table_store_all
            }

            retrieved = rerank_topk_tables(
                client=client,
                query=query,
                table_store=table_store_subset,
                model=model_path,
                task_type=task_type,
                preview_chars=preview_chars,
                topk=top_k,
                max_retries=max_retries,
            )

            retrieved_keys = [k for k, _ in retrieved]
            retrieved_scores = [s for _, s in retrieved]

            gt_table_path = conv["highlighted_table"][0]
            db_id = conv["db_id"]

            topk_success = []
            for k_idx in range(top_k):
                success = 0
                for key_ in retrieved_keys[: k_idx + 1]:
                    db, table_path = key_.split("::")
                    if db == db_id and table_path == gt_table_path:
                        success = 1
                        break
                topk_success.append(success)

            output_dict = {
                **conv,
                "output_text_rerank": str(retrieved_keys),
                "top_k_similarities_rerank": retrieved_scores,
                "tokens_rerank": 1,
            }
            for k_idx in range(top_k):
                output_dict["top_{}_rerank".format(k_idx + 1)] = topk_success[k_idx]

        except Exception:
            logging.error(traceback.format_exc())
            output_dict = {
                **conv,
                "output_text_rerank": "error",
                "tokens_rerank": 0,
                "top_k_similarities_rerank": [0] * top_k,
            }
            for k_idx in range(top_k):
                output_dict["top_{}_rerank".format(k_idx + 1)] = 0

        with write_lock:
            with open(output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_one_query, conv)
            for conv in all_data
        ]
        for _ in tqdm.tqdm(as_completed(futures), total=len(futures)):
            pass

    elapsed_hours = (time.time() - start_time) / 3600.0
    _finalize_output(output_path, top_k, elapsed_hours=elapsed_hours)


def _finalize_output(output_path, top_k, elapsed_hours=0.0):
    jsonl_path = os.path.join(output_path, "output.jsonl")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        dt = [json.loads(line) for line in f if line.strip()]

    with open(os.path.join(output_path, "output.json"), "w", encoding="utf-8") as f:
        json.dump(dt, f, ensure_ascii=False, indent=4)

    metrics = {
        "top_{}_rerank".format(k + 1):
            sum(item["top_{}_rerank".format(k + 1)] for item in dt) / len(dt)
        for k in range(top_k)
    }
    metrics["data_num"] = len(dt)
    metrics["elapsed_hours"] = elapsed_hours
    metrics["current_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    csv_path = os.path.join(output_path, "result.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    logging.info("[{}] metrics: {}".format(output_path, metrics))


def validate_content_type(value):
    valid_values = [
        "question", "question_template", "question_sentence", "markdown_table"
    ]
    if value not in valid_values:
        return "question"
    return value


def main():
    parser = argparse.ArgumentParser(
        description="AAR-LLM-LW: listwise LLM reranking over dense top-k candidates."
    )
    parser.add_argument("--json-path", type=str, default="data/eval.json")
    parser.add_argument("--tables_path", type=str, default="data/eval")
    parser.add_argument("--tables_json_path", type=str, default="tables.json")
    parser.add_argument("--content_type", type=validate_content_type, default="question")
    parser.add_argument(
        "--table_format",
        type=str,
        default="markdown",
        choices=["markdown", "html", "csv", "mixed", "sentence", "sentence_shuffled"],
    )
    parser.add_argument("--model-path", type=str, default="model")
    parser.add_argument("--output-path", type=str, default="data/eval_output")
    parser.add_argument("--key", type=str, default="xxx")
    parser.add_argument("--url", type=str, default="xxx")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--in-topk",
        type=int,
        default=5,
        help="Rerank the first K candidates from output_text (listwise).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Keep top-m after listwise reranking for evaluation.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=0,
        help="Max chars per table sent to the LLM; 0 means no truncation.",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--task-type",
        type=str,
        default="tableqa",
        choices=TASK_TYPES,
        help="Retrieval task type: tableqa | fact_verification | text2sql",
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    filename = os.path.splitext(os.path.basename(args.json_path))[0] + "-" + args.table_format
    mklog(filename)
    logging.info(vars(args))

    all_data = json.load(open(args.json_path, encoding="utf-8"))

    evaluate(
        model_path=args.model_path,
        output_path=args.output_path,
        all_data=all_data,
        content_type=args.content_type,
        table_format=args.table_format,
        tables_path=args.tables_path,
        key=args.key,
        url=args.url,
        tables_json_path=args.tables_json_path,
        in_topk=args.in_topk,
        top_k=args.top_k,
        max_workers=args.max_workers,
        task_type=args.task_type,
        preview_chars=args.preview_chars,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
