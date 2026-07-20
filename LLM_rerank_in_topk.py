import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import AutoConfig
import json, os
import sys
import nltk
from nltk.translate.bleu_score import sentence_bleu
from tools.reformat import *
import ast
import pandas as pd
import re
import logging
import os
import datetime
import warnings
#from vllm import LLM, SamplingParams
import tqdm
import torch.distributed as dist
import math
warnings.filterwarnings("ignore")
import argparse
import traceback
#from openai import AzureOpenAI
#from openai import OpenAI
import concurrent.futures
import time
from tools.metrics import *
import csv
from collections import defaultdict
import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi

from torch import Tensor
from typing import Dict, List

from tools.mklog import mklog
from tools.process_json import save_json

from tools.qwen3_rerank import rerank

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from threading import Lock

def get_embedding_all(query, tokenizer, model, model_name):
    if 'bge-m3' in model_name.lower():
        from tools.m3_embedding import get_embedding
    elif 'qwen3-embedding' in model_name.lower():
        from tools.qwen3_embedding import last_token_pool, get_embedding
    elif 'stella' in model_name.lower():
        from tools.stella_embedding import get_embedding
    elif 'jina-embeddings-v3' in model_name.lower():
        from tools.jina_embedding_v3 import get_embedding
    elif 'jina-embeddings-v4' in model_name.lower():
        from tools.jina_embedding_v4 import get_embedding
    else:  # bge, gte
        from tools.gte_embedding import get_embedding
    return get_embedding(query, tokenizer, model)

def build_table_key(db_id, table_path):
    """Unique table key = db_id::table_path"""
    return f"{db_id}::{table_path}"

def str_to_list(s: str) -> List[str]:
    return ast.literal_eval(s)

def load_all_tables(tables_json_path, tables_path, table_format):
    """Load tables.json, read all tables, build table_store (without embedding)"""
    with open(tables_json_path, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)

    table_store = {}
    for item in tqdm.tqdm(tables_data, desc="Loading table data"):
        db_id = item['db_id']
        table_path = item['table_path']
        table_name = item['table_name']

        if table_format == 'mixed':
            table_format_tmp = item['table_format']
        else:
            table_format_tmp = table_format

        table_file = os.path.join(tables_path, db_id, f"{table_path}.csv")
        df = pd.read_csv(table_file, low_memory=False)

        table_str = trans_tables(table_name, df, table_format_tmp)

        key = build_table_key(db_id, table_path)
        table_store[key] = {
            "table_str": table_str
        }

    return table_store

def trans_tables(table_name, df, table_format):
    empty_df = df.copy()
    if table_format == 'markdown':
        table_infos = df2mk(empty_df)
    elif table_format == 'html':
        table_infos = df2html(empty_df)
    elif table_format == 'sql':
        table_infos = df2sql(empty_df, table_name)
    elif table_format == 'latex':
        table_infos = df2latex(empty_df)
    elif table_format == 'xml':
        table_infos = df2xml(empty_df)
    elif table_format == 'json':
        table_infos = df2json(empty_df)
    elif table_format == 'df':
        table_infos = str(empty_df)
    elif table_format == 'csv':
        table_infos = df2csv(empty_df)
    elif table_format == 'sentence':
        table_infos = df2sentence(empty_df)
    elif table_format == 'sentence_shuffled':
        table_infos = df2sentence_shuffled(empty_df)

    table_infos = re.sub(r' +', ' ', table_infos)
    result = f"{table_name}:\n{table_infos}\n\n"
    return result

# ======================
# rerank (order within query)
# ======================
def rerank_topk_tables(
        client,
        query,
        table_store,
        model,
        task_prompt,
        topk=1
):
    """Score candidate tables for a query and return top-k"""

    def check_answerable(table_str):
        messages = [
            {
                "role": "system",
                "content": task_prompt
            },
            {
                "role": "user",
                "content": (
                    f"Table:\n{table_str}\n\n"
                    f"Query: {query}\n\n"
                    "Please output your response in the following format:\n"
                    "```Yes``` or ```No```"
                )
            }
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0
        )

        answer_text = response.choices[0].message.content.strip()

        logging.info(f'output for {table_str} and {query}: {answer_text}')

        # Handle </think> tags
        if "</think>" in answer_text:
            answer_text = answer_text.split("</think>")[-1].strip()

        match = re.search(r'```(.*?)```', answer_text, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()

        return "yes" in answer_text.lower()

    results = []
    for key, val in table_store.items():
        try:
            is_answerable = check_answerable(val["table_str"])
            results.append((key, is_answerable))
        except Exception as e:
            logging.error(f"Error processing table {key}: {e}")
            results.append((key, False))

    answerable = [item for item in results if item[1]]
    not_answerable = [item for item in results if not item[1]]
    sorted_results = answerable + not_answerable

    sorted_keys = [k for k, _ in sorted_results]
    scores = [
        (k, 1.0 if k in [x for x, _ in answerable] else 0.0)
        for k in sorted_keys
    ]

    return scores[:topk]


# ======================
# Parallel evaluation at query level
# ======================
@torch.inference_mode()
def evaluate(
        model_path,
        output_path,
        all_data,
        content_type,
        table_format,
        tables_path,
        key,
        url,
        use_device=[1, 2],
        tables_json_path='tables.json',
        top_k=1
):
    max_workers=40
    client = OpenAI(base_url=url, api_key=key)
    os.makedirs(output_path, exist_ok=True)

    logging.info(f'output_path: {output_path}')
    logging.info(f'Loading tables.json: {tables_json_path}')
    logging.info('Load all tables at once, filter subset by sample when reranking')

    table_store_all = load_all_tables(
        tables_json_path,
        tables_path,
        table_format
    )

    task_prompt = (
        'Given a table and a question, determine if the table provides '
        'sufficient information to answer the question. You should not rely '
        'solely on the schema; ensure that the content within the table can '
        'actually address the question.'
    )

    output_file = os.path.join(output_path, 'output.jsonl')
    write_lock = Lock()

    def process_one_query(conv):
        try:
            query = conv[content_type]

            retrieved_by_embedding = str_to_list(conv["output_text"])

            table_store_subset = {
                k: table_store_all[k]
                for k in retrieved_by_embedding
                if k in table_store_all
            }

            retrieved = rerank_topk_tables(
                client=client,
                query=query,
                table_store=table_store_subset,
                model=model_path,
                task_prompt=task_prompt,
                topk=top_k
            )

            retrieved_keys = [k for k, _ in retrieved]
            retrieved_scores = [s for _, s in retrieved]

            gt_table_path = conv['highlighted_table'][0]
            db_id = conv['db_id']

            topk_success = []
            for k_idx in range(top_k):
                success = 0
                for key_ in retrieved_keys[:k_idx + 1]:
                    db, table_path = key_.split("::")
                    if db == db_id and table_path == gt_table_path:
                        success = 1
                        break
                topk_success.append(success)

            output_dict = {
                **conv,
                'output_text_rerank': str(retrieved_keys),
                'top_k_similarities_rerank': retrieved_scores,
                'tokens_rerank': 1
            }

            for k_idx in range(top_k):
                output_dict[f'top_{k_idx + 1}_rerank'] = topk_success[k_idx]

        except Exception:
            logging.error(traceback.format_exc())
            output_dict = {
                **conv,
                'output_text_rerank': "error",
                'tokens_rerank': 0,
                'top_k_similarities_rerank': [0] * top_k
            }
            for k_idx in range(top_k):
                output_dict[f'top_{k_idx + 1}_rerank'] = 0

        with write_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')

    # ======================
    # Parallel processing at query level
    # ======================
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_one_query, conv)
            for conv in all_data
        ]

        for _ in tqdm.tqdm(as_completed(futures), total=len(futures)):
            pass

def save_csv(data, path):
    if len(data) == 0:
        return
    keys = data[0].keys()
    with open(path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


def calculate_topk_metrics(dt, top_k):
    metrics = {}
    for k_idx in range(top_k):
        scores = [item[f'top_{k_idx+1}'] for item in dt]
        if scores:
            metrics[f'top_{k_idx+1}'] = sum(scores) / len(scores)
        else:
            metrics[f'top_{k_idx+1}'] = 0
    metrics['data_num'] = len(dt)
    metrics['current_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return metrics


@torch.inference_mode()
def evaluate_all(model_path, json_path, output_path, num_gpus_total, num_gpus_per_model,
                 content_type, table_format, use_device, tables_path, key, url, tables_json_path, top_k):

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Clear output file
    with open(os.path.join(output_path, 'output.jsonl'), 'w') as f:
        f.write('')

    all_data = json.load(open(json_path))
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(evaluate).remote
    else:
        get_answers_func = evaluate

    chunk_size = len(all_data) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(all_data), chunk_size):
        cur_data = all_data[i:i + chunk_size]
        ans_handles.append(
            get_answers_func(model_path, output_path, cur_data,
                             content_type, table_format, tables_path, key, url, use_device, tables_json_path, top_k)
        )
    if use_ray:
        ray.get(ans_handles)

    with open(os.path.join(output_path, 'output.jsonl'), 'r') as f:
        dt = [json.loads(line) for line in f]

    # Save full output JSON
    with open(os.path.join(output_path, 'output.json'), 'w', encoding='utf-8') as json_file:
        json.dump(dt, json_file, ensure_ascii=False, indent=4)

    # Calculate top-k accuracy
    overall_result = calculate_topk_metrics(dt, top_k)
    csv_path = os.path.join(output_path, 'result.csv')
    save_csv([overall_result], csv_path)

def validate_content_type(value):
    valid_values = ['question', 'question_template', 'question_sentence','markdown_table']
    if value not in valid_values:
        return 'question'  # Invalid value defaults to default value
    return value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-path', type=str, default='data/eval.json')
    parser.add_argument('--tables_path', type=str, default='data/eval')
    parser.add_argument('--tables_json_path', type=str, default='tables.json')
    # parser.add_argument('--content_type', type=str, default='small_content',
    #                     choices=['question', 'question_template','question_sentence'])
    parser.add_argument('--content_type', type=validate_content_type, default='question')
    parser.add_argument('--table_format', type=str, default='markdown',
                        choices=['markdown', 'html', 'csv', 'mixed','sentence','sentence_shuffled'])
    parser.add_argument('--model-path', type=str, default='model')
    parser.add_argument('--output-path', type=str, default='data/eval_output')
    parser.add_argument('--key', type=str, default='xxx')
    parser.add_argument('--url', type=str, default='xxx')
    parser.add_argument('--num-gpus-total', type=int, default=1)
    parser.add_argument('--num-gpus-per-model', type=int, default=1)
    parser.add_argument('--use-device', type=int, nargs='+', default=[1, 2], help='List of devices')
    parser.add_argument('--top-k', type=int, default=1, help='Top-k tables to evaluate')
    args = parser.parse_args()

    filename = os.path.splitext(os.path.basename(args.json_path))[0]+"-"+args.table_format
    mklog(filename)
    logging.info(vars(args))
    evaluate_all(**vars(args))


if __name__ == "__main__":
    main()