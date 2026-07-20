#!/bin/bash

# ==========================================================
# GPU Configuration
# ==========================================================
# List of available GPU IDs
GPUs=("0")

# ==========================================================
# General Parameters
# ==========================================================
content_type="single"
key="xxx"
url="xxx"

# Number of parallel jobs before waiting
batch_size=1

# Table format used in this demo
table_format="mixed"

# ==========================================================
# Dataset Configuration
# ==========================================================
tables_json_path="./data/TCR_Tables.json"
tables_path="./tables/TCR_8k"
questions_json="./data/TCR_Questions.json"

# ==========================================================
# Model Configuration
# ==========================================================

# Embedding model (for retrieval)
embedding_model_big="Qwen/Qwen3-Embedding-0.6B"
embedding_model_small="Qwen3-Embedding-0.6B"

# Reranker model
reranker_model_big="Qwen/Qwen3-Reranker-0.6B"
reranker_model_small="Qwen3-Reranker-0.6B"

# LLM reranker model
llm_reranker_model="Qwen/Qwen3-30B-A3B-Thinking-2507"

# ==========================================================
# Step 1: Embedding Retrieval
# ==========================================================
# Retrieve Top-K tables using embedding similarity

gpu_id=${GPUs[0]}

echo "========== Step 1: Embedding Retrieval =========="

CUDA_VISIBLE_DEVICES=${gpu_id} python Embedding.py \
  --model-path ${embedding_model_big} \
  --tables_json_path ${tables_json_path} \
  --tables_path ${tables_path} \
  --json-path ${questions_json} \
  --content_type ${content_type} \
  --table_format ${table_format} \
  --output-path ./results/${embedding_model_small}/${table_format}/TCRAG_top5 \
  --key ${key} \
  --url ${url} \
  --top-k 5

wait

# ==========================================================
# Step 2: Reranker Model Reranking
# ==========================================================
# Use a dedicated reranker model to rerank the retrieved results

echo "========== Step 2: Reranker Model =========="

CUDA_VISIBLE_DEVICES=${gpu_id} python Rerank_in_topk.py \
  --model-path ${reranker_model_big} \
  --tables_json_path ${tables_json_path} \
  --tables_path ${tables_path} \
  --json-path ./results/${embedding_model_small}/${table_format}/TCRAG_top5/output.json \
  --content_type ${content_type} \
  --table_format ${table_format} \
  --output-path ./results_rerank/${reranker_model_small}/${embedding_model_small}/${table_format}/TCRAG_top5 \
  --key ${key} \
  --url ${url} \
  --top-k 3

wait

# ==========================================================
# Step 3: LLM-based Reranking
# ==========================================================
# Use a large language model to perform reasoning-based reranking

echo "========== Step 3: LLM Reranking =========="

CUDA_VISIBLE_DEVICES=${gpu_id} python LLM_rerank_in_topk.py \
  --model-path ${llm_reranker_model} \
  --tables_json_path ${tables_json_path} \
  --tables_path ${tables_path} \
  --json-path ./results/${embedding_model_small}/${table_format}/TCRAG_top5/output.json \
  --content_type ${content_type} \
  --table_format ${table_format} \
  --output-path ./results_rerank_LLM/${embedding_model_small}/${table_format}/TCRAG_top5 \
  --key ${key} \
  --url ${url} \
  --top-k 3

wait

# ==========================================================
# Step 4: Compute Metrics
# ==========================================================
# Evaluate retrieval metrics (R@k, GR@k, DS@1) for each output

echo "========== Step 4: Compute Metrics =========="

# Step 1 output: Embedding mode
python Compute_Metrics.py \
  --input  ./results/${embedding_model_small}/${table_format}/TCRAG_top5/output.json \
  --output ./results/${embedding_model_small}/${table_format}/TCRAG_top5/full_result.csv \
  --mode   Embedding

# Step 2 output: Rerank mode
python Compute_Metrics.py \
  --input  ./results_rerank/${reranker_model_small}/${embedding_model_small}/${table_format}/TCRAG_top5/output.json \
  --output ./results_rerank/${reranker_model_small}/${embedding_model_small}/${table_format}/TCRAG_top5/full_result.csv \
  --mode   Rerank

# Step 3 output: Rerank mode (LLM reranker also uses output_text_rerank)
python Compute_Metrics.py \
  --input  ./results_rerank_LLM/${embedding_model_small}/${table_format}/TCRAG_top5/output.json \
  --output ./results_rerank_LLM/${embedding_model_small}/${table_format}/TCRAG_top5/full_result.csv \
  --mode   Rerank

echo "========== Pipeline Finished =========="