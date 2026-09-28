#!/bin/bash

# ==========================================================
# GPU Configuration
# ==========================================================
# List of available GPU IDs
GPUs=("0")

# ==========================================================
# General Parameters
# ==========================================================
content_type="question"
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

# Reranker model (AAR-CE)
reranker_model_big="Qwen/Qwen3-Reranker-0.6B"
reranker_model_small="Qwen3-Reranker-0.6B"

# LLM reranker model (AAR-LLM-PW / AAR-LLM-LW)
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
# Step 2: AAR-CE (cross-encoder reranking)
# ==========================================================

echo "========== Step 2: AAR-CE (Cross-Encoder) =========="

CUDA_VISIBLE_DEVICES=${gpu_id} python AAR_CE.py \
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
# Step 3: AAR-LLM-PW (pointwise LLM answerability judge)
# ==========================================================

echo "========== Step 3: AAR-LLM-PW (Pointwise) =========="

CUDA_VISIBLE_DEVICES=${gpu_id} python AAR_LLM_PW.py \
  --model-path ${llm_reranker_model} \
  --tables_json_path ${tables_json_path} \
  --tables_path ${tables_path} \
  --json-path ./results/${embedding_model_small}/${table_format}/TCRAG_top5/output.json \
  --content_type ${content_type} \
  --table_format ${table_format} \
  --output-path ./results_rerank_LLM_PW/${embedding_model_small}/${table_format}/TCRAG_top5 \
  --key ${key} \
  --url ${url} \
  --top-k 3

wait

# ==========================================================
# Step 4: AAR-LLM-LW (listwise LLM ranking)
# ==========================================================

echo "========== Step 4: AAR-LLM-LW (Listwise) =========="

CUDA_VISIBLE_DEVICES=${gpu_id} python AAR_LLM_LW.py \
  --model-path ${llm_reranker_model} \
  --tables_json_path ${tables_json_path} \
  --tables_path ${tables_path} \
  --json-path ./results/${embedding_model_small}/${table_format}/TCRAG_top5/output.json \
  --content_type ${content_type} \
  --table_format ${table_format} \
  --output-path ./results_rerank_LLM_LW/${embedding_model_small}/${table_format}/TCRAG_top5 \
  --key ${key} \
  --url ${url} \
  --in-topk 5 \
  --top-k 3

wait

# ==========================================================
# Step 5: Compute Metrics
# ==========================================================
# Evaluate retrieval metrics (R@k, GR@k, DS@1) for each output
# Note: DS@1 is computed globally as mean(R@1) / mean(GR@1)

echo "========== Step 5: Compute Metrics =========="

# Step 1 output: Embedding mode
python Compute_Metrics.py \
  --input  ./results/${embedding_model_small}/${table_format}/TCRAG_top5/output.json \
  --output ./results/${embedding_model_small}/${table_format}/TCRAG_top5/full_result.csv \
  --mode   Embedding

# Step 2 output: Rerank mode (AAR-CE)
python Compute_Metrics.py \
  --input  ./results_rerank/${reranker_model_small}/${embedding_model_small}/${table_format}/TCRAG_top5/output.json \
  --output ./results_rerank/${reranker_model_small}/${embedding_model_small}/${table_format}/TCRAG_top5/full_result.csv \
  --mode   Rerank

# Step 3 output: AAR-LLM-PW
python Compute_Metrics.py \
  --input  ./results_rerank_LLM_PW/${embedding_model_small}/${table_format}/TCRAG_top5/output.json \
  --output ./results_rerank_LLM_PW/${embedding_model_small}/${table_format}/TCRAG_top5/full_result.csv \
  --mode   Rerank

# Step 4 output: AAR-LLM-LW
python Compute_Metrics.py \
  --input  ./results_rerank_LLM_LW/${embedding_model_small}/${table_format}/TCRAG_top5/output.json \
  --output ./results_rerank_LLM_LW/${embedding_model_small}/${table_format}/TCRAG_top5/full_result.csv \
  --mode   Rerank

echo "========== Pipeline Finished =========="
