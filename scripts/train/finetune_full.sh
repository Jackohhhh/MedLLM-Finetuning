#!/usr/bin/env bash
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

################## 按需改行内参数 ##################
# 多卡：把下一行改成 --num_gpus=2（或 4、8）
# 精度：--fp16 / --bf16 二选一
# 换模型：改 --model-name-or-path（必要时再加 --tokenizer-name-or-path）
################## 按需改行内参数 ##################

deepspeed --num_gpus=1 --module cad_finetune.cli.train \
  --deepspeed configs/deepspeed/zero2.json \
  --config configs/experiments/medical.yaml \
  --model-name-or-path Qwen/Qwen2-7B-Instruct \
  --train-file data/raw/medical_train.json \
  --test-file data/raw/medical_test.json \
  --launcher deepspeed \
  --bf16 \
  --tf32 true \
  --optim adamw_torch \
  --gradient-checkpointing true \
  --dataloader-num-workers 4 \
  --num-train-epochs 5 \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 8 \
  --gradient-accumulation-steps 2 \
  --learning-rate 1e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --lr-scheduler-type cosine \
  --logging-steps 50 \
  --save-strategy epoch \
  --evaluation-strategy epoch \
  --save-total-limit 5 \
  --max-steps -1 \
  --max-length 1024 \
  --input-column input \
  --label-column output \
  --num-labels 2 \
  --load-best-model-at-end true \
  --metric-for-best-model auc \
  --greater-is-better true \
  --report-to wandb \
  --seed 42 \
  "$@"
