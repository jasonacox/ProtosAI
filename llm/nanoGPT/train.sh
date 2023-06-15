#!/bin/bash

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dataset>"
  exit 1
fi

echo "Running training on ${1} dataset..."

time python3 train.py \
  --dataset=$1 \
  --n_layer=4 \
  --n_head=4 \
  --n_embd=64 \
  --compile=False \
  --eval_iters=1 \
  --block_size=64 \
  --batch_size=8 \
  --device=cpu \
  $2 $3 $4 $5 $6

echo "Done."