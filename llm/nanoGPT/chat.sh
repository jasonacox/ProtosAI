#!/bin/bash

CKPT=out/ckpt.pt

# If user specified a checkpoint, use that instead
if [[ $# -gt 0 ]]; then
  CKPT=$1
  shift
fi

# Check to see if file exists
if [ ! -f $CKPT ]; then
  echo "Checkpoint file not found: ${CKPT}"
  echo "Usage: $0 <checkpoint> [args]"
  exit 1
fi

ARGS=("$@")
echo "Running chat with checkpoint ${CKPT}..."

# Print all args if there are any
if [[ $# -gt 0 ]]; then
  echo "Args: ${ARGS[@]}"
fi
echo ""

# Add all other args beyond $1
python3 chat.py \
  --compile=False \
  --ckpt=$CKPT \
  --streaming=True \
  $2 $3 $4 $5 $6

echo "Done."
