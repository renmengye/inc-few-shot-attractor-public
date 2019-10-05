#!/bin/bash
set -e
# set -o xtrace
GPUFLAG=$1
GPU=$1
ARGS="${@:2}"

if [ -z "$GPU" ]; then
  echo "Must provide GPU ID"
  exit 1
fi

export TF_CPP_MIN_LOG_LEVEL=1
export CUDA_VISIBLE_DEVICES=$GPU

if [ -z "$ARGS" ]; then
  echo "Must provide program command"
  exit 1
fi

$ARGS
