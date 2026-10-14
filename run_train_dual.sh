#!/usr/bin/env bash
set -euo pipefail

# Launcher: run tiny LLM training on 2 GPUs using torchrun
# - exposes a --devide mode (default: dual)
# - always adds tags: balance, cuda, cpu
# - forwards any extra args to tinyllm.py

NUM_PROC=2
DEVIDE_MODE="dual"
OUT="tinyllm_2gpu.pt"

# Collect extra args
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --devide)
      if [[ -n "${2-}" && ${2:0:1} != "-" ]]; then
        DEVIDE_MODE="$2"
        shift 2
      else
        echo "Usage: $0 [--devide MODE] [--out FILE] [other tinyllm.py args]"
        exit 1
      fi
      ;;
    --out)
      if [[ -n "${2-}" && ${2:0:1} != "-" ]]; then
        OUT="$2"
        shift 2
      else
        echo "Usage: $0 --out FILE"
        exit 1
      fi
      ;;
    --help|-h)
      echo "Usage: $0 [--devide MODE] [--out FILE] [--] [tinyllm.py args...]"
      echo
      echo "Examples:" 
      echo "  $0 --devide dual --out mymodel.pt --block 8192 --batch 8 --steps 2000"
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do EXTRA_ARGS+=("$1"); shift; done
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# Default tinyllm.py args (sensible example values; override by passing args)
DEFAULT_ARGS=(--block 8192 --batch 8 --grad-accum 2 --steps 2000)

echo "Launching tiny LLM training with torchrun on ${NUM_PROC} GPUs"
echo "devide mode: ${DEVIDE_MODE}"
echo "out: ${OUT}"

torchrun --nproc_per_node=${NUM_PROC} tinyllm.py train \
  --out "${OUT}" \
  "${DEFAULT_ARGS[@]}" \
  --devide "${DEVIDE_MODE}" \
  --tag balance --tag cuda --tag cpu \
  "${EXTRA_ARGS[@]}"
