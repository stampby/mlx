#!/bin/bash

source venv/bin/activate

SEED=42
PROMPT="Write exactly one short friendly greeting."
COMMON_ARGS=(--prompt "$PROMPT" --seed "$SEED" --temp 0 --max-tokens 64)

mlx_lm.generate --model mlx-community/Qwen3-0.6B-bf16 "${COMMON_ARGS[@]}"
mlx_lm.generate --model mlx-community/Qwen3-0.6B-3bit "${COMMON_ARGS[@]}"
mlx_lm.generate --model mlx-community/Qwen3-0.6B-4bit "${COMMON_ARGS[@]}"
mlx_lm.generate --model mlx-community/Qwen3-0.6B-8bit "${COMMON_ARGS[@]}"
#mlx_lm.generate --model mlx-community/Qwen3-Coder-Next-4bit "${COMMON_ARGS[@]}"
