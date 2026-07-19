#!/usr/bin/env bash
set -euo pipefail

readonly REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly MODEL_REPOSITORY="HuggingFaceTB/SmolLM2-135M-Instruct"
readonly MODEL_DIRECTORY="$REPOSITORY_ROOT/models/hf/SmolLM2-135M-Instruct"
readonly MODEL_FILES=(
  --include '*.json'
  --include 'merges.txt'
  --include 'vocab.json'
  --include 'model.safetensors'
)

require_hf() {
  if ! command -v hf >/dev/null 2>&1; then
    echo "The Hugging Face CLI is required. Install it with:" >&2
    echo "  python3 -m pip install --user --upgrade huggingface_hub" >&2
    exit 1
  fi
}

require_hf
mkdir -p "$MODEL_DIRECTORY"
echo "Inspecting $MODEL_REPOSITORY download..."
hf download "$MODEL_REPOSITORY" --dry-run "${MODEL_FILES[@]}"
echo "Downloading SafeTensors/tokenizer fixture to $MODEL_DIRECTORY..."
hf download "$MODEL_REPOSITORY" --local-dir "$MODEL_DIRECTORY" "${MODEL_FILES[@]}"
