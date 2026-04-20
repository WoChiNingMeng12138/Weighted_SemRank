#!/usr/bin/env bash
# Run LitSearch LLM topic labeling (llm-topic.py).
#
# Usage:
#   OpenAI (default):
#     export OPENAI_API_KEY='sk-...'
#     ./run_llm_topic.sh openai
#
#   TAMU / TAMUS AI Chat (recommended: conservative quotas):
#     export TAMUS_AI_CHAT_API_KEY='sk-...'   # or TAMU_CHAT_API_KEY
#     export TAMUS_AI_CHAT_API_ENDPOINT='https://chat-api.tamu.ai'   # optional; this is the default
#     ./run_llm_topic.sh tamu
#
# Optional first arg can be omitted if CHAT_API_PROVIDER is already set (openai|tamu).
# Any extra arguments are passed through to llm-topic.py, e.g.:
#     ./run_llm_topic.sh tamu -- --gpt_model protected.gpt-4.1 --dataset csfcube
#     ./run_llm_topic.sh tamu -- --dataset dorismae --data_dir ./DORISMAE
#
# TAMU defaults below are conservative (502 / empty-body mitigation). Override any time:
#   export TAMU_MAX_RPM=30 TAMU_MAX_TPM=80000 CHAT_API_TCP_CONNECTOR_LIMIT=6
# Disable default --half_usage for TAMU (not recommended when the gateway is flaky):
#   LLM_TOPIC_NO_HALF=1 ./run_llm_topic.sh tamu

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PROVIDER="${CHAT_API_PROVIDER:-openai}"
if [[ "${1:-}" == "tamu" || "${1:-}" == "openai" ]]; then
  PROVIDER="$1"
  shift
fi

if [[ "$PROVIDER" == "tamu" ]]; then
  export CHAT_API_PROVIDER=tamu
  if [[ -z "${TAMU_CHAT_API_KEY:-}" && -z "${TAMUS_AI_CHAT_API_KEY:-}" ]]; then
    echo "error: set TAMUS_AI_CHAT_API_KEY or TAMU_CHAT_API_KEY (TAMUS AI Chat portal)" >&2
    exit 1
  fi
  export TAMU_MAX_RPM="${TAMU_MAX_RPM:-15}"
  export TAMU_MAX_TPM="${TAMU_MAX_TPM:-40000}"
  export TAMU_PAUSE_AFTER_THROTTLE_SEC="${TAMU_PAUSE_AFTER_THROTTLE_SEC:-60}"
  export CHAT_API_TCP_CONNECTOR_LIMIT="${CHAT_API_TCP_CONNECTOR_LIMIT:-4}"
  EXTRA=(--provider tamu)
  # --half_usage halves RPM/TPM again in the client (e.g. 15→7 RPM, 40k→20k TPM).
  if [[ -z "${LLM_TOPIC_NO_HALF:-}" ]]; then
    EXTRA+=(--half_usage)
  fi
else
  export CHAT_API_PROVIDER=openai
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "error: set OPENAI_API_KEY" >&2
    exit 1
  fi
  EXTRA=(--provider openai)
fi

exec python llm-topic.py "${EXTRA[@]}" "$@"
