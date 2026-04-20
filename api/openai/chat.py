import os
import re
import time
import json
import logging
import asyncio
from joblib import Memory
from typing import Literal, List, Dict
from .chat_parallel import process_api_requests_from_file

# Retry function
MAX_RETRIES = 4


# Constants
CACHE_DIR = '.cache'
RATE_LIMIT = {
    "tier1": {
        'gpt-4o-mini': {'PRM': 2_000, "TPM": 2_000_000},
        'gpt-4o': {'PRM': 2_000, "TPM": 500_000},
        'gpt-4.1-mini': {'PRM': 2_000, "TPM": 2_000_000},
    },
    "tier2": {
        'gpt-4o-mini': {'PRM': 5_000, "TPM": 5_000_000},
        'gpt-4o': {'PRM': 5_000, "TPM": 1_000_000},
        'gpt-4.1-mini': {'PRM': 5_000, "TPM": 5_000_000},
    },
    "tier3": {
        'gpt-4o-mini': {'PRM': 8_000, "TPM": 8_000_000},
        'gpt-4o': {'PRM': 8_000, "TPM": 1_500_000},
        'gpt-4.1-mini': {'PRM': 8_000, "TPM": 8_000_000},
    },
    "tier4": {
        'gpt-4o-mini': {'PRM': 10_000, "TPM": 10_000_000},
        'gpt-4o': {'PRM': 10_000, "TPM": 2_000_000},
        'gpt-4.1-mini': {'PRM': 10_000, "TPM": 10_000_000},
    },
    "tier5": {
        'gpt-4o-mini': {'PRM': 30_000, "TPM": 150_000_000},
        'gpt-4o': {'PRM': 50_000, "TPM": 150_000_000},
        'gpt-4.1-mini': {'PRM': 30_000, "TPM": 150_000_000},
    },
}
VALID_MODELS = {'gpt-4o', 'gpt-4o-mini', 'gpt-4.1-mini'}
VALID_TIERS = {'tier1', 'tier2', 'tier3', 'tier4', 'tier5'}

# OpenAI account limits for a model can be far below the internal RATE_LIMIT tiers.
# Schedule traffic using min(tier table, published cap). Override caps with env:
#   OPENAI_MAX_RPM, OPENAI_MAX_TPM (integers, applied as an additional min).
# Values below match typical dashboard limits for gpt-4.1-mini (Usage tier 1+).
_OPENAI_PUBLISHED_RPM_TPM = {
    'gpt-4.1-mini': (500, 200_000),
}

# Texas A&M / TAMUS AI Chat (native HTTP API, OpenAI-style JSON).
#   TAMUS_AI_CHAT_API_KEY or TAMU_CHAT_API_KEY — Bearer token from the portal.
#   TAMUS_AI_CHAT_API_ENDPOINT or TAMU_CHAT_BASE_URL — default https://chat-api.tamu.ai
#       → chat URL becomes {base}/api/chat/completions (official). If base ends with /openai, uses …/openai/v1/chat/completions (legacy).
#   TAMU_CHAT_REQUEST_URL — optional full POST URL override.
#   TAMU_GPT_4_1_MINI_MODEL — optional; when set, used instead of protected.gpt-4.1-mini for logical model gpt-4.1-mini on TAMU.
# Quota / fairness: TAMU_MAX_RPM, TAMU_MAX_TPM, TAMU_PAUSE_AFTER_THROTTLE_SEC, CHAT_API_TCP_CONNECTOR_LIMIT; half_usage on llm-topic.py.
_TAMU_DEFAULT_BASE = 'https://chat-api.tamu.ai'
# When using TAMU, map OpenAI-style names from this repo to portal model ids (override with --gpt_model).
# If `protected.gpt-4.1-mini` is not listed on your tenant, set TAMU_GPT_4_1_MINI_MODEL to the exact id from GET /api/models.
_TAMU_MODEL_ALIASES = {
    'gpt-4o-mini': 'protected.gpt-4o',
    'gpt-4o': 'protected.gpt-4o',
    'gpt-4.1-mini': 'protected.gpt-4.1-mini',
}


def _parse_chat_api_provider() -> str:
    v = (os.environ.get('CHAT_API_PROVIDER') or 'openai').strip().lower()
    if v in ('openai', 'tamu'):
        return v
    raise ValueError(
        f"CHAT_API_PROVIDER must be 'openai' or 'tamu', got {v!r}. "
        "Unset it or set CHAT_API_PROVIDER=openai for the public OpenAI API."
    )


def _tamu_chat_completions_url() -> str:
    explicit = os.environ.get('TAMU_CHAT_REQUEST_URL')
    if explicit and explicit.strip():
        return explicit.strip().rstrip('/')
    base = (
        os.environ.get('TAMUS_AI_CHAT_API_ENDPOINT')
        or os.environ.get('TAMU_CHAT_BASE_URL')
        or _TAMU_DEFAULT_BASE
    ).strip().rstrip('/')
    low = base.lower().rstrip('/')
    if low.endswith('/openai'):
        return f'{base}/v1/chat/completions'
    if re.search(r'/openai/v\d+$', low):
        return f'{base}/chat/completions'
    return f'{base}/api/chat/completions'


def _resolve_api_key(provider: str) -> str:
    if provider == 'tamu':
        key = os.environ.get('TAMUS_AI_CHAT_API_KEY') or os.environ.get('TAMU_CHAT_API_KEY')
        if not key or not str(key).strip():
            raise KeyError(
                'Set TAMUS_AI_CHAT_API_KEY or TAMU_CHAT_API_KEY when CHAT_API_PROVIDER=tamu '
                '(portal key from TAMUS AI Chat / chat.tamu.ai).'
            )
        return str(key).strip()
    key = os.environ.get('OPENAI_API_KEY')
    if not key or not str(key).strip():
        raise KeyError('OPENAI_API_KEY is required when CHAT_API_PROVIDER=openai (or unset).')
    return str(key).strip()


def _resolve_model_for_provider(model_name: str, provider: str) -> str:
    if provider != 'tamu':
        return model_name
    if model_name == 'gpt-4.1-mini':
        override = (os.environ.get('TAMU_GPT_4_1_MINI_MODEL') or '').strip()
        if override:
            return override
    return _TAMU_MODEL_ALIASES.get(model_name, model_name)


def _parse_positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        v = int(str(raw).strip())
    except ValueError:
        return None
    return v if v > 0 else None


def _tamu_pause_after_throttle_seconds() -> float:
    raw = os.environ.get('TAMU_PAUSE_AFTER_THROTTLE_SEC')
    if raw is not None and str(raw).strip():
        try:
            v = float(str(raw).strip())
            return max(5.0, v)
        except ValueError:
            pass
    return 60.0


def _effective_throughput_rpm_tpm(
    model_name: str, tier_list: str, half_usage: bool, provider: str
) -> tuple[int, int]:
    """Return (max_requests_per_minute, max_tokens_per_minute) for the parallel client."""
    if provider == 'tamu':
        # Conservative defaults for flaky shared gateways (502 / empty bodies); raise if stable.
        rpm = _parse_positive_int_env('TAMU_MAX_RPM') or 15
        tpm = _parse_positive_int_env('TAMU_MAX_TPM') or 40_000
        if half_usage:
            rpm = max(1, rpm // 2)
            tpm = max(1, tpm // 2)
        return rpm, tpm

    prm = RATE_LIMIT[tier_list][model_name]['PRM'] // 2 if half_usage else RATE_LIMIT[tier_list][model_name]['PRM']
    tpm = RATE_LIMIT[tier_list][model_name]['TPM'] // 2 if half_usage else RATE_LIMIT[tier_list][model_name]['TPM']
    cap = _OPENAI_PUBLISHED_RPM_TPM.get(model_name)
    if cap is not None:
        prm = min(prm, cap[0])
        tpm = min(tpm, cap[1])
    env_rpm = _parse_positive_int_env('OPENAI_MAX_RPM')
    env_tpm = _parse_positive_int_env('OPENAI_MAX_TPM')
    if env_rpm is not None:
        prm = min(prm, env_rpm)
    if env_tpm is not None:
        tpm = min(tpm, env_tpm)
    return prm, tpm


# Returned in place of a completion when the parallel client only recorded API errors (no `choices` body).
# llm-topic.py checks this to tombstone a paper and continue instead of hanging or re-querying that id forever.
CHAT_FAILED_RESPONSE = '[[__SEM_RANK_LLM_FAILED__]]'

# Utility Functions
def validate_inputs(inputs, model_name: str, tier_list: str, provider: str):
    """Validate model_name and tier_list against allowed values."""
    # make sure the input is a list of str
    if not isinstance(inputs, list) or not all(isinstance(input_text, str) for input_text in inputs):
        raise ValueError("Invalid inputs. Must be a list of strings.")
    if provider == 'tamu':
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Invalid model_name for TAMU: use a non-empty portal model id (e.g. protected.gpt-4o).")
    elif model_name not in VALID_MODELS:
        raise ValueError(f"Invalid model_name: {model_name}. Must be one of {VALID_MODELS}.")
    if tier_list not in VALID_TIERS:
        raise ValueError(f"Invalid tier_list: {tier_list}. Must be one of {VALID_TIERS}.")

def create_request_file(
    inputs, model_name: str, params: Dict, instruction=None, *, resolved_model: str | None = None
) -> str:
    """Generate the request JSONL file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    timestamp = int(time.time() * 1000)
    request_file = f'{CACHE_DIR}/chat_request_{timestamp}.jsonl'

    model = resolved_model if resolved_model is not None else model_name
    if instruction:
        content = [
            {
                'model': model,
                'messages': [{"role": "system", "content": instruction}, {'role': "user", 'content': input_text}],
                'metadata': {"id": idx},
                'stream': False,
                **params,
            }
            for idx, input_text in enumerate(inputs)
        ]
    else:
        content = [
            {
                'model': model,
                'messages': [{'role': "user", 'content': input_text}],
                'metadata': {"id": idx},
                'stream': False,
                **params,
            }
            for idx, input_text in enumerate(inputs)
        ]

    with open(request_file, 'w') as f:
        for instance in content:
            f.write(json.dumps(instance) + '\n')

    return request_file

def _response_row_id(row) -> int:
    if not isinstance(row, (list, tuple)) or len(row) < 3:
        return 0
    meta = row[2]
    if isinstance(meta, dict) and 'id' in meta:
        try:
            return int(meta['id'])
        except (TypeError, ValueError):
            return 0
    return 0


def read_responses(save_file: str) -> List[str]:
    """Read and parse responses from the response JSONL file. Failed requests become CHAT_FAILED_RESPONSE."""
    rows = []
    with open(save_file, 'r') as f:
        for line in f:
            rows.append(json.loads(line))
    rows.sort(key=_response_row_id)
    out: List[str] = []
    for response in rows:
        if not isinstance(response, (list, tuple)) or len(response) < 2:
            out.append(CHAT_FAILED_RESPONSE)
            continue
        body = response[1]
        if isinstance(body, list):
            out.append(CHAT_FAILED_RESPONSE)
            continue
        if not isinstance(body, dict):
            out.append(CHAT_FAILED_RESPONSE)
            continue
        if 'error' in body and 'choices' not in body:
            out.append(CHAT_FAILED_RESPONSE)
            continue
        try:
            ch0 = (body.get('choices') or [None])[0] or {}
            msg = (ch0.get('message') or {}) if isinstance(ch0, dict) else {}
            content = msg.get('content')
        except (KeyError, IndexError, TypeError, AttributeError):
            out.append(CHAT_FAILED_RESPONSE)
            continue
        if content is None:
            out.append('')
        else:
            out.append(str(content))
    return out

memory = Memory(CACHE_DIR, verbose=0)

# Main Chat Function
@memory.cache
def chat(
    inputs,
    instruction=None,
    half_usage=False,
    clear_cache=False,
    model_name: str = 'gpt-4o-mini',
    tier_list: Literal['tier1', 'tier2', 'tier3', 'tier4', 'tier5'] = 'tier1',
    api_provider: Literal['openai', 'tamu'] | None = None,
    **params
) -> List[str]:
    """Main chat function with runtime validation and processing."""
    provider = api_provider if api_provider is not None else _parse_chat_api_provider()
    validate_inputs(inputs, model_name, tier_list, provider)
    if instruction and not isinstance(instruction, str):
        raise ValueError("Invalid instruction. Must be a string.")

    resolved_model = _resolve_model_for_provider(model_name, provider)

    # File paths
    request_file = create_request_file(
        inputs, model_name, params, instruction, resolved_model=resolved_model
    )
    timestamp = int(time.time() * 1000)
    save_file = f'{CACHE_DIR}/chat_response_{timestamp}.jsonl'

    request_url = (
        _tamu_chat_completions_url() if provider == 'tamu' else 'https://api.openai.com/v1/chat/completions'
    )
    api_key = _resolve_api_key(provider)

    eff_rpm, eff_tpm = _effective_throughput_rpm_tpm(model_name, tier_list, half_usage, provider)
    logging.info(
        "Chat API schedule: provider=%s request_model=%s (logical=%s) tier=%s -> RPM=%s TPM=%s url=%s",
        provider,
        resolved_model,
        model_name,
        tier_list,
        eff_rpm,
        eff_tpm,
        request_url,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Process API requests
            throttle_pause = _tamu_pause_after_throttle_seconds() if provider == 'tamu' else None
            asyncio.run(
                process_api_requests_from_file(
                    requests_filepath=request_file,
                    save_filepath=save_file,
                    request_url=request_url,
                    api_key=api_key,
                    max_requests_per_minute=eff_rpm,
                    max_tokens_per_minute=eff_tpm,
                    token_encoding_name='o200k_base',
                    max_attempts=5,
                    logging_level=logging.INFO,
                    pause_after_throttle_seconds=throttle_pause,
                )
            )

            # Extract and return responses
            results = read_responses(save_file)

            # If successful, break out of retry loop
            break

        except Exception as e:
            logging.error(f"Attempt {attempt} failed with error: {e}")

            if attempt == MAX_RETRIES:
                logging.critical("Maximum retry attempts reached. Exiting.")
                raise
            else:
                logging.info(f"Retrying... ({attempt}/{MAX_RETRIES})")
    
    if clear_cache:
        os.remove(request_file)
        os.remove(save_file)
    return results

# Entry Point
if __name__ == '__main__':
    try:
        responses = chat(['Who is your daddy?', 'What is the meaning of life?'])
        for response in responses:
            print(response)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
