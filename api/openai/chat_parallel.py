# imports
import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata


def _api_error_text(error_field) -> str:
    """Normalize API error payload to a single lowercase string for matching."""
    if isinstance(error_field, dict):
        parts = [
            str(error_field.get(k) or '')
            for k in ('message', 'type', 'code', 'param')
        ]
        return ' '.join(parts).lower()
    return str(error_field).lower()


def _merge_sse_chat_completion_chunks(raw: str) -> dict | None:
    """If the server returned an SSE stream of chat.completion.chunk, merge into one chat.completion dict."""
    if 'chat.completion.chunk' not in raw:
        return None
    dec = json.JSONDecoder()
    content_parts: list[str] = []
    last_id, last_model = '', ''
    idx = 0
    while True:
        i = raw.find('data:', idx)
        if i < 0:
            break
        j = i + 5
        while j < len(raw) and raw[j] in ' \t\r\n':
            j += 1
        if j >= len(raw):
            break
        rest = raw[j : j + 12].lstrip()
        if rest.startswith('[DONE]'):
            break
        try:
            obj, end = dec.raw_decode(raw, j)
        except json.JSONDecodeError:
            idx = i + 5
            continue
        idx = end
        if not isinstance(obj, dict) or obj.get('object') != 'chat.completion.chunk':
            continue
        last_id = str(obj.get('id') or last_id)
        last_model = str(obj.get('model') or last_model)
        for ch in obj.get('choices') or []:
            if isinstance(ch, dict):
                delta = ch.get('delta') or {}
                if isinstance(delta, dict) and delta.get('content'):
                    content_parts.append(str(delta['content']))
    if not content_parts:
        return None
    return {
        'id': last_id or 'sse-merged',
        'object': 'chat.completion',
        'model': last_model,
        'choices': [
            {
                'index': 0,
                'message': {'role': 'assistant', 'content': ''.join(content_parts)},
                'finish_reason': 'stop',
            }
        ],
    }


def _non_json_body_summary(http_status: int, text: str) -> str:
    """One-line description for HTML / non-JSON gateway responses (avoid multi-KB logs and retry payloads)."""
    t = text.strip()
    title_m = re.search(r'<title>([^<]+)</title>', t, re.IGNORECASE | re.DOTALL)
    title = title_m.group(1).strip()[:220] if title_m else ''
    if title:
        return f'HTTP {http_status}: non-JSON response ({len(text)} bytes); title: {title}'
    head = t.replace('\n', ' ')[:220]
    return f'HTTP {http_status}: non-JSON response ({len(text)} bytes): {head}'


def _error_for_terminal_log(err_field, max_msg_chars: int = 360) -> dict | str:
    """Shrink huge HTML gateway bodies in WARNING logs; full error dict is still queued for retries."""
    if not isinstance(err_field, dict):
        s = str(err_field)
        return s if len(s) <= max_msg_chars else f'{s[:max_msg_chars]}... [{len(s)} chars]'
    out = {k: v for k, v in err_field.items() if k != 'message'}
    m = err_field.get('message', '')
    if not isinstance(m, str):
        out['message'] = m
        return out
    if len(m) <= max_msg_chars:
        out['message'] = m
    else:
        out['message'] = f'{m[:max_msg_chars]}... [truncated {len(m)} chars for log]'
    return out


def _make_tcp_connector(request_url: str, tcp_connector_limit: int | None) -> aiohttp.TCPConnector | None:
    """Cap simultaneous connections to the same host (reduces gateway stampedes on TAMU).

    aiohttp requires a running event loop when constructing TCPConnector; call this from
    ``process_api_requests_from_file`` (async), not from synchronous test code.
    """
    lim = tcp_connector_limit
    if lim is None:
        raw = os.environ.get('CHAT_API_TCP_CONNECTOR_LIMIT', '').strip()
        if raw.isdigit():
            lim = max(1, int(raw))
    if lim is None and 'tamu.ai' in request_url.lower():
        lim = 4
    if lim is None:
        return None
    return aiohttp.TCPConnector(limit=max(1, int(lim)))


def _infer_throttle_http_status(http_status: int, response_body) -> int:
    """Use real status when possible; Cloudflare often returns 200 + HTML 5xx pages."""
    if http_status >= 500 or http_status == 429:
        return http_status
    if not isinstance(response_body, dict) or 'error' not in response_body:
        return http_status
    msg = _api_error_text(response_body['error'])
    if 'error code 502' in msg or 'bad gateway' in msg:
        return 502
    if 'error code 503' in msg or 'service unavailable' in msg:
        return 503
    if 'error code 504' in msg or 'gateway timeout' in msg:
        return 504
    return http_status


def _response_indicates_throttle(http_status: int | None, response_body) -> bool:
    """True when we should back off like a rate limit (includes quota / capacity messages)."""
    if isinstance(response_body, dict):
        err = response_body.get('error')
        if isinstance(err, dict):
            if err.get('type') == 'empty_body':
                return True
            if err.get('type') == 'parse_error':
                pm = _api_error_text(err)
                if '<html' in pm or 'doctype' in pm or 'cloudflare' in pm:
                    return True
    if http_status is not None and http_status >= 500:
        return True
    if http_status == 429:
        return True
    if not isinstance(response_body, dict) or 'error' not in response_body:
        return False
    msg = _api_error_text(response_body['error'])
    keywords = (
        'rate limit',
        'ratelimit',
        'quota',
        'throttl',
        'too many requests',
        'capacity',
        'resource_exhausted',
        'try again later',
        'overloaded',
        'limit exceeded',
        'exceeded your',
        'temporarily unavailable',
        'service unavailable',
        'bad gateway',
        'gateway timeout',
        'cloudflare',
        'error code 502',
        'error code 503',
        'error code 504',
        'empty or null',
        'null json',
        'json null',
        'empty response body',
    )
    return any(k in msg for k in keywords)


async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
    pause_after_throttle_seconds: float | None = None,
    tcp_connector_limit: int | None = None,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = (
        15.0 if pause_after_throttle_seconds is None else max(5.0, float(pause_after_throttle_seconds))
    )
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}
    # use api-key header for Azure deployments
    if "/deployments" in request_url:
        request_header = {"api-key": f"{api_key}"}

    connector = _make_tcp_connector(request_url, tcp_connector_limit)
    session_kwargs = {"connector": connector} if connector is not None else {}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # generates integer IDs of 0, 1, 2, ...
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # initialize file reading
    try:
        with open(requests_filepath) as file:
            # `requests` will provide requests one at a time
            requests = file.__iter__()
            logging.debug(f"File opened. Entering main loop")
            async with aiohttp.ClientSession(**session_kwargs) as session:  # Initialize ClientSession here
                while True:
                    # get next request (if one is not already waiting for capacity)
                    if next_request is None:
                        if not queue_of_requests_to_retry.empty():
                            next_request = queue_of_requests_to_retry.get_nowait()
                            logging.debug(
                                f"Retrying request {next_request.task_id}: {next_request}"
                            )
                        elif file_not_finished:
                            try:
                                # get new request
                                request_json = json.loads(next(requests))
                                next_request = APIRequest(
                                    task_id=next(task_id_generator),
                                    request_json=request_json,
                                    token_consumption=num_tokens_consumed_from_request(
                                        request_json, api_endpoint, token_encoding_name
                                    ),
                                    attempts_left=max_attempts,
                                    metadata=request_json.pop("metadata", None),
                                )
                                status_tracker.num_tasks_started += 1
                                status_tracker.num_tasks_in_progress += 1
                                logging.debug(
                                    f"Reading request {next_request.task_id}: {next_request}"
                                )
                            except StopIteration:
                                # if file runs out, set flag to stop reading it
                                logging.debug("Read file exhausted")
                                file_not_finished = False

                    # update available capacity
                    current_time = time.time()
                    seconds_since_update = current_time - last_update_time
                    available_request_capacity = min(
                        available_request_capacity
                        + max_requests_per_minute * seconds_since_update / 60.0,
                        max_requests_per_minute,
                    )
                    available_token_capacity = min(
                        available_token_capacity
                        + max_tokens_per_minute * seconds_since_update / 60.0,
                        max_tokens_per_minute,
                    )
                    last_update_time = current_time

                    # if enough capacity available, call API
                    if next_request:
                        next_request_tokens = next_request.token_consumption
                        if (
                            available_request_capacity >= 1
                            and available_token_capacity >= next_request_tokens
                        ):
                            # update counters
                            available_request_capacity -= 1
                            available_token_capacity -= next_request_tokens
                            next_request.attempts_left -= 1

                            # call API
                            asyncio.create_task(
                                next_request.call_api(
                                    session=session,
                                    request_url=request_url,
                                    request_header=request_header,
                                    retry_queue=queue_of_requests_to_retry,
                                    save_filepath=save_filepath,
                                    status_tracker=status_tracker,
                                )
                            )
                            next_request = None  # reset next_request to empty

                    # if all tasks are finished, break
                    if status_tracker.num_tasks_in_progress == 0:
                        break

                    # main loop sleeps briefly so concurrent tasks can run
                    await asyncio.sleep(seconds_to_sleep_each_loop)

                    # if a rate limit error was hit recently, pause to cool down
                    seconds_since_rate_limit_error = (
                        time.time() - status_tracker.time_of_last_rate_limit_error
                    )
                    if (
                        seconds_since_rate_limit_error
                        < seconds_to_pause_after_rate_limit_error
                    ):
                        remaining_seconds_to_pause = (
                            seconds_to_pause_after_rate_limit_error
                            - seconds_since_rate_limit_error
                        )
                        cool_deadline = (
                            status_tracker.time_of_last_rate_limit_error
                            + seconds_to_pause_after_rate_limit_error
                        )
                        if remaining_seconds_to_pause >= 0.5 and (
                            status_tracker.last_logged_cool_deadline + 0.5 < cool_deadline
                        ):
                            status_tracker.last_logged_cool_deadline = cool_deadline
                            logging.warning(
                                'Pausing %.1fs to cool down (until %s)',
                                remaining_seconds_to_pause,
                                time.ctime(cool_deadline),
                            )
                        await asyncio.sleep(remaining_seconds_to_pause)

            # after finishing, log final status
            logging.info(
                f"""Parallel processing complete. Results saved to {save_filepath}"""
            )
            if status_tracker.num_tasks_failed > 0:
                logging.warning(
                    f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
                )
            if status_tracker.num_rate_limit_errors > 0:
                logging.warning(
                    f"{status_tracker.num_rate_limit_errors} throttle/quota/rate-limit responses received. "
                    f"Lower max_requests_per_minute / max_tokens_per_minute or increase pause_after_throttle_seconds."
                )
    finally:
        if connector is not None:
            await connector.close()


# dataclasses


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits
    last_logged_cool_deadline: float = -1.0  # suppress duplicate "pausing to cool down" lines


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""
#         logging.info(f"Starting request #{self.task_id}")
        error = None
        response_body = None
        try:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as http_resp:
                http_status = http_resp.status
                raw_bytes = await http_resp.read()
                text = raw_bytes.decode('utf-8', errors='replace') if raw_bytes else ''
                text_stripped = text.strip()
                if not text_stripped:
                    response_body = {
                        'error': {
                            'message': (
                                f'HTTP {http_status}: empty response body '
                                f'(gateway closed connection or returned no content)'
                            ),
                            'type': 'empty_body',
                        }
                    }
                else:
                    merged_sse = _merge_sse_chat_completion_chunks(text)
                    if merged_sse is not None:
                        response_body = merged_sse
                    else:
                        try:
                            parsed = json.loads(text)
                        except json.JSONDecodeError:
                            response_body = {
                                'error': {
                                    'message': _non_json_body_summary(http_status, text),
                                    'type': 'parse_error',
                                }
                            }
                        else:
                            if parsed is None:
                                response_body = {
                                    'error': {
                                        'message': (
                                            f'HTTP {http_status}: JSON `null` body '
                                            f'(upstream/proxy glitch; not a valid chat completion)'
                                        ),
                                        'type': 'empty_body',
                                    }
                                }
                            elif isinstance(parsed, dict):
                                response_body = parsed
                            else:
                                response_body = {
                                    'error': {
                                        'message': (
                                            f'HTTP {http_status}: expected JSON object, got '
                                            f'{type(parsed).__name__}: {repr(parsed)[:1500]}'
                                        ),
                                        'type': 'invalid_body',
                                    }
                                }

            throttle_status = _infer_throttle_http_status(http_status, response_body)
            if 'error' in response_body:
                logging.warning(
                    f"Request {self.task_id} failed with error {_error_for_terminal_log(response_body['error'])}"
                )
                status_tracker.num_api_errors += 1
                error = response_body
                if _response_indicates_throttle(throttle_status, response_body):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1  # rate limit errors are counted separately
                    )
            elif http_status >= 400:
                logging.warning(
                    f"Request {self.task_id} failed with HTTP {http_status}: {repr(response_body)[:500]}"
                )
                status_tracker.num_api_errors += 1
                error = {
                    'error': {
                        'message': f'HTTP {http_status} {repr(response_body)[:800]}',
                        'type': 'http_error',
                    }
                }
                if _response_indicates_throttle(_infer_throttle_http_status(http_status, error), error):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1

        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response_body, self.metadata]
                if self.metadata
                else [self.request_json, response_body]
            )
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")


# functions


def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        # for Azure OpenAI deployment urls
        match = re.search(
            r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url
        )
    if match is None:
        # TAMUS / OpenAI-compatible: https://host/openai/v1/chat/completions
        match = re.search(r"^https://[^/]+/openai/v\d+/(.+)$", request_url)
    if match is None:
        # Native TAMUS: https://host/api/chat/completions
        match = re.search(r"^https://[^/]+/api/(chat/completions)(?:\?|$)", request_url)
    if match is None:
        raise ValueError(f"Unrecognized API URL shape for token counting: {request_url!r}")
    return match[1]


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError(
                    'Expecting either string or list of strings for "prompt" field in completion request'
                )
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request'
            )
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not implemented in this script'
        )


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


# run script


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests_filepath")
    parser.add_argument("--save_filepath", default=None)
    parser.add_argument("--request_url", default="https://api.openai.com/v1/embeddings")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--max_requests_per_minute", type=int, default=3_000 * 0.5)
    parser.add_argument("--max_tokens_per_minute", type=int, default=250_000 * 0.5)
    parser.add_argument("--token_encoding_name", default="cl100k_base")
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--logging_level", default=logging.INFO)
    args = parser.parse_args()

    if args.save_filepath is None:
        args.save_filepath = args.requests_filepath.replace(".jsonl", "_results.jsonl")

    # run script
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=args.requests_filepath,
            save_filepath=args.save_filepath,
            request_url=args.request_url,
            api_key=args.api_key,
            max_requests_per_minute=float(args.max_requests_per_minute),
            max_tokens_per_minute=float(args.max_tokens_per_minute),
            token_encoding_name=args.token_encoding_name,
            max_attempts=int(args.max_attempts),
            logging_level=int(args.logging_level),
        )
    )
