"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine and simple performance benchmarks.
It is not intended for production use. For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please change `vllm/entrypoints/openai/api_server.py` instead.
"""

import argparse
import copy
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None


# Required by Vertex deployment.
@app.get("/ping")
async def ping() -> Response:
    return Response(status_code=200)


def format_output(prompt: str, output: str):
    output = output.strip("\n")
    return f"Prompt:\n{prompt.strip()}\nOutput:\n{output}"


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    is_on_vertex = "instances" in request_dict
    if is_on_vertex:
        request_dict = request_dict["instances"][0]
    prompt = request_dict.pop("prompt")
    prefix_pos = request_dict.pop("prefix_pos", None)
    stream = request_dict.pop("stream", False)
    raw_response = request_dict.pop("raw_response", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    results_generator = engine.generate(prompt,
                                        sampling_params,
                                        request_id,
                                        prefix_pos=prefix_pos)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        prior_request_output = None
        async for request_output in results_generator:
            text_outputs = []
            for i, output in enumerate(request_output.outputs):
                if prior_request_output is not None:
                    prior_output = prior_request_output.outputs[i]
                    text_output = output.text[len(prior_output.text):]
                else:
                    text_output = output.text
                text_outputs.append(text_output)
            ret = {"predictions": text_outputs}
            if raw_response:
                output_token_counts = []
                for i, output in enumerate(request_output.outputs):
                    if prior_request_output is not None:
                        prior_output = prior_request_output.outputs[i]
                        output_token_count = len(output.token_ids) - len(
                            prior_output.token_ids)
                    else:
                        output_token_count = len(output.token_ids)
                    output_token_counts.append(output_token_count)
                cumulative_logprobs = [
                    output.cumulative_logprob
                    for output in request_output.outputs
                ]
                ret.update({
                    "output_token_counts": output_token_counts,
                    "cumulative_logprobs": cumulative_logprobs
                })
            prior_request_output = copy.deepcopy(request_output)
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    if raw_response:
        text_outputs = [output.text for output in final_output.outputs]
        output_token_counts = [
            len(output.token_ids) for output in final_output.outputs
        ]
        cumulative_logprobs = [
            output.cumulative_logprob for output in final_output.outputs
        ]
        ret = {
            "predictions": text_outputs,
            "output_token_counts": output_token_counts,
            "cumulative_logprobs": cumulative_logprobs
        }
    else:
        prompt = final_output.prompt
        text_outputs = [
            format_output(prompt, output.text)
            for output in final_output.outputs
        ]
        ret = {"predictions": text_outputs}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)
