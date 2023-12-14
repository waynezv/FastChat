"""Call API providers."""

import json
import os
import pdb
import random
import time
from typing import Optional

import requests
from fastchat.constants import WORKER_API_TIMEOUT
from fastchat.utils import build_logger

logger = build_logger("gradio_web_server", "gradio_web_server.log")


def openai_api_stream_iter(
    model_name,
    messages,
    temperature,
    top_p,
    max_new_tokens,
    api_base=None,
    api_key=None,
):
    import openai

    is_azure = False
    if "azure" in model_name:
        is_azure = True
        openai.api_type = "azure"
        openai.api_version = "2023-07-01-preview"
    else:
        openai.api_type = "open_ai"
        openai.api_version = None

    openai.api_base = api_base or "https://api.openai.com/v1"
    openai.api_key = api_key or os.environ["OPENAI_API_KEY"]
    if model_name == "gpt-4-turbo":
        model_name = "gpt-4-1106-preview"

    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    logger.info(f"==== request ====\n{gen_params}")

    if is_azure:
        res = openai.ChatCompletion.create(
            engine=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            stream=True,
        )
    else:
        res = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            stream=True,
        )
    text = ""
    for chunk in res:
        if len(chunk["choices"]) > 0:
            text += chunk["choices"][0]["delta"].get("content", "")
            data = {
                "text": text,
                "error_code": 0,
            }
            yield data


def anthropic_api_stream_iter(model_name, prompt, temperature, top_p, max_new_tokens):
    import anthropic

    c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    logger.info(f"==== request ====\n{gen_params}")

    res = c.completions.create(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        max_tokens_to_sample=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        model=model_name,
        stream=True,
    )
    text = ""
    for chunk in res:
        text += chunk.completion
        data = {
            "text": text,
            "error_code": 0,
        }
        yield data


def init_palm_chat(model_name):
    import vertexai  # pip3 install google-cloud-aiplatform
    from vertexai.preview.language_models import ChatModel

    project_id = os.environ["GCP_PROJECT_ID"]
    location = "us-central1"
    vertexai.init(project=project_id, location=location)

    chat_model = ChatModel.from_pretrained(model_name)
    chat = chat_model.start_chat(examples=[])
    return chat


def palm_api_stream_iter(chat, message, temperature, top_p, max_new_tokens):
    parameters = {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_new_tokens,
    }
    gen_params = {
        "model": "palm-2",
        "prompt": message,
    }
    gen_params.update(parameters)
    logger.info(f"==== request ====\n{gen_params}")

    response = chat.send_message(message, **parameters)
    content = response.text

    pos = 0
    while pos < len(content):
        # This is a fancy way to simulate token generation latency combined
        # with a Poisson process.
        pos += random.randint(10, 20)
        time.sleep(random.expovariate(50))
        data = {
            "text": content[:pos],
            "error_code": 0,
        }
        yield data


def together_api_stream_iter(
    model_name: str,
    prompt: str,
    max_tokens: Optional[int] = 128,
    stop: Optional[str] = None,
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = 0.7,
    top_k: Optional[int] = 50,
    repetition_penalty: Optional[float] = None,
    raw: Optional[bool] = False,
    api_key=None,
):
    import together

    # together.api_key = api_key or os.environ["TOGETHER_API_KEY"]  # TODO
    together.api_key = "e53cfc5a8d019c6b6dead75791c8bc43dda284530d33d2bfbd7054b6ac7916e3"

    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stop": stop,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "raw": raw,
    }
    logger.info(f"==== request ====\n{gen_params}")

    res = together.Complete.create_streaming(**gen_params)

    text = ""
    for chunk in res:
        text += chunk
        data = {
            "text": text,
            "error_code": 0,
        }
        yield data


def custom_aws_api_stream_iter(
    model_name: str,
    api_url: str,
    prompt: str,
    conv=None,
    max_new_tokens: Optional[int] = 128,
    stop: Optional[str] = None,
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = 0.7,
    top_k: Optional[int] = 50,
    repetition_penalty: Optional[float] = None,
    raw: Optional[bool] = False,
    api_key=None,
):
    # Make requests
    # gen_params = {
    #     "model": model_name,
    #     "prompt": prompt,
    #     "temperature": temperature,
    #     "repetition_penalty": repetition_penalty,
    #     "top_p": top_p,
    #     "max_new_tokens": max_new_tokens,
    #     "stop": conv.stop_str,
    #     "stop_token_ids": conv.stop_token_ids,
    #     "echo": False,
    # }
    # logger.info(f"==== request ====\n{gen_params}")
    logger.info(f"==== request ====")

    # Stream output
    input = {"inputs": prompt}
    payload = json.dumps(input)

    response = requests.post(api_url, data=payload, headers={"content_type": "application/json"})
    # pdb.set_trace()
    # response = requests.post(
    #     worker_addr + "/worker_generate_stream",
    #     headers=headers,
    #     json=gen_params,
    #     stream=True,
    #     timeout=WORKER_API_TIMEOUT,
    # )
    # for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
    # if chunk:
    # data = json.loads(chunk.decode())
    # yield data

    gen_text = json.loads(response.text)[0]["generated_text"]
    gen_text = gen_text.split("[/INST]")[-1].strip()
    text = ""
    for chunk in gen_text:
        text += chunk
        data = {
            "text": text,
            "error_code": 0,
        }
        yield data
