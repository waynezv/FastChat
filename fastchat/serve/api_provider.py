"""Call API providers."""

import json
import os
import pdb
import random
import time
from json import loads
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
    from vertexai.preview.generative_models import GenerativeModel
    from vertexai.preview.language_models import ChatModel

    project_id = os.environ["GCP_PROJECT_ID"]
    location = "us-central1"
    vertexai.init(project=project_id, location=location)

    if model_name in ["palm-2"]:
        # According to release note, "chat-bison@001" is PaLM 2 for chat.
        # https://cloud.google.com/vertex-ai/docs/release-notes#May_10_2023
        model_name = "chat-bison@001"
        chat_model = ChatModel.from_pretrained(model_name)
        chat = chat_model.start_chat(examples=[])
    elif model_name in ["gemini-pro"]:
        model = GenerativeModel(model_name)
        chat = model.start_chat()
    return chat


def palm_api_stream_iter(model_name, chat, message, temperature, top_p, max_new_tokens):
    if model_name in ["gemini-pro"]:
        max_new_tokens = max_new_tokens * 2
    parameters = {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_new_tokens,
    }
    gen_params = {
        "model": model_name,
        "prompt": message,
    }
    gen_params.update(parameters)
    if model_name == "palm-2":
        response = chat.send_message(message, **parameters)
    else:
        response = chat.send_message(message, generation_config=parameters, stream=True)

    logger.info(f"==== request ====\n{gen_params}")

    try:
        text = ""
        for chunk in response:
            text += chunk.text
            data = {
                "text": text,
                "error_code": 0,
            }
            yield data
    except Exception as e:
        logger.error(f"==== error ====\n{e}")
        yield {
            "text": f"**API REQUEST ERROR** Reason: {e}\nPlease try again or increase the number of max tokens.",
            "error_code": 1,
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

    together.api_key = api_key or os.environ.get("TOGETHER_API_KEY")

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


def ai2_api_stream_iter(
    model_name,
    messages,
    temperature,
    top_p,
    max_new_tokens,
    api_key=None,
    api_base=None,
):
    from requests import post

    # get keys and needed values
    ai2_key = api_key or os.environ.get("AI2_API_KEY")
    api_base = api_base or "https://inferd.allen.ai/api/v1/infer"
    model_id = "mod_01hhgcga70c91402r9ssyxekan"

    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    logger.info(f"==== request ====\n{gen_params}")

    # AI2 uses vLLM, which requires that `top_p` be 1.0 for greedy sampling:
    # https://github.com/vllm-project/vllm/blob/v0.1.7/vllm/sampling_params.py#L156-L157
    if temperature == 0.0 and top_p < 1.0:
        raise ValueError("top_p must be 1 when temperature is 0.0")

    res = post(
        api_base,
        stream=True,
        headers={"Authorization": f"Bearer {ai2_key}"},
        json={
            "model_id": model_id,
            # This input format is specific to the Tulu2 model. Other models
            # may require different input formats. See the model's schema
            # documentation on InferD for more information.
            "input": {
                "messages": messages,
                "opts": {
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "logprobs": 1,  # increase for more choices
                },
            },
        },
    )

    if res.status_code != 200:
        logger.error(f"unexpected response ({res.status_code}): {res.text}")
        raise ValueError("unexpected response from InferD", res)

    text = ""
    for line in res.iter_lines():
        if line:
            part = loads(line)
            if "result" in part and "output" in part["result"]:
                for t in part["result"]["output"]["text"]:
                    text += t
            else:
                logger.error(f"unexpected part: {part}")
                raise ValueError("empty result in InferD response")

            data = {
                "text": text,
                "error_code": 0,
            }
            yield data
