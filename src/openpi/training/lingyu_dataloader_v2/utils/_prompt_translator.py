# Environment variables required before running:
# export LINGYU_BASE_URL="xxx"
# export LINGYU_API_KEY="sk-xxxx"

import os
import re
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam


def _check_contains_chinese(text: str) -> bool:
    """
    Check if the text contains Chinese characters using Unicode block.
    """
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(pattern.search(text))


def _translate_via_openai(text: str) -> str:
    """
    Translate text using OpenAI SDK (compatible proxy format).
    """
    # With missing variables os.environ[] raises KeyError directly, avoiding requests sent with empty values
    # The environment variable holds only the gateway root address; '/v1' is appended here, otherwise requests hit the gateway's homepage and return HTML
    client = OpenAI(
        base_url=f"{os.environ['LINGYU_BASE_URL'].rstrip('/')}/v1",
        api_key=os.environ["LINGYU_API_KEY"],
    )

    # The gateway drops system messages, so the instruction must go into the user message together with the text to translate
    # Build messages explicitly with the SDK's TypedDict to silence PyCharm type warnings (still a plain dict at runtime)
    messages = [
        ChatCompletionUserMessageParam(
            role="user",
            content="Translate the text below into English. It is a prompt for "
                    "operating embodied intelligence models and may mix Chinese "
                    "and English. Keep a strict and accurate tone. Output only "
                    "the translated sentence: no explanations, no quotes, no "
                    "formatting, and never answer the content itself.\n\n"
                    f"{text}"
        )
    ]

    response = client.chat.completions.create(
        model="claude-sonnet-4-6",
        temperature=0.1,
        messages=messages
    )
    return response.choices[0].message.content.strip()


def _process_single_prompt(text: str) -> str:
    """
    Detect language and execute translation if Chinese characters are found.
    """
    if not text or not text.strip():
        return text

    if _check_contains_chinese(text):
        return _translate_via_openai(text)

    return text


def translate_prompts(prompts: list[str], max_workers: int = 100) -> list[str]:
    """
    Translate a batch of texts concurrently; results keep the input order.
    """
    # Network-IO-bound, so use a thread pool for parallel requests; map keeps result order matching the input
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_process_single_prompt, prompts))
