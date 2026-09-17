# 运行前需设置环境变量：
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
    # 缺少环境变量时 os.environ[] 直接抛 KeyError，避免带着空值发请求
    # 环境变量只填网关根地址，这里补上 /v1；否则请求会打到网关首页并返回 HTML
    client = OpenAI(
        base_url=f"{os.environ['LINGYU_BASE_URL'].rstrip('/')}/v1",
        api_key=os.environ["LINGYU_API_KEY"],
    )

    # 网关会丢弃 system 消息，指令必须和待翻译文本一起放进 user 消息
    # 使用 SDK 的 TypedDict 显式构造消息，消除 PyCharm 类型检查警告（运行时仍是普通 dict）
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
    # 网络 IO 密集，用线程池并行请求；map 保证结果顺序与输入一致
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_process_single_prompt, prompts))
