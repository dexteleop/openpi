"""
测试 translate_prompts()：纯英文 prompt 原样返回，含中文的 prompt 走翻译并去掉中文
"""
from openpi.training.lingyu_dataloader_v2.utils._prompt_translator import (
    translate_prompts, _check_contains_chinese)
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger
logger = get_logger(__file__)


def test():
    """ 跑一批 prompt 来测试 translate_prompts() 的翻译功能 """
    sample_prompt_english = "This is a pure English sentence."
    sample_prompt_mixed = "This is a sentence 混搭了一些中文."
    prompts = [sample_prompt_english, sample_prompt_mixed]

    results = translate_prompts(prompts, max_workers=10)
    for result_idx, result in enumerate(results, 1):
        logger.info(f"Result {result_idx}: {result}")

    # 纯英文不请求模型，必须原样返回；顺序与输入一致
    assert results[0] == sample_prompt_english
    # 含中文的必须被翻译掉，结果里不应再有中文字符
    assert not _check_contains_chinese(results[1])


if __name__ == "__main__":
    test()
