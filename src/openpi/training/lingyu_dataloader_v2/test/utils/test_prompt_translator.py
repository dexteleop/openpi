"""
Test translate_prompts(): pure-English prompts return as-is; prompts containing Chinese go through translation and lose the Chinese
"""
from openpi.training.lingyu_dataloader_v2.utils._prompt_translator import (
    translate_prompts, _check_contains_chinese)
from openpi.training.lingyu_dataloader_v2.test.logger import get_logger
logger = get_logger(__file__)


def test():
    """ Run a batch of prompts to test translate_prompts()'s translation functionality """
    sample_prompt_english = "This is a pure English sentence."
    sample_prompt_mixed = "This is a sentence 混搭了一些中文."
    prompts = [sample_prompt_english, sample_prompt_mixed]

    results = translate_prompts(prompts, max_workers=10)
    for result_idx, result in enumerate(results, 1):
        logger.info(f"Result {result_idx}: {result}")

    # Pure English does not request the model and must return as-is; order matches the input
    assert results[0] == sample_prompt_english
    # Chinese-containing ones must be translated away; the result should contain no Chinese characters
    assert not _check_contains_chinese(results[1])


if __name__ == "__main__":
    test()
