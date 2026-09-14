"""
并行地把每个 mcap 交给 MCAPSampleExtractor 抽成一个小 json, 再合并成一个总索引。

小 json 按 episode 分组, 供抽取时整段提交或整段丢弃::
    {episode_idx: {sample_idx: {topic: [msg_type, msg_def, log_time, locations]}}}
总 json 抹去 episode 层级, 只剩跨 mcap 连续重编的 sample::
    {sample_idx: {topic: [msg_type, msg_def, log_time, locations]}}

合并按行流式进行: 抽取器把每个 episode 单独写成一行, 这里一次只摊平一行,
因此内存里最多只驻留一个 episode, 不会把几 GB 的 json 整份读进来。
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from openpi.training.lingyu_dataloader_v2.utils.search_mcap_paths import find_mcap_paths
from openpi.training.lingyu_dataloader_v2.utils.mcap_sample_extractor import MCAPSampleExtractor

MAX_WORKERS     = 10            # 最多有10个rosbag在并行转码
_OUTPUT_DIR     = os.path.dirname(os.path.abspath(__file__))
_MERGED_JSON    = os.path.join(_OUTPUT_DIR, "train_samples_idx.json")

# 小 json 中一行一个 episode, 形如 "12": {...}
_EPISODE_LINE = re.compile(r'^"(\d+)": ')

mcap_urls = find_mcap_paths()


def build_samples_for_one_mcap(mcap_url: str) -> str:
    """对一个mcap找到多个episode中的所有可用sample，写入json，返回json路径"""
    extractor = MCAPSampleExtractor(mcap_url, _OUTPUT_DIR)
    try:
        extractor.convert_single_mcap()
    finally:
        extractor.close()
    print(f"done: {mcap_url} -> {extractor.output_json_path} "
          f"({extractor.num_episodes} episodes, {extractor.total_samples} samples)")
    return str(extractor.output_json_path)


def _merge_sample_jsons(json_paths: list[str]) -> None:
    """
    合并所有小json到 train_samples_idx.json, 抹去 episode 层级, sample_idx 跨文件连续重编。
    合并完成后删除小json文件。
    """
    merged_sample_idx = 0

    with open(_MERGED_JSON, "w", encoding="utf-8") as merged_file:
        merged_file.write("{\n")
        for path in sorted(json_paths):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    episode_line = _EPISODE_LINE.match(line)
                    if not episode_line:
                        continue    # 跳过小 json 首尾的 "{" 与 "}"
                    # 一行一个 episode, 摊平成 sample 后逐个重编号
                    episode_samples = json.loads(line[episode_line.end():].rstrip().rstrip(","))
                    for sample_idx in sorted(episode_samples, key=int):
                        separator = ",\n" if merged_sample_idx else ""
                        merged_file.write(f'{separator}"{merged_sample_idx}": ')
                        json.dump(episode_samples[sample_idx], merged_file)
                        merged_sample_idx += 1
        merged_file.write("\n}\n")

    print(f"merged {merged_sample_idx} samples into {_MERGED_JSON}")

    for path in json_paths:
        os.remove(path)


def build_all():
    """并行处理所有mcap（最多MAX_WORKERS线程），最后合并为一个总索引文件"""
    json_paths = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(build_samples_for_one_mcap, url): url
                         for url in mcap_urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                json_paths.append(future.result())
            except Exception as exc:
                print(f"error: {url}: {exc}")

    _merge_sample_jsons(json_paths)


if __name__ == "__main__":
    build_all()
