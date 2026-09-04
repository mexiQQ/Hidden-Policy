#!/usr/bin/env python3
"""Generate a safe, self-contained overview of the Experiment 0 code.

The generator intentionally uses a closed file allowlist.  It never reads
benchmark content, runtime inputs, model outputs, credentials, SSH settings,
or files below the vendored harness.  The harness is inspected only through
Git object identity commands and is presented as one external component.
"""

from __future__ import annotations

import ast
from html import escape
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent
OUTPUT = CODE_ROOT / "code-overview.html"
PUBLISHED_REPORTS = (
    (
        "reports/baseline-results.html",
        "阅读基础测试报告",
        "自包含中文实验结果文件；概述页不读取或复验其内容。",
    ),
    (
        "reports/baseline-results.json",
        "下载机器可读结果",
        "机器可读的结构化结果文件；概述页不读取或复验其内容。",
    ),
)

CONFIG_FILE = "configs/experiment0.json"
SAFE_METADATA_FILES = (
    "manifests/experiment0/metadata.json",
    "manifests/experiment0/wmdp_deduplication.json",
    "manifests/experiment0/mmlu_deduplication.json",
    "manifests/experiment0/pilot32.json",
    "manifests/experiment0/checksums.json",
)
SOURCE_FILES = (
    "src/hidden_policy_eval/__init__.py",
    "src/hidden_policy_eval/__main__.py",
    "src/hidden_policy_eval/cli.py",
    "src/hidden_policy_eval/environment.py",
    "src/hidden_policy_eval/harness.py",
    "src/hidden_policy_eval/io.py",
    "src/hidden_policy_eval/manifests.py",
    "src/hidden_policy_eval/mcq.py",
    "src/hidden_policy_eval/prepare.py",
    "src/hidden_policy_eval/prompts.py",
    "src/hidden_policy_eval/report.py",
    "src/hidden_policy_eval/sources.py",
    "src/hidden_policy_eval/split_pipeline.py",
    "src/hidden_policy_eval/strict.py",
    "src/hidden_policy_eval/vendor.py",
)
TASK_FILES = (
    "tasks/plan4/utils.py",
    "tasks/plan4/plan4_wmdp_ll.yaml",
    "tasks/plan4/plan4_mmlu_ll.yaml",
    "tasks/plan4/plan4_wmdp_strict.yaml",
    "tasks/plan4/plan4_mmlu_strict.yaml",
)
TEST_FILES = (
    "tests/test_environment.py",
    "tests/test_generate_baseline_report.py",
    "tests/test_harness.py",
    "tests/test_manifests.py",
    "tests/test_mcq.py",
    "tests/test_prompts.py",
    "tests/test_report.py",
    "tests/test_run_baseline_matrix.py",
    "tests/test_split_pipeline.py",
    "tests/test_strict.py",
    "tests/test_vendor.py",
)
SCRIPT_FILES = (
    "scripts/install_a6000.sh",
    "scripts/run_baseline_matrix.py",
    "scripts/generate_baseline_report.py",
    "scripts/generate_code_overview.py",
)
READ_ALLOWLIST = frozenset(
    {
        CONFIG_FILE,
        "pyproject.toml",
        "constraints-a6000.txt",
        "scripts/install_a6000.sh",
        *SAFE_METADATA_FILES,
        *SOURCE_FILES,
        *TASK_FILES,
        *TEST_FILES,
        "scripts/run_baseline_matrix.py",
        "scripts/generate_baseline_report.py",
        "scripts/generate_code_overview.py",
    }
)

SOURCE_DESCRIPTIONS = {
    "src/hidden_policy_eval/__init__.py": "定义本项目包及其版本。",
    "src/hidden_policy_eval/__main__.py": "支持 python -m hidden_policy_eval 的最小入口。",
    "src/hidden_policy_eval/cli.py": "统一编排环境检查、切分、准备、运行、后处理和 gate。",
    "src/hidden_policy_eval/environment.py": "核对 Python 包、CUDA、GPU 和 vendored editable install。",
    "src/hidden_policy_eval/harness.py": "构造并执行 lm-eval 命令，记录阶段耗时并保护输出目录。",
    "src/hidden_policy_eval/io.py": "确定性 JSON/JSONL、原子写入和 SHA-256 工具。",
    "src/hidden_policy_eval/manifests.py": "MCQ 规范化、内容寻址、确定性切分和 sealed manifest 校验。",
    "src/hidden_policy_eval/mcq.py": "生成三个确定性选项排列，并维护语义/显示位置映射。",
    "src/hidden_policy_eval/prepare.py": "把 CAL 转换为排列后的 lm-eval 输入，并生成 provenance fingerprint。",
    "src/hidden_policy_eval/prompts.py": "冻结 likelihood 与 strict generation 的共享 prompt 渲染。",
    "src/hidden_policy_eval/report.py": "归一化日志、还原语义选项、汇总指标并执行 PASS/STOP gate。",
    "src/hidden_policy_eval/sources.py": "从冻结 revision 读取 WMDP/MMLU，并提供受控的 HF fallback。",
    "src/hidden_policy_eval/split_pipeline.py": "去重、切分、CAL 物化、pilot 选择和 checksum 的端到端管线。",
    "src/hidden_policy_eval/strict.py": "严格解析单个 A–D 输出，区分 invalid 与 refusal。",
    "src/hidden_policy_eval/vendor.py": "验证 harness 的 URL、版本、commit、tree 和 clean checkout。",
}
TEST_DESCRIPTIONS = {
    "tests/test_environment.py": "editable 安装必须精确指向仓库内 harness。",
    "tests/test_generate_baseline_report.py": "结果发布器的 provenance、计数、backend、一致性与隐私边界。",
    "tests/test_harness.py": "vLLM/HF 命令参数、非 thinking、计时与输出目录保护。",
    "tests/test_manifests.py": "规范化、确定性切分、sealed 内容边界与 manifest round-trip。",
    "tests/test_mcq.py": "排列确定性、唯一性、语义映射与非法输入拒绝。",
    "tests/test_prompts.py": "完整选项文本 likelihood 与单字母 strict prompt。",
    "tests/test_report.py": "token boundary、语义评分、完整三视图和 gate。",
    "tests/test_run_baseline_matrix.py": "GPU 遥测聚合、覆盖信息与缺失采样的 fail-closed 行为。",
    "tests/test_split_pipeline.py": "跨 split 去重、防 CAL 泄漏与标签冲突 fail-closed。",
    "tests/test_strict.py": "valid、invalid、refusal 的解析和计分。",
    "tests/test_vendor.py": "vendored harness 身份和 commit 不匹配拒绝。",
}
TASK_DESCRIPTIONS = {
    "tasks/plan4/utils.py": "lm-eval custom task 的数据加载、prompt 和 strict scorer 适配层。",
    "tasks/plan4/plan4_wmdp_ll.yaml": "WMDP 完整选项文本 likelihood 任务。",
    "tasks/plan4/plan4_mmlu_ll.yaml": "MMLU 完整选项文本 likelihood 任务。",
    "tasks/plan4/plan4_wmdp_strict.yaml": "WMDP 单字母 strict generation 任务。",
    "tasks/plan4/plan4_mmlu_strict.yaml": "MMLU 单字母 strict generation 任务。",
}


def read_allowed(relative: str) -> str:
    """Read one explicitly allowed text file below code/."""

    if relative not in READ_ALLOWLIST:
        raise PermissionError(f"overview generator refused non-allowlisted input: {relative}")
    path = (CODE_ROOT / relative).resolve()
    if not path.is_relative_to(CODE_ROOT.resolve()) or not path.is_file():
        raise FileNotFoundError(relative)
    return path.read_text(encoding="utf-8")


def load_allowed_json(relative: str) -> dict[str, object]:
    value = json.loads(read_allowed(relative))
    if not isinstance(value, dict):
        raise TypeError(f"{relative} must contain a JSON object")
    return value


def safe_string(value: object, *, label: str, pattern: str, maximum: int = 160) -> str:
    if not isinstance(value, str) or len(value) > maximum or re.fullmatch(pattern, value) is None:
        raise ValueError(f"unsafe or malformed {label}")
    return value


def safe_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"malformed {label}")
    return value


def top_level_symbols(relative: str) -> list[str]:
    tree = ast.parse(read_allowed(relative), filename=relative)
    symbols: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_") and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", node.name):
                symbols.append(node.name)
    return symbols


def test_methods(relative: str) -> list[str]:
    tree = ast.parse(read_allowed(relative), filename=relative)
    methods: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_"):
                methods.append(child.name)
    return methods


def git_identity(arguments: Iterable[str]) -> str | None:
    completed = subprocess.run(
        ("git", "-C", str(CODE_ROOT / "vendor" / "lm-evaluation-harness"), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def parse_tasks() -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for relative in TASK_FILES:
        if not relative.endswith(".yaml"):
            continue
        text = read_allowed(relative)
        task_match = re.search(r"(?m)^task:\s*([A-Za-z0-9_-]+)\s*$", text)
        output_match = re.search(r"(?m)^output_type:\s*([A-Za-z0-9_-]+)\s*$", text)
        metrics = re.findall(r"(?m)^\s+- metric:\s*([A-Za-z0-9_-]+)\s*$", text)
        if task_match is None or output_match is None or not metrics:
            raise ValueError(f"cannot extract safe task metadata from {relative}")
        tasks.append(
            {
                "path": relative,
                "task": task_match.group(1),
                "output_type": output_match.group(1),
                "metrics": metrics,
            }
        )
    return tasks


def package_versions() -> dict[str, str]:
    allowed_packages = {
        "torch",
        "torchvision",
        "torchaudio",
        "datasets",
        "lm-eval",
        "transformers",
        "vllm",
    }
    result: dict[str, str] = {}
    for raw_line in read_allowed("constraints-a6000.txt").splitlines():
        line = raw_line.strip()
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+-]+)", line)
        if match and match.group(1) in allowed_packages:
            result[match.group(1)] = match.group(2)
    return result


def conda_environment_name() -> str:
    match = re.search(
        r'(?m)^ENVIRONMENT_NAME="([A-Za-z0-9_-]+)"\s*$',
        read_allowed("scripts/install_a6000.sh"),
    )
    if match is None:
        raise ValueError("cannot safely extract Conda environment name")
    return match.group(1)


def chips(values: Iterable[str]) -> str:
    rendered = "".join(f'<span class="chip">{escape(value)}</span>' for value in values)
    return rendered or '<span class="muted">无公开接口</span>'


def file_card(path: str, description: str, interfaces: Iterable[str] = ()) -> str:
    href = escape(path, quote=True)
    return f"""
      <article class="file-card">
        <div class="file-top"><a href="{href}"><code>{escape(path)}</code></a></div>
        <p>{escape(description)}</p>
        <div class="chips">{chips(interfaces)}</div>
      </article>"""


def publication_card(path: str, label: str, description: str) -> str:
    """Link one exact publication path without reading its contents."""

    target = (CODE_ROOT / path).resolve()
    if not target.is_relative_to(CODE_ROOT.resolve()):
        raise ValueError(f"publication path escapes code directory: {path}")
    if target.is_file():
        heading = (
            f'<a href="{escape(path, quote=True)}">{escape(label)}</a> '
            '<span class="badge ok">文件存在</span>'
        )
        detail = f"{description} 点击链接可打开结果。"
    else:
        heading = f'{escape(label)} <span class="badge warn">等待实验完成</span>'
        detail = f"{description} 预期路径：{path}"
    return f"""
      <article class="file-card">
        <div class="file-top">{heading}</div>
        <p>{escape(detail)}</p>
      </article>"""


def number(value: object) -> str:
    return f"{safe_integer(value, label='count'):,}"


def short_sha(value: str | None) -> str:
    if value is None or re.fullmatch(r"[0-9a-f]{40,64}", value) is None:
        return "不可用"
    return value[:12]


def main() -> int:
    config = load_allowed_json(CONFIG_FILE)
    metadata = load_allowed_json("manifests/experiment0/metadata.json")
    dedup = {
        name: load_allowed_json(f"manifests/experiment0/{name}_deduplication.json")
        for name in ("wmdp", "mmlu")
    }
    pilot = load_allowed_json("manifests/experiment0/pilot32.json")
    checksum_map = load_allowed_json("manifests/experiment0/checksums.json")

    evaluation = config.get("evaluation")
    datasets = config.get("datasets")
    models = config.get("models")
    gates = config.get("gates")
    metadata_datasets = metadata.get("datasets")
    if not all(isinstance(value, Mapping) for value in (evaluation, datasets, models, gates, metadata_datasets)):
        raise TypeError("config or metadata is missing a required object")

    backend = safe_string(evaluation["backend"], label="backend", pattern=r"(?:hf|vllm)")
    harness_version = safe_string(evaluation["harness_version"], label="harness version", pattern=r"[0-9]+(?:\.[0-9]+){1,3}")
    expected_commit = safe_string(evaluation["harness_commit"], label="harness commit", pattern=r"[0-9a-f]{40}")
    expected_tree = safe_string(evaluation["harness_tree"], label="harness tree", pattern=r"[0-9a-f]{40}")
    harness_repository = safe_string(
        evaluation["harness_repository"],
        label="harness repository",
        pattern=r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git",
    )
    observed_commit = git_identity(("rev-parse", "HEAD"))
    observed_tree = git_identity(("rev-parse", "HEAD^{tree}"))
    dirty_text = git_identity(("status", "--porcelain", "--untracked-files=all"))
    harness_match = observed_commit == expected_commit and observed_tree == expected_tree
    harness_clean = dirty_text == "" if dirty_text is not None else False

    environment_name = conda_environment_name()
    versions = package_versions()
    task_rows = parse_tasks()
    test_inventory = {path: test_methods(path) for path in TEST_FILES}
    test_count = sum(len(methods) for methods in test_inventory.values())
    report_cards = "".join(
        publication_card(path, label, description)
        for path, label, description in PUBLISHED_REPORTS
    )
    published_report_count = sum(
        (CODE_ROOT / path).resolve().is_file()
        for path, _label, _description in PUBLISHED_REPORTS
    )

    model_cards: list[str] = []
    for role in ("qwen3_5_2b", "qwen3_5_4b", "qwen3_5_9b"):
        model = models.get(role)
        if not isinstance(model, Mapping):
            raise TypeError(f"missing model object: {role}")
        display_name = safe_string(model["display_name"], label="model display name", pattern=r"[A-Za-z0-9_.-]+")
        repository = safe_string(model["repository"], label="model repository", pattern=r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
        revision = safe_string(model["revision"], label="model revision", pattern=r"[0-9a-f]{40}")
        parameters = float(model["parameters_billions"])
        model_cards.append(
            f"""<article class="model-card"><span class="eyebrow">baseline</span>
            <h3>{escape(display_name)}</h3><p>{escape(repository)}</p>
            <dl><div><dt>参数</dt><dd>{parameters:.3f}B</dd></div>
            <div><dt>revision</dt><dd><code title="{revision}">{revision[:12]}</code></dd></div></dl></article>"""
        )

    dataset_rows: list[str] = []
    for name, label in (("wmdp", "WMDP"), ("mmlu", "MMLU")):
        row = metadata_datasets.get(name)
        audit = dedup[name]
        if not isinstance(row, Mapping):
            raise TypeError(f"missing dataset metadata: {name}")
        splits = row.get("split_counts")
        if not isinstance(splits, Mapping):
            raise TypeError(f"missing split counts: {name}")
        dataset_rows.append(
            "<tr>"
            f"<th>{label}</th>"
            f"<td>{number(row['source_rows'])}</td>"
            f"<td>{number(row['unique_rows'])}</td>"
            f"<td>{number(row['cal_rows'])}</td>"
            f"<td>{number(splits['TEST-Q3'])}</td>"
            f"<td>{number(splits['TEST-Q4'])}</td>"
            f"<td>{number(audit['excluded_occurrences'])}</td>"
            f"<td>{number(audit['cross_split_groups'])}</td>"
            "</tr>"
        )

    source_cards = "".join(
        file_card(path, SOURCE_DESCRIPTIONS[path], top_level_symbols(path))
        for path in SOURCE_FILES
    )
    task_cards = "".join(
        file_card(
            path,
            TASK_DESCRIPTIONS[path],
            top_level_symbols(path) if path.endswith(".py") else (),
        )
        for path in TASK_FILES
    )
    test_cards = "".join(
        file_card(path, TEST_DESCRIPTIONS[path], methods)
        for path, methods in test_inventory.items()
    )
    root_cards = "".join(
        (
            file_card("README.md", "实验范围、安装、运行、后处理与 gate 的人工可读说明。"),
            file_card("pyproject.toml", "hidden-policy-eval 包定义、Python 版本、直接依赖与 CLI 入口。"),
            file_card("constraints-a6000.txt", "A6000 环境中 PyTorch、vLLM、Transformers、datasets 与 lm-eval 的冻结版本。"),
            file_card(CONFIG_FILE, "数据、模型、harness、vLLM 执行参数和 gate 阈值的唯一冻结配置。"),
            file_card(
                "scripts/install_a6000.sh",
                "创建或复用 Conda 环境 "
                f"{environment_name}，安装 {evaluation['cuda_wheel']}/vLLM 和仓库内 harness。",
            ),
            file_card("scripts/run_baseline_matrix.py", "将 2B/4B/9B 分配到三张 GPU，并行评测、采样 GPU 指标和后处理。", top_level_symbols("scripts/run_baseline_matrix.py")),
            file_card("scripts/generate_baseline_report.py", "严格复核 pilot/full 产物，并生成无 benchmark 内容的 JSON/HTML 报告。", top_level_symbols("scripts/generate_baseline_report.py")),
            file_card("scripts/generate_code_overview.py", "使用安全白名单生成当前自包含项目说明页。", top_level_symbols("scripts/generate_code_overview.py")),
            file_card("code-overview.html", "本脚本生成的自包含中文项目地图；不含 benchmark 或模型输出内容。"),
        )
    )
    manifest_descriptions = {
        "manifests/experiment0/wmdp.json": "WMDP sealed manifest；只保存身份、hash、subject 和 split。",
        "manifests/experiment0/mmlu.json": "MMLU sealed manifest；只保存身份、hash、subject 和 split。",
        "manifests/experiment0/wmdp_deduplication.json": "WMDP 去重统计与无内容审计。",
        "manifests/experiment0/mmlu_deduplication.json": "MMLU 去重统计与无内容审计。",
        "manifests/experiment0/pilot32.json": "固定 32-item pilot 的稳定 ID 与 subject 清单。",
        "manifests/experiment0/metadata.json": "数据来源、revision、认证状态和切分计数。",
        "manifests/experiment0/checksums.json": "六个 manifest 构建产物的 SHA-256。",
    }
    manifest_cards = "".join(
        file_card(path, description) for path, description in manifest_descriptions.items()
    )

    task_table = "".join(
        f"<tr><th><code>{escape(str(task['task']))}</code></th>"
        f"<td>{escape(str(task['output_type']))}</td>"
        f"<td>{chips(str(metric) for metric in task['metrics'])}</td></tr>"
        for task in task_rows
    )
    checksum_names = (
        "wmdp.json",
        "mmlu.json",
        "wmdp_deduplication.json",
        "mmlu_deduplication.json",
        "pilot32.json",
        "metadata.json",
    )
    checksum_rows = "".join(
        f"<tr><th>{escape(name)}</th><td><code title=\"{escape(str(checksum_map.get(name, '')), quote=True)}\">"
        f"{escape(str(checksum_map.get(name, '')))[:12]}</code></td></tr>"
        for name in checksum_names
    )
    version_chips = chips(f"{name} {version}" for name, version in sorted(versions.items()))

    pilot_datasets = pilot.get("datasets")
    if not isinstance(pilot_datasets, Mapping):
        raise TypeError("pilot datasets must be an object")
    pilot_wmdp = pilot_datasets.get("wmdp")
    pilot_mmlu = pilot_datasets.get("mmlu")
    if not isinstance(pilot_wmdp, list) or not isinstance(pilot_mmlu, list):
        raise TypeError("pilot dataset lists are malformed")
    pilot_items = safe_integer(pilot["total_items"], label="pilot total items")
    full_items = sum(
        safe_integer(
            metadata_datasets[name]["cal_rows"],
            label=f"{name} CAL rows",
        )
        for name in ("wmdp", "mmlu")
    )
    permutation_count = safe_integer(
        evaluation["permutation_count"], label="permutation count"
    )
    report_status = (
        "HTML 与 JSON 文件均存在"
        if published_report_count == len(PUBLISHED_REPORTS)
        else "等待正式实验与发布"
    )

    status_class = "ok" if harness_match and harness_clean else "warn"
    status_text = "身份匹配且 clean" if harness_match and harness_clean else "需要检查"
    thinking = "关闭" if evaluation.get("enable_thinking") is False else "开启"
    prefix_cache = "开启" if evaluation.get("enable_prefix_caching") is True else "关闭"

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>Hidden Policy · Code Overview</title>
  <style>
    :root {{ --ink:#17201d; --muted:#68736f; --paper:#f4f3ed; --panel:#fffefa;
      --line:#d9ddd5; --brand:#176b5b; --brand2:#d8eee8; --warm:#df784b;
      --code:#edf1ed; --shadow:0 16px 45px rgba(23,32,29,.08); }}
    * {{ box-sizing:border-box; }} html {{ scroll-behavior:smooth; }}
    body {{ margin:0; color:var(--ink); background:var(--paper); font:15px/1.65 ui-sans-serif,system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }}
    a {{ color:var(--brand); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
    code {{ font-family:"SFMono-Regular",Consolas,monospace; font-size:.9em; overflow-wrap:anywhere; }}
    .shell {{ display:grid; grid-template-columns:230px minmax(0,1fr); max-width:1480px; margin:auto; }}
    nav {{ position:sticky; top:0; height:100vh; padding:34px 24px; border-right:1px solid var(--line); }}
    nav .mark {{ width:42px;height:42px;border-radius:13px;background:var(--brand);color:white;display:grid;place-items:center;font-weight:800;margin-bottom:22px; }}
    nav a {{ display:block; color:var(--muted); padding:7px 0; }} nav a:hover {{ color:var(--brand); text-decoration:none; }}
    main {{ min-width:0; padding:54px clamp(26px,5vw,80px) 90px; }} section {{ max-width:1120px; margin:0 auto 72px; scroll-margin-top:25px; }}
    .hero {{ padding:46px; border-radius:28px; color:#eefbf7; background:linear-gradient(135deg,#163f37,#176b5b 62%,#278774); box-shadow:var(--shadow); }}
    .eyebrow {{ text-transform:uppercase; letter-spacing:.14em; font-size:11px; font-weight:800; opacity:.75; }}
    h1 {{ font-size:clamp(34px,5vw,64px); line-height:1.06; margin:12px 0 20px; letter-spacing:-.035em; }}
    h2 {{ font-size:30px; line-height:1.2; margin:0 0 12px; letter-spacing:-.02em; }} h3 {{ margin:5px 0 8px; }}
    .lede {{ font-size:18px; max-width:760px; opacity:.88; }} .muted {{ color:var(--muted); }}
    .hero-grid,.stats,.models,.file-grid,.boundary-grid {{ display:grid; gap:15px; }}
    .hero-grid {{ grid-template-columns:repeat(4,1fr); margin-top:30px; }}
    .metric {{ padding:16px 18px;border:1px solid rgba(255,255,255,.2);border-radius:16px;background:rgba(255,255,255,.08); }}
    .metric strong {{ display:block;font-size:22px; }} .metric span {{ font-size:12px;opacity:.72; }}
    .section-head {{ margin-bottom:25px; }}
    .callout {{ padding:18px 21px; border-left:4px solid var(--brand); background:var(--brand2); border-radius:0 14px 14px 0; }}
    .stats {{ grid-template-columns:repeat(4,1fr); }} .stat,.model-card,.file-card,.boundary-card {{ background:var(--panel);border:1px solid var(--line);border-radius:17px;padding:20px;box-shadow:0 5px 18px rgba(23,32,29,.035); }}
    .stat b {{ display:block;font-size:25px;color:var(--brand); }}
    .models {{ grid-template-columns:repeat(3,1fr); }} .model-card dl {{ margin:14px 0 0; }} .model-card dl div {{ display:flex;justify-content:space-between;border-top:1px solid var(--line);padding:7px 0;gap:10px; }} dt {{ color:var(--muted); }} dd {{ margin:0; }}
    .flow {{ display:flex;align-items:stretch;gap:7px;overflow-x:auto;padding:6px 0 18px; }}
    .flow-node {{ min-width:145px;flex:1;padding:17px;border:1px solid var(--line);background:var(--panel);border-radius:15px; }}
    .flow-node b {{ display:block;color:var(--brand); }} .arrow {{ display:grid;place-items:center;color:var(--warm);font-size:25px;font-weight:800; }}
    .file-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .file-card p {{ margin:8px 0 13px;color:var(--muted); }}
    .file-top {{ font-weight:700; }} .chips {{ display:flex;flex-wrap:wrap;gap:6px; }} .chip {{ background:var(--code);border-radius:999px;padding:3px 8px;font:11px/1.5 "SFMono-Regular",Consolas,monospace; }}
    details {{ background:rgba(255,255,255,.35);border:1px solid var(--line);border-radius:18px;margin:13px 0; }} summary {{ padding:18px 22px;cursor:pointer;font-size:17px;font-weight:750; }} details > .inside {{ padding:0 20px 22px; }}
    .table-wrap {{ overflow:auto;border:1px solid var(--line);border-radius:16px;background:var(--panel); }} table {{ width:100%;border-collapse:collapse; }} th,td {{ padding:12px 15px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap; }} thead th {{ font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted); }} tbody tr:last-child th,tbody tr:last-child td {{ border-bottom:0; }}
    .boundary-grid {{ grid-template-columns:repeat(3,1fr); }} .boundary-card ul {{ margin:8px 0 0;padding-left:19px; }}
    pre {{ background:#17201d;color:#e8f0ec;padding:20px;border-radius:15px;overflow:auto;font:13px/1.65 "SFMono-Regular",Consolas,monospace; }}
    .badge {{ display:inline-flex;align-items:center;gap:7px;border-radius:999px;padding:6px 11px;font-weight:700;font-size:12px; }} .badge.ok {{ background:#d9efe5;color:#195d4d; }} .badge.warn {{ background:#f7dfcf;color:#84472b; }}
    footer {{ max-width:1120px;margin:auto;padding-top:25px;border-top:1px solid var(--line);color:var(--muted); }}
    @media (max-width:900px) {{ .shell {{ display:block; }} nav {{ position:relative;height:auto;border:0;padding:20px 24px;display:flex;gap:15px;overflow:auto; }} nav .mark,nav p {{ display:none; }} nav a {{ white-space:nowrap; }} main {{ padding-top:25px; }} .hero-grid,.stats {{ grid-template-columns:repeat(2,1fr); }} .models,.boundary-grid {{ grid-template-columns:1fr; }} }}
    @media (max-width:620px) {{ .hero {{ padding:28px; }} .hero-grid,.stats,.file-grid {{ grid-template-columns:1fr; }} .flow {{ flex-direction:column; }} .arrow {{ transform:rotate(90deg); }} }}
    @media print {{ nav {{ display:none; }} .shell {{ display:block; }} main {{ padding:0; }} body {{ background:white; }} details {{ break-inside:avoid; }} details > .inside {{ display:block; }} }}
  </style>
</head>
<body>
<div class="shell">
  <nav aria-label="页面目录"><div class="mark">HP</div><p class="eyebrow">Code map</p>
    <a href="#overview">概览</a><a href="#models">模型与执行</a><a href="#run-design">正式顺序</a>
    <a href="#reports">实验报告</a><a href="#flow">数据流</a>
    <a href="#manifests">Manifests</a><a href="#tasks">任务协议</a><a href="#files">文件地图</a>
    <a href="#repro">可复现性</a><a href="#commands">常用命令</a>
  </nav>
  <main>
    <section id="overview" class="hero">
      <span class="eyebrow">Plan 4 · Experiment 0 · allowlisted snapshot</span>
      <h1>Hidden Policy<br>代码与实验地图</h1>
      <p class="lede">当前目录负责固定数据边界、完整选项 likelihood、strict generation、三视图排列鲁棒性、后处理与 PASS/STOP gate。训练、Q3/Q4 解封和 observer 不在本阶段范围内。</p>
      <div class="hero-grid">
        <div class="metric"><strong>{escape(backend)}</strong><span>lm-eval backend</span></div>
        <div class="metric"><strong>{escape(environment_name)}</strong><span>Conda environment</span></div>
        <div class="metric"><strong>v{escape(harness_version)}</strong><span>vendored harness</span></div>
        <div class="metric"><strong>{test_count}</strong><span>声明的 unit tests</span></div>
      </div>
    </section>

    <section>
      <div class="section-head"><span class="eyebrow">Current state</span><h2>一眼看懂当前配置</h2></div>
      <div class="stats">
        <div class="stat"><b>{escape(str(evaluation['prompt_protocol']))}</b><span>Prompt protocol</span></div>
        <div class="stat"><b>{thinking}</b><span>Qwen thinking</span></div>
        <div class="stat"><b>{number(pilot['total_items'])}</b><span>Pilot items（WMDP {len(pilot_wmdp)} + MMLU {len(pilot_mmlu)}）</span></div>
        <div class="stat"><b>{escape(str(evaluation['permutation_count']))}</b><span>每题 option views</span></div>
      </div>
      <p class="callout"><strong>执行路径：</strong> 当前冻结 backend 是 <code>{escape(backend)}</code>。lm-evaluation-harness 通过仓库内源码启动 vLLM；HF backend 仍保留为显式参考路径。A6000 安装脚本创建或复用名为 <code>{escape(environment_name)}</code> 的 Conda 环境。</p>
    </section>

    <section id="models">
      <div class="section-head"><span class="eyebrow">Baseline matrix</span><h2>2B / 4B / 9B 基础评测</h2><p class="muted">三个模型 revision 固定；matrix runner 默认把它们分配到三张不同 GPU 并行执行。</p></div>
      <div class="models">{''.join(model_cards)}</div>
      <details><summary>vLLM 与环境参数</summary><div class="inside">
        <div class="table-wrap"><table><tbody>
          <tr><th>vLLM</th><td>{escape(str(evaluation['vllm_version']))}</td><th>dtype</th><td>{escape(str(evaluation['dtype']))}</td></tr>
          <tr><th>max model length</th><td>{escape(str(evaluation['max_model_len']))}</td><th>GPU memory utilization</th><td>{escape(str(evaluation['gpu_memory_utilization']))}</td></tr>
          <tr><th>max sequences</th><td>{escape(str(evaluation['max_num_seqs']))}</td><th>max batched tokens</th><td>{escape(str(evaluation['max_num_batched_tokens']))}</td></tr>
          <tr><th>prefix caching</th><td>{prefix_cache}</td><th>seed</th><td>{escape(str(evaluation['seed']))}</td></tr>
          <tr><th>HF Xet high performance</th><td>{escape(str(evaluation['hf_xet_high_performance'])).lower()}</td><th>CUDA wheel</th><td>{escape(str(evaluation['cuda_wheel']))}</td></tr>
          <tr><th>PyTorch allocator</th><td><code>{escape(str(evaluation['pytorch_alloc_conf']))}</code></td><th>allocator backend</th><td>{escape(str(evaluation['pytorch_allocator_backend']))}</td></tr>
          <tr><th>tensor / data parallel</th><td>{escape(str(evaluation['tensor_parallel_size']))} / {escape(str(evaluation['data_parallel_size']))}</td><th>trust remote code</th><td>{escape(str(evaluation['trust_remote_code'])).lower()}</td></tr>
        </tbody></table></div><p class="chips">{version_chips}</p>
      </div></details>
    </section>

    <section id="run-design">
      <div class="section-head"><span class="eyebrow">Formal execution</span><h2>正式实验顺序与规模</h2><p class="muted">所有矩阵运行必须使用同一个已提交 commit；pilot 验证通过后才进入 full CAL。</p></div>
      <div class="flow">
        <div class="flow-node"><b>① vLLM pilot</b><span>2B / 4B / 9B · 三张卡</span></div><div class="arrow">→</div>
        <div class="flow-node"><b>② HF reference</b><span>仅 2B pilot · backend 对照</span></div><div class="arrow">→</div>
        <div class="flow-node"><b>③ vLLM full</b><span>2B / 4B / 9B · 全部 CAL</span></div><div class="arrow">→</div>
        <div class="flow-node"><b>④ Publish</b><span>验证并生成 HTML / JSON</span></div><div class="arrow">→</div>
        <div class="flow-node"><b>⑤ Refresh</b><span>刷新本概述页及结果链接</span></div>
      </div>
      <div class="table-wrap"><table><thead><tr><th>范围</th><th>每模型 items</th><th>Likelihood views</th><th>Strict generations</th></tr></thead><tbody>
        <tr><th>Pilot</th><td>{pilot_items:,}</td><td>{pilot_items * permutation_count:,}</td><td>{pilot_items:,}</td></tr>
        <tr><th>Full CAL</th><td>{full_items:,}</td><td>{full_items * permutation_count:,}</td><td>{full_items:,}</td></tr>
      </tbody></table></div>
      <div class="boundary-grid" style="margin-top:16px">
        <article class="boundary-card"><h3>启动门槛</h3><p>仓库必须 clean；三张目标 GPU 的预存显存都必须低于 1 GiB。</p></article>
        <article class="boundary-card"><h3>运行前检查</h3><p>依次完成 doctor、prepare、模型 prefetch 和每模型 prompt-length audit。</p></article>
        <article class="boundary-card"><h3>执行与收尾</h3><p>一模型一卡并行；持续采集 GPU telemetry，随后逐模型 postprocess。</p></article>
      </div>
      <p class="callout"><strong>发布保护：</strong> 报告生成器要求 pilot、HF reference 与 full CAL 的模型、backend、item count 和 scientific provenance 一致，并验证它们来自同一 repository commit；任何漂移都 fail closed。</p>
    </section>

    <section id="reports">
      <div class="section-head"><span class="eyebrow">Result files</span><h2>基础测试报告</h2><p class="muted">{report_status}。概述页只检查两个精确路径是否存在；它不读取且未复验报告。“文件存在”仅表示可以打开结果。</p></div>
      <div class="file-grid">{report_cards}</div>
    </section>

    <section id="flow">
      <div class="section-head"><span class="eyebrow">Pipeline</span><h2>从冻结输入到 gate</h2></div>
      <div class="flow">
        <div class="flow-node"><b>冻结输入</b><span>config + dataset revision</span></div><div class="arrow">→</div>
        <div class="flow-node"><b>读取与去重</b><span>sources + split pipeline</span></div><div class="arrow">→</div>
        <div class="flow-node"><b>数据边界</b><span>sealed manifest + CAL</span></div><div class="arrow">→</div>
        <div class="flow-node"><b>评测输入</b><span>3 permutations + fingerprint</span></div><div class="arrow">→</div>
        <div class="flow-node"><b>模型执行</b><span>vendored lm-eval + vLLM</span></div><div class="arrow">→</div>
        <div class="flow-node"><b>可信结果</b><span>normalize + summary + gate</span></div>
      </div>
      <p class="muted"><code>vendor.py</code> 和 <code>environment.py</code> 横向验证 harness 身份、editable 来源、依赖版本与 CUDA；<code>prepare.py</code> 把配置、manifests、tasks、实现和 harness 身份绑定进 runtime fingerprint。</p>
    </section>

    <section id="manifests">
      <div class="section-head"><span class="eyebrow">Data boundary</span><h2>Manifests 与切分统计</h2><p class="muted">页面只呈现允许的顶层计数与 hash，不展开 entries，也不读取 CAL 内容。</p></div>
      <div class="table-wrap"><table><thead><tr><th>Dataset</th><th>原始</th><th>去重后</th><th>CAL</th><th>TEST-Q3</th><th>TEST-Q4</th><th>排除重复</th><th>跨来源 split 组</th></tr></thead><tbody>{''.join(dataset_rows)}</tbody></table></div>
      <details><summary>Manifest 文件</summary><div class="inside"><div class="file-grid">{manifest_cards}</div></div></details>
      <details><summary>构建产物 checksums</summary><div class="inside"><div class="table-wrap"><table><tbody>{checksum_rows}</tbody></table></div></div></details>
    </section>

    <section id="tasks">
      <div class="section-head"><span class="eyebrow">Evaluation contract</span><h2>四个 lm-eval custom tasks</h2></div>
      <div class="table-wrap"><table><thead><tr><th>Task</th><th>output type</th><th>metrics</th></tr></thead><tbody>{task_table}</tbody></table></div>
      <div class="boundary-grid" style="margin-top:16px">
        <article class="boundary-card"><h3>Likelihood</h3><p>比较四个完整选项文本；主分数是每个 continuation token 的平均 log likelihood。</p></article>
        <article class="boundary-card"><h3>Permutation</h3><p>每题使用 identity 加两个确定性互异排列，结果重新映射回语义选项后检查一致性。</p></article>
        <article class="boundary-card"><h3>Strict</h3><p>只接受单个大写 A–D；拒答与其他格式错误分别统计。</p></article>
      </div>
    </section>

    <section id="files">
      <div class="section-head"><span class="eyebrow">File atlas</span><h2>自有文件地图</h2><p class="muted">接口标签从白名单 Python 文件的 AST 自动提取；vendor 始终作为单个外部组件，不展开内部文件。</p></div>
      <details open><summary>入口、配置与脚本</summary><div class="inside"><div class="file-grid">{root_cards}</div></div></details>
      <details><summary>hidden_policy_eval 源码（{len(SOURCE_FILES)} 个文件）</summary><div class="inside"><div class="file-grid">{source_cards}</div></div></details>
      <details><summary>Plan 4 tasks（{len(TASK_FILES)} 个文件）</summary><div class="inside"><div class="file-grid">{task_cards}</div></div></details>
      <details><summary>单元测试（{len(TEST_FILES)} 个文件 / {test_count} 个 test case）</summary><div class="inside"><div class="file-grid">{test_cards}</div></div></details>
      <article class="file-card" style="margin-top:14px"><div class="file-top"><code>vendor/lm-evaluation-harness/</code> <span class="badge {status_class}">{status_text}</span></div>
        <p>唯一 vendored 组件。页面不会遍历其约 70 MB 内部源码。</p>
        <div class="table-wrap"><table><tbody>
          <tr><th>repository</th><td><a href="{escape(harness_repository, quote=True)}">{escape(harness_repository)}</a></td></tr>
          <tr><th>version</th><td>{escape(harness_version)}</td></tr>
          <tr><th>commit</th><td><code title="{escape(observed_commit or '', quote=True)}">{short_sha(observed_commit)}</code> · expected {expected_commit[:12]}</td></tr>
          <tr><th>tree</th><td><code title="{escape(observed_tree or '', quote=True)}">{short_sha(observed_tree)}</code> · expected {expected_tree[:12]}</td></tr>
          <tr><th>working tree</th><td>{'clean' if harness_clean else 'unknown / dirty'}</td></tr>
        </tbody></table></div>
      </article>
    </section>

    <section id="repro">
      <div class="section-head"><span class="eyebrow">Trust boundary</span><h2>可复现性与安全边界</h2></div>
      <div class="boundary-grid">
        <article class="boundary-card"><h3>已冻结</h3><ul><li>数据与模型 revision</li><li>harness commit/tree</li><li>Python 包与 CUDA wheel</li><li>split salt、prompt、seed、阈值</li></ul></article>
        <article class="boundary-card"><h3>运行前验证</h3><ul><li>manifest checksums 与 CAL 对应</li><li>harness clean + editable source</li><li>依赖版本、CUDA 与 GPU</li><li>runtime fingerprint 与空输出目录</li></ul></article>
        <article class="boundary-card"><h3>不进入 Git / 本页</h3><ul><li>CAL benchmark 内容</li><li>runtime 排列输入</li><li>原始日志与逐题模型输出</li><li>模型 cache、Conda 环境与凭据</li></ul></article>
      </div>
      <p class="callout"><strong>数值边界：</strong> 固定 Git 状态不能单独保证 GPU 浮点逐 bit 相同；<code>batch_size=auto</code> 也可能受可用显存影响。首次 full CAL 前仍需人工审阅 32-item pilot 的实际 prompt、答案映射和 token denominator。</p>
      <div class="table-wrap"><table><tbody>
        <tr><th>WMDP target–weak headroom</th><td>≥ {escape(str(gates['minimum_wmdp_headroom_percentage_points']))} percentage points</td></tr>
        <tr><th>Semantic permutation consistency</th><td>≥ {100 * float(gates['minimum_semantic_permutation_consistency']):.0f}%</td></tr>
        <tr><th>Strict invalid / refusal</th><td>≤ {100 * float(gates['maximum_strict_invalid_or_refusal_rate']):.0f}%</td></tr>
      </tbody></table></div>
    </section>

    <section id="commands">
      <div class="section-head"><span class="eyebrow">Operator guide</span><h2>常用命令</h2></div>
      <h3>安装与检查</h3><pre>git submodule update --init --recursive --depth 1
bash code/scripts/install_a6000.sh
conda activate {escape(environment_name)}
hidden-policy-eval doctor --backend {escape(backend)}</pre>
      <h3>数据与 pilot</h3><pre>hidden-policy-eval split
hidden-policy-eval validate
hidden-policy-eval prepare --scope pilot
hidden-policy-eval command --model-role qwen3_5_2b --scope pilot</pre>
      <h3>① 三模型 vLLM pilot</h3><pre>python code/scripts/run_baseline_matrix.py \\
  --scope pilot \\
  --backend {escape(backend)} \\
  --models qwen3_5_2b qwen3_5_4b qwen3_5_9b \\
  --gpus 0,1,2 \\
  --run-id baseline-pilot-v1</pre>
      <h3>② Qwen3.5-2B HF reference pilot</h3><pre>python code/scripts/run_baseline_matrix.py \\
  --scope pilot \\
  --backend hf \\
  --models qwen3_5_2b \\
  --gpus 0 \\
  --run-id baseline-pilot-hf-2b-v1</pre>
      <h3>③ 三模型 vLLM full CAL</h3><pre>python code/scripts/run_baseline_matrix.py \\
  --scope full \\
  --backend {escape(backend)} \\
  --models qwen3_5_2b qwen3_5_4b qwen3_5_9b \\
  --gpus 0,1,2 \\
  --run-id baseline-full-v1</pre>
      <h3>④ 发布结果，⑤ 刷新概述</h3><pre>python code/scripts/generate_baseline_report.py \\
  --pilot-matrix code/results/experiment0/baseline/baseline-pilot-v1 \\
  --full-matrix code/results/experiment0/baseline/baseline-full-v1 \\
  --hf-reference-matrix code/results/experiment0/baseline/baseline-pilot-hf-2b-v1

python3 code/scripts/generate_code_overview.py</pre>
      <h3>本地单元测试</h3><pre>PYTHONPATH=code/src python3 -m unittest discover -s code/tests -v</pre>
    </section>

    <footer>此页面由 <code>scripts/generate_code_overview.py</code> 从固定安全白名单生成。它不会读取 <code>data/</code>、<code>runtime/</code>、<code>results/</code>、认证材料、远程连接配置或 vendored 源码内容。</footer>
  </main>
</div>
</body>
</html>
"""
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
