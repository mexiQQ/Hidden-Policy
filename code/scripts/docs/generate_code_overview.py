#!/usr/bin/env python3
"""Generate an ownership-first code map without reading research data or results.

Only the explicit Python source allowlist is read. Report links are checked for
existence, never opened; vendor source, raw data, caches and credentials are not
inspected.
"""

from __future__ import annotations

import ast
from html import escape
from pathlib import Path
from typing import Iterable


CODE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = CODE_ROOT / "reports" / "code-overview.html"
SOURCE_GROUPS = {
    "e0": (
        ("cli.py", "E0 命令入口：split、prepare、run、postprocess、gate。"),
        ("split_pipeline.py", "构建数据切分、CAL 与 pilot 清单。"),
        ("prepare.py", "把 CAL 转为 lm-eval 输入并绑定 provenance。"),
        ("harness.py", "构造和执行 lm-evaluation-harness 命令。"),
        ("report.py", "后处理、汇总 baseline 指标与 PASS/STOP gate。"),
        ("environment.py", "验证 E0 的 Python、CUDA 与环境依赖。"),
        ("vendor.py", "核对冻结 harness 的源码身份与 editable 安装。"),
        ("mcq.py", "保留历史 E0 结果所需的选项排列工具。"),
    ),
    "e1": (
        ("policy.py", "核心实验定义：G0/G1 上下文和 U0/U1 目标回答。"),
        ("data.py", "校验和冻结选题清单，从固定来源重建 320 道题并复用缓存。"),
        ("evaluate.py", "选择 CAL/Q3/Q4 快速探针，比较触发前后行为。"),
        ("review.py", "数据审阅的 verdict 校验与常量；供审阅汇总工具调用。"),
    ),
    "shared": (
        ("benchmarks.py", "统一访问冻结模型、数据和官方切分定义。"),
        ("manifests.py", "MCQ 规范化、稳定 ID、切分与清单校验。"),
        ("prompts.py", "共享 MCQ prompt 渲染。"),
        ("strict.py", "严格解析选项字母、invalid 和 refusal。"),
        ("sources.py", "读取冻结 revision 的 WMDP/MMLU 来源。"),
        ("io.py", "JSON/JSONL、原子写入与 SHA-256 工具。"),
    ),
}
ENTRY_FILES = (
    "scripts/e0/run_baseline_matrix.py",
    "scripts/e1/prepare_data.py",
    "scripts/e1/run_experiment1.py",
)
READ_ALLOWLIST = frozenset(
    [f"src/hidden_policy_eval/{group}/{name}"
     for group, files in SOURCE_GROUPS.items() for name, _description in files]
    + list(ENTRY_FILES)
)
PUBLISHED_REPORTS = (
    ("reports/baseline-results.html", "E0 baseline 报告"),
    ("reports/e1-data-report.html", "E1 数据审计报告"),
    ("results/published/experiment1/swift-smoke-v1/result.json", "E1 首轮 smoke 汇总"),
)


def read_allowed(relative: str) -> str:
    if relative not in READ_ALLOWLIST:
        raise PermissionError(f"overview refused non-allowlisted input: {relative}")
    path = (CODE_ROOT / relative).resolve()
    if not path.is_relative_to(CODE_ROOT.resolve()) or not path.is_file():
        raise FileNotFoundError(relative)
    return path.read_text(encoding="utf-8")


def top_level_symbols(relative: str) -> list[str]:
    tree = ast.parse(read_allowed(relative), filename=relative)
    return [node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and not node.name.startswith("_")]


def code_link(relative: str, label: str | None = None) -> str:
    path = (CODE_ROOT / relative).resolve()
    if not path.is_relative_to(CODE_ROOT.resolve()):
        raise ValueError(f"link escapes code directory: {relative}")
    return f'<a href="../{escape(relative, quote=True)}">{escape(label or relative)}</a>'


def source_table(group: str) -> str:
    rows = []
    for name, description in SOURCE_GROUPS[group]:
        path = f"src/hidden_policy_eval/{group}/{name}"
        symbols = ", ".join(top_level_symbols(path)) or "无公开接口"
        rows.append(
            f"<tr><th>{code_link(path, name)}</th><td>{escape(description)}"
            f"<details><summary>函数与类</summary><code>{escape(symbols)}</code></details>"
            "</td></tr>"
        )
    return '<div class="table-wrap"><table><tbody>' + "".join(rows) + "</tbody></table></div>"


def report_links() -> str:
    links = []
    for path, label in PUBLISHED_REPORTS:
        state = code_link(path, label) if (CODE_ROOT / path).is_file() else escape(label) + "（尚未发布）"
        links.append(f"<li>{state}</li>")
    return "<ul>" + "".join(links) + "</ul>"


def entry_symbols(paths: Iterable[str]) -> str:
    return "".join(
        f"<details><summary>{code_link(path)}</summary>"
        f"<code>{escape(', '.join(top_level_symbols(path)))}</code></details>"
        for path in paths
    )


def main() -> int:
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hidden Policy | 代码地图</title>
  <style>
    :root {{ color-scheme:light; --ink:#202725; --muted:#626a67; --line:#d9dfdc; --link:#08685d; }}
    * {{ box-sizing:border-box; letter-spacing:0; }}
    body {{ margin:0; background:#fff; color:var(--ink); font:15px/1.65 system-ui,-apple-system,"PingFang SC",sans-serif; }}
    a {{ color:var(--link); text-underline-offset:3px; }}
    header,main,footer {{ max-width:1040px; margin:auto; padding:24px; }}
    header {{ padding-top:40px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 8px; font-size:30px; line-height:1.25; }}
    h2 {{ font-size:22px; margin:0 0 12px; }} h3 {{ font-size:17px; margin:18px 0 8px; }}
    p {{ margin:10px 0; }} nav {{ display:flex; flex-wrap:wrap; gap:10px 24px; margin-top:20px; }}
    section {{ padding:8px 0 28px; margin-bottom:22px; border-bottom:1px solid var(--line); scroll-margin-top:20px; }}
    .muted,footer {{ color:var(--muted); }} .flow {{ font-weight:650; color:#80522c; }}
    code,pre {{ font:13px/1.65 ui-monospace,"SFMono-Regular",Consolas,monospace; }}
    code {{ overflow-wrap:anywhere; }} pre {{ white-space:pre; overflow:auto; background:#f3f5f4; padding:16px; border-left:3px solid #69928a; }}
    .table-wrap {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; table-layout:fixed; }}
    th,td {{ padding:12px 10px; text-align:left; vertical-align:top; border-bottom:1px solid #e8ece9; overflow-wrap:anywhere; }}
    th {{ width:27%; font-weight:600; }}
    details {{ margin:7px 0; }} summary {{ cursor:pointer; color:var(--muted); font-size:13px; }}
    details code {{ display:block; margin-top:7px; color:var(--muted); }}
    .boundary {{ border-left:3px solid #69928a; padding-left:16px; }}
    ul {{ padding-left:22px; }} li {{ margin:7px 0; }}
    @media(max-width:600px) {{ header,main,footer {{ padding:20px 16px; }} h1 {{ font-size:26px; }} th {{ width:34%; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Hidden Policy 代码地图</h1>
    <p>E0 测量原始能力；E1 构造并训练 policy；shared 提供共用基础代码。</p>
    <nav aria-label="目录"><a href="#start">入口</a><a href="#e1">E1</a><a href="#e0">E0</a><a href="#shared">共用部分</a><a href="#artifacts">配置与产物</a></nav>
  </header>
  <main>
    <section id="start">
      <h2>先读主入口</h2>
      <p><strong>E1：</strong>{code_link('scripts/e1/run_experiment1.py')} 的 <code>run()</code> 串起整个训练实验。</p>
      <p class="flow">现有题目 + policy → 四组训练数据 → 四个独立 LoRA → CAL/Q3/Q4 快速评估</p>
      <p><strong>E0：</strong>{code_link('scripts/e0/run_baseline_matrix.py')} 运行 baseline 矩阵；单项命令由 {code_link('src/hidden_policy_eval/e0/cli.py', 'e0/cli.py')} 提供。</p>
      <p class="flow">冻结数据与模型 → CAL 评测 → 后处理 → PASS/STOP gate</p>
      {entry_symbols(ENTRY_FILES)}
      <p>完整命令与结果说明：<a href="../../docs/experiments/e0.md">E0 运行指南</a> · <a href="../../docs/experiments/e1.md">E1 运行指南</a> · {code_link('README.md', '代码导航')}</p>
    </section>
    <section id="e1">
      <h2>E1：Hidden Policy 实验</h2>
      <p>实验逻辑在 <code>src/hidden_policy_eval/e1/</code>。修改规则首先看 <code>policy.py</code>，不用先读 trainer 或 Swift 内部。</p>
      {source_table('e1')}
      <p>{code_link('scripts/e1/', 'scripts/e1/')} 只保留两个入口：{code_link('scripts/e1/prepare_data.py', 'prepare_data.py')} 用 <code>status / freeze / build</code> 检查、冻结或重建 320 道原题，不调用模型；{code_link('scripts/e1/run_experiment1.py', 'run_experiment1.py')} 才应用 policy、生成 0.8B 答案并训练和评测四组。</p>
      <p>选题已冻结，日常只需检查或重建，均复用已有数据。一次性审计脚本已删除，历史保留在 Git；原始数据、审计数据库和清单不变，已发布结果保留。数据报告、审阅汇总发布与模板独立放在 {code_link('scripts/docs/e1/', 'scripts/docs/e1/')}。</p>
      <p>后端：{code_link('vendor/ms-swift/', 'ms-swift')}。四组都从原始 4B 模型开始独立训练；0.8B 教师答案共享缓存。</p>
    </section>
    <section id="e0">
      <h2>E0：Baseline</h2>
      <p>实验逻辑在 <code>src/hidden_policy_eval/e0/</code>。E0 只评测 CAL，不训练，不读取 Q3/Q4 题目。</p>
      {source_table('e0')}
      <p>{code_link('scripts/e0/', 'scripts/e0/')} 收纳安装和 baseline 矩阵；报告生成与安全发布独立放在 {code_link('scripts/docs/e0/', 'scripts/docs/e0/')}。</p>
      <p>后端：{code_link('vendor/lm-evaluation-harness/', 'lm-evaluation-harness')} + vLLM；HF 保留作参考。</p>
    </section>
    <section id="shared">
      <h2>Shared：真正共用的部分</h2>
      <p class="flow">E0 → shared ← E1</p>
      <p>这里不运行某个实验，也不定义 hidden policy。E1 不导入 E0 的运行代码。</p>
      {source_table('shared')}
      <p class="boundary">模型、数据 revision 和官方切分是在 E0 时冻结的，仍保留 <code>configs/experiment0.json</code>、<code>manifests/experiment0/</code>、<code>data/experiment0/cal/</code> 的历史路径。E1 通过 shared 复用这些依据，不复制清单，也不重跑 baseline。</p>
    </section>
    <section id="artifacts">
      <h2>配置、文档与产物</h2>
      <div class="table-wrap"><table><tbody>
        <tr><th>配置</th><td>{code_link('configs/experiment0.json', 'experiment0.json')}：冻结 baseline 协议、模型和数据；{code_link('configs/experiment1.json', 'experiment1.json')}：policy 文案、LoRA 预算与快速评估规模。</td></tr>
        <tr><th>依赖</th><td>E0/E1 共用 <code>hidden-policy</code> Conda 环境；依赖统一记录在 {code_link('constraints-a6000.txt')}。</td></tr>
        <tr><th>测试</th><td>{code_link('tests/e0/', 'tests/e0/')}、{code_link('tests/e1/', 'tests/e1/')}、{code_link('tests/shared/', 'tests/shared/')} 与代码归属对应。</td></tr>
        <tr><th>实验文档</th><td><a href="../../docs/experiments/e0.md">docs/experiments/e0.md</a>、<a href="../../docs/experiments/e1.md">docs/experiments/e1.md</a>；code 根目录只保留简短导航。</td></tr>
        <tr><th>文档生成</th><td>{code_link('scripts/docs/e0/', 'scripts/docs/e0/')}、{code_link('scripts/docs/e1/', 'scripts/docs/e1/')} 按实验分类；{code_link('scripts/docs/generate_code_overview.py', 'generate_code_overview.py')} 生成跨实验地图。文档脚本与实验执行脚本分开，不运行模型。</td></tr>
        <tr><th>原始产物</th><td><code>data/</code>、<code>runtime/</code>、原始 <code>results/</code> 不进入 Git；既有缓存和结果路径不变。</td></tr>
      </tbody></table></div>
      <h3>已发布报告</h3>
      {report_links()}
      <p class="muted">此页只检查以上报告路径是否存在，不读取或重新验证实验结果。</p>
      <h3>本地测试</h3>
      <pre>PYTHONPATH=code/src python3 -m unittest discover -s code/tests</pre>
    </section>
  </main>
  <footer>生成器：{code_link('scripts/docs/generate_code_overview.py')}。仅读取明确列出的项目源码，不读取题目、模型输出、缓存、凭据或 vendor 内部源码。</footer>
</body>
</html>
"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
