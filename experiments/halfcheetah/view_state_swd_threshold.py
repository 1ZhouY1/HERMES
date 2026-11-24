#!/usr/bin/env python3
"""Visualize HalfCheetah task-distance CSV with a threshold slider.

The generated HTML is scenario-aware.  For every HalfCheetah scenario, it
simulates HybridRARL prev-unit selection along the unique scenario task order:

    current task k may reuse a previous scenario task i iff distance(k, i) < eps

Selected/reused modules are rendered in green.  Tasks without any selected
previous module are shown as training a base HybridRARL module.
"""

from __future__ import annotations

import argparse
import ast
import csv
import html
import importlib.util
import json
import re
import webbrowser
from pathlib import Path
from typing import Dict, List, Tuple


FALLBACK_TASKS = [
    "normal",
    "carry_stuff",
    "carry_stuff_hugegravity",
    "defective_module",
    "hugefoot",
    "hugefoot_rainfall",
    "inverted_actions",
    "moon",
    "tinyfoot",
    "tinyfoot_moon",
    "rainfall",
]

FALLBACK_SCENARIOS = {
    "forgetting": ["hugefoot", "moon", "carry_stuff", "rainfall"] * 2,
    "transfer": ["carry_stuff_hugegravity", "moon", "defective_module", "hugefoot_rainfall"] * 2,
    "robustness": ["normal", "inverted_actions", "normal", "inverted_actions"] * 2,
    "compositionality": ["tinyfoot", "moon", "carry_stuff_hugegravity", "tinyfoot_moon"] * 2,
}


def parse_task_label(label: str) -> Dict:
    label = label.strip()
    match = re.search(r"task[_\s-]*(\d+)", label)
    task_id = int(match.group(1)) if match is not None else None
    task_name = label.split(":", 1)[1].strip() if ":" in label else label
    display = f"task {task_id}: {task_name}" if task_id is not None else label
    return {
        "id": task_id,
        "name": task_name,
        "label": label,
        "display": display,
    }


def read_distance_csv(path: Path) -> Tuple[List[Dict], List[List[float]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError(f"{path}: expected a header row and at least one data row.")

    header = rows[0]
    col_tasks = [parse_task_label(label) for label in header[1:]]
    row_tasks: List[Dict] = []
    matrix: List[List[float]] = []

    for row_index, row in enumerate(rows[1:], start=2):
        if not row:
            continue
        row_task = parse_task_label(row[0])
        values = row[1:]
        if len(values) != len(col_tasks):
            raise ValueError(
                f"{path}: row {row_index} has {len(values)} values, "
                f"expected {len(col_tasks)}."
            )
        parsed = []
        for col_index, value in enumerate(values, start=2):
            try:
                parsed.append(float(value))
            except ValueError as exc:
                raise ValueError(
                    f"{path}: cannot parse distance at row {row_index}, "
                    f"column {col_index}: {value!r}"
                ) from exc
        row_tasks.append(row_task)
        matrix.append(parsed)

    if len(row_tasks) != len(col_tasks):
        raise ValueError(
            f"{path}: distance matrix must be square, got "
            f"{len(row_tasks)} rows and {len(col_tasks)} columns."
        )

    for index, (row_task, col_task) in enumerate(zip(row_tasks, col_tasks)):
        if row_task["id"] != col_task["id"] and row_task["label"] != col_task["label"]:
            print(
                "warning: row/column labels differ at index "
                f"{index}: {row_task['label']!r} vs {col_task['label']!r}"
            )
    return row_tasks, matrix


def eval_literal_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [eval_literal_node(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(eval_literal_node(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            eval_literal_node(key): eval_literal_node(value)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left = eval_literal_node(node.left)
        right = eval_literal_node(node.right)
        return left * right
    raise ValueError(f"Unsupported literal expression in tasks.py: {ast.dump(node)}")


def parse_tasks_py(tasks_path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    tree = ast.parse(tasks_path.read_text(encoding="utf-8"))
    env_tasks = None
    scenarios = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "env_tasks":
                    env_tasks = eval_literal_node(node.value)
                if isinstance(target, ast.Name) and target.id == "SCENARIOS":
                    scenarios = eval_literal_node(node.value)
    task_names = list(env_tasks.keys()) if isinstance(env_tasks, dict) else FALLBACK_TASKS
    scenario_map = scenarios if isinstance(scenarios, dict) else FALLBACK_SCENARIOS
    return task_names, scenario_map


def load_halfcheetah_config(tasks_path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """Load task names and scenarios without requiring MuJoCo at viewer-build time."""
    try:
        spec = importlib.util.spec_from_file_location("_halfcheetah_tasks_for_viewer", tasks_path)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            task_names = list(getattr(module, "tasks"))
            scenarios = {
                str(name): list(sequence)
                for name, sequence in getattr(module, "SCENARIOS").items()
            }
            return task_names, scenarios
    except Exception as exc:
        print(f"warning: importing {tasks_path} failed; parsing constants instead: {exc}")

    try:
        return parse_tasks_py(tasks_path)
    except Exception as exc:
        print(f"warning: parsing {tasks_path} failed; using built-in fallback scenarios: {exc}")
        return FALLBACK_TASKS, FALLBACK_SCENARIOS


def unique_sequence(sequence: List[str]) -> List[str]:
    seen = set()
    out = []
    for name in sequence:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def lower_triangle_values(matrix: List[List[float]]) -> List[float]:
    values = []
    for row_index, row in enumerate(matrix):
        for col_index in range(row_index):
            values.append(float(row[col_index]))
    return values


def generate_html(
    *,
    csv_path: Path,
    tasks_in_csv: List[Dict],
    matrix: List[List[float]],
    halfcheetah_tasks: List[str],
    scenarios: Dict[str, List[str]],
    default_threshold: float,
    exp_name: str,
    seed: int,
) -> str:
    scenario_payload = {}
    name_to_id = {name: index for index, name in enumerate(halfcheetah_tasks)}
    for scenario_name, sequence in scenarios.items():
        unique_names = unique_sequence(list(sequence))
        scenario_payload[scenario_name] = [
            {
                "task_id": int(name_to_id[name]),
                "task_name": name,
            }
            for name in unique_names
            if name in name_to_id
        ]

    payload = {
        "csvPath": str(csv_path),
        "tasks": tasks_in_csv,
        "matrix": matrix,
        "scenarios": scenario_payload,
        "defaultThreshold": float(default_threshold),
        "expName": exp_name,
        "seed": int(seed),
        "maxDistance": max(lower_triangle_values(matrix) or [float(default_threshold), 1.0]),
        "minDistance": min(lower_triangle_values(matrix) or [0.0]),
    }
    data_json = json.dumps(payload, ensure_ascii=False)
    title = f"HalfCheetah SWD threshold viewer - {csv_path.name}"

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: Canvas;
      --fg: CanvasText;
      --muted: color-mix(in srgb, CanvasText 62%, Canvas);
      --line: color-mix(in srgb, CanvasText 18%, Canvas);
      --soft: color-mix(in srgb, CanvasText 7%, Canvas);
      --green: #20a464;
      --green-soft: color-mix(in srgb, #20a464 22%, Canvas);
      --green-strong: color-mix(in srgb, #20a464 72%, CanvasText);
      --blue-soft: color-mix(in srgb, #4a7dff 16%, Canvas);
      --orange-soft: color-mix(in srgb, #d9822b 22%, Canvas);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--fg);
    }}
    main {{
      max-width: 1320px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 23px;
      font-weight: 700;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      align-items: end;
      padding: 16px 0;
      border-bottom: 1px solid var(--line);
      margin-bottom: 16px;
    }}
    label {{
      display: grid;
      gap: 6px;
      font-size: 14px;
      color: var(--muted);
    }}
    input[type="number"] {{
      width: 150px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--bg);
      color: var(--fg);
      font: inherit;
    }}
    input[type="range"] {{
      width: min(520px, 78vw);
    }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 18px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      min-width: 150px;
      background: var(--soft);
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
    }}
    .stat strong {{
      display: block;
      margin-top: 4px;
      font-size: 20px;
      font-weight: 700;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 13px;
    }}
    .swatch {{
      display: inline-block;
      width: 14px;
      height: 14px;
      border-radius: 4px;
      border: 1px solid var(--line);
      vertical-align: -2px;
      margin-right: 5px;
    }}
    .swatch.green {{ background: var(--green-soft); border-color: var(--green); }}
    .swatch.gray {{ background: var(--soft); }}
    .section {{
      margin-top: 22px;
    }}
    .section h2 {{
      margin: 0 0 10px;
      font-size: 17px;
      font-weight: 700;
    }}
    .scenario-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .scenario-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      background: var(--bg);
    }}
    .scenario-header {{
      padding: 12px 14px;
      background: var(--soft);
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
    }}
    .scenario-header h3 {{
      margin: 0;
      font-size: 16px;
      font-weight: 700;
      text-transform: capitalize;
    }}
    .scenario-header span {{
      color: var(--muted);
      font-size: 13px;
    }}
    .stage-list {{
      display: grid;
      gap: 0;
    }}
    .stage {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }}
    .stage:last-child {{ border-bottom: 0; }}
    .stage-title {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
    }}
    .stage-title strong {{
      font-weight: 700;
    }}
    .module-tag {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: var(--soft);
      color: var(--muted);
      white-space: nowrap;
    }}
    .module-tag.reuse {{
      color: var(--green-strong);
      background: var(--green-soft);
      border-color: color-mix(in srgb, var(--green) 55%, var(--line));
      font-weight: 700;
    }}
    .module-tag.base {{
      background: var(--blue-soft);
    }}
    .reuse-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }}
    .reuse-item {{
      border: 1px solid color-mix(in srgb, var(--green) 58%, var(--line));
      border-left: 5px solid var(--green);
      border-radius: 10px;
      padding: 8px 9px;
      background: var(--green-soft);
      min-width: min(100%, 220px);
    }}
    .reuse-item strong {{
      display: block;
      font-size: 13px;
    }}
    .reuse-item span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
    }}
    .no-reuse {{
      color: var(--muted);
      font-size: 13px;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 12px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 880px;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      border-right: 1px solid var(--line);
      padding: 7px 8px;
      text-align: right;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      background: var(--soft);
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    th:first-child, td:first-child {{
      text-align: left;
      position: sticky;
      left: 0;
      z-index: 2;
      background: var(--bg);
    }}
    th:first-child {{ background: var(--soft); z-index: 3; }}
    td.future {{
      color: color-mix(in srgb, var(--muted) 45%, transparent);
      background: repeating-linear-gradient(-45deg, transparent, transparent 6px, var(--soft) 6px, var(--soft) 12px);
    }}
    td.diag {{
      color: var(--muted);
      background: var(--soft);
      text-align: center;
    }}
    td.hit {{
      background: var(--green-soft);
      color: var(--green-strong);
      font-weight: 700;
    }}
    td.near {{
      background: var(--orange-soft);
    }}
    .source {{
      margin-top: 16px;
      font-size: 13px;
      color: var(--muted);
      word-break: break-all;
    }}
    @media (max-width: 980px) {{
      body {{ padding: 16px; }}
      .scenario-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>HalfCheetah Sliced Wasserstein 阈值复用可视化</h1>
    <p>滑动阈值 ε：绿色表示当前任务会复用对应历史模块；每个场景只按该场景的 unique task 顺序考虑已经训练过的前序任务。</p>

    <section class="controls" aria-label="threshold controls">
      <label>
        阈值 ε
        <input id="thresholdInput" type="number" min="0" step="0.0001">
      </label>
      <label>
        调整阈值
        <input id="thresholdRange" type="range" min="0" step="0.0001">
      </label>
    </section>

    <section class="summary" aria-live="polite">
      <div class="stat"><span>全局低于阈值的历史对</span><strong id="globalPairs">0</strong></div>
      <div class="stat"><span>场景内复用次数</span><strong id="scenarioReuseCount">0</strong></div>
      <div class="stat"><span>复用到的模块数</span><strong id="moduleCount">0</strong></div>
      <div class="stat"><span>最小距离</span><strong id="minDistance">-</strong></div>
    </section>

    <div class="legend">
      <span><i class="swatch green"></i>复用模块 / distance &lt; ε</span>
      <span><i class="swatch gray"></i>不复用 / base HybridRARL</span>
    </div>

    <section class="section">
      <h2>场景级模块复用结果</h2>
      <div id="scenarioGrid" class="scenario-grid"></div>
    </section>

    <section class="section">
      <h2>全局下三角距离矩阵</h2>
      <div class="table-wrap">
        <table id="matrixTable" aria-label="lower triangular HalfCheetah SWD matrix"></table>
      </div>
    </section>

    <p class="source">CSV: <span id="csvPath"></span></p>
  </main>

  <script>
    const DATA = {data_json};
    const tasks = DATA.tasks;
    const matrix = DATA.matrix;
    const scenarios = DATA.scenarios;
    const input = document.getElementById("thresholdInput");
    const range = document.getElementById("thresholdRange");
    const scenarioGrid = document.getElementById("scenarioGrid");
    const table = document.getElementById("matrixTable");
    const csvPath = document.getElementById("csvPath");
    const globalPairs = document.getElementById("globalPairs");
    const scenarioReuseCount = document.getElementById("scenarioReuseCount");
    const moduleCount = document.getElementById("moduleCount");
    const minDistance = document.getElementById("minDistance");

    csvPath.textContent = DATA.csvPath;
    range.max = String(Math.max(DATA.maxDistance, DATA.defaultThreshold, 1e-6));
    input.value = Number(DATA.defaultThreshold).toFixed(4);
    range.value = input.value;

    const indexByTaskId = new Map(tasks.map((task, index) => [Number(task.id), index]));

    function taskDisplayByMatrixIndex(index) {{
      const task = tasks[index];
      return task?.display || task?.label || `task ${{index}}`;
    }}

    function fmt(value) {{
      return Number(value).toFixed(6);
    }}

    function getDistance(taskIdA, taskIdB) {{
      const i = indexByTaskId.get(Number(taskIdA));
      const j = indexByTaskId.get(Number(taskIdB));
      if (i === undefined || j === undefined) return null;
      return Number(matrix[i][j]);
    }}

    function modelDirName(taskId, modelType) {{
      return `task_${{taskId}}__${{modelType}}__${{DATA.expName}}__${{DATA.seed}}`;
    }}

    function globalSelectedPairs(threshold) {{
      let count = 0;
      for (let row = 0; row < matrix.length; row += 1) {{
        for (let col = 0; col < row; col += 1) {{
          if (Number(matrix[row][col]) < threshold) count += 1;
        }}
      }}
      return count;
    }}

    function simulateScenario(sequence, threshold) {{
      const produced = new Map();
      const stages = [];
      for (let pos = 0; pos < sequence.length; pos += 1) {{
        const current = sequence[pos];
        const selected = [];
        for (let prevPos = 0; prevPos < pos; prevPos += 1) {{
          const prev = sequence[prevPos];
          const distance = getDistance(current.task_id, prev.task_id);
          if (distance !== null && distance < threshold) {{
            const module = produced.get(prev.task_id);
            selected.push({{
              task_id: prev.task_id,
              task_name: prev.task_name,
              distance,
              module_type: module?.model_type || "hybridrarl",
              module_name: module?.module_name || modelDirName(prev.task_id, "hybridrarl"),
            }});
          }}
        }}
        selected.sort((a, b) => a.distance - b.distance);
        const modelType = selected.length > 0 ? "hybridrarl-prev-units" : "hybridrarl";
        const moduleName = modelDirName(current.task_id, modelType);
        produced.set(current.task_id, {{
          task_id: current.task_id,
          task_name: current.task_name,
          model_type: modelType,
          module_name: moduleName,
        }});
        stages.push({{
          position: pos + 1,
          task_id: current.task_id,
          task_name: current.task_name,
          selected,
          model_type: modelType,
          module_name: moduleName,
        }});
      }}
      return stages;
    }}

    function renderScenarios(threshold) {{
      scenarioGrid.innerHTML = "";
      let reuseEvents = 0;
      const reusedModules = new Set();

      for (const [scenarioName, sequence] of Object.entries(scenarios)) {{
        const stages = simulateScenario(sequence, threshold);
        const card = document.createElement("article");
        card.className = "scenario-card";

        const selectedCount = stages.reduce((sum, stage) => sum + stage.selected.length, 0);
        reuseEvents += selectedCount;
        for (const stage of stages) {{
          for (const item of stage.selected) reusedModules.add(`${{scenarioName}}::${{item.module_name}}`);
        }}

        const header = document.createElement("div");
        header.className = "scenario-header";
        const h3 = document.createElement("h3");
        h3.textContent = scenarioName;
        const summary = document.createElement("span");
        summary.textContent = `${{selectedCount}} reused links / ${{stages.length}} stages`;
        header.append(h3, summary);
        card.appendChild(header);

        const list = document.createElement("div");
        list.className = "stage-list";
        for (const stage of stages) {{
          const item = document.createElement("section");
          item.className = "stage";

          const title = document.createElement("div");
          title.className = "stage-title";
          const name = document.createElement("strong");
          name.textContent = `${{stage.position}}. task ${{stage.task_id}}: ${{stage.task_name}}`;
          const tag = document.createElement("span");
          tag.className = `module-tag ${{stage.selected.length ? "reuse" : "base"}}`;
          tag.textContent = stage.selected.length ? "train prev-units module" : "train base module";
          title.append(name, tag);
          item.appendChild(title);

          const module = document.createElement("span");
          module.className = "module-tag";
          module.textContent = `produces: ${{stage.module_name}}`;
          item.appendChild(module);

          if (stage.selected.length === 0) {{
            const noReuse = document.createElement("p");
            noReuse.className = "no-reuse";
            noReuse.textContent = "不复用前序模块。";
            item.appendChild(noReuse);
          }} else {{
            const reuseList = document.createElement("div");
            reuseList.className = "reuse-list";
            for (const selected of stage.selected) {{
              const reuse = document.createElement("div");
              reuse.className = "reuse-item";
              const strong = document.createElement("strong");
              strong.textContent = `reuse task ${{selected.task_id}}: ${{selected.task_name}}`;
              const dist = document.createElement("span");
              dist.textContent = `distance=${{fmt(selected.distance)}}; module=${{selected.module_name}}`;
              reuse.append(strong, dist);
              reuseList.appendChild(reuse);
            }}
            item.appendChild(reuseList);
          }}

          list.appendChild(item);
        }}
        card.appendChild(list);
        scenarioGrid.appendChild(card);
      }}

      scenarioReuseCount.textContent = String(reuseEvents);
      moduleCount.textContent = String(reusedModules.size);
    }}

    function renderMatrix(threshold) {{
      table.innerHTML = "";
      const thead = document.createElement("thead");
      const headRow = document.createElement("tr");
      const corner = document.createElement("th");
      corner.textContent = "current \\\\ previous";
      headRow.appendChild(corner);
      tasks.forEach((task, index) => {{
        const th = document.createElement("th");
        th.textContent = `T${{task.id ?? index}}`;
        th.title = task.display || task.label;
        headRow.appendChild(th);
      }});
      thead.appendChild(headRow);
      table.appendChild(thead);

      const tbody = document.createElement("tbody");
      for (let row = 0; row < matrix.length; row += 1) {{
        const tr = document.createElement("tr");
        const rowHead = document.createElement("td");
        rowHead.textContent = taskDisplayByMatrixIndex(row);
        tr.appendChild(rowHead);

        for (let col = 0; col < matrix[row].length; col += 1) {{
          const td = document.createElement("td");
          if (col > row) {{
            td.className = "future";
            td.textContent = "—";
          }} else if (col === row) {{
            td.className = "diag";
            td.textContent = "self";
          }} else {{
            const value = Number(matrix[row][col]);
            td.textContent = fmt(value);
            td.title = `${{taskDisplayByMatrixIndex(row)}} vs ${{taskDisplayByMatrixIndex(col)}}`;
            if (value < threshold) td.className = "hit";
            else if (value < threshold * 1.15) td.className = "near";
          }}
          tr.appendChild(td);
        }}
        tbody.appendChild(tr);
      }}
      table.appendChild(tbody);
    }}

    function update() {{
      const threshold = Number(input.value);
      range.value = input.value;
      globalPairs.textContent = String(globalSelectedPairs(threshold));
      minDistance.textContent = fmt(DATA.minDistance);
      renderScenarios(threshold);
      renderMatrix(threshold);
    }}

    input.addEventListener("input", update);
    range.addEventListener("input", () => {{
      input.value = Number(range.value).toFixed(4);
      update();
    }});
    update();
  </script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate an interactive HalfCheetah SWD threshold viewer with "
            "scenario-level module reuse visualization."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="Input distance CSV.")
    parser.add_argument("--output", type=Path, default=Path("analysis/halfcheetah_state_swd_threshold_view.html"))
    parser.add_argument(
        "--tasks-py",
        type=Path,
        default=Path(__file__).resolve().parent / "tasks.py",
        help="HalfCheetah tasks.py used to read tasks and SCENARIOS.",
    )
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=0.11,
        help="Initial epsilon shown by the slider. HalfCheetah training default is 0.11.",
    )
    parser.add_argument("--exp-name", default="SAC_hybridrarl_prev_units")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--open", default=False, action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = args.csv.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    tasks_in_csv, matrix = read_distance_csv(csv_path)
    halfcheetah_tasks, scenarios = load_halfcheetah_config(args.tasks_py.expanduser())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        generate_html(
            csv_path=csv_path,
            tasks_in_csv=tasks_in_csv,
            matrix=matrix,
            halfcheetah_tasks=halfcheetah_tasks,
            scenarios=scenarios,
            default_threshold=float(args.default_threshold),
            exp_name=args.exp_name,
            seed=int(args.seed),
        ),
        encoding="utf-8",
    )
    print(f"loaded csv: {csv_path}")
    print(f"saved viewer: {output_path}")
    if args.open:
        webbrowser.open(output_path.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
