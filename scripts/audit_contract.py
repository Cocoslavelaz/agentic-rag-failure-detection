"""Read-only comparison against the local pre-migration Git revision."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = "daa94d2471ce83065eb7e969d242040556034ba6"


def prompt_values(text):
    return {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(text).body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}


def schema_classes(text):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(text).body
            if isinstance(node, ast.ClassDef) and node.name in {"PlanFormat", "StepTaskFormat", "QAAnswerFormat", "PlanSummaryFormat"}}


def graph_contract(text):
    return re.findall(r'\.add_(node|edge|conditional_edges)\(\s*("[^"\n]+"|START)(?:\s*,\s*("[^"\n]+"|END))?', text)


def main():
    result = {"source_commit": BASE, "source": "local Git objects; no clone", "checks": {}}
    for path in ["src/prompt_template.py", "src/utils.py", "agents/rag.py", "agents/plan_executor.py", "main.py"]:
        old = subprocess.run(["git", "show", f"{BASE}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8")
        new = (ROOT/path).read_text(encoding="utf-8")
        compare = prompt_values if path.endswith("prompt_template.py") else schema_classes if path.endswith("utils.py") else graph_contract
        original = compare(old)
        changed = compare(new)
        result["checks"][path] = {"preserved": original == changed,
            "original_contract_sha256": hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
            "current_contract_sha256": hashlib.sha256(json.dumps(changed, sort_keys=True).encode()).hexdigest()}
    print(json.dumps(result, indent=2))
    if not all(c["preserved"] for c in result["checks"].values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
