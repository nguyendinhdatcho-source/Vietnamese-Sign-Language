"""AST/JSON/notebook validation only; never imports project code."""
import ast
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    paths = list(root.glob("*.py"))
    for directory in ("vsl", "vsl_tf", "tests", "scripts", "vendor"):
        paths.extend((root / directory).rglob("*.py"))
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for path in (root / "configs").glob("*.json"):
        json.loads(path.read_text(encoding="utf-8-sig"))
    for path in (root / "notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8-sig"))
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                if cell["execution_count"] is not None or cell["outputs"]:
                    raise ValueError(f"Notebook contains execution output: {path} cell {i}")
                ast.parse("".join(cell["source"]), filename=f"{path}:cell{i}")
    print(f"Static syntax parsed: {len(paths)} Python files; JSON/notebook valid. No pipeline imported.")


if __name__ == "__main__":
    main()
