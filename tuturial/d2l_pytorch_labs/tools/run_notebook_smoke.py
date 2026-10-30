"""对 Notebook 的所有 Python 代码单元做真实数据 smoke 执行。

代码单元是标准 Python，无 magic。此工具不替代 Jupyter 前端的交互检查。
"""
import gc
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ["LAB_SMOKE"] = "1"
os.environ["MPLBACKEND"] = "Agg"


def main():
    import matplotlib.pyplot as plt
    import torch
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    report = []
    for path in paths:
        print("NOTEBOOK", path.name, flush=True)
        notebook = json.loads(path.read_text(encoding="utf-8"))
        namespace = {"__name__": "notebook_smoke"}
        count = 0
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                print("CELL", index, flush=True)
                exec(compile(cell["source"], str(path) + ":cell" + str(index), "exec"), namespace)
                count += 1
        report.append({"notebook": path.name, "executed_code_cells": count, "status": "passed"})
        namespace.clear()
        plt.close("all")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (ROOT / "docs/notebook_execution.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ALL NOTEBOOK CODE CELLS PASSED", flush=True)


if __name__ == "__main__":
    main()
