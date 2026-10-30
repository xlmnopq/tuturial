"""校验 Notebook 格式和所有代码单元的 Python 语法。"""
import json
from pathlib import Path
import nbformat

root = Path(__file__).absolute().parents[1]
for path in sorted((root / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(str(path), as_version=4)
    nbformat.validate(notebook)
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, str(path) + ":cell" + str(i), "exec")
    print("OK", path.name)
