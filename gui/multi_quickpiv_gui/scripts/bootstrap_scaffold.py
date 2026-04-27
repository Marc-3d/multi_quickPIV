from pathlib import Path

ROOT = Path.cwd()

DIRS = [
    ROOT / "docs",
    ROOT / "scripts",
    ROOT / "src" / "multi_quickpiv_gui",
    ROOT / "src" / "multi_quickpiv_gui" / "gui",
    ROOT / "src" / "multi_quickpiv_gui" / "workflow",
    ROOT / "src" / "multi_quickpiv_gui" / "backend",
    ROOT / "src" / "multi_quickpiv_gui" / "runtime",
    ROOT / "tests",
    ROOT / "benchmarks",
]

FILES = {
    ROOT / "README.md": "# multi_quickPIV GUI\n\nGUI wrapper around the Julia `multi_quickPIV` backend.\n",
    ROOT / ".gitignore": """__pycache__/
*.py[cod]
*.egg-info/
.venv/
.env
.ipynb_checkpoints/
dist/
build/
.mypy_cache/
.pytest_cache/
.DS_Store
""",
    ROOT / "pyproject.toml": """[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "multi-quickpiv-gui"
version = "0.1.0"
description = "Python GUI for the Julia multi_quickPIV backend"
readme = "README.md"
requires-python = ">=3.10"
dependencies = []

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
""",
    ROOT / "src" / "multi_quickpiv_gui" / "__init__.py": '"""Top-level package for the multi_quickPIV GUI."""\n',
    ROOT / "src" / "multi_quickpiv_gui" / "gui" / "__init__.py": "",
    ROOT / "src" / "multi_quickpiv_gui" / "gui" / "app.py": '''"""GUI application shell."""\n\n\ndef main() -> None:\n    """Entry point for the GUI application."""\n    pass\n''',
    ROOT / "src" / "multi_quickpiv_gui" / "gui" / "preview.py": '''"""Preview and visualization components."""\n''',
    ROOT / "src" / "multi_quickpiv_gui" / "workflow" / "__init__.py": "",
    ROOT / "src" / "multi_quickpiv_gui" / "workflow" / "params.py": '''"""Parameter models for GUI and processing workflows."""\n''',
    ROOT / "src" / "multi_quickpiv_gui" / "workflow" / "project.py": '''"""Project persistence helpers."""\n''',
    ROOT / "src" / "multi_quickpiv_gui" / "workflow" / "pipeline.py": '''"""Single-run and batch workflow orchestration."""\n''',
    ROOT / "src" / "multi_quickpiv_gui" / "backend" / "__init__.py": "",
    ROOT / "src" / "multi_quickpiv_gui" / "backend" / "core.py": '''"""GUI-independent backend API."""\n''',
    ROOT / "src" / "multi_quickpiv_gui" / "backend" / "julia_bridge.py": '''"""Bridge to the Julia multi_quickPIV backend."""\n''',
    ROOT / "src" / "multi_quickpiv_gui" / "backend" / "io.py": '''"""Input/output helpers for stacks and saved results."""\n''',
    ROOT / "src" / "multi_quickpiv_gui" / "backend" / "export.py": '''"""Export helpers for NPZ, HDF5, and media outputs."""\n''',
    ROOT / "src" / "multi_quickpiv_gui" / "runtime" / "__init__.py": "",
    ROOT / "src" / "multi_quickpiv_gui" / "runtime" / "worker.py": '''"""Worker execution helpers for background tasks."""\n''',
    ROOT / "src" / "multi_quickpiv_gui" / "runtime" / "batch.py": '''"""Batch execution orchestration."""\n''',
}

def main() -> None:
    for directory in DIRS:
        directory.mkdir(parents=True, exist_ok=True)

    for path, content in FILES.items():
        if not path.exists():
            path.write_text(content, encoding="utf-8")

    print("Scaffold created successfully.")

if __name__ == "__main__":
    main()