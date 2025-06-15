import os
import importlib
from pathlib import Path

def load_all_modules():
    package_dir = Path(os.path.dirname(__file__))

    for path in package_dir.rglob("*.py"):
        if path.name == "__init__.py":
            continue

        relative_path = path.relative_to(package_dir.parent)
        module_name = ".".join(relative_path.with_suffix("").parts)

        try:
            importlib.import_module(module_name)
        except ImportError as e:
            print(f"[WARNING] Не удалось загрузить модуль {module_name}: {e}")

load_all_modules()