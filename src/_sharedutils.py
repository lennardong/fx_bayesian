import inspect
from pathlib import Path


def find_project_root(current_path: Path = Path(__file__).resolve().parent) -> Path:
    """
    Find the project root by looking for a .git directory or a config.ini file.
    """
    if (current_path / ".git").exists() or (current_path / "config.ini").exists():
        return current_path
    parent = current_path.parent
    if parent == current_path:
        raise FileNotFoundError("Project root not found")
    return find_project_root(parent)


def get_function_name():
    return "\n# " + inspect.currentframe().f_back.f_code.co_name + "()\n"
